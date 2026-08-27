"""PyTorch conditional variational autoencoder for ssDNA tokens."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence


@dataclass(frozen=True)
class SequenceCVAEConfig:
    """Serializable architecture configuration for :class:`ConditionalSequenceVAE`."""

    vocabulary_size: int
    condition_dimension: int
    pad_token_id: int = 0
    embedding_dim: int = 32
    encoder_hidden_dim: int = 64
    decoder_hidden_dim: int = 128
    condition_hidden_dim: int = 32
    latent_dim: int = 16
    dropout: float = 0.10

    def __post_init__(self) -> None:
        integer_fields = (
            "vocabulary_size",
            "condition_dimension",
            "embedding_dim",
            "encoder_hidden_dim",
            "decoder_hidden_dim",
            "condition_hidden_dim",
            "latent_dim",
        )
        for field_name in integer_fields:
            if int(getattr(self, field_name)) < 1:
                raise ValueError(f"{field_name} must be positive.")
        if not 0 <= self.pad_token_id < self.vocabulary_size:
            raise ValueError("pad_token_id must be within the vocabulary.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "SequenceCVAEConfig":
        return cls(**state)


@dataclass
class SequenceCVAEOutput:
    """Output tensors from a training-time CVAE forward pass."""

    logits: Tensor
    mean: Tensor
    log_variance: Tensor


@dataclass
class SequenceCVAELoss:
    """Differentiable loss components and detached reporting metrics."""

    loss: Tensor
    reconstruction_loss: Tensor
    kl_divergence: Tensor
    token_accuracy: Tensor
    token_count: Tensor


class ConditionalSequenceVAE(nn.Module):
    """Generate variable-length DNA directly while conditioning on a molecule.

    The bidirectional encoder maps a known sequence and target condition to a
    posterior distribution. The autoregressive decoder receives a sampled
    latent vector plus the same target condition at every nucleotide step.
    """

    def __init__(self, config: SequenceCVAEConfig) -> None:
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(
            config.vocabulary_size,
            config.embedding_dim,
            padding_idx=config.pad_token_id,
        )
        self.encoder = nn.GRU(
            input_size=config.embedding_dim,
            hidden_size=config.encoder_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.condition_encoder = nn.Sequential(
            nn.Linear(config.condition_dimension, config.condition_hidden_dim),
            nn.LayerNorm(config.condition_hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )

        posterior_input_dim = (
            2 * config.encoder_hidden_dim + config.condition_hidden_dim
        )
        self.posterior_mean = nn.Linear(posterior_input_dim, config.latent_dim)
        self.posterior_log_variance = nn.Linear(
            posterior_input_dim, config.latent_dim
        )

        context_dim = config.latent_dim + config.condition_hidden_dim
        self.decoder_initial_state = nn.Linear(context_dim, config.decoder_hidden_dim)
        self.decoder = nn.GRU(
            input_size=config.embedding_dim + context_dim,
            hidden_size=config.decoder_hidden_dim,
            batch_first=True,
        )
        self.output_projection = nn.Linear(
            config.decoder_hidden_dim, config.vocabulary_size
        )

    def encode(
        self,
        token_ids: Tensor,
        lengths: Tensor,
        conditions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode a batch and return posterior mean, log variance, and condition."""

        embedded = self.embedding(token_ids)
        packed = pack_padded_sequence(
            embedded,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.encoder(packed)
        sequence_representation = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        condition_representation = self.condition_encoder(conditions)
        posterior_input = torch.cat(
            [sequence_representation, condition_representation], dim=-1
        )
        mean = self.posterior_mean(posterior_input)
        log_variance = self.posterior_log_variance(posterior_input).clamp(-12.0, 12.0)
        return mean, log_variance, condition_representation

    @staticmethod
    def reparameterize(mean: Tensor, log_variance: Tensor) -> Tensor:
        """Draw a differentiable posterior sample."""

        standard_deviation = torch.exp(0.5 * log_variance)
        return mean + torch.randn_like(standard_deviation) * standard_deviation

    def decode_teacher_forced(
        self,
        decoder_input_ids: Tensor,
        latent: Tensor,
        condition_representation: Tensor,
    ) -> Tensor:
        """Decode with known preceding tokens and return vocabulary logits."""

        context = torch.cat([latent, condition_representation], dim=-1)
        embedded = self.embedding(decoder_input_ids)
        repeated_context = context.unsqueeze(1).expand(-1, embedded.shape[1], -1)
        decoder_input = torch.cat([embedded, repeated_context], dim=-1)
        initial_state = torch.tanh(self.decoder_initial_state(context)).unsqueeze(0)
        decoded, _ = self.decoder(decoder_input, initial_state)
        return self.output_projection(decoded)

    def forward(
        self,
        token_ids: Tensor,
        lengths: Tensor,
        conditions: Tensor,
        *,
        sample_latent: bool = True,
    ) -> SequenceCVAEOutput:
        """Run the posterior encoder and teacher-forced decoder."""

        mean, log_variance, condition_representation = self.encode(
            token_ids, lengths, conditions
        )
        latent = (
            self.reparameterize(mean, log_variance) if sample_latent else mean
        )
        logits = self.decode_teacher_forced(
            token_ids[:, :-1], latent, condition_representation
        )
        return SequenceCVAEOutput(
            logits=logits,
            mean=mean,
            log_variance=log_variance,
        )

    @torch.no_grad()
    def sample_prior(
        self,
        conditions: Tensor,
        *,
        bos_token_id: int,
        eos_token_id: int,
        maximum_sequence_length: int,
        minimum_sequence_length: int = 1,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> list[list[int]]:
        """Autoregressively sample token lists from the conditioned prior."""

        if conditions.ndim != 2:
            raise ValueError("conditions must have shape (batch, condition_dimension).")
        if conditions.shape[1] != self.config.condition_dimension:
            raise ValueError("Condition dimension does not match the model.")
        if maximum_sequence_length < 1:
            raise ValueError("maximum_sequence_length must be positive.")
        if not 0 <= minimum_sequence_length <= maximum_sequence_length:
            raise ValueError(
                "minimum_sequence_length must be between zero and the maximum."
            )
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive when supplied.")

        batch_size = conditions.shape[0]
        device = conditions.device
        condition_representation = self.condition_encoder(conditions)
        latent = torch.randn(batch_size, self.config.latent_dim, device=device)
        context = torch.cat([latent, condition_representation], dim=-1)
        hidden = torch.tanh(self.decoder_initial_state(context)).unsqueeze(0)

        current_tokens = torch.full(
            (batch_size,), bos_token_id, dtype=torch.long, device=device
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        generated: list[list[int]] = [[bos_token_id] for _ in range(batch_size)]

        for base_index in range(maximum_sequence_length + 1):
            if base_index == maximum_sequence_length:
                sampled = torch.full_like(current_tokens, eos_token_id)
            else:
                embedded = self.embedding(current_tokens).unsqueeze(1)
                decoder_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
                decoded, hidden = self.decoder(decoder_input, hidden)
                logits = self.output_projection(decoded[:, 0, :]) / temperature
                logits[:, self.config.pad_token_id] = -torch.inf
                logits[:, bos_token_id] = -torch.inf
                if base_index < minimum_sequence_length:
                    logits[:, eos_token_id] = -torch.inf

                if top_k is not None and top_k < logits.shape[-1]:
                    top_values = torch.topk(logits, k=top_k, dim=-1).values
                    cutoff = top_values[:, -1].unsqueeze(-1)
                    logits = logits.masked_fill(logits < cutoff, -torch.inf)

                probabilities = torch.softmax(logits, dim=-1)
                sampled = torch.multinomial(probabilities, num_samples=1).squeeze(1)

            sampled = torch.where(
                finished, torch.full_like(sampled, eos_token_id), sampled
            )
            for row_index, token_id in enumerate(sampled.detach().cpu().tolist()):
                if not bool(finished[row_index]):
                    generated[row_index].append(int(token_id))
            finished = finished | sampled.eq(eos_token_id)
            current_tokens = sampled
            if bool(finished.all()):
                break

        return generated


def sequence_cvae_loss(
    output: SequenceCVAEOutput,
    target_token_ids: Tensor,
    *,
    pad_token_id: int,
    beta: float,
) -> SequenceCVAELoss:
    """Compute token reconstruction loss plus beta-weighted KL divergence."""

    vocabulary_size = output.logits.shape[-1]
    flat_logits = output.logits.reshape(-1, vocabulary_size)
    flat_targets = target_token_ids.reshape(-1)
    token_mask = flat_targets.ne(pad_token_id)
    token_count = token_mask.sum().clamp_min(1)

    reconstruction_sum = F.cross_entropy(
        flat_logits,
        flat_targets,
        ignore_index=pad_token_id,
        reduction="sum",
    )
    reconstruction_loss = reconstruction_sum / token_count
    kl_per_sample = -0.5 * torch.sum(
        1.0
        + output.log_variance
        - output.mean.pow(2)
        - output.log_variance.exp(),
        dim=-1,
    )
    kl_divergence = kl_per_sample.mean()
    loss = reconstruction_loss + float(beta) * kl_divergence

    predictions = output.logits.argmax(dim=-1).reshape(-1)
    correct = predictions.eq(flat_targets) & token_mask
    token_accuracy = correct.sum() / token_count
    return SequenceCVAELoss(
        loss=loss,
        reconstruction_loss=reconstruction_loss,
        kl_divergence=kl_divergence,
        token_accuracy=token_accuracy,
        token_count=token_count,
    )
