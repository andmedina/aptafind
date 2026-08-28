"""Deterministic training and evaluation utilities for the sequence CVAE."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from aptafind.generation.data import AptamerSequenceDataset
from aptafind.generation.model import ConditionalSequenceVAE, sequence_cvae_loss


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 40
    batch_size: int = 16
    learning_rate: float = 1e-3
    beta_max: float = 0.05
    beta_warmup_epochs: int = 15
    free_bits_per_dimension: float = 0.0
    decoder_token_dropout: float = 0.0
    permute_training_targets: bool = False
    condition_diagnostic_permutations: int = 5
    patience: int = 8
    gradient_clip_norm: float = 5.0
    seed: int = 42
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "patience"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.beta_warmup_epochs < 0:
            raise ValueError("beta_warmup_epochs cannot be negative.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.beta_max < 0:
            raise ValueError("beta_max cannot be negative.")
        if self.free_bits_per_dimension < 0:
            raise ValueError("free_bits_per_dimension cannot be negative.")
        if not 0.0 <= self.decoder_token_dropout < 1.0:
            raise ValueError("decoder_token_dropout must be in [0, 1).")
        if self.condition_diagnostic_permutations < 1:
            raise ValueError("condition_diagnostic_permutations must be positive.")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive.")

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpochMetrics:
    loss: float
    reconstruction_loss: float
    kl_divergence: float
    effective_kl_divergence: float
    token_accuracy: float
    perplexity: float
    sample_count: int
    token_count: int
    active_latent_units: int
    posterior_mean_variance_mean: float
    posterior_mean_variance_max: float

    def state_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass
class TrainingResult:
    history: list[dict[str, Any]]
    best_epoch: int
    stopped_epoch: int
    best_validation_loss: float


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch generators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Resolve a requested device and fail instead of silently changing it."""

    normalized = requested.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if normalized == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS was requested but is unavailable.")
    if normalized not in ("cpu", "cuda", "mps"):
        raise ValueError("device must be one of: auto, cpu, cuda, mps.")
    return torch.device(normalized)


def beta_for_epoch(epoch: int, config: TrainingConfig) -> float:
    """Linearly warm the KL term to reduce posterior collapse pressure."""

    if config.beta_warmup_epochs == 0:
        return config.beta_max
    return config.beta_max * min(1.0, epoch / config.beta_warmup_epochs)


def _run_epoch(
    model: ConditionalSequenceVAE,
    loader: DataLoader[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    beta: float,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float,
    free_bits_per_dimension: float = 0.0,
    decoder_token_dropout: float = 0.0,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    reconstruction_sum = 0.0
    kl_sum = 0.0
    effective_kl_sum = 0.0
    correct_token_sum = 0.0
    token_count_sum = 0
    sample_count_sum = 0
    posterior_mean_sum: torch.Tensor | None = None
    posterior_mean_square_sum: torch.Tensor | None = None

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            token_ids = batch["token_ids"].to(device)
            lengths = batch["length"].to(device)
            conditions = batch["condition"].to(device)
            decoder_conditions = batch.get("decoder_condition")
            if decoder_conditions is not None:
                decoder_conditions = decoder_conditions.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)

            output = model(
                token_ids,
                lengths,
                conditions,
                sample_latent=training,
                decoder_conditions=decoder_conditions,
                decoder_token_dropout=(decoder_token_dropout if training else 0.0),
            )
            loss = sequence_cvae_loss(
                output,
                token_ids[:, 1:],
                pad_token_id=model.config.pad_token_id,
                beta=beta,
                free_bits_per_dimension=free_bits_per_dimension,
            )
            if training:
                loss.loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()

            batch_token_count = int(loss.token_count.detach().cpu())
            batch_size = int(token_ids.shape[0])
            reconstruction_sum += (
                float(loss.reconstruction_loss.detach().cpu()) * batch_token_count
            )
            kl_sum += float(loss.kl_divergence.detach().cpu()) * batch_size
            effective_kl_sum += (
                float(loss.effective_kl_divergence.detach().cpu()) * batch_size
            )
            correct_token_sum += (
                float(loss.token_accuracy.detach().cpu()) * batch_token_count
            )
            token_count_sum += batch_token_count
            sample_count_sum += batch_size
            batch_mean = output.mean.detach().cpu()
            if posterior_mean_sum is None:
                posterior_mean_sum = batch_mean.sum(dim=0)
                posterior_mean_square_sum = batch_mean.pow(2).sum(dim=0)
            else:
                posterior_mean_sum += batch_mean.sum(dim=0)
                assert posterior_mean_square_sum is not None
                posterior_mean_square_sum += batch_mean.pow(2).sum(dim=0)

    if sample_count_sum == 0 or token_count_sum == 0:
        raise ValueError("Cannot evaluate an empty dataset partition.")
    reconstruction_loss = reconstruction_sum / token_count_sum
    kl_divergence = kl_sum / sample_count_sum
    effective_kl_divergence = effective_kl_sum / sample_count_sum
    assert posterior_mean_sum is not None
    assert posterior_mean_square_sum is not None
    posterior_mean_variance = (
        posterior_mean_square_sum / sample_count_sum
        - (posterior_mean_sum / sample_count_sum).pow(2)
    ).clamp_min(0.0)
    total_loss = reconstruction_loss + beta * effective_kl_divergence
    return EpochMetrics(
        loss=total_loss,
        reconstruction_loss=reconstruction_loss,
        kl_divergence=kl_divergence,
        effective_kl_divergence=effective_kl_divergence,
        token_accuracy=correct_token_sum / token_count_sum,
        perplexity=math.exp(min(reconstruction_loss, 20.0)),
        sample_count=sample_count_sum,
        token_count=token_count_sum,
        active_latent_units=int((posterior_mean_variance > 0.01).sum()),
        posterior_mean_variance_mean=float(posterior_mean_variance.mean()),
        posterior_mean_variance_max=float(posterior_mean_variance.max()),
    )


def evaluate_model(
    model: ConditionalSequenceVAE,
    loader: DataLoader[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    beta: float,
    free_bits_per_dimension: float = 0.0,
) -> EpochMetrics:
    """Evaluate deterministically by decoding from the posterior mean."""

    return _run_epoch(
        model,
        loader,
        device=device,
        beta=beta,
        optimizer=None,
        gradient_clip_norm=1.0,
        free_bits_per_dimension=free_bits_per_dimension,
    )


def evaluate_reconstruction_examples(
    model: ConditionalSequenceVAE,
    dataset: AptamerSequenceDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, list[float] | list[int]]:
    """Return deterministic posterior reconstruction totals for each example."""

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    nll_sums: list[float] = []
    token_counts: list[int] = []
    with torch.no_grad():
        for batch in loader:
            token_ids = batch["token_ids"].to(device)
            lengths = batch["length"].to(device)
            conditions = batch["condition"].to(device)
            output = model(
                token_ids,
                lengths,
                conditions,
                sample_latent=False,
            )
            targets = token_ids[:, 1:]
            token_mask = targets.ne(model.config.pad_token_id)
            token_losses = F.cross_entropy(
                output.logits.transpose(1, 2),
                targets,
                ignore_index=model.config.pad_token_id,
                reduction="none",
            )
            nll_sums.extend(
                (token_losses * token_mask).sum(dim=1).detach().cpu().tolist()
            )
            token_counts.extend(token_mask.sum(dim=1).detach().cpu().tolist())
    return {"nll_sums": nll_sums, "token_counts": token_counts}


class _ConditionControlDataset(Dataset[dict[str, torch.Tensor]]):
    """Reuse tokens while supplying controlled encoder/decoder conditions."""

    def __init__(
        self,
        base: AptamerSequenceDataset,
        *,
        encoder_conditions: torch.Tensor,
        decoder_conditions: torch.Tensor | None = None,
    ) -> None:
        if encoder_conditions.shape != base.conditions.shape:
            raise ValueError("Controlled encoder conditions have the wrong shape.")
        if (
            decoder_conditions is not None
            and decoder_conditions.shape != base.conditions.shape
        ):
            raise ValueError("Controlled decoder conditions have the wrong shape.")
        self.base = base
        self.encoder_conditions = encoder_conditions
        self.decoder_conditions = decoder_conditions

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = {
            "token_ids": self.base.token_ids[index],
            "length": self.base.lengths[index],
            "condition": self.encoder_conditions[index],
        }
        if self.decoder_conditions is not None:
            row["decoder_condition"] = self.decoder_conditions[index]
        return row


def _condition_derangement(
    conditions: torch.Tensor, *, seed: int
) -> tuple[torch.Tensor, int]:
    unique_conditions, inverse = torch.unique(
        conditions, dim=0, sorted=True, return_inverse=True
    )
    unique_count = int(unique_conditions.shape[0])
    if unique_count < 2:
        raise ValueError("At least two unique conditions are required.")
    generator = torch.Generator().manual_seed(seed)
    identity = torch.arange(unique_count)
    for _ in range(10_000):
        permutation = torch.randperm(unique_count, generator=generator)
        if bool(torch.all(permutation != identity)):
            break
    else:  # pragma: no cover - a derangement exists for every n >= 2.
        raise RuntimeError("Unable to construct a condition derangement.")
    return unique_conditions[permutation][inverse], unique_count


def _control_loader(
    dataset: AptamerSequenceDataset,
    *,
    encoder_conditions: torch.Tensor,
    decoder_conditions: torch.Tensor | None,
    batch_size: int,
) -> DataLoader[dict[str, torch.Tensor]]:
    controlled = _ConditionControlDataset(
        dataset,
        encoder_conditions=encoder_conditions,
        decoder_conditions=decoder_conditions,
    )
    return DataLoader(controlled, batch_size=batch_size, shuffle=False, num_workers=0)


def evaluate_condition_controls(
    model: ConditionalSequenceVAE,
    dataset: AptamerSequenceDataset,
    *,
    batch_size: int,
    device: torch.device,
    beta: float,
    free_bits_per_dimension: float = 0.0,
    seed: int = 42,
    permutations: int = 5,
) -> dict[str, Any]:
    """Measure reconstruction sensitivity to zeroed and wrong target conditions.

    ``full`` controls replace the condition in both the posterior encoder and
    decoder. ``decoder_only`` holds the sequence posterior fixed and changes
    only the condition delivered to the decoder, isolating decoder dependence.
    """

    if permutations < 1:
        raise ValueError("permutations must be positive.")
    matched_loader = _control_loader(
        dataset,
        encoder_conditions=dataset.conditions,
        decoder_conditions=None,
        batch_size=batch_size,
    )
    matched = evaluate_model(
        model,
        matched_loader,
        device=device,
        beta=beta,
        free_bits_per_dimension=free_bits_per_dimension,
    )
    zero_conditions = torch.zeros_like(dataset.conditions)
    zero_full = evaluate_model(
        model,
        _control_loader(
            dataset,
            encoder_conditions=zero_conditions,
            decoder_conditions=None,
            batch_size=batch_size,
        ),
        device=device,
        beta=beta,
        free_bits_per_dimension=free_bits_per_dimension,
    )
    zero_decoder = evaluate_model(
        model,
        _control_loader(
            dataset,
            encoder_conditions=dataset.conditions,
            decoder_conditions=zero_conditions,
            batch_size=batch_size,
        ),
        device=device,
        beta=beta,
        free_bits_per_dimension=free_bits_per_dimension,
    )

    unique_condition_count = int(torch.unique(dataset.conditions, dim=0).shape[0])
    if unique_condition_count < 2:
        return {
            "matched": matched.state_dict(),
            "zero_vector": {
                "full": zero_full.state_dict(),
                "decoder_only": zero_decoder.state_dict(),
                "full_reconstruction_nll_delta": (
                    zero_full.reconstruction_loss - matched.reconstruction_loss
                ),
                "decoder_only_reconstruction_nll_delta": (
                    zero_decoder.reconstruction_loss - matched.reconstruction_loss
                ),
            },
            "target_permutations": [],
            "summary": {
                "available": False,
                "reason": "At least two unique condition groups are required.",
                "permutations": 0,
                "unique_condition_groups": unique_condition_count,
                "full_reconstruction_nll_delta_mean": None,
                "full_reconstruction_nll_delta_std": None,
                "decoder_only_reconstruction_nll_delta_mean": None,
                "decoder_only_reconstruction_nll_delta_std": None,
            },
        }

    permutation_rows: list[dict[str, Any]] = []
    for index in range(permutations):
        permutation_seed = seed + index
        permuted, unique_condition_count = _condition_derangement(
            dataset.conditions, seed=permutation_seed
        )
        full = evaluate_model(
            model,
            _control_loader(
                dataset,
                encoder_conditions=permuted,
                decoder_conditions=None,
                batch_size=batch_size,
            ),
            device=device,
            beta=beta,
            free_bits_per_dimension=free_bits_per_dimension,
        )
        decoder_only = evaluate_model(
            model,
            _control_loader(
                dataset,
                encoder_conditions=dataset.conditions,
                decoder_conditions=permuted,
                batch_size=batch_size,
            ),
            device=device,
            beta=beta,
            free_bits_per_dimension=free_bits_per_dimension,
        )
        permutation_rows.append(
            {
                "seed": permutation_seed,
                "full": full.state_dict(),
                "decoder_only": decoder_only.state_dict(),
                "full_reconstruction_nll_delta": (
                    full.reconstruction_loss - matched.reconstruction_loss
                ),
                "decoder_only_reconstruction_nll_delta": (
                    decoder_only.reconstruction_loss - matched.reconstruction_loss
                ),
            }
        )

    full_deltas = np.asarray(
        [row["full_reconstruction_nll_delta"] for row in permutation_rows]
    )
    decoder_deltas = np.asarray(
        [row["decoder_only_reconstruction_nll_delta"] for row in permutation_rows]
    )
    return {
        "matched": matched.state_dict(),
        "zero_vector": {
            "full": zero_full.state_dict(),
            "decoder_only": zero_decoder.state_dict(),
            "full_reconstruction_nll_delta": (
                zero_full.reconstruction_loss - matched.reconstruction_loss
            ),
            "decoder_only_reconstruction_nll_delta": (
                zero_decoder.reconstruction_loss - matched.reconstruction_loss
            ),
        },
        "target_permutations": permutation_rows,
        "summary": {
            "available": True,
            "permutations": permutations,
            "unique_condition_groups": unique_condition_count,
            "full_reconstruction_nll_delta_mean": float(full_deltas.mean()),
            "full_reconstruction_nll_delta_std": float(full_deltas.std()),
            "decoder_only_reconstruction_nll_delta_mean": float(
                decoder_deltas.mean()
            ),
            "decoder_only_reconstruction_nll_delta_std": float(
                decoder_deltas.std()
            ),
        },
    }


def train_model(
    model: ConditionalSequenceVAE,
    train_loader: DataLoader[dict[str, torch.Tensor]],
    validation_loader: DataLoader[dict[str, torch.Tensor]],
    config: TrainingConfig,
) -> TrainingResult:
    """Train with KL warmup and validation-loss early stopping."""

    seed_everything(config.seed)
    device = resolve_device(config.device)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_epoch = config.epochs

    for epoch in range(1, config.epochs + 1):
        beta = beta_for_epoch(epoch, config)
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device,
            beta=beta,
            optimizer=optimizer,
            gradient_clip_norm=config.gradient_clip_norm,
            free_bits_per_dimension=config.free_bits_per_dimension,
            decoder_token_dropout=config.decoder_token_dropout,
        )
        validation_metrics = evaluate_model(
            model,
            validation_loader,
            device=device,
            beta=beta,
            free_bits_per_dimension=config.free_bits_per_dimension,
        )
        history.append(
            {
                "epoch": epoch,
                "beta": beta,
                "train": train_metrics.state_dict(),
                "validation": validation_metrics.state_dict(),
            }
        )

        if validation_metrics.loss < best_validation_loss - 1e-8:
            best_validation_loss = validation_metrics.loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                stopped_epoch = epoch
                break

    if best_state is None:
        raise RuntimeError("Training completed without a valid model state.")
    model.load_state_dict(best_state)
    return TrainingResult(
        history=history,
        best_epoch=best_epoch,
        stopped_epoch=stopped_epoch,
        best_validation_loss=best_validation_loss,
    )
