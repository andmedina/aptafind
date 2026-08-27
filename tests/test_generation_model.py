import torch

from aptafind.generation.model import (
    ConditionalSequenceVAE,
    SequenceCVAEConfig,
    sequence_cvae_loss,
)
from aptafind.generation.tokenizer import DNATokenizer


def test_sequence_cvae_forward_backward_and_sampling() -> None:
    torch.manual_seed(3)
    tokenizer = DNATokenizer(maximum_sequence_length=8)
    encoded = [tokenizer.encode(value) for value in ("ACGT", "TGCATG")]
    token_ids = torch.tensor([value[0] for value in encoded], dtype=torch.long)
    lengths = torch.tensor([value[1] for value in encoded], dtype=torch.long)
    conditions = torch.randn(2, 12)
    model = ConditionalSequenceVAE(
        SequenceCVAEConfig(
            vocabulary_size=tokenizer.vocabulary_size,
            condition_dimension=12,
            pad_token_id=tokenizer.pad_id,
            embedding_dim=8,
            encoder_hidden_dim=8,
            decoder_hidden_dim=12,
            condition_hidden_dim=6,
            latent_dim=4,
            dropout=0.0,
        )
    )

    output = model(token_ids, lengths, conditions)
    loss = sequence_cvae_loss(
        output,
        token_ids[:, 1:],
        pad_token_id=tokenizer.pad_id,
        beta=0.1,
    )
    loss.loss.backward()

    assert output.logits.shape == (2, tokenizer.encoded_length - 1, 7)
    assert torch.isfinite(loss.loss)
    assert model.embedding.weight.grad is not None

    sampled = model.sample_prior(
        conditions,
        bos_token_id=tokenizer.bos_id,
        eos_token_id=tokenizer.eos_id,
        maximum_sequence_length=8,
        minimum_sequence_length=3,
        top_k=5,
    )
    decoded = [tokenizer.decode(row) for row in sampled]
    assert len(decoded) == 2
    assert all(3 <= len(sequence) <= 8 for sequence in decoded)
    assert all(set(sequence) <= set("ACGT") for sequence in decoded)
