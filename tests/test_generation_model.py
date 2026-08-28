import pandas as pd
import pytest
import torch

from aptafind.generation.chemistry import MoleculeFeaturizer
from aptafind.generation.data import AptamerSequenceDataset
from aptafind.generation.demo import create_synthetic_aptamer_table
from aptafind.generation.model import (
    ConditionalSequenceVAE,
    SequenceCVAEConfig,
    SequenceCVAEOutput,
    sequence_cvae_loss,
)
from aptafind.generation.tokenizer import DNATokenizer
from aptafind.generation.training import evaluate_condition_controls


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


def test_free_bits_report_raw_and_effective_kl_separately() -> None:
    output = SequenceCVAEOutput(
        logits=torch.zeros((2, 3, 7)),
        mean=torch.zeros((2, 4)),
        log_variance=torch.zeros((2, 4)),
    )
    targets = torch.ones((2, 3), dtype=torch.long)

    loss = sequence_cvae_loss(
        output,
        targets,
        pad_token_id=0,
        beta=0.1,
        free_bits_per_dimension=0.05,
    )

    assert float(loss.kl_divergence) == pytest.approx(0.0)
    assert float(loss.effective_kl_divergence) == pytest.approx(0.2)
    assert float(loss.loss) == pytest.approx(
        float(loss.reconstruction_loss) + 0.02
    )


def test_condition_controls_use_deranged_target_groups() -> None:
    frame: pd.DataFrame = create_synthetic_aptamer_table(
        samples_per_target=2, sequence_length=16, seed=31
    )
    tokenizer = DNATokenizer(maximum_sequence_length=16)
    featurizer = MoleculeFeaturizer(fingerprint_bits=16).fit(
        frame["target_smiles"]
    )
    dataset = AptamerSequenceDataset(frame, tokenizer, featurizer)
    model = ConditionalSequenceVAE(
        SequenceCVAEConfig(
            vocabulary_size=tokenizer.vocabulary_size,
            condition_dimension=featurizer.condition_dimension,
            pad_token_id=tokenizer.pad_id,
            embedding_dim=8,
            encoder_hidden_dim=8,
            decoder_hidden_dim=12,
            condition_hidden_dim=6,
            latent_dim=4,
            dropout=0.0,
        )
    )

    diagnostics = evaluate_condition_controls(
        model,
        dataset,
        batch_size=4,
        device=torch.device("cpu"),
        beta=0.01,
        seed=37,
        permutations=2,
    )

    assert diagnostics["summary"]["available"]
    assert diagnostics["summary"]["unique_condition_groups"] == 4
    assert len(diagnostics["target_permutations"]) == 2
    assert diagnostics["matched"]["sample_count"] == len(frame)
