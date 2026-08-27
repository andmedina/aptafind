from pathlib import Path

import pandas as pd

from aptafind.generation.baselines import (
    evaluate_bigram_baseline,
    evaluate_unigram_baseline,
)
from aptafind.generation.chemistry import MoleculeFeaturizer
from aptafind.generation.data import (
    AptamerSequenceDataset,
    load_aptamer_table,
    split_aptamer_table,
)
from aptafind.generation.demo import write_synthetic_aptamer_table
from aptafind.generation.tokenizer import DNATokenizer


def test_molecule_featurizer_round_trip_has_stable_schema() -> None:
    featurizer = MoleculeFeaturizer(fingerprint_bits=32).fit(["CCO", "CC(=O)O"])

    condition = featurizer.transform_one("OCC")
    restored = MoleculeFeaturizer.from_state_dict(featurizer.state_dict())

    assert condition.shape == (40,)
    assert restored.transform_one("CCO").tolist() == condition.tolist()


def test_target_split_has_no_molecule_overlap(tmp_path: Path) -> None:
    dataset_path = write_synthetic_aptamer_table(
        tmp_path / "demo.csv", samples_per_target=4, sequence_length=24
    )
    loaded = load_aptamer_table(dataset_path)

    partitions = split_aptamer_table(
        loaded.frame,
        validation_fraction=0.25,
        test_fraction=0.25,
        seed=5,
        strategy="target",
    )

    train_targets = set(partitions.train["target_smiles"])
    validation_targets = set(partitions.validation["target_smiles"])
    test_targets = set(partitions.test["target_smiles"])
    assert train_targets.isdisjoint(validation_targets)
    assert train_targets.isdisjoint(test_targets)
    assert validation_targets.isdisjoint(test_targets)
    assert len(partitions.train) == 8
    assert len(partitions.validation) == 4
    assert len(partitions.test) == 4


def test_legacy_loader_validates_row_aligned_smiles(tmp_path: Path) -> None:
    aptamer_path = tmp_path / "aptamers.csv"
    target_path = tmp_path / "targets.csv"
    pd.DataFrame(
        {
            "sequence": ["ACGT", "TGCA"],
            "target": ["ethanol", "acetic acid"],
        }
    ).to_csv(aptamer_path, index=False)
    pd.DataFrame({"Smiles": ["CCO", "CC(=O)O"]}).to_csv(
        target_path, index=False
    )

    loaded = load_aptamer_table(
        aptamer_path, legacy_target_features_path=target_path
    )

    assert loaded.used_legacy_row_alignment
    assert loaded.frame["target_name"].tolist() == ["ethanol", "acetic acid"]


def test_target_split_balances_rows_and_target_counts() -> None:
    records = []
    for target_index, group_size in enumerate(range(1, 11), start=1):
        records.extend(
            {
                "sequence": "A" * sequence_index + "C",
                "target_name": f"target_{target_index}",
                "target_smiles": f"smiles_{target_index}",
            }
            for sequence_index in range(1, group_size + 1)
        )
    frame = pd.DataFrame.from_records(records)

    partitions = split_aptamer_table(
        frame,
        validation_fraction=0.20,
        test_fraction=0.20,
        seed=19,
        strategy="target",
    )

    assert len(partitions.validation) == 11
    assert len(partitions.test) == 11
    assert partitions.validation["target_smiles"].nunique() == 2
    assert partitions.test["target_smiles"].nunique() == 2


def test_bigram_baseline_detects_simple_transition_pattern() -> None:
    frame = pd.DataFrame(
        {
            "sequence": ["ACAC", "ACAC", "ACAC"],
            "target_name": ["ethanol"] * 3,
            "target_smiles": ["CCO"] * 3,
        }
    )
    tokenizer = DNATokenizer(maximum_sequence_length=4)
    featurizer = MoleculeFeaturizer(fingerprint_bits=16).fit(["CCO"])
    dataset = AptamerSequenceDataset(frame, tokenizer, featurizer)

    unigram = evaluate_unigram_baseline(dataset, dataset, tokenizer)
    bigram = evaluate_bigram_baseline(dataset, dataset, tokenizer)

    assert bigram.negative_log_likelihood < unigram.negative_log_likelihood
    assert bigram.token_count == 15
