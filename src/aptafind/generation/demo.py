"""Generate a clearly labeled synthetic dataset for software smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SYNTHETIC_TARGETS = (
    (
        "synthetic_estradiol_demo",
        "C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@H]2O)CCC4=C3C=CC(=C4)O",
        "ACGCTA",
    ),
    (
        "synthetic_cortisol_demo",
        "C[C@]12CCC(=O)C=C1CC[C@@H]3[C@@H]2[C@H](C[C@]4([C@H]3CC[C@@]4(C(=O)CO)O)C)O",
        "GGTACC",
    ),
    (
        "synthetic_testosterone_demo",
        "C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@H]2O)CCC4=CC(=O)CC[C@]34C",
        "TTGCGA",
    ),
    (
        "synthetic_progesterone_demo",
        "CC(=O)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CCC4=CC(=O)CC[C@]34C)C",
        "CATGGC",
    ),
)


def create_synthetic_aptamer_table(
    *, samples_per_target: int = 12, sequence_length: int = 40, seed: int = 7
) -> pd.DataFrame:
    """Create artificial motif-bearing sequences with no claimed binding activity."""

    if samples_per_target < 2:
        raise ValueError("samples_per_target must be at least two.")
    longest_motif = max(len(target[2]) for target in SYNTHETIC_TARGETS)
    if sequence_length < longest_motif + 4:
        raise ValueError("sequence_length is too short for the synthetic motifs.")

    rng = np.random.default_rng(seed)
    alphabet = np.asarray(list("ACGT"))
    records: list[dict[str, object]] = []
    observed: set[tuple[str, str]] = set()
    for target_name, target_smiles, motif in SYNTHETIC_TARGETS:
        while sum(record["target_name"] == target_name for record in records) < samples_per_target:
            sequence_array = rng.choice(alphabet, size=sequence_length)
            motif_start = int(rng.integers(2, sequence_length - len(motif) - 1))
            sequence_array[motif_start : motif_start + len(motif)] = list(motif)
            sequence = "".join(sequence_array.tolist())
            identity = (sequence, target_smiles)
            if identity in observed:
                continue
            observed.add(identity)
            records.append(
                {
                    "sequence": sequence,
                    "target_name": target_name,
                    "target_smiles": target_smiles,
                    "data_origin": "synthetic_demo_no_binding_evidence",
                }
            )
    return pd.DataFrame.from_records(records)


def write_synthetic_aptamer_table(
    path: str | Path,
    *,
    samples_per_target: int = 12,
    sequence_length: int = 40,
    seed: int = 7,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    create_synthetic_aptamer_table(
        samples_per_target=samples_per_target,
        sequence_length=sequence_length,
        seed=seed,
    ).to_csv(output_path, index=False)
    return output_path
