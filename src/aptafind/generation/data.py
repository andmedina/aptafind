"""Validated dataset ingestion and leakage-resistant train/test splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from aptafind.generation.chemistry import MoleculeFeaturizer, canonicalize_smiles
from aptafind.generation.tokenizer import DNATokenizer, normalize_dna


SplitStrategy = Literal["target", "random"]


@dataclass
class LoadedAptamerTable:
    """Canonical table and audit information from one input dataset."""

    frame: pd.DataFrame
    source_rows: int
    duplicate_rows_removed: int
    used_legacy_row_alignment: bool

    def summary(self) -> dict[str, Any]:
        return {
            "source_rows": self.source_rows,
            "validated_rows": int(len(self.frame)),
            "duplicate_rows_removed": self.duplicate_rows_removed,
            "unique_sequences": int(self.frame["sequence"].nunique()),
            "unique_targets": int(self.frame["target_smiles"].nunique()),
            "minimum_sequence_length": int(self.frame["sequence"].str.len().min()),
            "maximum_sequence_length": int(self.frame["sequence"].str.len().max()),
            "used_legacy_row_alignment": self.used_legacy_row_alignment,
        }


@dataclass
class DatasetPartitions:
    """Dataframes assigned to model development partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    strategy: SplitStrategy

    def summary(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "train_rows": int(len(self.train)),
            "validation_rows": int(len(self.validation)),
            "test_rows": int(len(self.test)),
            "train_targets": int(self.train["target_smiles"].nunique()),
            "validation_targets": int(self.validation["target_smiles"].nunique()),
            "test_targets": int(self.test["target_smiles"].nunique()),
        }


def _resolve_target_name_column(
    frame: pd.DataFrame, requested_column: str | None
) -> str | None:
    if requested_column is not None:
        if requested_column not in frame.columns:
            raise ValueError(
                f"Target-name column {requested_column!r} is not in the dataset."
            )
        return requested_column
    for candidate in ("target_name", "target"):
        if candidate in frame.columns:
            return candidate
    return None


def load_aptamer_table(
    path: str | Path,
    *,
    sequence_column: str = "sequence",
    smiles_column: str = "target_smiles",
    target_name_column: str | None = None,
    legacy_target_features_path: str | Path | None = None,
    legacy_smiles_column: str = "Smiles",
) -> LoadedAptamerTable:
    """Load, normalize, validate, and deduplicate positive aptamer examples.

    Modern input files should contain one sequence and target SMILES per row.
    The optional legacy target-feature file supports the historical Aptafind
    dataset, whose SMILES values were saved in a separate row-aligned CSV. Row
    alignment is validated explicitly and recorded in the returned audit.
    """

    source_path = Path(path)
    frame = pd.read_csv(source_path)
    source_rows = len(frame)
    if source_rows == 0:
        raise ValueError(f"Aptamer dataset is empty: {source_path}")
    if sequence_column not in frame.columns:
        raise ValueError(
            f"Sequence column {sequence_column!r} is not in {source_path}."
        )

    used_legacy_alignment = False
    if smiles_column in frame.columns:
        raw_smiles = frame[smiles_column]
    elif legacy_target_features_path is not None:
        target_features_path = Path(legacy_target_features_path)
        target_columns = pd.read_csv(target_features_path, nrows=0).columns
        if legacy_smiles_column not in target_columns:
            raise ValueError(
                f"Legacy SMILES column {legacy_smiles_column!r} is not in "
                f"{target_features_path}."
            )
        # Historical fingerprint strings can exceed native integer ranges in
        # newer Pandas versions. They are not needed because RDKit rebuilds the
        # conditioning features from SMILES deterministically.
        target_frame = pd.read_csv(
            target_features_path, usecols=[legacy_smiles_column]
        )
        if len(target_frame) != source_rows:
            raise ValueError(
                "Legacy aptamer and target-feature files must have identical row "
                f"counts; found {source_rows} and {len(target_frame)}."
            )
        raw_smiles = target_frame[legacy_smiles_column]
        used_legacy_alignment = True
    else:
        raise ValueError(
            f"Target-SMILES column {smiles_column!r} is missing. Supply a modern "
            "table containing that column or use legacy_target_features_path."
        )

    resolved_name_column = _resolve_target_name_column(frame, target_name_column)
    if resolved_name_column is None:
        raw_target_names = raw_smiles
    else:
        raw_target_names = frame[resolved_name_column]

    records: list[dict[str, str]] = []
    errors: list[str] = []
    for row_position in range(source_rows):
        try:
            sequence = normalize_dna(frame.iloc[row_position][sequence_column])
            target_smiles = canonicalize_smiles(raw_smiles.iloc[row_position])
            raw_name = raw_target_names.iloc[row_position]
            target_name = str(raw_name).strip()
            if not target_name or target_name.lower() == "nan":
                target_name = target_smiles
        except (TypeError, ValueError) as error:
            errors.append(f"row {row_position + 2}: {error}")
            continue
        records.append(
            {
                "sequence": sequence,
                "target_name": target_name,
                "target_smiles": target_smiles,
            }
        )

    if errors:
        preview = "; ".join(errors[:8])
        remainder = len(errors) - min(len(errors), 8)
        suffix = f"; plus {remainder} more" if remainder else ""
        raise ValueError(f"Dataset validation failed: {preview}{suffix}.")

    canonical = pd.DataFrame.from_records(records)
    before_deduplication = len(canonical)
    canonical = canonical.drop_duplicates(
        subset=["sequence", "target_smiles"], keep="first"
    ).reset_index(drop=True)
    duplicate_rows_removed = before_deduplication - len(canonical)
    if canonical.empty:
        raise ValueError("No aptamer records remain after validation and deduplication.")

    target_name_counts = canonical.groupby("target_smiles")["target_name"].nunique()
    if bool((target_name_counts > 1).any()):
        # Aliases are scientifically harmless; use a stable label in reports.
        stable_names = canonical.groupby("target_smiles")["target_name"].transform(
            lambda values: sorted(set(values))[0]
        )
        canonical["target_name"] = stable_names

    return LoadedAptamerTable(
        frame=canonical,
        source_rows=source_rows,
        duplicate_rows_removed=duplicate_rows_removed,
        used_legacy_row_alignment=used_legacy_alignment,
    )


def _partition_counts(
    item_count: int, validation_fraction: float, test_fraction: float
) -> tuple[int, int, int]:
    requested_partitions = 1 + int(validation_fraction > 0) + int(test_fraction > 0)
    if item_count < requested_partitions:
        raise ValueError(
            f"At least {requested_partitions} items are required for the requested "
            f"partitions; found {item_count}."
        )

    validation_count = (
        max(1, int(round(item_count * validation_fraction)))
        if validation_fraction > 0
        else 0
    )
    test_count = (
        max(1, int(round(item_count * test_fraction))) if test_fraction > 0 else 0
    )
    while validation_count + test_count >= item_count:
        if validation_count >= test_count and validation_count > int(
            validation_fraction > 0
        ):
            validation_count -= 1
        elif test_count > int(test_fraction > 0):
            test_count -= 1
        else:
            raise ValueError("Fractions leave no items for model training.")
    train_count = item_count - validation_count - test_count
    return train_count, validation_count, test_count


def _choose_target_groups(
    group_sizes: dict[str, int],
    *,
    desired_rows: int,
    desired_groups: int,
    minimum_groups_to_leave: int,
    rng: np.random.Generator,
) -> set[str]:
    """Choose a deterministic subset close to row and group-count targets."""

    groups = list(group_sizes)
    rng.shuffle(groups)
    maximum_selected_groups = len(groups) - minimum_groups_to_leave
    if desired_groups < 1 or maximum_selected_groups < 1:
        return set()

    # Dynamic programming keeps one seeded subset for each (rows, groups)
    # state. Dataset row counts are small, making this exact search inexpensive.
    states: dict[tuple[int, int], tuple[str, ...]] = {(0, 0): ()}
    for group in groups:
        size = group_sizes[group]
        additions: dict[tuple[int, int], tuple[str, ...]] = {}
        for (row_count, group_count), selected in list(states.items()):
            if group_count >= maximum_selected_groups:
                continue
            state = (row_count + size, group_count + 1)
            if state not in states and state not in additions:
                additions[state] = (*selected, group)
        states.update(additions)

    candidates = [state for state in states if state[1] > 0]
    if not candidates:
        raise ValueError("Unable to assign a non-empty target holdout partition.")
    best_state = min(
        candidates,
        key=lambda state: (
            abs(state[0] - desired_rows) / max(desired_rows, 1)
            + abs(state[1] - desired_groups) / max(desired_groups, 1),
            abs(state[0] - desired_rows),
            abs(state[1] - desired_groups),
        ),
    )
    return set(states[best_state])


def split_aptamer_table(
    frame: pd.DataFrame,
    *,
    validation_fraction: float = 0.10,
    test_fraction: float = 0.10,
    seed: int = 42,
    strategy: SplitStrategy = "target",
) -> DatasetPartitions:
    """Create deterministic development partitions.

    ``target`` (the default) assigns each canonical target molecule to exactly
    one partition, providing a more honest unseen-target evaluation. ``random``
    is retained for explicitly labeled exploratory comparisons.
    """

    for name, fraction in (
        ("validation_fraction", validation_fraction),
        ("test_fraction", test_fraction),
    ):
        if not 0.0 <= fraction < 1.0:
            raise ValueError(f"{name} must be in [0, 1).")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("Validation and test fractions must sum to less than one.")
    if strategy not in ("target", "random"):
        raise ValueError("strategy must be 'target' or 'random'.")

    rng = np.random.default_rng(seed)
    if strategy == "target":
        group_sizes = frame.groupby("target_smiles").size().astype(int).to_dict()
        train_group_count, validation_group_count, test_group_count = _partition_counts(
            len(group_sizes), validation_fraction, test_fraction
        )
        test_items = (
            _choose_target_groups(
                group_sizes,
                desired_rows=max(1, int(round(len(frame) * test_fraction))),
                desired_groups=test_group_count,
                minimum_groups_to_leave=validation_group_count + train_group_count,
                rng=rng,
            )
            if test_group_count
            else set()
        )
        remaining_sizes = {
            group: size for group, size in group_sizes.items() if group not in test_items
        }
        validation_items = (
            _choose_target_groups(
                remaining_sizes,
                desired_rows=max(1, int(round(len(frame) * validation_fraction))),
                desired_groups=validation_group_count,
                minimum_groups_to_leave=train_group_count,
                rng=rng,
            )
            if validation_group_count
            else set()
        )
        train_items = set(group_sizes).difference(test_items, validation_items)
        selector = frame["target_smiles"]
    else:
        items = np.arange(len(frame))
        rng.shuffle(items)
        train_count, validation_count, _ = _partition_counts(
            len(items), validation_fraction, test_fraction
        )
        train_items = set(items[:train_count].tolist())
        validation_items = set(
            items[train_count : train_count + validation_count].tolist()
        )
        test_items = set(items[train_count + validation_count :].tolist())
        selector = pd.Series(np.arange(len(frame)), index=frame.index)

    train = frame[selector.isin(train_items)].reset_index(drop=True)
    validation = frame[selector.isin(validation_items)].reset_index(drop=True)
    test = frame[selector.isin(test_items)].reset_index(drop=True)
    return DatasetPartitions(
        train=train,
        validation=validation,
        test=test,
        strategy=strategy,
    )


def permute_target_assignments(
    target_smiles: Iterable[str], *, seed: int
) -> tuple[list[str], dict[str, str]]:
    """Map every target to one different target for a label-permutation control.

    The mapping is a seeded derangement over unique canonical targets, so all
    sequences associated with one target receive the same incorrect condition
    and no target remains mapped to itself.
    """

    values = [str(value) for value in target_smiles]
    unique_targets = sorted(set(values))
    if len(unique_targets) < 2:
        raise ValueError("At least two targets are required for label permutation.")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(unique_targets))
    for _ in range(10_000):
        shuffled = rng.permutation(indices)
        if bool(np.all(shuffled != indices)):
            break
    else:  # pragma: no cover - a derangement exists for every n >= 2.
        raise RuntimeError("Unable to construct a target-label derangement.")
    mapping = {
        target: unique_targets[int(shuffled[index])]
        for index, target in enumerate(unique_targets)
    }
    return [mapping[value] for value in values], mapping


class AptamerSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Pre-tokenized sequence and molecule-condition tensors."""

    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: DNATokenizer,
        molecule_featurizer: MoleculeFeaturizer,
        *,
        condition_smiles: Iterable[str] | None = None,
    ) -> None:
        token_rows: list[list[int]] = []
        lengths: list[int] = []
        for sequence in frame["sequence"]:
            token_ids, length = tokenizer.encode(sequence)
            token_rows.append(token_ids)
            lengths.append(length)

        resolved_condition_smiles = (
            list(frame["target_smiles"])
            if condition_smiles is None
            else list(condition_smiles)
        )
        if len(resolved_condition_smiles) != len(frame):
            raise ValueError("condition_smiles must contain one value per row.")

        if token_rows:
            self.token_ids = torch.tensor(token_rows, dtype=torch.long)
            self.lengths = torch.tensor(lengths, dtype=torch.long)
            self.conditions = torch.from_numpy(
                molecule_featurizer.transform(resolved_condition_smiles)
            )
        else:
            self.token_ids = torch.empty(
                (0, tokenizer.encoded_length), dtype=torch.long
            )
            self.lengths = torch.empty((0,), dtype=torch.long)
            self.conditions = torch.empty(
                (0, molecule_featurizer.condition_dimension), dtype=torch.float32
            )

    def __len__(self) -> int:
        return int(self.token_ids.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "token_ids": self.token_ids[index],
            "length": self.lengths[index],
            "condition": self.conditions[index],
        }


def make_data_loader(
    dataset: AptamerSequenceDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Build a seeded data loader without multiprocessing side effects."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and len(dataset) > 0,
        num_workers=0,
        generator=generator,
    )
