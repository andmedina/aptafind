"""Strict provenance-aware grouping for repeated sequence-CVAE evaluation."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from rapidfuzz.distance import Levenshtein


class _DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


@dataclass(frozen=True)
class RepeatedEvaluationConfig:
    """Configuration for strict repeated target-condition experiments."""

    fold_count: int = 5
    training_seeds: tuple[int, ...] = (42, 43, 44)
    fold_assignment_seed: int = 20260828
    validation_fold_offset: int = 1
    sequence_family_identity_threshold: float = 0.90
    excluded_publications: tuple[str, ...] = ()
    bootstrap_replicates: int = 5_000
    bootstrap_seed: int = 20260828

    def __post_init__(self) -> None:
        if self.fold_count < 3:
            raise ValueError("fold_count must be at least three.")
        if not self.training_seeds:
            raise ValueError("training_seeds cannot be empty.")
        if len(set(self.training_seeds)) != len(self.training_seeds):
            raise ValueError("training_seeds must be unique.")
        if not 0 < self.validation_fold_offset < self.fold_count:
            raise ValueError("validation_fold_offset must identify another fold.")
        if not 0.0 < self.sequence_family_identity_threshold <= 1.0:
            raise ValueError("sequence family identity threshold must be in (0, 1].")
        if self.bootstrap_replicates < 1:
            raise ValueError("bootstrap_replicates must be positive.")

    def state_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["training_seeds"] = list(self.training_seeds)
        payload["excluded_publications"] = list(self.excluded_publications)
        return payload


@dataclass
class GroupedFoldTable:
    """Filtered rows, strict group identities, fold assignments, and audit."""

    frame: pd.DataFrame
    audit: dict[str, Any]


def load_repeated_evaluation_config(
    path: str | Path,
) -> RepeatedEvaluationConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    validation = payload.get("validation", {})
    if not isinstance(validation, dict):
        raise ValueError("Configuration validation section must be a mapping.")
    normalized = dict(validation)
    if "training_seeds" in normalized:
        normalized["training_seeds"] = tuple(
            int(value) for value in normalized["training_seeds"]
        )
    if "excluded_publications" in normalized:
        normalized["excluded_publications"] = tuple(
            str(value) for value in normalized["excluded_publications"]
        )
    return RepeatedEvaluationConfig(**normalized)


def publication_tokens(value: object) -> tuple[str, ...]:
    """Return stable pipe-delimited publication identifiers."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ()
    return tuple(
        sorted(
            {
                token.strip()
                for token in str(value).split("|")
                if token.strip() and token.strip().lower() != "nan"
            }
        )
    )


def _stable_identifier(prefix: str, values: Iterable[str]) -> str:
    payload = "\n".join(sorted(set(values))).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:16]}"


def cluster_sequence_families(
    sequences: Iterable[str], *, identity_threshold: float
) -> dict[str, str]:
    """Single-linkage families using normalized Levenshtein identity."""

    unique_sequences = sorted(set(str(value) for value in sequences))
    disjoint = _DisjointSet()
    for left_index, left in enumerate(unique_sequences):
        disjoint.find(left)
        for right in unique_sequences[left_index + 1 :]:
            if min(len(left), len(right)) / max(len(left), len(right)) < identity_threshold:
                continue
            if (
                Levenshtein.normalized_similarity(
                    left, right, score_cutoff=identity_threshold
                )
                >= identity_threshold
            ):
                disjoint.union(left, right)

    members_by_root: dict[str, list[str]] = defaultdict(list)
    for sequence in unique_sequences:
        members_by_root[disjoint.find(sequence)].append(sequence)
    family_by_sequence: dict[str, str] = {}
    for members in members_by_root.values():
        family_id = _stable_identifier("family", members)
        for sequence in members:
            family_by_sequence[sequence] = family_id
    return family_by_sequence


def _assign_components_to_folds(
    frame: pd.DataFrame, *, fold_count: int, seed: int
) -> dict[str, int]:
    stats = (
        frame.groupby("independence_group_id")
        .agg(rows=("sequence", "size"), targets=("target_smiles", "nunique"))
        .reset_index()
    )
    if len(stats) < fold_count:
        raise ValueError(
            f"At least {fold_count} independence groups are required; found {len(stats)}."
        )
    rng = np.random.default_rng(seed)
    stats["tie_breaker"] = rng.random(len(stats))
    stats = stats.sort_values(
        ["rows", "targets", "tie_breaker"],
        ascending=[False, False, True],
    )
    fold_rows = [0] * fold_count
    fold_targets = [0] * fold_count
    fold_groups = [0] * fold_count
    mapping: dict[str, int] = {}
    for row in stats.itertuples(index=False):
        selected = min(
            range(fold_count),
            key=lambda fold: (
                fold_rows[fold],
                fold_targets[fold],
                fold_groups[fold],
                fold,
            ),
        )
        mapping[row.independence_group_id] = selected
        fold_rows[selected] += int(row.rows)
        fold_targets[selected] += int(row.targets)
        fold_groups[selected] += 1
    return mapping


def build_strict_group_folds(
    frame: pd.DataFrame, config: RepeatedEvaluationConfig
) -> GroupedFoldTable:
    """Exclude declared lineages and group target/family/publication components."""

    required = {"sequence", "target_smiles", "publication_ids"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Strict grouping requires columns: " + ", ".join(sorted(missing))
        )
    working = frame.copy().reset_index(drop=True)
    working["publication_tokens"] = working["publication_ids"].map(
        publication_tokens
    )
    excluded = set(config.excluded_publications)
    exclusion_mask = working["publication_tokens"].map(
        lambda values: bool(excluded.intersection(values))
    )
    excluded_frame = working[exclusion_mask]
    working = working[~exclusion_mask].reset_index(drop=True)
    if working.empty:
        raise ValueError("All rows were removed by publication exclusions.")

    family_by_sequence = cluster_sequence_families(
        working["sequence"],
        identity_threshold=config.sequence_family_identity_threshold,
    )
    working["sequence_family_id"] = working["sequence"].map(family_by_sequence)

    disjoint = _DisjointSet()
    for row in working.itertuples(index=False):
        target_node = f"target:{row.target_smiles}"
        disjoint.union(target_node, f"family:{row.sequence_family_id}")
        for publication in row.publication_tokens:
            disjoint.union(target_node, f"publication:{publication}")

    targets_by_root: dict[str, list[str]] = defaultdict(list)
    for target in sorted(working["target_smiles"].unique()):
        targets_by_root[disjoint.find(f"target:{target}")].append(target)
    group_by_target: dict[str, str] = {}
    for targets in targets_by_root.values():
        group_id = _stable_identifier("group", targets)
        for target in targets:
            group_by_target[target] = group_id
    working["independence_group_id"] = working["target_smiles"].map(group_by_target)
    fold_by_group = _assign_components_to_folds(
        working,
        fold_count=config.fold_count,
        seed=config.fold_assignment_seed,
    )
    working["fold"] = working["independence_group_id"].map(fold_by_group).astype(int)

    exploded_publications = working.explode("publication_tokens")
    exploded_publications = exploded_publications[
        exploded_publications["publication_tokens"].astype(bool)
    ]
    overlap_audit = {
        "maximum_folds_per_target": int(
            working.groupby("target_smiles")["fold"].nunique().max()
        ),
        "maximum_folds_per_sequence_family": int(
            working.groupby("sequence_family_id")["fold"].nunique().max()
        ),
        "maximum_folds_per_publication": (
            int(
                exploded_publications.groupby("publication_tokens")["fold"]
                .nunique()
                .max()
            )
            if not exploded_publications.empty
            else 0
        ),
    }
    if any(value > 1 for value in overlap_audit.values()):
        raise RuntimeError(f"Strict fold grouping failed: {overlap_audit}")

    family_sizes = Counter(working["sequence_family_id"])
    group_sizes = Counter(working["independence_group_id"])
    fold_summary = [
        {
            "fold": fold,
            "rows": int((working["fold"] == fold).sum()),
            "targets": int(
                working.loc[working["fold"] == fold, "target_smiles"].nunique()
            ),
            "sequence_families": int(
                working.loc[
                    working["fold"] == fold, "sequence_family_id"
                ].nunique()
            ),
            "independence_groups": int(
                working.loc[
                    working["fold"] == fold, "independence_group_id"
                ].nunique()
            ),
        }
        for fold in range(config.fold_count)
    ]
    audit = {
        "source_rows": int(len(frame)),
        "retained_rows": int(len(working)),
        "retained_targets": int(working["target_smiles"].nunique()),
        "retained_sequences": int(working["sequence"].nunique()),
        "excluded_rows": int(len(excluded_frame)),
        "excluded_targets": int(excluded_frame["target_smiles"].nunique()),
        "excluded_publications": sorted(excluded),
        "sequence_family_identity_threshold": (
            config.sequence_family_identity_threshold
        ),
        "sequence_families": len(family_sizes),
        "largest_sequence_family_rows": max(family_sizes.values()),
        "independence_groups": len(group_sizes),
        "largest_independence_group_rows": max(group_sizes.values()),
        "folds": fold_summary,
        "overlap_audit": overlap_audit,
    }
    return GroupedFoldTable(frame=working, audit=audit)
