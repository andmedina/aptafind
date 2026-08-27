"""Candidate generation, novelty checks, and transparent sequence filters."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from itertools import groupby
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from rapidfuzz.distance import Levenshtein

from aptafind.generation.checkpoint import LoadedGenerator, sequence_digest
from aptafind.generation.chemistry import canonicalize_smiles
from aptafind.generation.tokenizer import DNA_ALPHABET, normalize_dna
from aptafind.generation.training import seed_everything


@dataclass(frozen=True)
class CandidateFilterConfig:
    """Sequence-only plausibility filters; none are evidence of target binding."""

    minimum_length: int = 12
    maximum_length: int | None = None
    minimum_gc_fraction: float = 0.25
    maximum_gc_fraction: float = 0.75
    maximum_homopolymer: int = 4
    maximum_reference_identity: float = 0.95

    def __post_init__(self) -> None:
        if self.minimum_length < 1:
            raise ValueError("minimum_length must be positive.")
        if self.maximum_length is not None and self.maximum_length < self.minimum_length:
            raise ValueError("maximum_length cannot be smaller than minimum_length.")
        if not 0 <= self.minimum_gc_fraction <= self.maximum_gc_fraction <= 1:
            raise ValueError("GC fractions must satisfy 0 <= minimum <= maximum <= 1.")
        if self.maximum_homopolymer < 1:
            raise ValueError("maximum_homopolymer must be positive.")
        if not 0 <= self.maximum_reference_identity <= 1:
            raise ValueError("maximum_reference_identity must be in [0, 1].")


@dataclass
class CandidateGenerationResult:
    candidates: pd.DataFrame
    draws: int
    unique_sequences: int
    rejected_sequences: int


def maximum_homopolymer_length(sequence: str) -> int:
    """Return the longest run of one nucleotide."""

    normalized = normalize_dna(sequence)
    return max(sum(1 for _ in group) for _, group in groupby(normalized))


def shannon_entropy_bits(sequence: str) -> float:
    """Return mononucleotide Shannon entropy on a zero-to-two-bit scale."""

    normalized = normalize_dna(sequence)
    probabilities = [normalized.count(base) / len(normalized) for base in DNA_ALPHABET]
    return -sum(value * math.log2(value) for value in probabilities if value > 0)


def gc_fraction(sequence: str) -> float:
    normalized = normalize_dna(sequence)
    return (normalized.count("G") + normalized.count("C")) / len(normalized)


def nearest_reference_identity(
    sequence: str, reference_sequences: Iterable[str]
) -> float | None:
    """Return maximum normalized Levenshtein similarity to supplied references."""

    references = list(reference_sequences)
    if not references:
        return None
    return max(
        float(Levenshtein.normalized_similarity(sequence, reference))
        for reference in references
    )


def _candidate_record(
    sequence: str,
    *,
    target_name: str,
    target_smiles: str,
    training_sequence_hashes: set[str],
    references: list[str],
    filters: CandidateFilterConfig,
    target_median_length: float | None,
) -> dict[str, object]:
    normalized = normalize_dna(sequence)
    sequence_gc = gc_fraction(normalized)
    homopolymer = maximum_homopolymer_length(normalized)
    entropy = shannon_entropy_bits(normalized)
    seen_in_training = sequence_digest(normalized) in training_sequence_hashes
    reference_identity = nearest_reference_identity(normalized, references)

    reasons: list[str] = []
    if len(normalized) < filters.minimum_length:
        reasons.append("below_minimum_length")
    if filters.maximum_length is not None and len(normalized) > filters.maximum_length:
        reasons.append("above_maximum_length")
    if not filters.minimum_gc_fraction <= sequence_gc <= filters.maximum_gc_fraction:
        reasons.append("gc_outside_range")
    if homopolymer > filters.maximum_homopolymer:
        reasons.append("homopolymer_too_long")
    if seen_in_training:
        reasons.append("seen_in_training")
    if (
        reference_identity is not None
        and reference_identity > filters.maximum_reference_identity
    ):
        reasons.append("too_similar_to_reference")

    candidate_id = hashlib.sha256(
        f"{target_smiles}|{normalized}".encode("utf-8")
    ).hexdigest()[:16]
    return {
        "candidate_id": candidate_id,
        "target_name": target_name,
        "target_smiles": target_smiles,
        "sequence": normalized,
        "length": len(normalized),
        "length_distance_from_target_median": (
            abs(len(normalized) - target_median_length)
            if target_median_length is not None
            else None
        ),
        "gc_fraction": sequence_gc,
        "shannon_entropy_bits": entropy,
        "maximum_homopolymer": homopolymer,
        "seen_in_training": seen_in_training,
        "nearest_reference_identity": reference_identity,
        "passes_sequence_filters": not reasons,
        "rejection_reasons": ";".join(reasons),
    }


def generate_candidate_table(
    generator: LoadedGenerator,
    *,
    target_smiles: str,
    target_name: str | None = None,
    candidate_count: int = 10,
    temperature: float = 0.9,
    top_k: int | None = 5,
    filters: CandidateFilterConfig | None = None,
    reference_sequences: Iterable[str] = (),
    seed: int = 42,
    maximum_draw_multiplier: int = 30,
    ranking_pool_multiplier: int = 4,
) -> CandidateGenerationResult:
    """Sample, deduplicate, screen, and rank computational candidates."""

    if candidate_count < 1:
        raise ValueError("candidate_count must be positive.")
    if maximum_draw_multiplier < 1:
        raise ValueError("maximum_draw_multiplier must be positive.")
    if ranking_pool_multiplier < 1:
        raise ValueError("ranking_pool_multiplier must be positive.")

    tokenizer = generator.tokenizer
    active_filters = filters or CandidateFilterConfig(
        maximum_length=tokenizer.maximum_sequence_length
    )
    if (
        active_filters.maximum_length is not None
        and active_filters.maximum_length > tokenizer.maximum_sequence_length
    ):
        raise ValueError("Filter maximum length exceeds the trained tokenizer limit.")

    canonical_smiles = canonicalize_smiles(target_smiles)
    resolved_target_name = target_name.strip() if target_name else canonical_smiles
    target_range = generator.metadata.get("training_target_length_ranges", {}).get(
        canonical_smiles, {}
    )
    target_median_length = target_range.get("median_length")
    references = sorted({normalize_dna(value) for value in reference_sequences})
    device = next(generator.model.parameters()).device
    condition = generator.molecule_featurizer.transform_one(canonical_smiles)
    sampling_maximum_length = (
        active_filters.maximum_length
        if active_filters.maximum_length is not None
        else tokenizer.maximum_sequence_length
    )
    seed_everything(seed)

    records_by_sequence: dict[str, dict[str, object]] = {}
    accepted_count = 0
    draws = 0
    maximum_draws = candidate_count * maximum_draw_multiplier
    minimum_ranking_draws = min(
        maximum_draws, candidate_count * ranking_pool_multiplier
    )
    while draws < maximum_draws and (
        accepted_count < candidate_count or draws < minimum_ranking_draws
    ):
        batch_size = min(max(candidate_count * 2, 8), maximum_draws - draws)
        condition_batch = torch.from_numpy(
            np.repeat(condition[None, :], batch_size, axis=0)
        ).to(device)
        sampled_token_rows = generator.model.sample_prior(
            condition_batch,
            bos_token_id=tokenizer.bos_id,
            eos_token_id=tokenizer.eos_id,
            maximum_sequence_length=sampling_maximum_length,
            minimum_sequence_length=active_filters.minimum_length,
            temperature=temperature,
            top_k=top_k,
        )
        draws += batch_size
        for token_ids in sampled_token_rows:
            sequence = tokenizer.decode(token_ids)
            if not sequence or sequence in records_by_sequence:
                continue
            record = _candidate_record(
                sequence,
                target_name=resolved_target_name,
                target_smiles=canonical_smiles,
                training_sequence_hashes=generator.training_sequence_hashes,
                references=references,
                filters=active_filters,
                target_median_length=target_median_length,
            )
            records_by_sequence[sequence] = record
            if bool(record["passes_sequence_filters"]):
                accepted_count += 1

    all_records = pd.DataFrame.from_records(list(records_by_sequence.values()))
    if all_records.empty:
        accepted = all_records
        rejected_count = 0
    else:
        all_records["_gc_distance"] = (all_records["gc_fraction"] - 0.5).abs()
        all_records["_reference_sort"] = all_records[
            "nearest_reference_identity"
        ].fillna(0.0)
        all_records["_length_sort"] = all_records[
            "length_distance_from_target_median"
        ].fillna(0.0)
        all_records = all_records.sort_values(
            by=[
                "passes_sequence_filters",
                "_length_sort",
                "_reference_sort",
                "_gc_distance",
                "shannon_entropy_bits",
                "sequence",
            ],
            ascending=[False, True, True, True, False, True],
        ).drop(columns=["_gc_distance", "_reference_sort", "_length_sort"])
        accepted = all_records[all_records["passes_sequence_filters"]].head(
            candidate_count
        )
        rejected_count = int((~all_records["passes_sequence_filters"]).sum())

    accepted = accepted.reset_index(drop=True)
    if not accepted.empty:
        accepted.insert(0, "candidate_rank", range(1, len(accepted) + 1))
    return CandidateGenerationResult(
        candidates=accepted,
        draws=draws,
        unique_sequences=len(records_by_sequence),
        rejected_sequences=rejected_count,
    )
