"""Canonical entities and measurements for heterogeneous aptamer sources.

The Bronze inputs are immutable.  Source adapters emit :class:`SourceMeasurement`
objects, and this module constructs normalized Silver tables without inventing
missing sequence-to-aptamer or name-to-molecule joins.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import re
from typing import Any, Iterable

import pandas as pd

from aptafind.generation.chemistry import canonicalize_smiles


BINDING_LABELS = {
    "positive",
    "negative",
    "cross_reactive",
    "measured_nonresponse",
    "screen_positive",
    "screen_below_threshold",
    "unlabeled",
}


@dataclass(frozen=True)
class SourceMeasurement:
    """One source-level aptamer-target observation.

    Sequence and chemical structure are optional because some publications
    expose measurements under local aptamer or target identifiers only.  Those
    records remain useful and receive explicitly provisional entity IDs.
    """

    source_dataset: str
    source_file: str
    source_sheet: str | None
    source_row: int | None
    source_record_id: str
    source_aptamer_id: str | None
    sequence_raw: str | None
    polymer_type: str | None
    sequence_role: str | None
    source_target_id: str | None
    target_name_raw: str | None
    target_smiles_raw: str | None
    publication_id: str | None
    measurement_type: str
    measurement_value: float | None
    measurement_unit: str | None
    measurement_error: float | None = None
    measurement_error_unit: str | None = None
    binding_label: str = "unlabeled"
    evidence_type: str = "published_candidate"
    is_target_measurement: bool | None = None
    assay: str | None = None
    buffer: str | None = None
    replicate_count: int | None = None
    value_qualifier: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    transform_notes: str | None = None


@dataclass
class HarmonizedTables:
    """Normalized Silver tables and model-ready views."""

    aptamers: pd.DataFrame
    targets: pd.DataFrame
    measurements: pd.DataFrame
    model_interactions: pd.DataFrame
    generation_positive_pairs: pd.DataFrame


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _clean_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = " ".join(str(value).strip().split())
    return text or None


def normalize_polymer_type(value: Any) -> str | None:
    """Normalize source polymer labels conservatively."""

    text = _clean_text(value)
    if text is None:
        return None
    compact = text.casefold().replace("-", "")
    if compact in {"dna", "ssdna", "singlestrandeddna"}:
        return "DNA"
    if compact in {"rna", "ssrna", "singlestrandedrna"}:
        return "RNA"
    return text


def normalize_source_sequence(
    value: Any, polymer_type: str | None
) -> tuple[str | None, str]:
    """Return a validated unmodified sequence and an explicit status.

    Common terminal ``5'``/``3'`` annotations and whitespace are presentation
    artifacts and are removed.  Internal modification notation or ambiguous
    bases are never silently deleted; such records retain their raw text and a
    non-valid status.
    """

    text = _clean_text(value)
    if text is None:
        return None, "missing"
    normalized = text.upper()
    normalized = re.sub(r"^5\s*[′’']\s*[-:]?\s*", "", normalized)
    normalized = re.sub(r"\s*[-:]?\s*3\s*[′’']$", "", normalized)
    normalized = re.sub(r"\s+", "", normalized)

    polymer = normalize_polymer_type(polymer_type)
    alphabet = set(normalized)
    if polymer == "DNA" and alphabet <= set("ACGT"):
        return normalized, "valid_dna"
    if polymer == "RNA" and alphabet <= set("ACGU"):
        return normalized, "valid_rna"
    if not normalized:
        return None, "missing"
    return None, "unsupported_or_modified_sequence"


def normalize_target_name(value: Any) -> str | None:
    """Create a readable, whitespace-normalized target label."""

    return _clean_text(value)


def _aptamer_identity(
    *,
    source_dataset: str,
    source_aptamer_id: str | None,
    sequence: str | None,
    polymer_type: str | None,
) -> tuple[str, str]:
    if sequence is not None:
        polymer = normalize_polymer_type(polymer_type) or "unknown"
        return f"sequence:{polymer.lower()}:{_sha256_text(sequence)}", "exact_sequence"
    alias = source_aptamer_id or "missing-source-aptamer-id"
    seed = f"{source_dataset}|{alias}"
    return f"provisional_aptamer:{_sha256_text(seed)}", "source_local_identifier"


def _target_identity(
    *,
    source_dataset: str,
    source_target_id: str | None,
    target_name: str | None,
    canonical_smiles: str | None,
) -> tuple[str, str]:
    if canonical_smiles is not None:
        return f"smiles:{_sha256_text(canonical_smiles)}", "canonical_smiles"
    source_id = _clean_text(source_target_id)
    if source_id is not None and source_id.isdigit():
        return f"pubchem:{source_id}", "pubchem_cid"
    seed_value = source_id or target_name or "missing-source-target-id"
    seed = f"{source_dataset}|{seed_value.casefold()}"
    return f"provisional_target:{_sha256_text(seed)}", "source_local_identifier"


def _measurement_identity(record: SourceMeasurement, aptamer_id: str, target_id: str) -> str:
    parts = [
        record.source_dataset,
        record.source_file,
        record.source_sheet or "",
        str(record.source_row or ""),
        record.source_record_id,
        aptamer_id,
        target_id,
        record.measurement_type,
    ]
    return f"measurement:{_sha256_text('|'.join(parts))}"


def _first_non_null(values: pd.Series) -> Any:
    for value in values:
        if value is not None and not pd.isna(value) and value != "":
            return value
    return None


def build_harmonized_tables(records: Iterable[SourceMeasurement]) -> HarmonizedTables:
    """Build validated entity, measurement, and modeling tables."""

    measurement_rows: list[dict[str, Any]] = []
    aptamer_rows: dict[str, dict[str, Any]] = {}
    target_rows: dict[str, dict[str, Any]] = {}

    for record in records:
        if record.binding_label not in BINDING_LABELS:
            raise ValueError(
                f"Unsupported binding label {record.binding_label!r} in "
                f"{record.source_dataset}:{record.source_record_id}."
            )

        polymer_type = normalize_polymer_type(record.polymer_type)
        sequence_raw = _clean_text(record.sequence_raw)
        sequence, sequence_status = normalize_source_sequence(
            sequence_raw, polymer_type
        )
        target_name = normalize_target_name(record.target_name_raw)
        target_smiles = None
        target_smiles_status = "missing"
        if _clean_text(record.target_smiles_raw) is not None:
            try:
                target_smiles = canonicalize_smiles(str(record.target_smiles_raw))
                target_smiles_status = "valid"
            except (TypeError, ValueError):
                target_smiles_status = "invalid"

        aptamer_id, aptamer_identity_basis = _aptamer_identity(
            source_dataset=record.source_dataset,
            source_aptamer_id=_clean_text(record.source_aptamer_id),
            sequence=sequence,
            polymer_type=polymer_type,
        )
        target_id, target_identity_basis = _target_identity(
            source_dataset=record.source_dataset,
            source_target_id=_clean_text(record.source_target_id),
            target_name=target_name,
            canonical_smiles=target_smiles,
        )
        measurement_id = _measurement_identity(record, aptamer_id, target_id)

        aptamer_row = {
            "aptamer_id": aptamer_id,
            "sequence": sequence,
            "sequence_length": len(sequence) if sequence is not None else None,
            "polymer_type": polymer_type,
            "sequence_status": sequence_status,
            "identity_basis": aptamer_identity_basis,
        }
        existing_aptamer = aptamer_rows.get(aptamer_id)
        if (
            existing_aptamer is not None
            and existing_aptamer["sequence"] is not None
            and sequence is not None
            and existing_aptamer["sequence"] != sequence
        ):
            raise ValueError("One aptamer identity resolved to multiple sequences.")
        if existing_aptamer is None or existing_aptamer["sequence"] is None:
            aptamer_rows[aptamer_id] = aptamer_row

        target_row = {
            "target_id": target_id,
            "target_name": target_name,
            "target_smiles": target_smiles,
            "target_smiles_status": target_smiles_status,
            "identity_basis": target_identity_basis,
        }
        existing_target = target_rows.get(target_id)
        if (
            existing_target is not None
            and existing_target["target_smiles"] is not None
            and target_smiles is not None
            and existing_target["target_smiles"] != target_smiles
        ):
            raise ValueError("One target identity resolved to multiple structures.")
        if existing_target is None or existing_target["target_name"] is None:
            target_rows[target_id] = target_row
        row = asdict(record)
        row.pop("details")
        row.update(
            {
                "measurement_id": measurement_id,
                "aptamer_id": aptamer_id,
                "target_id": target_id,
                "sequence_raw": sequence_raw,
                "target_name_raw": _clean_text(record.target_name_raw),
                "target_smiles_raw": _clean_text(record.target_smiles_raw),
                "measurement_details_json": json.dumps(
                    record.details, sort_keys=True, separators=(",", ":")
                ),
            }
        )
        measurement_rows.append(row)

    measurements = pd.DataFrame.from_records(measurement_rows)
    if measurements.empty:
        raise ValueError("No source measurements were supplied for harmonization.")
    if not measurements["measurement_id"].is_unique:
        duplicates = measurements.loc[
            measurements["measurement_id"].duplicated(), "measurement_id"
        ].tolist()
        raise ValueError(f"Duplicate measurement identities: {duplicates[:5]}")

    aptamers = (
        pd.DataFrame.from_records(list(aptamer_rows.values()))
        .sort_values("aptamer_id")
        .reset_index(drop=True)
    )
    targets = (
        pd.DataFrame.from_records(list(target_rows.values()))
        .sort_values("target_id")
        .reset_index(drop=True)
    )

    if not set(measurements["aptamer_id"]).issubset(set(aptamers["aptamer_id"])):
        raise ValueError("Measurement table contains an unknown aptamer ID.")
    if not set(measurements["target_id"]).issubset(set(targets["target_id"])):
        raise ValueError("Measurement table contains an unknown target ID.")

    joined = measurements.merge(
        aptamers[["aptamer_id", "sequence", "polymer_type"]].rename(
            columns={"polymer_type": "canonical_polymer_type"}
        ),
        on="aptamer_id",
        how="left",
        validate="many_to_one",
    ).merge(
        targets[["target_id", "target_name", "target_smiles"]],
        on="target_id",
        how="left",
        validate="many_to_one",
    )
    model_interactions = joined.loc[
        joined["sequence"].notna()
        & joined["target_smiles"].notna()
        & joined["binding_label"].isin({"positive", "negative"})
        & joined["canonical_polymer_type"].eq("DNA"),
        [
            "measurement_id",
            "sequence",
            "target_name",
            "target_smiles",
            "binding_label",
            "source_dataset",
            "publication_id",
            "evidence_type",
            "buffer",
        ],
    ].copy()
    model_interactions["label"] = model_interactions["binding_label"].map(
        {"negative": 0, "positive": 1}
    )
    model_interactions["pair_label_conflict"] = (
        model_interactions.groupby(["sequence", "target_smiles"])["label"]
        .transform("nunique")
        .gt(1)
    )
    model_interactions = model_interactions.sort_values(
        ["source_dataset", "measurement_id"]
    ).reset_index(drop=True)

    positives = model_interactions.loc[
        model_interactions["label"].eq(1)
        & ~model_interactions["pair_label_conflict"]
    ].copy()
    positive_groups = positives.groupby(
        ["sequence", "target_smiles"], as_index=False, sort=True, dropna=False
    )
    generation_positive_pairs = positive_groups.agg(
        target_name=("target_name", _first_non_null),
        source_datasets=(
            "source_dataset",
            lambda values: "|".join(sorted(set(values))),
        ),
        publication_ids=(
            "publication_id",
            lambda values: "|".join(
                sorted({str(value) for value in values if not pd.isna(value)})
            ),
        ),
        supporting_measurements=("measurement_id", "nunique"),
    )
    generation_positive_pairs = generation_positive_pairs[
        [
            "sequence",
            "target_smiles",
            "target_name",
            "source_datasets",
            "publication_ids",
            "supporting_measurements",
        ]
    ]

    measurements = measurements.sort_values(
        ["source_dataset", "source_file", "source_sheet", "source_row"],
        na_position="last",
    ).reset_index(drop=True)
    return HarmonizedTables(
        aptamers=aptamers,
        targets=targets,
        measurements=measurements,
        model_interactions=model_interactions,
        generation_positive_pairs=generation_positive_pairs,
    )
