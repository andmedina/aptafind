"""Run the provenance-preserving Bronze-to-Silver endpoint harmonization."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yaml

from aptafind.data.harmonization_schema import (
    HarmonizedTables,
    SourceMeasurement,
    build_harmonized_tables,
)
from aptafind.data.source_adapters import ADAPTERS, AdapterResult


OUTPUT_FILENAMES = {
    "aptamers": "aptamers.csv",
    "targets": "targets.csv",
    "measurements": "measurements.csv",
    "model_interactions": "model_interactions.csv",
    "generation_positive_pairs": "generation_positive_pairs.csv",
}


def sha256_file(path: Path) -> str:
    """Hash a source or output without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_bronze_path(bronze_root: Path, relative_path: str) -> Path:
    root = bronze_root.resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Configured path escapes the Bronze root: {relative_path}")
    if not path.is_file():
        raise FileNotFoundError(f"Configured Bronze source is missing: {path}")
    return path


def load_source_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate the source-adapter manifest."""

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "sources" not in config:
        raise ValueError("Harmonization config must contain a sources list.")
    if config.get("manifest_version") != "0.1.0":
        raise ValueError("Unsupported harmonization manifest version.")
    source_keys: list[tuple[str, str]] = []
    for source in config["sources"]:
        if not isinstance(source, dict):
            raise ValueError("Every configured source must be a mapping.")
        missing = {"dataset_id", "adapter", "paths"} - set(source)
        if missing:
            raise ValueError(
                f"Configured source is missing fields: {sorted(missing)}"
            )
        if source["adapter"] not in ADAPTERS:
            raise ValueError(f"Unknown source adapter: {source['adapter']}")
        source_keys.append((str(source["dataset_id"]), str(source["adapter"])))
    if len(source_keys) != len(set(source_keys)):
        raise ValueError("Harmonization dataset/adapter pairs must be unique.")
    return config


def _run_adapter(
    source: dict[str, Any], bronze_root: Path
) -> tuple[AdapterResult, list[Path]]:
    dataset_id = str(source["dataset_id"])
    adapter_name = str(source["adapter"])
    configured_paths = source["paths"]
    if not isinstance(configured_paths, dict):
        raise ValueError(f"Paths for {dataset_id} must be a mapping.")
    resolved = {
        name: _resolve_bronze_path(bronze_root, str(relative_path))
        for name, relative_path in configured_paths.items()
    }

    adapter = ADAPTERS[adapter_name]
    if adapter_name == "aptadb":
        required = {"interaction", "aptamer", "molecule"}
        if set(resolved) != required:
            raise ValueError(
                f"AptaDB paths must be exactly {sorted(required)}; "
                f"received {sorted(resolved)}."
            )
        result = adapter(
            resolved["interaction"],
            resolved["aptamer"],
            resolved["molecule"],
            dataset_id,
        )
    else:
        if set(resolved) != {"input"}:
            raise ValueError(
                f"Adapter {adapter_name} requires one path named input."
            )
        result = adapter(resolved["input"], dataset_id)
    return result, list(resolved.values())


def _table_summary(tables: HarmonizedTables) -> dict[str, Any]:
    measurements = tables.measurements
    aptamers = tables.aptamers
    targets = tables.targets
    model = tables.model_interactions
    pair_label_counts = (
        model.groupby(["sequence", "target_smiles"])["label"].nunique()
        if not model.empty
        else pd.Series(dtype=int)
    )
    return {
        "aptamer_entities": int(len(aptamers)),
        "aptamers_with_valid_sequence": int(aptamers["sequence"].notna().sum()),
        "provisional_aptamer_entities": int(
            aptamers["identity_basis"].eq("source_local_identifier").sum()
        ),
        "target_entities": int(len(targets)),
        "targets_with_canonical_smiles": int(targets["target_smiles"].notna().sum()),
        "measurement_records": int(len(measurements)),
        "measurements_by_source": {
            str(key): int(value)
            for key, value in measurements["source_dataset"]
            .value_counts()
            .sort_index()
            .items()
        },
        "binding_labels": {
            str(key): int(value)
            for key, value in measurements["binding_label"]
            .value_counts()
            .sort_index()
            .items()
        },
        "evidence_types": {
            str(key): int(value)
            for key, value in measurements["evidence_type"]
            .value_counts()
            .sort_index()
            .items()
        },
        "model_ready_binary_interactions": int(len(model)),
        "model_ready_positive_interactions": int(model["label"].eq(1).sum()),
        "model_ready_negative_interactions": int(model["label"].eq(0).sum()),
        "exact_sequence_target_pairs_with_conflicting_labels": int(
            (pair_label_counts > 1).sum()
        ),
        "deduplicated_generation_positive_pairs": int(
            len(tables.generation_positive_pairs)
        ),
    }


def _write_tables(
    tables: HarmonizedTables, output_directory: Path, overwrite: bool
) -> dict[str, dict[str, Any]]:
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, Any]] = {}
    for table_name, filename in OUTPUT_FILENAMES.items():
        path = output_directory / filename
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Silver output already exists: {path}. Use --overwrite explicitly."
            )
        frame = getattr(tables, table_name)
        frame.to_csv(path, index=False)
        outputs[table_name] = {
            "path": str(path),
            "rows": int(len(frame)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return outputs


def run_harmonization(
    *,
    bronze_root: Path,
    config_path: Path,
    output_directory: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Execute configured adapters, validate tables, and write Silver CSVs."""

    config = load_source_config(config_path)
    all_records: list[SourceMeasurement] = []
    source_audits: list[dict[str, Any]] = []

    for source in config["sources"]:
        if source.get("enabled", True) is False:
            continue
        result, source_paths = _run_adapter(source, bronze_root)
        all_records.extend(result.records)
        source_audits.append(
            {
                "dataset_id": source["dataset_id"],
                "adapter": source["adapter"],
                "files": [
                    {
                        "path": str(path),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                    for path in source_paths
                ],
                "audit": result.audit,
            }
        )

    tables = build_harmonized_tables(all_records)
    outputs = _write_tables(tables, output_directory, overwrite)
    report = {
        "report_version": "0.1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bronze_root": str(bronze_root.resolve()),
        "config": {
            "path": str(config_path.resolve()),
            "sha256": sha256_file(config_path),
        },
        "sources": source_audits,
        "summary": _table_summary(tables),
        "outputs": outputs,
        "known_overlap_warnings": [
            "AptaBench includes a Specificity lineage derived from DOI "
            "10.1093/nar/gkaf219; Xiao source measurements are retained for "
            "provenance and must not be counted as independent supervision.",
            "Source-local aptamer IDs without sequences are provisional identities "
            "until an explicit sequence mapping is verified.",
            "N2A2 z-score threshold flags do not apply the publication's separate "
            "specificity-ratio criterion and are not binary binding labels.",
        ],
    }
    report_path = output_directory / "harmonization_report.json"
    if report_path.exists() and not overwrite:
        raise FileExistsError(
            f"Silver report already exists: {report_path}. Use --overwrite explicitly."
        )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Harmonize Aptafind Bronze endpoint sources into Silver tables."
    )
    parser.add_argument("--bronze-root", type=Path, default=Path("data_lake/bronze"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/thesis_data_sources.yaml"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data_lake/silver/thesis_endpoints"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_harmonization(
        bronze_root=args.bronze_root,
        config_path=args.config,
        output_directory=args.output_directory,
        overwrite=args.overwrite,
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
