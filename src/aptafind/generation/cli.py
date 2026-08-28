"""Command-line interface for training and using the Aptafind sequence CVAE."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from aptafind.generation.candidates import (
    CandidateFilterConfig,
    generate_candidate_table,
)
from aptafind.generation.checkpoint import load_generator_checkpoint
from aptafind.generation.chemistry import canonicalize_smiles
from aptafind.generation.data import load_aptamer_table, split_aptamer_table
from aptafind.generation.demo import write_synthetic_aptamer_table
from aptafind.generation.pipeline import (
    diagnose_checkpoint_conditions,
    file_sha256,
    load_run_config,
    train_sequence_generator,
)
from aptafind.generation.tokenizer import normalize_dna
from aptafind.generation.training import resolve_device


def _add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", required=True, help="Aptamer CSV input path.")
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--smiles-column", default="target_smiles")
    parser.add_argument("--target-name-column")
    parser.add_argument(
        "--legacy-target-features",
        help="Historical row-aligned target-feature CSV containing SMILES.",
    )
    parser.add_argument("--legacy-smiles-column", default="Smiles")


def _loaded_table_from_args(args: argparse.Namespace) -> Any:
    return load_aptamer_table(
        args.data,
        sequence_column=args.sequence_column,
        smiles_column=args.smiles_column,
        target_name_column=args.target_name_column,
        legacy_target_features_path=args.legacy_target_features,
        legacy_smiles_column=args.legacy_smiles_column,
    )


def _command_inspect_data(args: argparse.Namespace) -> int:
    config = load_run_config(args.config)
    loaded = _loaded_table_from_args(args)
    partitions = split_aptamer_table(
        loaded.frame,
        validation_fraction=config.data.validation_fraction,
        test_fraction=config.data.test_fraction,
        seed=config.training.seed,
        strategy=config.data.split_strategy,
    )
    print(
        json.dumps(
            {"dataset": loaded.summary(), "partitions": partitions.summary()},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _command_train(args: argparse.Namespace) -> int:
    config = load_run_config(args.config)
    result = train_sequence_generator(
        data_path=args.data,
        output_directory=args.output_directory,
        config=config,
        sequence_column=args.sequence_column,
        smiles_column=args.smiles_column,
        target_name_column=args.target_name_column,
        legacy_target_features_path=args.legacy_target_features,
        legacy_smiles_column=args.legacy_smiles_column,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "checkpoint": str(result.checkpoint_path),
                "summary": str(result.summary_path),
                "history": str(result.history_path),
                "split_manifest": str(result.split_manifest_path),
                "best_epoch": result.summary["best_epoch"],
                "test_metrics": result.summary["test_metrics"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _command_diagnose_condition(args: argparse.Namespace) -> int:
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    report = diagnose_checkpoint_conditions(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        sequence_column=args.sequence_column,
        smiles_column=args.smiles_column,
        target_name_column=args.target_name_column,
        legacy_target_features_path=args.legacy_target_features,
        legacy_smiles_column=args.legacy_smiles_column,
        permutations=args.permutations,
        seed=args.seed,
        device_name=args.device,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "checkpoint_sha256": report["checkpoint_sha256"],
                "summary": report["condition_diagnostics"]["summary"],
                "scope": report["scientific_scope"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _metadata_default(
    metadata: dict[str, Any], key: str, fallback: Any, override: Any
) -> Any:
    if override is not None:
        return override
    return metadata.get("generation_config", {}).get(key, fallback)


def _load_reference_sequences(path: str | None, sequence_column: str) -> list[str]:
    if path is None:
        return []
    frame = pd.read_csv(path)
    if sequence_column not in frame.columns:
        raise ValueError(
            f"Reference sequence column {sequence_column!r} is not in {path}."
        )
    return [normalize_dna(value) for value in frame[sequence_column]]


def _write_fasta(path: str | Path, candidates: pd.DataFrame) -> None:
    lines: list[str] = []
    for row in candidates.itertuples(index=False):
        lines.extend(
            [
                f">{row.candidate_id}|target={row.target_name}",
                row.sequence,
            ]
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _command_generate(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    loaded = load_generator_checkpoint(args.checkpoint, device=device)
    metadata = loaded.metadata
    candidate_count = int(
        _metadata_default(metadata, "samples_per_target", 10, args.count)
    )
    top_k = _metadata_default(metadata, "top_k", 5, args.top_k)
    if top_k == 0:
        top_k = None
    canonical_target = canonicalize_smiles(args.target_smiles)
    target_range = metadata.get("training_target_length_ranges", {}).get(
        canonical_target, {}
    )
    minimum_length = args.minimum_length
    if minimum_length is None:
        configured_minimum = metadata.get("generation_config", {}).get(
            "minimum_length", 12
        )
        minimum_length = max(
            int(configured_minimum), int(target_range.get("minimum_length", 0))
        )
    maximum_length = args.maximum_length
    if maximum_length is None:
        maximum_length = metadata.get("generation_config", {}).get("maximum_length")
    if maximum_length is None:
        maximum_length = target_range.get("maximum_length")
    if maximum_length is None:
        maximum_length = metadata.get("dataset", {}).get(
            "maximum_sequence_length", loaded.tokenizer.maximum_sequence_length
        )
    filters = CandidateFilterConfig(
        minimum_length=minimum_length,
        maximum_length=int(maximum_length),
        minimum_gc_fraction=float(
            _metadata_default(
                metadata, "minimum_gc_fraction", 0.25, args.minimum_gc_fraction
            )
        ),
        maximum_gc_fraction=float(
            _metadata_default(
                metadata, "maximum_gc_fraction", 0.75, args.maximum_gc_fraction
            )
        ),
        maximum_homopolymer=int(
            _metadata_default(
                metadata, "maximum_homopolymer", 4, args.maximum_homopolymer
            )
        ),
        maximum_reference_identity=float(
            _metadata_default(
                metadata,
                "maximum_reference_identity",
                0.95,
                args.maximum_reference_identity,
            )
        ),
    )
    references = _load_reference_sequences(
        args.reference_data, args.reference_sequence_column
    )
    result = generate_candidate_table(
        loaded,
        target_smiles=args.target_smiles,
        target_name=args.target_name,
        candidate_count=candidate_count,
        temperature=float(
            _metadata_default(metadata, "temperature", 0.9, args.temperature)
        ),
        top_k=top_k,
        filters=filters,
        reference_sequences=references,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.candidates.to_csv(output_path, index=False)
    if args.fasta_output:
        Path(args.fasta_output).parent.mkdir(parents=True, exist_ok=True)
        _write_fasta(args.fasta_output, result.candidates)
    metadata_output = (
        Path(args.metadata_output)
        if args.metadata_output
        else output_path.with_suffix(".metadata.json")
    )
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    generation_metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "target_name": args.target_name or canonical_target,
        "target_smiles": canonical_target,
        "candidate_count_requested": candidate_count,
        "accepted_candidates": int(len(result.candidates)),
        "draws": result.draws,
        "unique_sequences": result.unique_sequences,
        "rejected_sequences": result.rejected_sequences,
        "temperature": float(
            _metadata_default(metadata, "temperature", 0.9, args.temperature)
        ),
        "top_k": top_k,
        "seed": args.seed,
        "filters": asdict(filters),
        "reference_data": str(Path(args.reference_data)) if args.reference_data else None,
        "reference_data_sha256": (
            file_sha256(args.reference_data) if args.reference_data else None
        ),
        "reference_sequence_count": len(references),
        "scope": (
            "Computational candidates only; sequence filters are not evidence "
            "of binding, specificity, or efficacy."
        ),
    }
    metadata_output.write_text(
        json.dumps(generation_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "output": str(output_path),
        "metadata": str(metadata_output),
        "accepted_candidates": int(len(result.candidates)),
        "draws": result.draws,
        "unique_sequences": result.unique_sequences,
        "rejected_sequences": result.rejected_sequences,
        "scope": "Computational candidates only; binding requires experimental testing.",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if result.candidates.empty:
        raise RuntimeError(
            "No sampled sequence passed the configured filters; inspect or relax the "
            "filters explicitly rather than treating rejected samples as candidates."
        )
    return 0


def _command_make_demo_data(args: argparse.Namespace) -> int:
    output = write_synthetic_aptamer_table(
        args.output,
        samples_per_target=args.samples_per_target,
        sequence_length=args.sequence_length,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "scope": "Synthetic software-demo data with no binding claims.",
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aptafind-generate",
        description="Train and use the Aptafind target-conditioned ssDNA generator.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect-data", help="Validate a dataset and preview its split."
    )
    _add_dataset_arguments(inspect_parser)
    inspect_parser.add_argument("--config")
    inspect_parser.set_defaults(handler=_command_inspect_data)

    train_parser = subparsers.add_parser(
        "train", help="Train and evaluate a sequence CVAE."
    )
    _add_dataset_arguments(train_parser)
    train_parser.add_argument("--output-directory", required=True)
    train_parser.add_argument("--config")
    train_parser.add_argument("--overwrite", action="store_true")
    train_parser.set_defaults(handler=_command_train)

    diagnostic_parser = subparsers.add_parser(
        "diagnose-condition",
        help="Audit a checkpoint with zeroed and permuted target conditions.",
    )
    _add_dataset_arguments(diagnostic_parser)
    diagnostic_parser.add_argument("--checkpoint", required=True)
    diagnostic_parser.add_argument("--output", required=True)
    diagnostic_parser.add_argument("--permutations", type=int, default=10)
    diagnostic_parser.add_argument("--seed", type=int)
    diagnostic_parser.add_argument("--overwrite", action="store_true")
    diagnostic_parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu"
    )
    diagnostic_parser.set_defaults(handler=_command_diagnose_condition)

    generation_parser = subparsers.add_parser(
        "generate", help="Generate filtered candidates for one target SMILES."
    )
    generation_parser.add_argument("--checkpoint", required=True)
    generation_parser.add_argument("--target-smiles", required=True)
    generation_parser.add_argument("--target-name")
    generation_parser.add_argument("--output", required=True)
    generation_parser.add_argument("--fasta-output")
    generation_parser.add_argument("--metadata-output")
    generation_parser.add_argument("--count", type=int)
    generation_parser.add_argument("--temperature", type=float)
    generation_parser.add_argument(
        "--top-k", type=int, help="Use zero to disable top-k truncation."
    )
    generation_parser.add_argument("--minimum-length", type=int)
    generation_parser.add_argument("--maximum-length", type=int)
    generation_parser.add_argument("--minimum-gc-fraction", type=float)
    generation_parser.add_argument("--maximum-gc-fraction", type=float)
    generation_parser.add_argument("--maximum-homopolymer", type=int)
    generation_parser.add_argument("--maximum-reference-identity", type=float)
    generation_parser.add_argument("--reference-data")
    generation_parser.add_argument(
        "--reference-sequence-column", default="sequence"
    )
    generation_parser.add_argument("--seed", type=int, default=42)
    generation_parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu"
    )
    generation_parser.set_defaults(handler=_command_generate)

    demo_parser = subparsers.add_parser(
        "make-demo-data", help="Write a deterministic synthetic smoke-test dataset."
    )
    demo_parser.add_argument("--output", required=True)
    demo_parser.add_argument("--samples-per-target", type=int, default=12)
    demo_parser.add_argument("--sequence-length", type=int, default=40)
    demo_parser.add_argument("--seed", type=int, default=7)
    demo_parser.set_defaults(handler=_command_make_demo_data)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileExistsError, KeyError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
