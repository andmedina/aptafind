"""Resumable repeated evaluation with strict provenance-aware folds."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from aptafind.generation.checkpoint import (
    load_generator_checkpoint,
    sequence_digest,
)
from aptafind.generation.data import (
    AptamerSequenceDataset,
    DatasetPartitions,
    LoadedAptamerTable,
    load_aptamer_table,
)
from aptafind.generation.pipeline import (
    file_sha256,
    load_run_config,
    train_sequence_generator,
)
from aptafind.generation.training import (
    evaluate_reconstruction_examples,
    resolve_device,
)
from aptafind.generation.validation import (
    RepeatedEvaluationConfig,
    build_strict_group_folds,
    load_repeated_evaluation_config,
)


@dataclass(frozen=True)
class RepeatedEvaluationResult:
    summary_path: Path
    grouping_report_path: Path
    grouping_manifest_path: Path
    summary: dict[str, Any]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _hash_values(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(set(values))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _grouping_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        publications = tuple(row.publication_tokens)
        records.append(
            {
                "sequence_sha256": sequence_digest(row.sequence),
                "target_sha256": hashlib.sha256(
                    row.target_smiles.encode("utf-8")
                ).hexdigest(),
                "sequence_family_id": row.sequence_family_id,
                "independence_group_id": row.independence_group_id,
                "publication_set_sha256": _hash_values(publications),
                "fold": int(row.fold),
            }
        )
    return pd.DataFrame.from_records(records).sort_values(
        ["fold", "independence_group_id", "target_sha256", "sequence_sha256"]
    )


def _expected_run_paths(directory: Path) -> dict[str, Path]:
    return {
        "checkpoint": directory / "sequence_cvae.pt",
        "summary": directory / "run_summary.json",
        "history": directory / "training_history.csv",
        "split_manifest": directory / "split_manifest.csv",
    }


def _train_or_resume(
    *,
    data_path: str | Path,
    directory: Path,
    config: Any,
    loaded: LoadedAptamerTable,
    partitions: DatasetPartitions,
    partition_metadata: dict[str, Any],
    overwrite: bool,
) -> tuple[dict[str, Path], dict[str, Any]]:
    paths = _expected_run_paths(directory)
    existing = [path for path in paths.values() if path.exists()]
    if len(existing) == len(paths) and not overwrite:
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        if summary.get("source_sha256", {}).get("aptamer_data") != file_sha256(
            data_path
        ):
            raise ValueError(f"Existing run source hash mismatch: {directory}")
        if summary.get("partition_metadata") != partition_metadata:
            raise ValueError(f"Existing run partition metadata mismatch: {directory}")
        return paths, summary
    if existing and not overwrite:
        raise FileExistsError(
            "Incomplete repeated run exists; pass --overwrite to replace it: "
            + ", ".join(str(path) for path in existing)
        )
    result = train_sequence_generator(
        data_path=data_path,
        output_directory=directory,
        config=config,
        overwrite=overwrite,
        preloaded_table=loaded,
        preassigned_partitions=partitions,
        partition_metadata=partition_metadata,
    )
    return paths, result.summary


def _paired_example_metrics(
    *,
    primary_checkpoint: Path,
    control_checkpoint: Path,
    test_frame: pd.DataFrame,
    batch_size: int,
    device_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = resolve_device(device_name)
    primary = load_generator_checkpoint(primary_checkpoint, device=device)
    control = load_generator_checkpoint(control_checkpoint, device=device)
    primary_dataset = AptamerSequenceDataset(
        test_frame, primary.tokenizer, primary.molecule_featurizer
    )
    control_dataset = AptamerSequenceDataset(
        test_frame, control.tokenizer, control.molecule_featurizer
    )
    primary_metrics = evaluate_reconstruction_examples(
        primary.model,
        primary_dataset,
        batch_size=batch_size,
        device=device,
    )
    control_metrics = evaluate_reconstruction_examples(
        control.model,
        control_dataset,
        batch_size=batch_size,
        device=device,
    )
    primary_sums = np.asarray(primary_metrics["nll_sums"], dtype=float)
    control_sums = np.asarray(control_metrics["nll_sums"], dtype=float)
    primary_counts = np.asarray(primary_metrics["token_counts"], dtype=int)
    control_counts = np.asarray(control_metrics["token_counts"], dtype=int)
    if not np.array_equal(primary_counts, control_counts):
        raise RuntimeError("Paired checkpoints produced different test token counts.")
    return primary_sums, control_sums, primary_counts


def _aggregate_results(
    *,
    run_records: list[dict[str, Any]],
    target_records: dict[str, dict[str, Any]],
    validation_config: RepeatedEvaluationConfig,
    selected_folds: list[int],
) -> dict[str, Any]:
    target_rows = list(target_records.values())
    if not target_rows:
        raise ValueError("No repeated-evaluation target results were collected.")
    primary_sums = np.asarray([row["primary_nll_sum"] for row in target_rows])
    control_sums = np.asarray([row["control_nll_sum"] for row in target_rows])
    token_counts = np.asarray([row["token_count"] for row in target_rows])
    total_tokens = int(token_counts.sum())
    primary_nll = float(primary_sums.sum() / total_tokens)
    control_nll = float(control_sums.sum() / total_tokens)
    delta = control_nll - primary_nll

    rng = np.random.default_rng(validation_config.bootstrap_seed)
    bootstrap_deltas = np.empty(
        validation_config.bootstrap_replicates, dtype=float
    )
    for replicate in range(validation_config.bootstrap_replicates):
        sampled = rng.integers(0, len(target_rows), size=len(target_rows))
        bootstrap_deltas[replicate] = (
            control_sums[sampled].sum() - primary_sums[sampled].sum()
        ) / token_counts[sampled].sum()
    interval = np.quantile(bootstrap_deltas, [0.025, 0.975])

    paired_deltas = np.asarray(
        [row["control_minus_primary_nll"] for row in run_records]
    )
    primary_active_units = np.asarray(
        [row["primary_test_metrics"]["active_latent_units"] for row in run_records]
    )
    control_active_units = np.asarray(
        [row["control_test_metrics"]["active_latent_units"] for row in run_records]
    )
    primary_condition_deltas = np.asarray(
        [
            row["primary_condition_summary"][
                "decoder_only_reconstruction_nll_delta_mean"
            ]
            for row in run_records
        ]
    )
    control_condition_deltas = np.asarray(
        [
            row["control_condition_summary"][
                "decoder_only_reconstruction_nll_delta_mean"
            ]
            for row in run_records
        ]
    )
    complete = set(selected_folds) == set(range(validation_config.fold_count))
    return {
        "complete": complete,
        "completed_fold_seed_pairs": len(run_records),
        "expected_fold_seed_pairs": (
            validation_config.fold_count * len(validation_config.training_seeds)
        ),
        "evaluated_folds": selected_folds,
        "evaluated_targets": len(target_rows),
        "evaluated_tokens_including_seed_repeats": total_tokens,
        "primary_reconstruction_nll": primary_nll,
        "control_reconstruction_nll": control_nll,
        "control_minus_primary_nll": delta,
        "relative_nll_reduction_vs_control": delta / control_nll,
        "paired_run_delta": {
            "mean": float(paired_deltas.mean()),
            "standard_deviation": float(paired_deltas.std()),
            "minimum": float(paired_deltas.min()),
            "maximum": float(paired_deltas.max()),
            "positive_pairs": int((paired_deltas > 0.0).sum()),
        },
        "target_cluster_bootstrap": {
            "replicates": validation_config.bootstrap_replicates,
            "seed": validation_config.bootstrap_seed,
            "confidence_level": 0.95,
            "control_minus_primary_nll_interval": [
                float(interval[0]),
                float(interval[1]),
            ],
            "fraction_at_or_below_zero": float(
                np.mean(bootstrap_deltas <= 0.0)
            ),
        },
        "targets_where_control_nll_is_higher": int(
            sum(
                row["control_nll_sum"] / row["token_count"]
                > row["primary_nll_sum"] / row["token_count"]
                for row in target_rows
            )
        ),
        "primary_active_latent_units": {
            "minimum": int(primary_active_units.min()),
            "mean": float(primary_active_units.mean()),
            "maximum": int(primary_active_units.max()),
        },
        "control_active_latent_units": {
            "minimum": int(control_active_units.min()),
            "mean": float(control_active_units.mean()),
            "maximum": int(control_active_units.max()),
        },
        "primary_wrong_target_nll_delta": {
            "mean": float(primary_condition_deltas.mean()),
            "standard_deviation": float(primary_condition_deltas.std()),
        },
        "control_wrong_target_nll_delta": {
            "mean": float(control_condition_deltas.mean()),
            "standard_deviation": float(control_condition_deltas.std()),
        },
    }


def run_repeated_evaluation(
    *,
    data_path: str | Path,
    config_path: str | Path,
    output_directory: str | Path,
    fold_indices: Iterable[int] | None = None,
    overwrite: bool = False,
) -> RepeatedEvaluationResult:
    """Run or resume paired real/permuted models over strict grouped folds."""

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    run_config = load_run_config(config_path)
    validation_config = load_repeated_evaluation_config(config_path)
    loaded = load_aptamer_table(data_path)
    grouped = build_strict_group_folds(loaded.frame, validation_config)
    filtered_loaded = LoadedAptamerTable(
        frame=grouped.frame,
        source_rows=loaded.source_rows,
        duplicate_rows_removed=loaded.duplicate_rows_removed,
        used_legacy_row_alignment=loaded.used_legacy_row_alignment,
    )

    grouping_report_path = output_path / "grouping_report.json"
    grouping_manifest_path = output_path / "grouping_manifest.csv"
    summary_path = output_path / "repeated_summary.json"
    _write_json(grouping_report_path, grouped.audit)
    _grouping_manifest(grouped.frame).to_csv(grouping_manifest_path, index=False)

    if fold_indices is None:
        selected_folds = list(range(validation_config.fold_count))
    else:
        selected_folds = sorted(set(int(value) for value in fold_indices))
    if not selected_folds:
        raise ValueError("At least one fold must be selected.")
    invalid_folds = [
        fold
        for fold in selected_folds
        if not 0 <= fold < validation_config.fold_count
    ]
    if invalid_folds:
        raise ValueError(f"Fold indices are out of range: {invalid_folds}")

    target_records: dict[str, dict[str, Any]] = {}
    run_records: list[dict[str, Any]] = []
    for test_fold in selected_folds:
        validation_fold = (
            test_fold + validation_config.validation_fold_offset
        ) % validation_config.fold_count
        test_frame = grouped.frame[grouped.frame["fold"] == test_fold].reset_index(
            drop=True
        )
        validation_frame = grouped.frame[
            grouped.frame["fold"] == validation_fold
        ].reset_index(drop=True)
        train_frame = grouped.frame[
            ~grouped.frame["fold"].isin([test_fold, validation_fold])
        ].reset_index(drop=True)
        partitions = DatasetPartitions(
            train=train_frame,
            validation=validation_frame,
            test=test_frame,
            strategy="preassigned_group",
        )
        partition_metadata = {
            "origin": "strict_repeated_group_fold",
            "test_fold": test_fold,
            "validation_fold": validation_fold,
            "fold_count": validation_config.fold_count,
            "fold_assignment_seed": validation_config.fold_assignment_seed,
            "sequence_family_identity_threshold": (
                validation_config.sequence_family_identity_threshold
            ),
            "excluded_publications": list(
                validation_config.excluded_publications
            ),
        }

        for training_seed in validation_config.training_seeds:
            pair_directory = (
                output_path / f"fold_{test_fold}" / f"seed_{training_seed}"
            )
            primary_directory = pair_directory / "primary"
            control_directory = pair_directory / "permuted_control"
            primary_config = replace(
                run_config,
                training=replace(
                    run_config.training,
                    seed=training_seed,
                    permute_training_targets=False,
                ),
            )
            control_config = replace(
                run_config,
                training=replace(
                    run_config.training,
                    seed=training_seed,
                    permute_training_targets=True,
                ),
            )
            primary_paths, primary_summary = _train_or_resume(
                data_path=data_path,
                directory=primary_directory,
                config=primary_config,
                loaded=filtered_loaded,
                partitions=partitions,
                partition_metadata=partition_metadata,
                overwrite=overwrite,
            )
            control_paths, control_summary = _train_or_resume(
                data_path=data_path,
                directory=control_directory,
                config=control_config,
                loaded=filtered_loaded,
                partitions=partitions,
                partition_metadata=partition_metadata,
                overwrite=overwrite,
            )
            if file_sha256(primary_paths["split_manifest"]) != file_sha256(
                control_paths["split_manifest"]
            ):
                raise RuntimeError("Primary and control split manifests differ.")

            primary_sums, control_sums, token_counts = _paired_example_metrics(
                primary_checkpoint=primary_paths["checkpoint"],
                control_checkpoint=control_paths["checkpoint"],
                test_frame=test_frame,
                batch_size=primary_config.training.batch_size,
                device_name=primary_config.training.device,
            )
            primary_nll = float(primary_sums.sum() / token_counts.sum())
            control_nll = float(control_sums.sum() / token_counts.sum())
            comparison = {
                "test_fold": test_fold,
                "validation_fold": validation_fold,
                "training_seed": training_seed,
                "test_rows": int(len(test_frame)),
                "test_targets": int(test_frame["target_smiles"].nunique()),
                "test_tokens": int(token_counts.sum()),
                "primary_reconstruction_nll": primary_nll,
                "control_reconstruction_nll": control_nll,
                "control_minus_primary_nll": control_nll - primary_nll,
                "primary_checkpoint_sha256": file_sha256(
                    primary_paths["checkpoint"]
                ),
                "control_checkpoint_sha256": file_sha256(
                    control_paths["checkpoint"]
                ),
                "split_manifest_sha256": file_sha256(
                    primary_paths["split_manifest"]
                ),
            }
            _write_json(pair_directory / "paired_comparison.json", comparison)
            run_records.append(
                {
                    **comparison,
                    "primary_test_metrics": primary_summary["test_metrics"],
                    "control_test_metrics": control_summary["test_metrics"],
                    "primary_condition_summary": primary_summary[
                        "test_condition_diagnostics"
                    ]["summary"],
                    "control_condition_summary": control_summary[
                        "test_condition_diagnostics"
                    ]["summary"],
                    "control_training_mapping_sha256": control_summary[
                        "training_condition_control"
                    ]["mapping_sha256"],
                }
            )

            target_values = test_frame["target_smiles"].astype(str).to_numpy()
            for target in sorted(set(target_values)):
                mask = target_values == target
                record = target_records.setdefault(
                    target,
                    {
                        "target_sha256": hashlib.sha256(
                            target.encode("utf-8")
                        ).hexdigest(),
                        "test_fold": test_fold,
                        "primary_nll_sum": 0.0,
                        "control_nll_sum": 0.0,
                        "token_count": 0,
                        "training_seed_count": 0,
                    },
                )
                if record["test_fold"] != test_fold:
                    raise RuntimeError("A target appeared in more than one test fold.")
                record["primary_nll_sum"] += float(primary_sums[mask].sum())
                record["control_nll_sum"] += float(control_sums[mask].sum())
                record["token_count"] += int(token_counts[mask].sum())
                record["training_seed_count"] += 1

    aggregate = _aggregate_results(
        run_records=run_records,
        target_records=target_records,
        validation_config=validation_config,
        selected_folds=selected_folds,
    )
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_data": str(Path(data_path)),
        "source_sha256": file_sha256(data_path),
        "config": str(Path(config_path)),
        "config_sha256": file_sha256(config_path),
        "validation_config": validation_config.state_dict(),
        "grouping_audit": grouped.audit,
        "aggregate": aggregate,
        "runs": run_records,
        "per_target": sorted(
            target_records.values(), key=lambda row: row["target_sha256"]
        ),
        "scientific_scope": (
            "Repeated reconstruction and condition controls do not establish "
            "binding, affinity, specificity, or candidate efficacy."
        ),
    }
    _write_json(summary_path, summary)
    return RepeatedEvaluationResult(
        summary_path=summary_path,
        grouping_report_path=grouping_report_path,
        grouping_manifest_path=grouping_manifest_path,
        summary=summary,
    )
