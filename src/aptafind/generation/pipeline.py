"""End-to-end orchestration for training a reproducible sequence generator."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rdkit
import torch
import yaml

from aptafind.generation.baselines import (
    evaluate_bigram_baseline,
    evaluate_unigram_baseline,
)
from aptafind.generation.checkpoint import (
    load_generator_checkpoint,
    save_generator_checkpoint,
    sequence_digest,
)
from aptafind.generation.chemistry import MoleculeFeaturizer
from aptafind.generation.data import (
    AptamerSequenceDataset,
    DatasetPartitions,
    LoadedAptamerTable,
    make_data_loader,
    load_aptamer_table,
    permute_target_assignments,
    split_aptamer_table,
)
from aptafind.generation.model import ConditionalSequenceVAE, SequenceCVAEConfig
from aptafind.generation.tokenizer import DNATokenizer
from aptafind.generation.training import (
    TrainingConfig,
    evaluate_condition_controls,
    evaluate_model,
    evaluate_reconstruction_examples,
    resolve_device,
    seed_everything,
    train_model,
)


@dataclass(frozen=True)
class DataConfig:
    maximum_sequence_length: int = 128
    validation_fraction: float = 0.10
    test_fraction: float = 0.10
    split_strategy: str = "target"
    fingerprint_bits: int = 128
    fingerprint_radius: int = 2

    def __post_init__(self) -> None:
        if self.maximum_sequence_length < 1:
            raise ValueError("maximum_sequence_length must be positive.")
        if self.fingerprint_bits < 16:
            raise ValueError("fingerprint_bits must be at least 16.")
        if self.fingerprint_radius < 1:
            raise ValueError("fingerprint_radius must be positive.")
        if self.split_strategy not in ("target", "random"):
            raise ValueError("split_strategy must be 'target' or 'random'.")


@dataclass(frozen=True)
class GenerationConfig:
    samples_per_target: int = 10
    temperature: float = 0.90
    top_k: int | None = 5
    minimum_length: int = 12
    maximum_length: int | None = None
    minimum_gc_fraction: float = 0.25
    maximum_gc_fraction: float = 0.75
    maximum_homopolymer: int = 4
    maximum_reference_identity: float = 0.95

    def __post_init__(self) -> None:
        if self.samples_per_target < 1:
            raise ValueError("samples_per_target must be positive.")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be positive when supplied.")
        if self.minimum_length < 1:
            raise ValueError("minimum_length must be positive.")
        if self.maximum_length is not None and self.maximum_length < self.minimum_length:
            raise ValueError("maximum_length cannot be smaller than minimum_length.")
        if not 0 <= self.minimum_gc_fraction <= self.maximum_gc_fraction <= 1:
            raise ValueError("Generation GC fractions are invalid.")
        if self.maximum_homopolymer < 1:
            raise ValueError("maximum_homopolymer must be positive.")
        if not 0 <= self.maximum_reference_identity <= 1:
            raise ValueError("maximum_reference_identity must be in [0, 1].")


@dataclass(frozen=True)
class RunConfig:
    model: dict[str, Any]
    training: TrainingConfig
    data: DataConfig
    generation: GenerationConfig


@dataclass(frozen=True)
class TrainPipelineResult:
    checkpoint_path: Path
    summary_path: Path
    history_path: Path
    split_manifest_path: Path
    summary: dict[str, Any]


DEFAULT_MODEL_CONFIG: dict[str, Any] = {
    "embedding_dim": 32,
    "encoder_hidden_dim": 64,
    "decoder_hidden_dim": 128,
    "condition_hidden_dim": 32,
    "latent_dim": 16,
    "dropout": 0.10,
}


def load_run_config(path: str | Path | None = None) -> RunConfig:
    """Load YAML configuration, falling back to reviewed defaults."""

    payload: dict[str, Any] = {}
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        if loaded is not None and not isinstance(loaded, dict):
            raise ValueError("Configuration root must be a mapping.")
        payload = loaded or {}

    model_payload = {**DEFAULT_MODEL_CONFIG, **payload.get("model", {})}
    return RunConfig(
        model=model_payload,
        training=TrainingConfig(**payload.get("training", {})),
        data=DataConfig(**payload.get("data", {})),
        generation=GenerationConfig(**payload.get("generation", {})),
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "numpy": str(np.__version__),
        "pandas": str(pd.__version__),
        "rdkit": str(rdkit.__version__),
        "pyyaml": str(yaml.__version__),
    }


def _build_split_manifest(partitions: Any) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for partition_name in ("train", "validation", "test"):
        partition = getattr(partitions, partition_name)
        for row in partition.itertuples(index=False):
            rows.append(
                {
                    "sequence_sha256": sequence_digest(row.sequence),
                    "target_name": row.target_name,
                    "target_smiles": row.target_smiles,
                    "partition": partition_name,
                }
            )
    return pd.DataFrame.from_records(rows).sort_values(
        ["partition", "target_smiles", "sequence_sha256"]
    )


def _training_target_length_ranges(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    ranges: dict[str, dict[str, Any]] = {}
    for target_smiles, target_frame in frame.groupby("target_smiles", sort=True):
        lengths = target_frame["sequence"].str.len()
        ranges[target_smiles] = {
            "target_name": sorted(set(target_frame["target_name"]))[0],
            "sample_count": int(len(target_frame)),
            "minimum_length": int(lengths.min()),
            "median_length": float(lengths.median()),
            "maximum_length": int(lengths.max()),
        }
    return ranges


def diagnose_checkpoint_conditions(
    *,
    checkpoint_path: str | Path,
    data_path: str | Path,
    sequence_column: str = "sequence",
    smiles_column: str = "target_smiles",
    target_name_column: str | None = None,
    legacy_target_features_path: str | Path | None = None,
    legacy_smiles_column: str = "Smiles",
    permutations: int = 10,
    seed: int | None = None,
    device_name: str = "cpu",
) -> dict[str, Any]:
    """Rebuild a checkpoint's test fold and audit target-condition dependence."""

    if permutations < 1:
        raise ValueError("permutations must be positive.")
    device = resolve_device(device_name)
    loaded_checkpoint = load_generator_checkpoint(checkpoint_path, device=device)
    metadata = loaded_checkpoint.metadata
    source_hashes = {"aptamer_data": file_sha256(data_path)}
    if legacy_target_features_path is not None:
        source_hashes["legacy_target_features"] = file_sha256(
            legacy_target_features_path
        )
    recorded_hashes = metadata.get("source_sha256", {})
    if source_hashes != recorded_hashes:
        raise ValueError(
            "Diagnostic data hashes do not match the checkpoint metadata: "
            f"observed={source_hashes}, recorded={recorded_hashes}."
        )

    loaded = load_aptamer_table(
        data_path,
        sequence_column=sequence_column,
        smiles_column=smiles_column,
        target_name_column=target_name_column,
        legacy_target_features_path=legacy_target_features_path,
        legacy_smiles_column=legacy_smiles_column,
    )
    data_config = metadata.get("data_config", {})
    training_config = metadata.get("training_config", {})
    split_seed = int(training_config.get("seed", 42))
    partitions = split_aptamer_table(
        loaded.frame,
        validation_fraction=float(data_config.get("validation_fraction", 0.10)),
        test_fraction=float(data_config.get("test_fraction", 0.10)),
        seed=split_seed,
        strategy=str(data_config.get("split_strategy", "target")),
    )
    observed_partition_summary = partitions.summary()
    recorded_partition_summary = metadata.get("partitions")
    if (
        recorded_partition_summary is not None
        and observed_partition_summary != recorded_partition_summary
    ):
        raise ValueError(
            "Reconstructed partitions do not match the checkpoint metadata."
        )
    test_dataset = AptamerSequenceDataset(
        partitions.test,
        loaded_checkpoint.tokenizer,
        loaded_checkpoint.molecule_featurizer,
    )
    diagnostic_seed = split_seed + 10_000 if seed is None else int(seed)
    diagnostics = evaluate_condition_controls(
        loaded_checkpoint.model,
        test_dataset,
        batch_size=int(training_config.get("batch_size", 16)),
        device=device,
        beta=float(training_config.get("beta_max", 0.05)),
        free_bits_per_dimension=float(
            training_config.get("free_bits_per_dimension", 0.0)
        ),
        seed=diagnostic_seed,
        permutations=permutations,
    )
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(Path(checkpoint_path)),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "source_sha256": source_hashes,
        "dataset": loaded.summary(),
        "partitions": observed_partition_summary,
        "diagnostic_seed": diagnostic_seed,
        "condition_diagnostics": diagnostics,
        "scientific_scope": (
            "Condition sensitivity in posterior reconstruction is a diagnostic, "
            "not evidence that prior samples bind a target."
        ),
    }


def compare_checkpoint_reconstruction(
    *,
    primary_checkpoint_path: str | Path,
    control_checkpoint_path: str | Path,
    data_path: str | Path,
    sequence_column: str = "sequence",
    smiles_column: str = "target_smiles",
    target_name_column: str | None = None,
    legacy_target_features_path: str | Path | None = None,
    legacy_smiles_column: str = "Smiles",
    bootstrap_replicates: int = 5_000,
    bootstrap_seed: int = 20260828,
    device_name: str = "cpu",
) -> dict[str, Any]:
    """Compare checkpoints on identical examples with target-cluster bootstrap."""

    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive.")
    device = resolve_device(device_name)
    primary = load_generator_checkpoint(primary_checkpoint_path, device=device)
    control = load_generator_checkpoint(control_checkpoint_path, device=device)
    source_hashes = {"aptamer_data": file_sha256(data_path)}
    if legacy_target_features_path is not None:
        source_hashes["legacy_target_features"] = file_sha256(
            legacy_target_features_path
        )
    for label, checkpoint in (("primary", primary), ("control", control)):
        if checkpoint.metadata.get("source_sha256", {}) != source_hashes:
            raise ValueError(f"{label} checkpoint source hashes do not match the data.")

    primary_metadata = primary.metadata
    control_metadata = control.metadata
    comparison_keys = ("data_config", "partitions", "source_sha256")
    for key in comparison_keys:
        if primary_metadata.get(key) != control_metadata.get(key):
            raise ValueError(f"Checkpoint metadata differ for required key {key!r}.")
    primary_training = primary_metadata.get("training_config", {})
    control_training = control_metadata.get("training_config", {})
    if primary_training.get("seed", 42) != control_training.get("seed", 42):
        raise ValueError("Checkpoint split seeds differ.")

    loaded = load_aptamer_table(
        data_path,
        sequence_column=sequence_column,
        smiles_column=smiles_column,
        target_name_column=target_name_column,
        legacy_target_features_path=legacy_target_features_path,
        legacy_smiles_column=legacy_smiles_column,
    )
    data_config = primary_metadata.get("data_config", {})
    split_seed = int(primary_training.get("seed", 42))
    partitions = split_aptamer_table(
        loaded.frame,
        validation_fraction=float(data_config.get("validation_fraction", 0.10)),
        test_fraction=float(data_config.get("test_fraction", 0.10)),
        seed=split_seed,
        strategy=str(data_config.get("split_strategy", "target")),
    )
    if partitions.summary() != primary_metadata.get("partitions"):
        raise ValueError("Reconstructed partitions do not match checkpoint metadata.")

    primary_dataset = AptamerSequenceDataset(
        partitions.test, primary.tokenizer, primary.molecule_featurizer
    )
    control_dataset = AptamerSequenceDataset(
        partitions.test, control.tokenizer, control.molecule_featurizer
    )
    primary_examples = evaluate_reconstruction_examples(
        primary.model,
        primary_dataset,
        batch_size=int(primary_training.get("batch_size", 16)),
        device=device,
    )
    control_examples = evaluate_reconstruction_examples(
        control.model,
        control_dataset,
        batch_size=int(control_training.get("batch_size", 16)),
        device=device,
    )
    primary_sums = np.asarray(primary_examples["nll_sums"], dtype=float)
    control_sums = np.asarray(control_examples["nll_sums"], dtype=float)
    primary_counts = np.asarray(primary_examples["token_counts"], dtype=int)
    control_counts = np.asarray(control_examples["token_counts"], dtype=int)
    if not np.array_equal(primary_counts, control_counts):
        raise ValueError("Checkpoint evaluations produced different token counts.")

    target_values = partitions.test["target_smiles"].astype(str).to_numpy()
    unique_targets = np.asarray(sorted(set(target_values)))
    target_primary_sums: list[float] = []
    target_control_sums: list[float] = []
    target_token_counts: list[int] = []
    per_target: list[dict[str, Any]] = []
    for target in unique_targets:
        mask = target_values == target
        primary_sum = float(primary_sums[mask].sum())
        control_sum = float(control_sums[mask].sum())
        token_count = int(primary_counts[mask].sum())
        primary_nll = primary_sum / token_count
        control_nll = control_sum / token_count
        target_primary_sums.append(primary_sum)
        target_control_sums.append(control_sum)
        target_token_counts.append(token_count)
        per_target.append(
            {
                "target_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
                "rows": int(mask.sum()),
                "tokens": token_count,
                "primary_nll": primary_nll,
                "control_nll": control_nll,
                "control_minus_primary_nll": control_nll - primary_nll,
            }
        )

    target_primary_array = np.asarray(target_primary_sums)
    target_control_array = np.asarray(target_control_sums)
    target_count_array = np.asarray(target_token_counts)
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_deltas = np.empty(bootstrap_replicates, dtype=float)
    for replicate in range(bootstrap_replicates):
        sampled = rng.integers(0, len(unique_targets), size=len(unique_targets))
        tokens = target_count_array[sampled].sum()
        bootstrap_deltas[replicate] = (
            target_control_array[sampled].sum()
            - target_primary_array[sampled].sum()
        ) / tokens

    total_tokens = int(primary_counts.sum())
    primary_nll = float(primary_sums.sum() / total_tokens)
    control_nll = float(control_sums.sum() / total_tokens)
    delta = control_nll - primary_nll
    confidence_interval = np.quantile(bootstrap_deltas, [0.025, 0.975])
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_checkpoint": str(Path(primary_checkpoint_path)),
        "primary_checkpoint_sha256": file_sha256(primary_checkpoint_path),
        "control_checkpoint": str(Path(control_checkpoint_path)),
        "control_checkpoint_sha256": file_sha256(control_checkpoint_path),
        "source_sha256": source_hashes,
        "partitions": partitions.summary(),
        "comparison": {
            "test_rows": int(len(partitions.test)),
            "test_targets": int(len(unique_targets)),
            "test_tokens": total_tokens,
            "primary_reconstruction_nll": primary_nll,
            "control_reconstruction_nll": control_nll,
            "control_minus_primary_nll": delta,
            "relative_nll_reduction_vs_control": delta / control_nll,
            "targets_where_control_nll_is_higher": int(
                sum(row["control_minus_primary_nll"] > 0.0 for row in per_target)
            ),
            "target_cluster_bootstrap": {
                "replicates": bootstrap_replicates,
                "seed": bootstrap_seed,
                "confidence_level": 0.95,
                "control_minus_primary_nll_interval": [
                    float(confidence_interval[0]),
                    float(confidence_interval[1]),
                ],
                "fraction_at_or_below_zero": float(
                    np.mean(bootstrap_deltas <= 0.0)
                ),
            },
        },
        "per_target": per_target,
        "scientific_scope": (
            "A paired reconstruction comparison on held-out targets is not a "
            "binding, affinity, or candidate-efficacy measurement."
        ),
    }


def train_sequence_generator(
    *,
    data_path: str | Path,
    output_directory: str | Path,
    config: RunConfig,
    sequence_column: str = "sequence",
    smiles_column: str = "target_smiles",
    target_name_column: str | None = None,
    legacy_target_features_path: str | Path | None = None,
    legacy_smiles_column: str = "Smiles",
    overwrite: bool = False,
    preloaded_table: LoadedAptamerTable | None = None,
    preassigned_partitions: DatasetPartitions | None = None,
    partition_metadata: dict[str, Any] | None = None,
) -> TrainPipelineResult:
    """Validate data, train the model, evaluate it, and save auditable outputs."""

    output_path = Path(output_directory)
    checkpoint_path = output_path / "sequence_cvae.pt"
    summary_path = output_path / "run_summary.json"
    history_path = output_path / "training_history.csv"
    split_manifest_path = output_path / "split_manifest.csv"
    expected_outputs = (
        checkpoint_path,
        summary_path,
        history_path,
        split_manifest_path,
    )
    existing_outputs = [path for path in expected_outputs if path.exists()]
    if existing_outputs and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite an existing run: "
            + ", ".join(str(path) for path in existing_outputs)
        )
    output_path.mkdir(parents=True, exist_ok=True)

    loaded: LoadedAptamerTable = preloaded_table or load_aptamer_table(
        data_path,
        sequence_column=sequence_column,
        smiles_column=smiles_column,
        target_name_column=target_name_column,
        legacy_target_features_path=legacy_target_features_path,
        legacy_smiles_column=legacy_smiles_column,
    )
    observed_maximum = int(loaded.frame["sequence"].str.len().max())
    if observed_maximum > config.data.maximum_sequence_length:
        raise ValueError(
            f"Dataset contains a length-{observed_maximum} sequence, exceeding "
            f"maximum_sequence_length={config.data.maximum_sequence_length}."
        )

    partitions = preassigned_partitions or split_aptamer_table(
        loaded.frame,
        validation_fraction=config.data.validation_fraction,
        test_fraction=config.data.test_fraction,
        seed=config.training.seed,
        strategy=config.data.split_strategy,
    )
    if any(
        partition.empty
        for partition in (partitions.train, partitions.validation, partitions.test)
    ):
        raise ValueError("Training, validation, and test partitions must be non-empty.")
    if sum(
        len(partition)
        for partition in (partitions.train, partitions.validation, partitions.test)
    ) != len(loaded.frame):
        raise ValueError("Preassigned partitions must cover every validated row once.")
    tokenizer = DNATokenizer(config.data.maximum_sequence_length)
    molecule_featurizer = MoleculeFeaturizer(
        fingerprint_bits=config.data.fingerprint_bits,
        fingerprint_radius=config.data.fingerprint_radius,
    ).fit(partitions.train["target_smiles"])

    condition_control: dict[str, Any] = {
        "permute_training_targets": config.training.permute_training_targets,
        "permutation_seed": None,
        "mapping_sha256": None,
        "target_count": int(partitions.train["target_smiles"].nunique()),
        "fixed_points": 0,
    }
    training_condition_smiles: list[str] | None = None
    if config.training.permute_training_targets:
        permutation_seed = config.training.seed
        training_condition_smiles, target_mapping = permute_target_assignments(
            partitions.train["target_smiles"], seed=permutation_seed
        )
        mapping_payload = json.dumps(
            target_mapping, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        condition_control.update(
            {
                "permutation_seed": permutation_seed,
                "mapping_sha256": hashlib.sha256(mapping_payload).hexdigest(),
            }
        )
    train_dataset = AptamerSequenceDataset(
        partitions.train,
        tokenizer,
        molecule_featurizer,
        condition_smiles=training_condition_smiles,
    )
    validation_dataset = AptamerSequenceDataset(
        partitions.validation, tokenizer, molecule_featurizer
    )
    test_dataset = AptamerSequenceDataset(
        partitions.test, tokenizer, molecule_featurizer
    )
    train_loader = make_data_loader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        seed=config.training.seed,
    )
    validation_loader = make_data_loader(
        validation_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        seed=config.training.seed,
    )
    test_loader = make_data_loader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        seed=config.training.seed,
    )

    model_config = SequenceCVAEConfig(
        vocabulary_size=tokenizer.vocabulary_size,
        condition_dimension=molecule_featurizer.condition_dimension,
        pad_token_id=tokenizer.pad_id,
        **config.model,
    )
    seed_everything(config.training.seed)
    model = ConditionalSequenceVAE(model_config)
    training_result = train_model(
        model, train_loader, validation_loader, config.training
    )
    device = resolve_device(config.training.device)
    test_metrics = evaluate_model(
        model,
        test_loader,
        device=device,
        beta=config.training.beta_max,
        free_bits_per_dimension=config.training.free_bits_per_dimension,
    )
    condition_diagnostics = evaluate_condition_controls(
        model,
        test_dataset,
        batch_size=config.training.batch_size,
        device=device,
        beta=config.training.beta_max,
        free_bits_per_dimension=config.training.free_bits_per_dimension,
        seed=config.training.seed + 10_000,
        permutations=config.training.condition_diagnostic_permutations,
    )
    unigram_baseline = evaluate_unigram_baseline(
        train_dataset, test_dataset, tokenizer
    )
    bigram_baseline = evaluate_bigram_baseline(train_dataset, test_dataset, tokenizer)

    source_hashes = {"aptamer_data": file_sha256(data_path)}
    if legacy_target_features_path is not None:
        source_hashes["legacy_target_features"] = file_sha256(
            legacy_target_features_path
        )
    summary: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": loaded.summary(),
        "partitions": partitions.summary(),
        "source_sha256": source_hashes,
        "software_versions": _software_versions(),
        "model_config": model_config.state_dict(),
        "training_config": config.training.state_dict(),
        "training_condition_control": condition_control,
        "partition_metadata": partition_metadata
        or {
            "origin": "generated",
            "strategy": partitions.strategy,
        },
        "data_config": asdict(config.data),
        "generation_config": asdict(config.generation),
        "training_target_length_ranges": _training_target_length_ranges(
            partitions.train
        ),
        "best_epoch": training_result.best_epoch,
        "stopped_epoch": training_result.stopped_epoch,
        "best_validation_loss": training_result.best_validation_loss,
        "test_metrics": test_metrics.state_dict(),
        "test_condition_diagnostics": condition_diagnostics,
        "test_token_baselines": {
            "unigram": unigram_baseline.state_dict(),
            "bigram": bigram_baseline.state_dict(),
        },
        "test_reconstruction_improvement_over_bigram": (
            bigram_baseline.negative_log_likelihood
            - test_metrics.reconstruction_loss
        )
        / bigram_baseline.negative_log_likelihood,
        "scientific_scope": (
            "Generated sequences are computational hypotheses. Sequence-level "
            "plausibility is not evidence of binding, specificity, or efficacy."
        ),
    }

    history_rows: list[dict[str, Any]] = []
    for entry in training_result.history:
        history_rows.append(
            {
                "epoch": entry["epoch"],
                "beta": entry["beta"],
                **{f"train_{key}": value for key, value in entry["train"].items()},
                **{
                    f"validation_{key}": value
                    for key, value in entry["validation"].items()
                },
            }
        )
    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    _build_split_manifest(partitions).to_csv(split_manifest_path, index=False)
    _write_json(summary_path, summary)
    save_generator_checkpoint(
        checkpoint_path,
        model=model,
        tokenizer=tokenizer,
        molecule_featurizer=molecule_featurizer,
        training_sequences=partitions.train["sequence"],
        metadata=summary,
    )
    return TrainPipelineResult(
        checkpoint_path=checkpoint_path,
        summary_path=summary_path,
        history_path=history_path,
        split_manifest_path=split_manifest_path,
        summary=summary,
    )
