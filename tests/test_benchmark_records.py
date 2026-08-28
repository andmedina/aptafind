import hashlib
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = (
    REPOSITORY_ROOT / "benchmarks" / "thesis_cvae_baseline_v0.1.0.json"
)
CONTROLLED_PATH = (
    REPOSITORY_ROOT / "benchmarks" / "thesis_cvae_controlled_v0.3.0.json"
)


def test_thesis_cvae_baseline_record_is_internally_consistent() -> None:
    record = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    assert record["baseline_id"] == "thesis-cvae-baseline-v0.1.0"
    assert record["status"] == "frozen"
    assert record["code"]["tag"] == record["baseline_id"]

    dataset = record["dataset"]
    evaluation = record["evaluation"]
    partition_rows = sum(
        evaluation[partition]["rows"]
        for partition in ("training", "validation", "test")
    )
    assert partition_rows == dataset["validated_sequence_target_pairs"]

    for source_hash in dataset["sources_sha256"].values():
        assert len(source_hash) == hashlib.sha256().digest_size * 2

    results = record["results"]
    assert (
        results["conditional_sequence_cvae"]["negative_log_likelihood"]
        < results["bigram"]["negative_log_likelihood"]
    )


def test_controlled_cvae_record_preserves_matched_comparison() -> None:
    record = json.loads(CONTROLLED_PATH.read_text(encoding="utf-8"))

    assert record["benchmark_id"] == "thesis-cvae-controlled-v0.3.0"
    assert record["status"] == "frozen"
    assert record["code"]["tag"] == record["benchmark_id"]
    assert (
        record["artifact_hashes"]["primary_split_manifest"]
        == record["artifact_hashes"]["permuted_split_manifest"]
        == record["evaluation"]["split_manifest_sha256"]
    )
    partition_rows = sum(
        record["evaluation"][partition]["rows"]
        for partition in ("training", "validation", "test")
    )
    assert partition_rows == record["dataset"]["validated_sequence_target_pairs"]
    assert record["primary_results"]["test"]["active_latent_units"] == 16
    assert record["historical_v0_2_diagnostic"]["active_latent_units"] == 0
    assert record["target_label_comparison"]["control_minus_primary_nll"] > 0
    interval = record["target_label_comparison"]["target_cluster_bootstrap"][
        "control_minus_primary_nll_interval"
    ]
    assert interval[0] < 0 < interval[1]
    for artifact_hash in record["artifact_hashes"].values():
        assert len(artifact_hash) == hashlib.sha256().digest_size * 2
