import hashlib
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = (
    REPOSITORY_ROOT / "benchmarks" / "thesis_cvae_baseline_v0.1.0.json"
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
