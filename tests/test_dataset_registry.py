from pathlib import Path

import yaml


REGISTRY_PATH = Path("manifests/datasets.yaml")


def _load_registry() -> dict:
    return yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))


def test_dataset_registry_has_unique_valid_entries() -> None:
    registry = _load_registry()
    datasets = registry["datasets"]
    dataset_ids = [dataset["dataset_id"] for dataset in datasets]

    assert registry["manifest_version"] == "0.2.0"
    assert len(dataset_ids) == len(set(dataset_ids))
    assert len(dataset_ids) >= 20

    valid_statuses = set(registry["status_vocabulary"])
    valid_evidence = set(registry["evidence_vocabulary"])
    valid_roles = set(registry["source_role_vocabulary"])

    for dataset in datasets:
        assert dataset["status"] in valid_statuses
        assert dataset["source_role"] in valid_roles
        assert set(dataset["evidence_types"]) <= valid_evidence
        assert dataset["source"]["repository"]
        assert {"access", "license", "license_status", "redistribution_status"} <= set(
            dataset["rights"]
        )

        bronze_uri = dataset.get("local", {}).get("bronze_uri")
        if bronze_uri is not None:
            assert bronze_uri == f"bronze/{dataset['dataset_id']}"

        for entry in dataset.get("files", {}).get("entries", []):
            assert entry["name"]


def test_high_value_small_molecule_sources_are_registered() -> None:
    registry = _load_registry()
    by_id = {dataset["dataset_id"]: dataset for dataset in registry["datasets"]}

    expected = {
        "aptabench_current_review_release",
        "n2a2_kynurenine_specificity_2022",
        "xiao_thermodynamics_specificity_2025",
        "dl_selex_steroid_endpoints_2025",
        "utexas_aptamer_database_v1_1_0",
        "aptadb_2023_12_03",
        "aptamer_base_archive",
    }
    assert expected <= set(by_id)

    assert (
        by_id["n2a2_kynurenine_specificity_2022"]["reported_counts"][
            "screened_clusters"
        ]
        == 2_800_000
    )
    assert (
        by_id["xiao_thermodynamics_specificity_2025"]["reported_counts"][
            "dna_aptamers_with_specificity_profiles"
        ]
        == 218
    )
    assert (
        by_id["dl_selex_steroid_endpoints_2025"]["reported_counts"][
            "literature_derived_steroid_aptamer_pairs"
        ]
        == 195
    )


def test_dl_selex_trajectory_inventories_are_checksumed() -> None:
    registry = _load_registry()
    trajectories = [
        dataset
        for dataset in registry["datasets"]
        if dataset["source_role"] == "selection_trajectory"
    ]

    assert len(trajectories) == 3
    total_size = sum(
        dataset["files"]["expected_total_size_bytes"] for dataset in trajectories
    )
    assert total_size > 13_000_000_000

    for dataset in trajectories:
        assert dataset["rights"]["license"] == "CC-BY-4.0"
        fastq_entries = [
            entry
            for entry in dataset["files"]["entries"]
            if entry["name"].endswith((".fq.gz", ".fastq.gz"))
        ]
        assert fastq_entries
        assert all(len(entry["md5"]) == 32 for entry in fastq_entries)


def test_profiled_steroid_supplements_have_frozen_source_metadata() -> None:
    registry = _load_registry()
    by_id = {dataset["dataset_id"]: dataset for dataset in registry["datasets"]}

    expected = {
        "one_pot_selex_steroids_2019": {
            "filename": "ao9b02412_si_001.pdf",
            "md5": "8435f5da2ad55836c6479c1fbd2d0dd9",
        },
        "estradiol_capture_selex_2022": {
            "filename": "es2c05808_si_001.pdf",
            "md5": "44964ce923b548e9b0575b199c721395",
        },
    }

    for dataset_id, source in expected.items():
        dataset = by_id[dataset_id]
        entry = dataset["files"]["entries"][0]

        assert dataset["status"] == "profiled"
        assert dataset["local"]["download_status"] == (
            "downloaded_checksum_verified_and_profiled"
        )
        assert dataset["rights"]["license"] == "CC-BY-NC-4.0"
        assert entry["name"] == source["filename"]
        assert entry["md5"] == source["md5"]
