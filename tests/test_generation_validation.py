from pathlib import Path

import pandas as pd

from aptafind.generation.validation import (
    RepeatedEvaluationConfig,
    build_strict_group_folds,
    cluster_sequence_families,
    load_repeated_evaluation_config,
)
from aptafind.generation.repeated import run_repeated_evaluation


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_sequence_family_clustering_uses_single_linkage_identity() -> None:
    families = cluster_sequence_families(
        ["AAAAAAAAAA", "AAAAAAAAAT", "CCCCCCCCCC"],
        identity_threshold=0.90,
    )

    assert families["AAAAAAAAAA"] == families["AAAAAAAAAT"]
    assert families["AAAAAAAAAA"] != families["CCCCCCCCCC"]


def test_strict_groups_keep_targets_families_and_publications_in_one_fold() -> None:
    frame = pd.DataFrame(
        {
            "sequence": [
                "AAAAAAAAAA",
                "AAAAAAAAAT",
                "CCCCCCCCCC",
                "GGGGGGGGGG",
                "TTTTTTTTTT",
                "ACACACACAC",
                "CGCGCGCGCG",
                "AGAGAGAGAG",
            ],
            "target_smiles": [f"target_{index}" for index in range(8)],
            "target_name": [f"target_{index}" for index in range(8)],
            "publication_ids": [
                "pub_shared",
                "pub_1",
                "pub_shared",
                "pub_3",
                "pub_4",
                "pub_5",
                "pub_excluded",
                "pub_7",
            ],
        }
    )
    config = RepeatedEvaluationConfig(
        fold_count=3,
        training_seeds=(3,),
        sequence_family_identity_threshold=0.90,
        excluded_publications=("pub_excluded",),
        bootstrap_replicates=10,
    )

    grouped = build_strict_group_folds(frame, config)

    assert grouped.audit["excluded_rows"] == 1
    assert grouped.audit["overlap_audit"] == {
        "maximum_folds_per_target": 1,
        "maximum_folds_per_sequence_family": 1,
        "maximum_folds_per_publication": 1,
    }
    assert set(grouped.frame["fold"]) == {0, 1, 2}
    shared_publication_folds = grouped.frame.loc[
        grouped.frame["publication_ids"] == "pub_shared", "fold"
    ]
    assert shared_publication_folds.nunique() == 1
    similar_family_folds = grouped.frame.loc[
        grouped.frame["sequence"].isin(["AAAAAAAAAA", "AAAAAAAAAT"]), "fold"
    ]
    assert similar_family_folds.nunique() == 1


def test_repeated_validation_config_loads_reviewed_repository_config() -> None:
    config = load_repeated_evaluation_config(
        REPOSITORY_ROOT / "configs" / "repeated_controlled_cvae.yaml"
    )

    assert config.fold_count == 5
    assert config.training_seeds == (42, 43, 44)
    assert config.sequence_family_identity_threshold == 0.90
    assert config.excluded_publications == ("doi:10.1093/nar/gkaf219",)


def test_repeated_runner_trains_one_resumable_paired_fold(tmp_path: Path) -> None:
    sequences = [
        "AAAACCCCGGGG",
        "TTTTGGGGCCCC",
        "ACGTACGTACGT",
        "TGCATGCATGCA",
        "AATTCCGGAATT",
        "CCGGAATTCCGG",
        "AGCTAGCTAGCT",
        "TCGATCGATCGA",
        "ATATCGCGATAT",
        "CGCGATATCGCG",
        "AACCGGTTAACC",
        "GGCCAATTGGCC",
    ]
    smiles = ["CCO", "CCN", "CCC", "CC(=O)O", "CCOC", "CCCl"]
    records = []
    for target_index, target_smiles in enumerate(smiles):
        for sequence in sequences[2 * target_index : 2 * target_index + 2]:
            records.append(
                {
                    "sequence": sequence,
                    "target_name": f"target_{target_index}",
                    "target_smiles": target_smiles,
                    "source_datasets": "synthetic_test",
                    "publication_ids": f"publication_{target_index}",
                }
            )
    data_path = tmp_path / "grouped_demo.csv"
    pd.DataFrame.from_records(records).to_csv(data_path, index=False)
    config_path = tmp_path / "repeated.yaml"
    config_path.write_text(
        """model:
  embedding_dim: 4
  encoder_hidden_dim: 4
  decoder_hidden_dim: 8
  condition_hidden_dim: 4
  latent_dim: 2
  dropout: 0.0
training:
  epochs: 1
  batch_size: 4
  learning_rate: 0.01
  beta_max: 0.01
  beta_warmup_epochs: 1
  free_bits_per_dimension: 0.01
  decoder_token_dropout: 0.10
  condition_diagnostic_permutations: 2
  patience: 1
  seed: 7
  device: cpu
data:
  maximum_sequence_length: 16
  validation_fraction: 0.20
  test_fraction: 0.20
  split_strategy: target
  fingerprint_bits: 16
  fingerprint_radius: 2
generation:
  samples_per_target: 2
  minimum_length: 4
validation:
  fold_count: 3
  training_seeds: [7]
  fold_assignment_seed: 11
  validation_fold_offset: 1
  sequence_family_identity_threshold: 0.95
  excluded_publications: []
  bootstrap_replicates: 20
  bootstrap_seed: 13
""",
        encoding="utf-8",
    )

    result = run_repeated_evaluation(
        data_path=data_path,
        config_path=config_path,
        output_directory=tmp_path / "repeated_run",
        fold_indices=[0],
    )
    resumed = run_repeated_evaluation(
        data_path=data_path,
        config_path=config_path,
        output_directory=tmp_path / "repeated_run",
        fold_indices=[0],
    )

    assert result.summary_path.exists()
    assert result.grouping_report_path.exists()
    assert result.grouping_manifest_path.exists()
    assert not result.summary["aggregate"]["complete"]
    assert result.summary["aggregate"]["completed_fold_seed_pairs"] == 1
    assert result.summary["aggregate"]["expected_fold_seed_pairs"] == 3
    assert resumed.summary["aggregate"] == result.summary["aggregate"]
