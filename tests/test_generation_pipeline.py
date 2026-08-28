from pathlib import Path

from aptafind.generation.candidates import (
    CandidateFilterConfig,
    generate_candidate_table,
)
from aptafind.generation.checkpoint import load_generator_checkpoint
from aptafind.generation.demo import SYNTHETIC_TARGETS, write_synthetic_aptamer_table
from aptafind.generation.pipeline import (
    DataConfig,
    GenerationConfig,
    RunConfig,
    diagnose_checkpoint_conditions,
    train_sequence_generator,
)
from aptafind.generation.training import TrainingConfig


def test_tiny_pipeline_trains_checkpoints_and_generates(tmp_path: Path) -> None:
    dataset_path = write_synthetic_aptamer_table(
        tmp_path / "demo.csv",
        samples_per_target=4,
        sequence_length=20,
        seed=11,
    )
    config = RunConfig(
        model={
            "embedding_dim": 8,
            "encoder_hidden_dim": 8,
            "decoder_hidden_dim": 12,
            "condition_hidden_dim": 6,
            "latent_dim": 4,
            "dropout": 0.0,
        },
        training=TrainingConfig(
            epochs=2,
            batch_size=4,
            learning_rate=0.01,
            beta_max=0.01,
            beta_warmup_epochs=1,
            free_bits_per_dimension=0.02,
            decoder_token_dropout=0.20,
            condition_diagnostic_permutations=2,
            patience=2,
            seed=13,
            device="cpu",
        ),
        data=DataConfig(
            maximum_sequence_length=20,
            validation_fraction=0.25,
            test_fraction=0.25,
            split_strategy="target",
            fingerprint_bits=16,
        ),
        generation=GenerationConfig(samples_per_target=2, minimum_length=5),
    )

    result = train_sequence_generator(
        data_path=dataset_path,
        output_directory=tmp_path / "run",
        config=config,
    )
    loaded = load_generator_checkpoint(result.checkpoint_path)
    diagnostic = diagnose_checkpoint_conditions(
        checkpoint_path=result.checkpoint_path,
        data_path=dataset_path,
        permutations=2,
    )
    generated = generate_candidate_table(
        loaded,
        target_name=SYNTHETIC_TARGETS[0][0],
        target_smiles=SYNTHETIC_TARGETS[0][1],
        candidate_count=2,
        temperature=1.0,
        top_k=5,
        filters=CandidateFilterConfig(
            minimum_length=5,
            maximum_length=20,
            minimum_gc_fraction=0.0,
            maximum_gc_fraction=1.0,
            maximum_homopolymer=20,
            maximum_reference_identity=1.0,
        ),
        seed=17,
    )

    assert result.checkpoint_path.exists()
    assert result.summary_path.exists()
    assert result.history_path.exists()
    assert result.split_manifest_path.exists()
    assert result.summary["software_versions"]["pytorch"]
    assert result.summary["test_metrics"]["effective_kl_divergence"] >= 0.08
    assert not result.summary["test_condition_diagnostics"]["summary"]["available"]
    assert diagnostic["checkpoint_sha256"]
    assert len(loaded.metadata["training_target_length_ranges"]) == 2
    assert len(generated.candidates) == 2
    assert generated.candidates["candidate_rank"].tolist() == [1, 2]
    assert generated.candidates["passes_sequence_filters"].all()
