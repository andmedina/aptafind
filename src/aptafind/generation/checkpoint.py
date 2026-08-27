"""Portable, versioned checkpoints for trained Aptafind generators."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from aptafind.generation.chemistry import MoleculeFeaturizer
from aptafind.generation.model import ConditionalSequenceVAE, SequenceCVAEConfig
from aptafind.generation.tokenizer import DNATokenizer, normalize_dna


CHECKPOINT_FORMAT = "aptafind-sequence-cvae"
CHECKPOINT_VERSION = 1


def sequence_digest(sequence: str) -> str:
    """Return a stable digest used for exact training-set novelty checks."""

    return hashlib.sha256(normalize_dna(sequence).encode("ascii")).hexdigest()


def sequence_digests(sequences: Iterable[str]) -> list[str]:
    return sorted({sequence_digest(sequence) for sequence in sequences})


@dataclass
class LoadedGenerator:
    model: ConditionalSequenceVAE
    tokenizer: DNATokenizer
    molecule_featurizer: MoleculeFeaturizer
    metadata: dict[str, Any]
    training_sequence_hashes: set[str]


def save_generator_checkpoint(
    path: str | Path,
    *,
    model: ConditionalSequenceVAE,
    tokenizer: DNATokenizer,
    molecule_featurizer: MoleculeFeaturizer,
    training_sequences: Iterable[str],
    metadata: dict[str, Any],
) -> Path:
    """Write an atomic checkpoint containing no raw training sequences."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "model_config": model.config.state_dict(),
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "tokenizer": tokenizer.state_dict(),
        "molecule_featurizer": molecule_featurizer.state_dict(),
        "training_sequence_hashes": sequence_digests(training_sequences),
        "metadata": metadata,
    }
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(output_path)
    return output_path


def load_generator_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> LoadedGenerator:
    """Load and validate a generator checkpoint."""

    checkpoint_path = Path(path)
    try:
        payload = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:  # PyTorch versions before ``weights_only`` was available.
        payload = torch.load(checkpoint_path, map_location=device)

    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Not an Aptafind sequence-CVAE checkpoint: {checkpoint_path}")
    if int(payload.get("version", -1)) != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {payload.get('version')!r}; "
            f"expected {CHECKPOINT_VERSION}."
        )

    model_config = SequenceCVAEConfig.from_state_dict(payload["model_config"])
    model = ConditionalSequenceVAE(model_config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    tokenizer = DNATokenizer.from_state_dict(payload["tokenizer"])
    molecule_featurizer = MoleculeFeaturizer.from_state_dict(
        payload["molecule_featurizer"]
    )
    return LoadedGenerator(
        model=model,
        tokenizer=tokenizer,
        molecule_featurizer=molecule_featurizer,
        metadata=dict(payload.get("metadata", {})),
        training_sequence_hashes=set(payload.get("training_sequence_hashes", [])),
    )
