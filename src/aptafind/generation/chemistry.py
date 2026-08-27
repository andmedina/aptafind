"""Deterministic small-molecule conditioning features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator


DESCRIPTOR_NAMES = (
    "molecular_weight",
    "log_p",
    "hydrogen_bond_donors",
    "hydrogen_bond_acceptors",
    "topological_polar_surface_area",
    "rotatable_bonds",
    "ring_count",
    "fraction_csp3",
)


def molecule_from_smiles(smiles: str) -> Chem.Mol:
    """Parse a SMILES string or fail with a useful validation error."""

    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"Invalid molecular SMILES: {smiles!r}")
    return molecule


def canonicalize_smiles(smiles: str) -> str:
    """Return an isomeric canonical SMILES representation."""

    return Chem.MolToSmiles(
        molecule_from_smiles(smiles), canonical=True, isomericSmiles=True
    )


def molecule_descriptors(molecule: Chem.Mol) -> np.ndarray:
    """Calculate the small descriptor set used beside the Morgan fingerprint."""

    return np.asarray(
        [
            Descriptors.MolWt(molecule),
            Descriptors.MolLogP(molecule),
            Lipinski.NumHDonors(molecule),
            Lipinski.NumHAcceptors(molecule),
            Descriptors.TPSA(molecule),
            Lipinski.NumRotatableBonds(molecule),
            Lipinski.RingCount(molecule),
            Lipinski.FractionCSP3(molecule),
        ],
        dtype=np.float32,
    )


@dataclass
class MoleculeFeaturizer:
    """Fit training-only descriptor scaling and emit molecule conditions."""

    fingerprint_bits: int = 128
    fingerprint_radius: int = 2
    descriptor_mean: np.ndarray | None = None
    descriptor_scale: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.fingerprint_bits < 16:
            raise ValueError("fingerprint_bits must be at least 16.")
        if self.fingerprint_radius < 1:
            raise ValueError("fingerprint_radius must be positive.")

    @property
    def condition_dimension(self) -> int:
        return self.fingerprint_bits + len(DESCRIPTOR_NAMES)

    @property
    def is_fitted(self) -> bool:
        return self.descriptor_mean is not None and self.descriptor_scale is not None

    def fit(self, smiles_values: Iterable[str]) -> "MoleculeFeaturizer":
        """Fit descriptor scaling using unique training molecules only."""

        canonical = sorted({canonicalize_smiles(value) for value in smiles_values})
        if not canonical:
            raise ValueError("At least one molecule is required to fit the featurizer.")
        descriptors = np.vstack(
            [molecule_descriptors(molecule_from_smiles(value)) for value in canonical]
        )
        self.descriptor_mean = descriptors.mean(axis=0).astype(np.float32)
        scale = descriptors.std(axis=0).astype(np.float32)
        scale[scale < 1e-8] = 1.0
        self.descriptor_scale = scale
        return self

    def transform_one(self, smiles: str) -> np.ndarray:
        """Create a standardized descriptor plus Morgan-fingerprint vector."""

        if not self.is_fitted:
            raise RuntimeError("MoleculeFeaturizer must be fitted before transform.")
        molecule = molecule_from_smiles(smiles)
        descriptors = molecule_descriptors(molecule)
        standardized = (descriptors - self.descriptor_mean) / self.descriptor_scale

        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.fingerprint_radius,
            fpSize=self.fingerprint_bits,
        )
        fingerprint = generator.GetFingerprint(molecule)
        fingerprint_array = np.zeros(self.fingerprint_bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_array)
        return np.concatenate([standardized, fingerprint_array]).astype(np.float32)

    def transform(self, smiles_values: Iterable[str]) -> np.ndarray:
        values = list(smiles_values)
        if not values:
            return np.empty((0, self.condition_dimension), dtype=np.float32)
        return np.vstack([self.transform_one(value) for value in values])

    def state_dict(self) -> dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError("Cannot serialize an unfitted MoleculeFeaturizer.")
        return {
            "fingerprint_bits": self.fingerprint_bits,
            "fingerprint_radius": self.fingerprint_radius,
            "descriptor_names": list(DESCRIPTOR_NAMES),
            "descriptor_mean": self.descriptor_mean.tolist(),
            "descriptor_scale": self.descriptor_scale.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "MoleculeFeaturizer":
        names = tuple(state["descriptor_names"])
        if names != DESCRIPTOR_NAMES:
            raise ValueError(
                f"Descriptor schema mismatch: expected {DESCRIPTOR_NAMES}, found {names}."
            )
        return cls(
            fingerprint_bits=int(state["fingerprint_bits"]),
            fingerprint_radius=int(state["fingerprint_radius"]),
            descriptor_mean=np.asarray(state["descriptor_mean"], dtype=np.float32),
            descriptor_scale=np.asarray(state["descriptor_scale"], dtype=np.float32),
        )
