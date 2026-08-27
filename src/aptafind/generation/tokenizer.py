"""Tokenization for variable-length ssDNA sequences."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


DNA_ALPHABET = ("A", "C", "G", "T")
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"


def normalize_dna(sequence: str) -> str:
    """Return uppercase DNA with whitespace removed and validate its alphabet."""

    normalized = re.sub(r"\s+", "", str(sequence)).upper()
    if not normalized:
        raise ValueError("DNA sequence is empty after normalization.")
    invalid = sorted(set(normalized).difference(DNA_ALPHABET))
    if invalid:
        raise ValueError(
            f"DNA sequence contains invalid symbols {invalid}: {sequence!r}"
        )
    return normalized


@dataclass(frozen=True)
class DNATokenizer:
    """Encode DNA directly as categorical tokens with explicit boundaries."""

    maximum_sequence_length: int = 128

    def __post_init__(self) -> None:
        if self.maximum_sequence_length < 1:
            raise ValueError("maximum_sequence_length must be positive.")

    @property
    def token_to_id(self) -> dict[str, int]:
        return {
            PAD_TOKEN: 0,
            BOS_TOKEN: 1,
            EOS_TOKEN: 2,
            "A": 3,
            "C": 4,
            "G": 5,
            "T": 6,
        }

    @property
    def id_to_token(self) -> dict[int, str]:
        return {value: key for key, value in self.token_to_id.items()}

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS_TOKEN]

    @property
    def vocabulary_size(self) -> int:
        return len(self.token_to_id)

    @property
    def encoded_length(self) -> int:
        return self.maximum_sequence_length + 2

    def encode(self, sequence: str) -> tuple[list[int], int]:
        """Encode one sequence and return padded token ids plus true length."""

        normalized = normalize_dna(sequence)
        if len(normalized) > self.maximum_sequence_length:
            raise ValueError(
                f"Sequence length {len(normalized)} exceeds configured maximum "
                f"{self.maximum_sequence_length}."
            )
        ids = [self.bos_id]
        ids.extend(self.token_to_id[base] for base in normalized)
        ids.append(self.eos_id)
        length = len(ids)
        ids.extend([self.pad_id] * (self.encoded_length - length))
        return ids, length

    def decode(self, token_ids: Iterable[int]) -> str:
        """Decode tokens until EOS, ignoring boundary and padding tokens."""

        bases: list[str] = []
        for raw_token_id in token_ids:
            token_id = int(raw_token_id)
            if token_id == self.eos_id:
                break
            token = self.id_to_token.get(token_id)
            if token in DNA_ALPHABET:
                bases.append(token)
        return "".join(bases)

    def state_dict(self) -> dict[str, int]:
        return {"maximum_sequence_length": self.maximum_sequence_length}

    @classmethod
    def from_state_dict(cls, state: dict[str, int]) -> "DNATokenizer":
        return cls(maximum_sequence_length=int(state["maximum_sequence_length"]))
