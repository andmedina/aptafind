import pytest

from aptafind.generation.tokenizer import DNATokenizer, normalize_dna


def test_tokenizer_round_trip_preserves_variable_length_dna() -> None:
    tokenizer = DNATokenizer(maximum_sequence_length=8)

    token_ids, length = tokenizer.encode(" acgt ac ")

    assert length == 8
    assert len(token_ids) == 10
    assert tokenizer.decode(token_ids) == "ACGTAC"


def test_normalize_dna_rejects_ambiguous_bases() -> None:
    with pytest.raises(ValueError, match="invalid symbols"):
        normalize_dna("ACNT")


def test_tokenizer_rejects_sequences_above_configured_maximum() -> None:
    tokenizer = DNATokenizer(maximum_sequence_length=3)

    with pytest.raises(ValueError, match="exceeds configured maximum"):
        tokenizer.encode("ACGT")
