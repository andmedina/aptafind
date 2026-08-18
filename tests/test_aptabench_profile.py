from aptafind.data.aptabench_profile import (
    cluster_sequences,
    connectivity_smiles,
    has_steroid_fused_ring_nucleus,
    normalize_origin,
    normalize_sequence,
    normalized_edit_identity,
)


def test_normalize_sequence_converts_rna_and_whitespace() -> None:
    assert normalize_sequence(" ac gu\n") == "ACGU"


def test_normalized_edit_identity_uses_global_length() -> None:
    assert normalized_edit_identity("AAAA", "AAAT") == 0.75
    assert normalized_edit_identity("", "") == 1.0


def test_sequence_clustering_joins_near_identical_sequences() -> None:
    assignments, sizes = cluster_sequences(["AAAAAAAAAA", "AAAAAAAAAT", "CCCCCCCCCC"], 0.90)

    assert assignments["AAAAAAAAAA"] == assignments["AAAAAAAAAT"]
    assert assignments["AAAAAAAAAA"] != assignments["CCCCCCCCCC"]
    assert sizes == [2, 1]


def test_steroid_screen_recognizes_estradiol_but_not_caffeine() -> None:
    estradiol = "CC12CCC3c4ccc(O)cc4CCC3C1CCC2O"
    caffeine = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"

    assert has_steroid_fused_ring_nucleus(estradiol)
    assert not has_steroid_fused_ring_nucleus(caffeine)


def test_steroid_screen_rejects_opioid_polycycles() -> None:
    naloxone = "C=CCN1CC[C@]23c4c5ccc(O)c4O[C@H]2C(=O)CC[C@@]3(O)[C@H]1C5"

    assert not has_steroid_fused_ring_nucleus(naloxone)


def test_connectivity_smiles_merges_specified_and_unspecified_stereo() -> None:
    specified = "C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1CC[C@@H]2O"
    unspecified = "CC12CCC3c4ccc(O)cc4CCC3C1CCC2O"

    assert connectivity_smiles(specified) == connectivity_smiles(unspecified)


def test_normalize_origin_prefers_doi_and_pubmed_identifiers() -> None:
    assert normalize_origin("https://doi.org/10.1021/acs.est.2c05808") == (
        "doi:10.1021/acs.est.2c05808"
    )
    assert normalize_origin("https://pubmed.ncbi.nlm.nih.gov/9115371/") == (
        "pmid:9115371"
    )
