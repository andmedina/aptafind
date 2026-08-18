"""Profile a frozen AptaBench release and audit its evaluation units.

The module deliberately separates immutable source data from derived reports.
It reads the Bronze CSV and fixed split files but never modifies them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml
from rapidfuzz.distance import Levenshtein
from rdkit import Chem


EXPECTED_COLUMNS = [
    "type",
    "sequence",
    "canonical_smiles",
    "pKd_value",
    "label",
    "buffer",
    "origin",
    "source",
]
FAMILY_THRESHOLDS = (0.80, 0.90, 0.95)
DL_SELEX_PUBLICATION_MARKERS = ("10.1093/bib/bbaf680", "bbaf680", "dl-selex")


class UnionFind:
    """Small disjoint-set structure used for single-linkage sequence families."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_sequence(sequence: str) -> str:
    """Normalize nucleic-acid text for identity and family comparisons."""

    return re.sub(r"\s+", "", str(sequence)).upper()


def normalized_edit_identity(left: str, right: str) -> float:
    """Compute global edit identity as 1 - distance / maximum length."""

    maximum_length = max(len(left), len(right))
    if maximum_length == 0:
        return 1.0
    return 1.0 - (Levenshtein.distance(left, right) / maximum_length)


def cluster_sequences(
    sequences: Iterable[str], threshold: float
) -> tuple[dict[str, int], list[int]]:
    """Create single-linkage families using normalized global edit identity.

    The method is intentionally transparent and suitable for this small audit.
    Results are a sensitivity proxy, not a replacement for a later alignment-
    aware biological family analysis.
    """

    unique_sequences = sorted({normalize_sequence(value) for value in sequences})
    families = UnionFind(len(unique_sequences))

    for left_index, left in enumerate(unique_sequences):
        for right_index in range(left_index + 1, len(unique_sequences)):
            right = unique_sequences[right_index]
            length_ratio = min(len(left), len(right)) / max(len(left), len(right))
            if length_ratio < threshold:
                continue
            if normalized_edit_identity(left, right) >= threshold:
                families.union(left_index, right_index)

    roots = [families.find(index) for index in range(len(unique_sequences))]
    root_to_cluster = {
        root: cluster_id for cluster_id, root in enumerate(sorted(set(roots)))
    }
    assignments = {
        sequence: root_to_cluster[root]
        for sequence, root in zip(unique_sequences, roots, strict=True)
    }
    sizes = sorted(Counter(assignments.values()).values(), reverse=True)
    return assignments, sizes


def _fused_ring_components(molecule: Chem.Mol) -> list[list[tuple[int, ...]]]:
    rings = [tuple(ring) for ring in molecule.GetRingInfo().AtomRings()]
    neighbors = {index: set() for index in range(len(rings))}

    for left_index, left in enumerate(rings):
        for right_index in range(left_index + 1, len(rings)):
            if len(set(left).intersection(rings[right_index])) >= 2:
                neighbors[left_index].add(right_index)
                neighbors[right_index].add(left_index)

    components: list[list[tuple[int, ...]]] = []
    visited: set[int] = set()
    for start in range(len(rings)):
        if start in visited:
            continue
        pending = [start]
        component: list[tuple[int, ...]] = []
        while pending:
            index = pending.pop()
            if index in visited:
                continue
            visited.add(index)
            component.append(rings[index])
            pending.extend(neighbors[index] - visited)
        components.append(component)
    return components


def has_steroid_fused_ring_nucleus(smiles: str) -> bool:
    """Identify a steroid-like fused 6-6-6-5 carbon-ring nucleus.

    This chemistry-aware rule supports aromatic or unsaturated steroids. It is
    a structural screening definition, not a claim about biological activity.
    Candidate structures are exported for manual chemical review.
    """

    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return False

    for component in _fused_ring_components(molecule):
        ring_sizes = sorted(len(ring) for ring in component)
        if ring_sizes != [5, 6, 6, 6]:
            continue
        atoms = set().union(*(set(ring) for ring in component))
        if len(atoms) != 17:
            continue
        if not all(
            molecule.GetAtomWithIdx(index).GetAtomicNum() == 6 for index in atoms
        ):
            continue
        fusion_degrees = []
        for ring in component:
            degree = sum(
                len(set(ring).intersection(other_ring)) >= 2
                for other_ring in component
                if other_ring != ring
            )
            fusion_degrees.append(degree)
        if sorted(fusion_degrees) == [1, 1, 2, 2]:
            return True
    return False


def connectivity_smiles(smiles: str) -> str:
    """Canonicalize molecular connectivity while intentionally ignoring stereo."""

    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    Chem.RemoveStereochemistry(molecule)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def normalize_origin(origin: Any) -> str | None:
    """Create a conservative publication identifier from heterogeneous text."""

    if pd.isna(origin):
        return None
    text = " ".join(str(origin).strip().lower().split())
    doi_match = re.search(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", text)
    if doi_match:
        return f"doi:{doi_match.group(0).rstrip('.,;') }"
    pubmed_match = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", text)
    if pubmed_match:
        return f"pmid:{pubmed_match.group(1)}"
    if re.fullmatch(r"\d{7,9}", text):
        return f"pmid:{text}"
    return f"text:{text}"


def validate_schema(data: pd.DataFrame) -> None:
    """Fail early if the frozen release no longer has the expected schema."""

    if list(data.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Unexpected AptaBench schema. Expected {EXPECTED_COLUMNS}; "
            f"received {list(data.columns)}"
        )
    if not set(data["label"].unique()).issubset({0, 1}):
        raise ValueError("AptaBench labels must be binary values 0 or 1.")


def profile_split(
    data: pd.DataFrame,
    split_path: Path,
    family_assignments: dict[float, dict[str, int]],
) -> dict[str, Any]:
    """Verify split indices, exact disjointness, and family-level leakage."""

    folds = json.loads(split_path.read_text())
    fold_profiles = []
    for fold in folds:
        train = data.iloc[fold["train_idx"]]
        validation = data.iloc[fold["val_idx"]]
        train_sequences = {normalize_sequence(value) for value in train["sequence"]}
        validation_sequences = {
            normalize_sequence(value) for value in validation["sequence"]
        }
        train_ligands = set(train["canonical_smiles"])
        validation_ligands = set(validation["canonical_smiles"])

        family_overlap: dict[str, int] = {}
        for threshold, assignments in family_assignments.items():
            train_families = {assignments[sequence] for sequence in train_sequences}
            validation_families = {
                assignments[sequence] for sequence in validation_sequences
            }
            family_overlap[f"{threshold:.2f}"] = len(
                train_families.intersection(validation_families)
            )

        fold_profiles.append(
            {
                "fold": int(fold["fold"]),
                "train_records": len(train),
                "validation_records": len(validation),
                "index_overlap": len(
                    set(fold["train_idx"]).intersection(fold["val_idx"])
                ),
                "exact_sequence_overlap": len(
                    train_sequences.intersection(validation_sequences)
                ),
                "exact_ligand_overlap": len(
                    train_ligands.intersection(validation_ligands)
                ),
                "sequence_family_overlap": family_overlap,
            }
        )

    return {"file": split_path.name, "folds": fold_profiles}


def _git_value(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_frozen_release(repository: Path, manifest_path: Path) -> None:
    """Fail if a checkout differs from its recorded revision or file hashes."""

    manifest = yaml.safe_load(manifest_path.read_text())
    actual_revision = _git_value(repository, "rev-parse", "HEAD")
    if actual_revision != manifest["git_revision"]:
        raise ValueError(
            f"AptaBench revision mismatch: expected {manifest['git_revision']}, "
            f"found {actual_revision}"
        )

    for item in manifest["files"]:
        path = repository / item["path"]
        if not path.exists():
            raise FileNotFoundError(f"Frozen AptaBench file is missing: {path}")
        actual_hash = sha256_file(path)
        allowed_hashes = {
            item[key]
            for key in ("sha256", "local_pointer_sha256", "lfs_sha256")
            if key in item
        }
        if actual_hash not in allowed_hashes:
            raise ValueError(
                f"AptaBench file hash mismatch for {item['path']}: {actual_hash}"
            )


def build_profile(
    repository: Path,
    ligand_manifest_path: Path | None = None,
    release_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Build a machine-readable profile for a pinned AptaBench repository."""

    if release_manifest_path is not None:
        verify_frozen_release(repository, release_manifest_path)

    csv_path = repository / "dataset" / "AptaBench_dataset.csv"
    data = pd.read_csv(csv_path)
    validate_schema(data)

    data = data.copy()
    data["normalized_sequence"] = data["sequence"].map(normalize_sequence)
    data["publication_unit"] = data["origin"].map(normalize_origin)
    data["is_steroid"] = data["canonical_smiles"].map(
        has_steroid_fused_ring_nucleus
    )
    data["ligand_connectivity"] = data["canonical_smiles"].map(connectivity_smiles)

    ligand_names: dict[str, dict[str, Any]] = {}
    if ligand_manifest_path is not None and ligand_manifest_path.exists():
        ligand_manifest = json.loads(ligand_manifest_path.read_text())
        ligand_names = {
            item["canonical_smiles"]: item for item in ligand_manifest["ligands"]
        }

    all_family_assignments: dict[float, dict[str, int]] = {}
    all_family_summary: dict[str, Any] = {}
    for threshold in FAMILY_THRESHOLDS:
        assignments, sizes = cluster_sequences(data["normalized_sequence"], threshold)
        all_family_assignments[threshold] = assignments
        all_family_summary[f"{threshold:.2f}"] = {
            "clusters": len(sizes),
            "largest_cluster": max(sizes, default=0),
            "multi_sequence_clusters": sum(size > 1 for size in sizes),
        }

    steroid = data[data["is_steroid"]].copy()
    steroid_family_summary: dict[str, Any] = {}
    steroid_family_assignments: dict[float, dict[str, int]] = {}
    for threshold in FAMILY_THRESHOLDS:
        assignments, sizes = cluster_sequences(
            steroid["normalized_sequence"], threshold
        )
        steroid_family_assignments[threshold] = assignments
        steroid_family_summary[f"{threshold:.2f}"] = {
            "clusters": len(sizes),
            "largest_cluster": max(sizes, default=0),
            "multi_sequence_clusters": sum(size > 1 for size in sizes),
        }

    pair_groups = data.groupby(["normalized_sequence", "canonical_smiles"])
    repeated_pairs = pair_groups.size()
    conflicting_labels = pair_groups["label"].nunique()
    quantitative_pairs = data[data["pKd_value"].notna()].groupby(
        ["normalized_sequence", "canonical_smiles"]
    )["pKd_value"]

    split_profiles = {}
    for split_path in sorted((repository / "dataset" / "splits").glob("*.json")):
        split_profiles[split_path.stem] = profile_split(
            data, split_path, all_family_assignments
        )

    tracked_files = _git_value(repository, "ls-tree", "-r", "--name-only", "HEAD")
    file_profiles = []
    for relative_name in tracked_files.splitlines():
        path = repository / relative_name
        item: dict[str, Any] = {"path": relative_name}
        if path.exists():
            item.update({"bytes": path.stat().st_size, "sha256": sha256_file(path)})
            if path.stat().st_size < 1024:
                first_line = path.read_bytes().splitlines()[:1]
                if first_line == [b"version https://git-lfs.github.com/spec/v1"]:
                    pointer = path.read_text().splitlines()
                    item["git_lfs_pointer"] = True
                    item["lfs_sha256"] = pointer[1].removeprefix("oid sha256:")
                    item["lfs_bytes"] = int(pointer[2].removeprefix("size "))
        else:
            item["missing_from_checkout"] = True
        file_profiles.append(item)

    source_counts = {
        str(key): int(value) for key, value in data["source"].value_counts().items()
    }
    type_counts = {
        str(key): int(value) for key, value in data["type"].value_counts().items()
    }
    label_counts = {
        str(int(key)): int(value) for key, value in data["label"].value_counts().items()
    }
    steroid_by_ligand = []
    for smiles, group in steroid.groupby("canonical_smiles", sort=True):
        identity = ligand_names.get(smiles, {})
        steroid_by_ligand.append(
            {
                "canonical_smiles": smiles,
                "connectivity_smiles": group["ligand_connectivity"].iloc[0],
                "name": identity.get("name"),
                "pubchem_cid": identity.get("pubchem_cid"),
                "records": len(group),
                "unique_sequences": int(group["normalized_sequence"].nunique()),
                "positive_records": int((group["label"] == 1).sum()),
                "negative_records": int((group["label"] == 0).sum()),
                "quantitative_affinity_records": int(group["pKd_value"].notna().sum()),
                "publication_units": int(group["publication_unit"].nunique()),
                "sources": sorted(group["source"].unique().tolist()),
            }
        )

    steroid_by_connectivity = []
    for connectivity, group in steroid.groupby("ligand_connectivity", sort=True):
        smiles_values = sorted(group["canonical_smiles"].unique().tolist())
        names = sorted(
            {
                ligand_names[smiles]["name"]
                for smiles in smiles_values
                if smiles in ligand_names
            }
        )
        publications = group["publication_unit"].dropna().unique()
        family_90 = {
            steroid_family_assignments[0.90][sequence]
            for sequence in group["normalized_sequence"].unique()
        }
        steroid_by_connectivity.append(
            {
                "connectivity_smiles": connectivity,
                "names": names,
                "smiles_variants": len(smiles_values),
                "records": len(group),
                "unique_sequences": int(group["normalized_sequence"].nunique()),
                "sequence_families_90pct": len(family_90),
                "positive_records": int((group["label"] == 1).sum()),
                "negative_records": int((group["label"] == 0).sum()),
                "quantitative_affinity_records": int(group["pKd_value"].notna().sum()),
                "provisional_publication_units": len(publications),
                "records_missing_origin": int(group["publication_unit"].isna().sum()),
            }
        )

    steroid_target_family_pairs_90 = {
        (
            row.ligand_connectivity,
            steroid_family_assignments[0.90][row.normalized_sequence],
        )
        for row in steroid.itertuples()
    }

    origin_text = data["origin"].fillna("").str.lower()
    dl_selex_publication_overlap = data[
        origin_text.map(
            lambda value: any(marker in value for marker in DL_SELEX_PUBLICATION_MARKERS)
        )
    ]

    return {
        "profile_generated_utc": datetime.now(timezone.utc).isoformat(),
        "release": {
            "dataset": "AptaBench",
            "repository_url": _git_value(repository, "remote", "get-url", "origin"),
            "git_revision": _git_value(repository, "rev-parse", "HEAD"),
            "git_commit_date": _git_value(repository, "show", "-s", "--format=%cI", "HEAD"),
            "license": "CC BY 4.0 (declared in Hugging Face dataset-card metadata)",
            "files": file_profiles,
        },
        "schema": [
            {"name": column, "dtype": str(data[column].dtype)}
            for column in EXPECTED_COLUMNS
        ],
        "overall": {
            "records": len(data),
            "unique_normalized_aptamers": int(data["normalized_sequence"].nunique()),
            "unique_ligands": int(data["canonical_smiles"].nunique()),
            "type_counts": type_counts,
            "source_counts": source_counts,
            "label_counts": label_counts,
            "quantitative_affinity_records": int(data["pKd_value"].notna().sum()),
            "publication_units_nonmissing": int(data["publication_unit"].nunique()),
            "records_missing_publication_unit": int(data["publication_unit"].isna().sum()),
            "missing_values": {
                column: int(data[column].isna().sum()) for column in EXPECTED_COLUMNS
            },
            "exact_duplicate_rows": int(data[EXPECTED_COLUMNS].duplicated().sum()),
            "repeated_sequence_ligand_pairs": int((repeated_pairs > 1).sum()),
            "records_beyond_one_per_pair": int((repeated_pairs - 1).clip(lower=0).sum()),
            "pairs_with_conflicting_labels": int((conflicting_labels > 1).sum()),
            "quantitative_pairs_with_multiple_values": int(
                (quantitative_pairs.nunique() > 1).sum()
            ),
            "sequence_family_sensitivity": all_family_summary,
        },
        "dna_only": {
            "records": int((data["type"] == "DNA").sum()),
            "unique_normalized_aptamers": int(
                data.loc[data["type"] == "DNA", "normalized_sequence"].nunique()
            ),
            "unique_ligands": int(
                data.loc[data["type"] == "DNA", "canonical_smiles"].nunique()
            ),
        },
        "steroid_structural_subset": {
            "definition": (
                "RDKit ring-topology screen for an all-carbon, 17-atom nucleus "
                "of exactly four path-fused rings sized 6-6-6-5"
            ),
            "records": len(steroid),
            "dna_records": int((steroid["type"] == "DNA").sum()),
            "unique_normalized_aptamers": int(steroid["normalized_sequence"].nunique()),
            "unique_ligands": int(steroid["canonical_smiles"].nunique()),
            "unique_connectivity_targets": int(
                steroid["ligand_connectivity"].nunique()
            ),
            "positive_records": int((steroid["label"] == 1).sum()),
            "negative_records": int((steroid["label"] == 0).sum()),
            "quantitative_affinity_records": int(steroid["pKd_value"].notna().sum()),
            "publication_units_nonmissing": int(steroid["publication_unit"].nunique()),
            "records_missing_publication_unit": int(
                steroid["publication_unit"].isna().sum()
            ),
            "target_family_pairs_90pct": len(steroid_target_family_pairs_90),
            "sequence_family_sensitivity": steroid_family_summary,
            "ligands": steroid_by_ligand,
            "connectivity_targets": steroid_by_connectivity,
        },
        "fixed_split_audit": split_profiles,
        "planned_pretraining_overlap": {
            "dl_selex_publication_doi": "10.1093/bib/bbaf680",
            "matching_endpoint_records": len(dl_selex_publication_overlap),
            "sequence_level_status": (
                "unresolved until a derived unique-sequence inventory is produced "
                "from the planned DL-SELEX data"
            ),
        },
        "limitations": [
            "AptaBench has no explicit ligand-name or chemical-class column.",
            "The structurally screened identities were checked with PubChem, but their biological roles and experimental units still require source-publication review.",
            "Origin is absent for some records and is not equivalent to a verified experiment ID.",
            "Edit-identity single-linkage clusters are a sensitivity proxy, not curated biological lineages.",
            "Sequence-level pretraining overlap cannot be resolved from publication metadata alone.",
        ],
    }


def render_markdown(profile: dict[str, Any]) -> str:
    """Render the core audit findings as a readable research report."""

    release = profile["release"]
    overall = profile["overall"]
    dna = profile["dna_only"]
    steroid = profile["steroid_structural_subset"]
    overlap = profile["planned_pretraining_overlap"]

    lines = [
        "# Frozen AptaBench Profile Report",
        "",
        f"Generated: `{profile['profile_generated_utc']}`",
        "",
        "## Frozen release",
        "",
        f"- Repository: `{release['repository_url']}`",
        f"- Git revision: `{release['git_revision']}`",
        f"- Commit date: `{release['git_commit_date']}`",
        f"- License: {release['license']}",
        "",
        "The CSV and fixed split files are pinned by both Git revision and SHA-256 hashes in the machine-readable profile. The Parquet file and logo are represented by Git LFS pointers; their declared object hashes and sizes are also recorded.",
        "",
        "## Benchmark composition",
        "",
        "| Measure | Result |",
        "|---|---:|",
        f"| Interaction records | {overall['records']:,} |",
        f"| Unique normalized aptamers | {overall['unique_normalized_aptamers']:,} |",
        f"| Unique ligands | {overall['unique_ligands']:,} |",
        f"| DNA records | {dna['records']:,} |",
        f"| RNA records | {overall['type_counts'].get('RNA', 0):,} |",
        f"| Positive records | {overall['label_counts'].get('1', 0):,} |",
        f"| Negative records | {overall['label_counts'].get('0', 0):,} |",
        f"| Quantitative affinity records | {overall['quantitative_affinity_records']:,} |",
        f"| Publication units with origin metadata | {overall['publication_units_nonmissing']:,} |",
        f"| Records lacking origin metadata | {overall['records_missing_publication_unit']:,} |",
        "",
        "## Data-quality findings",
        "",
        f"- Exact duplicate rows: **{overall['exact_duplicate_rows']:,}**.",
        f"- Repeated normalized sequence-ligand pairs: **{overall['repeated_sequence_ligand_pairs']:,}** pairs, representing **{overall['records_beyond_one_per_pair']:,}** records beyond one row per pair.",
        f"- Repeated pairs with conflicting binary labels: **{overall['pairs_with_conflicting_labels']:,}**.",
        f"- Quantitative pairs with multiple distinct pKd values: **{overall['quantitative_pairs_with_multiple_values']:,}**.",
        "- Repeated measurements should be reconciled at the experiment level, not silently dropped.",
        "",
        "## Structurally identified steroid subset",
        "",
        f"Screening definition: {steroid['definition']}.",
        "",
        "| Measure | Result |",
        "|---|---:|",
        f"| Steroid interaction records | {steroid['records']:,} |",
        f"| DNA steroid records | {steroid['dna_records']:,} |",
        f"| Unique steroid ligands | {steroid['unique_ligands']:,} |",
        f"| Unique connectivity-level targets | {steroid['unique_connectivity_targets']:,} |",
        f"| Unique normalized aptamers | {steroid['unique_normalized_aptamers']:,} |",
        f"| Positive records | {steroid['positive_records']:,} |",
        f"| Negative records | {steroid['negative_records']:,} |",
        f"| Quantitative affinity records | {steroid['quantitative_affinity_records']:,} |",
        f"| Nonmissing publication units | {steroid['publication_units_nonmissing']:,} |",
        "",
        "### Connectivity-level target evidence",
        "",
        "| Target identity | Records | 90% families | Positive | Negative | Quantitative | Origin groups |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for target in steroid["connectivity_targets"]:
        display_name = " / ".join(target["names"]) or "Unresolved steroid structure"
        lines.append(
            f"| {display_name} | {target['records']} | {target['sequence_families_90pct']} | {target['positive_records']} | {target['negative_records']} | {target['quantitative_affinity_records']} | {target['provisional_publication_units']} |"
        )

    lines.extend(
        [
        "",
        "### Sequence-family sensitivity",
        "",
        "| Minimum normalized edit identity | All-dataset families | Steroid-subset families |",
        "|---:|---:|---:|",
        ]
    )
    for threshold in FAMILY_THRESHOLDS:
        key = f"{threshold:.2f}"
        lines.append(
            f"| {threshold:.0%} | {overall['sequence_family_sensitivity'][key]['clusters']:,} | {steroid['sequence_family_sensitivity'][key]['clusters']:,} |"
        )

    lines.extend(
        [
            "",
            "These are single-linkage sensitivity estimates. Final model splits should use a documented, alignment-aware clustering method and should be grouped by publication or experiment where possible.",
            "",
            "## Existing fixed-split audit",
            "",
        ]
    )
    for split_name, split in profile["fixed_split_audit"].items():
        exact_sequence = max(fold["exact_sequence_overlap"] for fold in split["folds"])
        exact_ligand = max(fold["exact_ligand_overlap"] for fold in split["folds"])
        family_90 = max(
            fold["sequence_family_overlap"]["0.90"] for fold in split["folds"]
        )
        lines.append(
            f"- `{split_name}`: maximum fold overlap = {exact_sequence} exact sequences, {exact_ligand} exact ligands, and {family_90} sequence families at the 90% threshold."
        )

    lines.extend(
        [
            "",
            "The supplied splits are useful benchmark protocols, but exact aptamer disjointness must not be interpreted as sequence-family disjointness.",
            "",
            "## Planned DL-SELEX overlap",
            "",
            f"- Endpoint records explicitly citing `{overlap['dl_selex_publication_doi']}`: **{overlap['matching_endpoint_records']:,}**.",
            f"- Sequence-level status: {overlap['sequence_level_status']}.",
            "",
            "## Gate status",
        "",
            "**The frozen benchmark is suitable for general small-molecule endpoint modeling, but AptaBench alone is insufficient for a rigorous steroid-specific Model A/B/C comparison.**",
            "",
            f"After structural validation, stereochemistry-insensitive target consolidation, exact deduplication, 90% family clustering, and removal of the **0** records with explicit DL-SELEX publication overlap, the current provisional evaluation units are **{steroid['unique_connectivity_targets']} steroid targets**, **{steroid['sequence_family_sensitivity']['0.90']['clusters']} sequence families**, **{steroid['target_family_pairs_90pct']} target-family pairs**, and **{steroid['publication_units_nonmissing']} nonmissing origin groups**. The origin groups are not yet verified independent experiments, and sequence-level DL-SELEX overlap remains unresolved.",
            "",
            "The decisive weakness is label structure: only 10 steroid negative records were found, all for one connectivity-level target. Eight of nine target units contain positives only. Consequently, a steroid-only classifier could learn target or source shortcuts and cannot yet support a defensible broad specificity claim.",
            "",
            "**Gate 1A — benchmark characterization: complete.** The benchmark has been frozen, profiled, chemically screened, and documented.",
            "",
            "**Gate 1B — cross-dataset independence audit: in progress.** Before large FASTQ acquisition, the project needs (1) publication-level confirmation of steroid identities and experimental units, and (2) a compact DL-SELEX sequence inventory for exact and family-overlap testing. Model development and training-split freezing remain blocked. The primary experiment may need to evaluate transfer on the broader AptaBench small-molecule task, with steroid results reported as a limited subgroup, unless additional measured steroid negatives are curated.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {limitation}" for limitation in profile["limitations"])
    return "\n".join(lines) + "\n"


def write_outputs(profile: dict[str, Any], json_path: Path, report_path: Path) -> None:
    """Write deterministic derived artifacts outside the frozen Bronze snapshot."""

    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(profile, indent=2) + "\n")
    report_path.write_text(render_markdown(profile))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument(
        "--ligand-manifest",
        type=Path,
        default=Path("manifests/aptabench_steroid_ligands.json"),
    )
    parser.add_argument(
        "--release-manifest",
        type=Path,
        default=Path("manifests/aptabench_frozen_release.yaml"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports/aptabench_frozen_profile.json"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("docs/aptabench_frozen_profile.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = build_profile(
        args.repository.resolve(),
        args.ligand_manifest.resolve(),
        args.release_manifest.resolve(),
    )
    write_outputs(profile, args.json_output, args.report_output)


if __name__ == "__main__":
    main()
