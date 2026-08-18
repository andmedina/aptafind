# Frozen AptaBench Profile Report

Generated: `2026-08-14T02:38:00.832132+00:00`

## Frozen release

- Repository: `https://huggingface.co/datasets/aptabench-anonymous/AptaBench_dataset`
- Git revision: `e4a4623f97975ea0a0526632fa253427f29372c9`
- Commit date: `2026-05-05T23:57:26Z`
- License: CC BY 4.0 (declared in Hugging Face dataset-card metadata)

The CSV and fixed split files are pinned by both Git revision and SHA-256 hashes in the machine-readable profile. The Parquet file and logo are represented by Git LFS pointers; their declared object hashes and sizes are also recorded.

## Benchmark composition

| Measure | Result |
|---|---:|
| Interaction records | 6,289 |
| Unique normalized aptamers | 1,610 |
| Unique ligands | 942 |
| DNA records | 4,721 |
| RNA records | 1,568 |
| Positive records | 3,240 |
| Negative records | 3,049 |
| Quantitative affinity records | 2,087 |
| Publication units with origin metadata | 397 |
| Records lacking origin metadata | 1,157 |

## Data-quality findings

- Exact duplicate rows: **0**.
- Repeated normalized sequence-ligand pairs: **208** pairs, representing **237** records beyond one row per pair.
- Repeated pairs with conflicting binary labels: **23**.
- Quantitative pairs with multiple distinct pKd values: **100**.
- Repeated measurements should be reconciled at the experiment level, not silently dropped.

## Structurally identified steroid subset

Screening definition: RDKit ring-topology screen for an all-carbon, 17-atom nucleus of exactly four path-fused rings sized 6-6-6-5.

| Measure | Result |
|---|---:|
| Steroid interaction records | 79 |
| DNA steroid records | 78 |
| Unique steroid ligands | 12 |
| Unique connectivity-level targets | 9 |
| Unique normalized aptamers | 67 |
| Positive records | 69 |
| Negative records | 10 |
| Quantitative affinity records | 40 |
| Nonmissing publication units | 22 |

### Connectivity-level target evidence

| Target identity | Records | 90% families | Positive | Negative | Quantitative | Origin groups |
|---|---:|---:|---:|---:|---:|---:|
| Desoxy-tetraene steroid derivative | 1 | 1 | 1 | 0 | 1 | 0 |
| Progesterone / Progesterone (stereochemistry unspecified) | 12 | 7 | 12 | 0 | 7 | 3 |
| Cholic acid / Ox bile extract / cholic-acid connectivity (stereochemistry unspecified) | 14 | 12 | 14 | 0 | 8 | 3 |
| Cortisol (stereochemistry unspecified) | 4 | 2 | 4 | 0 | 3 | 3 |
| Testosterone | 11 | 11 | 11 | 0 | 11 | 2 |
| Dehydroepiandrosterone sulfate | 14 | 11 | 4 | 10 | 0 | 2 |
| Nandrolone | 2 | 2 | 2 | 0 | 0 | 1 |
| Estradiol / Estradiol connectivity (stereochemistry unspecified) | 10 | 9 | 10 | 0 | 5 | 8 |
| Lanicor (cardiac glycoside) | 11 | 6 | 11 | 0 | 5 | 1 |

### Sequence-family sensitivity

| Minimum normalized edit identity | All-dataset families | Steroid-subset families |
|---:|---:|---:|
| 80% | 1,009 | 59 |
| 90% | 1,220 | 61 |
| 95% | 1,348 | 62 |

These are single-linkage sensitivity estimates. Final model splits should use a documented, alignment-aware clustering method and should be grouped by publication or experiment where possible.

## Existing fixed-split audit

- `disjoint_aptamer`: maximum fold overlap = 0 exact sequences, 222 exact ligands, and 0 sequence families at the 90% threshold.
- `disjoint_molecule`: maximum fold overlap = 325 exact sequences, 0 exact ligands, and 240 sequence families at the 90% threshold.
- `stratified`: maximum fold overlap = 363 exact sequences, 237 exact ligands, and 305 sequence families at the 90% threshold.

The supplied splits are useful benchmark protocols, but exact aptamer disjointness must not be interpreted as sequence-family disjointness.

## Planned DL-SELEX overlap

- Endpoint records explicitly citing `10.1093/bib/bbaf680`: **0**.
- Sequence-level status: unresolved until a derived unique-sequence inventory is produced from the planned DL-SELEX data.

## Gate status

**The frozen benchmark is suitable for general small-molecule endpoint modeling, but AptaBench alone is insufficient for a rigorous steroid-specific Model A/B/C comparison.**

After structural validation, stereochemistry-insensitive target consolidation, exact deduplication, 90% family clustering, and removal of the **0** records with explicit DL-SELEX publication overlap, the current provisional evaluation units are **9 steroid targets**, **61 sequence families**, **61 target-family pairs**, and **22 nonmissing origin groups**. The origin groups are not yet verified independent experiments, and sequence-level DL-SELEX overlap remains unresolved.

The decisive weakness is label structure: only 10 steroid negative records were found, all for one connectivity-level target. Eight of nine target units contain positives only. Consequently, a steroid-only classifier could learn target or source shortcuts and cannot yet support a defensible broad specificity claim.

**Gate 1A — benchmark characterization: complete.** The benchmark has been frozen, profiled, chemically screened, and documented.

**Gate 1B — cross-dataset independence audit: in progress.** Before large FASTQ acquisition, the project needs (1) publication-level confirmation of steroid identities and experimental units, and (2) a compact DL-SELEX sequence inventory for exact and family-overlap testing. Model development and training-split freezing remain blocked. The primary experiment may need to evaluate transfer on the broader AptaBench small-molecule task, with steroid results reported as a limited subgroup, unless additional measured steroid negatives are curated.

## Limitations

- AptaBench has no explicit ligand-name or chemical-class column.
- The structurally screened identities were checked with PubChem, but their biological roles and experimental units still require source-publication review.
- Origin is absent for some records and is not equivalent to a verified experiment ID.
- Edit-identity single-linkage clusters are a sensitivity proxy, not curated biological lineages.
- Sequence-level pretraining overlap cannot be resolved from publication metadata alone.
