# Aptafind Dataset Registry

## Purpose

This registry inventories candidate data sources for Aptafind's steroid-focused dataset sufficiency audit.

Inclusion in this document does not mean that a source has been approved for training, redistribution, or publication. Each source must pass provenance, licensing, schema, evidence, and leakage review before use.

Status terms:

- **Discovered**: a potentially relevant source has been identified.
- **Metadata verified**: core repository or publication metadata has been checked.
- **Downloaded**: source files have been acquired without modification.
- **Profiled**: contents, fields, record counts, targets, and evidence types have been measured.
- **Approved for analysis**: provenance and permitted use have been reviewed.
- **Approved for redistribution**: files or derived records may be included publicly under documented terms.

## Priority sources

### AptaBench small-molecule interaction benchmark

| Field | Value |
|---|---|
| Registry ID | `aptabench_current_review_release` |
| Source | [AptaBench Hugging Face dataset](https://huggingface.co/datasets/aptabench-anonymous/AptaBench_dataset) |
| Associated manuscript | [Comprehensive Benchmark for Tailored Small Molecule-Binding Aptamer Design](https://openreview.net/forum?id=MsZa6NgqWJ) |
| Release status | Anonymous peer-review release; versioning must be captured at download time |
| Records currently reported | 6,289 aptamer-ligand pairs |
| Unique aptamers currently reported | 1,610 |
| Unique ligands currently reported | 942 |
| Formats | CSV and Parquet |
| Evidence | Experimentally grounded active/inactive labels and quantitative affinity where available |
| Fixed splits | Stratified, molecule-disjoint, and aptamer-disjoint five-fold splits |
| Frozen revision | `e4a4623f97975ea0a0526632fa253427f29372c9` (2026-05-05 commit) |
| Current status | Frozen and profiled |
| License | CC BY 4.0, declared in the frozen dataset card |

Reported fields include aptamer type, standardized sequence, canonical ligand SMILES, pKd where available, activity label, buffer or assay condition, origin, and curated source.

Audit value:

- Directly supports aptamer-small-molecule binding classification and affinity regression.
- Molecule-disjoint splits align with the held-out-target objective.
- Reported inactive labels are experimentally grounded rather than random cross-pairing.
- Provides a stronger starting benchmark than independently rebuilding a comparable endpoint table from several databases.

Audit cautions:

- Current counts differ across manuscript revisions, so the exact dataset revision and file checksums must be frozen.
- Aptamer-disjoint splits prevent exact sequence reuse but may not prevent closely related sequence families from crossing folds.
- The steroid subset, steroid hard-negative structure, DNA-only coverage, source overlap, and assay comparability remain unprofiled.
- It may include records derived from UTexas, AptaDB, or sources overlapping historical Aptafind data.
- Its existing benchmark does not replace the proposed steroid-specific trajectory-pretraining ablation.

Profiling result:

- The complete benchmark contains 6,289 records, 1,610 exact aptamers, and 942 ligands.
- A chemistry-aware screen followed by PubChem identity validation found 79 steroid records representing 9 connectivity-level targets and 61 sequence families at 90% edit identity.
- Only 10 steroid negative records were present, all for one target connectivity; the other eight target units contain positives only.
- No endpoint record explicitly cited the planned DL-SELEX publication, but sequence-level overlap remains unresolved.
- See [Frozen AptaBench Profile Report](aptabench_frozen_profile.md) and `manifests/aptabench_frozen_release.yaml`.

### DL-SELEX hydrocortisone HT-NGS

| Field | Value |
|---|---|
| Registry ID | `dl_selex_hydrocortisone_guided` |
| Target | Hydrocortisone/cortisol (`CS` in the source study) |
| Source | [Zenodo record 14272647](https://zenodo.org/records/14272647) |
| Related publication | [Structure-enhanced deep learning accelerates aptamer selection for small molecule families like steroids](https://doi.org/10.1093/bib/bbaf680) |
| Evidence | Multi-round target-selection observations |
| Rounds | 3, 5, and 7 |
| Format | Paired compressed FASTQ |
| Files | Six FASTQ files, forward and reverse reads for each round |
| Approximate size | 4.9 GB compressed |
| Checksums | MD5 supplied per file by Zenodo |
| Current status | Metadata verified; not downloaded |
| License | Must be verified from the record's Rights metadata before download/use |

Known files:

| Round | Read | Filename | Size | MD5 |
|---:|---:|---|---:|---|
| 3 | 1 | `3th-cs-LFL5718_L3_1.fq.gz` | 909.8 MB | `ce663275300a74b03b6d18d3aa90c487` |
| 3 | 2 | `3th-cs-LFL5718_L3_2.fq.gz` | 877.8 MB | `013dc9d317bee24c77b17e3da985908a` |
| 5 | 1 | `5th-cs-LFL5722_L3_1.fq.gz` | 872.3 MB | `e31888a485180c23e79cf99db8538fb6` |
| 5 | 2 | `5th-cs-LFL5722_L3_2.fq.gz` | 774.0 MB | `2ba9be3a8785359366c574193027d0e3` |
| 7 | 1 | `7th-cs-LFL5726_L3_1.fq.gz` | 768.6 MB | `fe5dc9eb7158e56fe0b90c18b8f4efd4` |
| 7 | 2 | `7th-cs-LFL5726_L3_2.fq.gz` | 723.8 MB | `8b70263b612310443a559ac6c38f7651` |

Audit value:

- Enables sequence abundance, frequency, enrichment, persistence, and family-convergence analysis.
- Does not by itself provide direct affinity labels for every observed sequence.
- All rounds belong to one target and experimental lineage, so millions of reads do not represent millions of independent target-aptamer experiments.

### DL-SELEX testosterone HT-NGS

| Field | Value |
|---|---|
| Registry ID | `dl_selex_testosterone_guided` |
| Target | Testosterone (`TES` in the source study) |
| Source | [Zenodo record 14272757](https://zenodo.org/records/14272757) |
| Related publication | [DL-SELEX paper](https://doi.org/10.1093/bib/bbaf680) |
| Evidence | Multi-round target-selection observations |
| Expected rounds | 3, 5, and 7, subject to file-level verification |
| Expected format | Paired compressed FASTQ |
| Current status | Discovered; file metadata not yet fully recorded |
| License | Verification required |

Audit value:

- Supplies a second chemically related target from the same research program.
- Enables cross-target comparison with hydrocortisone under more comparable methods than unrelated studies.
- Two SELEX targets remain insufficient by themselves for a strong leave-one-steroid-out generalization claim.

### Manually designed hydrocortisone HT-SELEX

| Field | Value |
|---|---|
| Registry ID | `dl_selex_hydrocortisone_manual` |
| Target | Hydrocortisone/cortisol |
| Source | [Zenodo record 14272347](https://zenodo.org/records/14272347) |
| Related publication | [DL-SELEX paper](https://doi.org/10.1093/bib/bbaf680) |
| Evidence | Multi-round selection from a manually designed library |
| Current status | Discovered; file and round metadata require verification |
| License | Verification required |

Audit value:

- Provides a same-target comparison between different initial-library strategies.
- May help quantify library-design effects separately from target-selection effects.
- Is not an independent target and must not be counted as one in target-level evaluation.

### UTexas Aptamer Dataset

| Field | Value |
|---|---|
| Registry ID | `utexas_aptamer_database_v1_1_0` |
| Source | [Zenodo record 8387047](https://zenodo.org/records/8387047) |
| Version | 1.1.0 |
| Publication date | 2023-08-19 |
| Records reported | 1,415 aptamer sequences from 489 papers covering 1990-2022 |
| Format | Two Excel snapshots |
| Approximate size | 1.1 MB total |
| Current status | Metadata verified; not downloaded |
| License | Must be recorded from the Zenodo Rights field before use or redistribution |

Files:

| Filename | Size | MD5 |
|---|---:|---|
| `UTexas Aptamer Database dataset.xlsx` | 511.0 kB | `65b1430f6bad16bec09e72ebf202a824` |
| `UTexas Aptamer Database dataset_Sept2023.xlsx` | 579.8 kB | `39e4838d14d7837be021f24daa87021d` |

Reported fields include sequence, target, nucleic-acid type, affinity, buffer, original library, modifications, publication information, and parent-sequence identifiers.

Audit cautions:

- Some sequences emerged from selection experiments but were not individually tested for binding.
- Records from the same paper or parent sequence are not independent observations.
- Steroid subset size and measured-negative coverage must be determined after download.
- Original publications remain the authoritative source for evidence interpretation.

### AptaDB

| Field | Value |
|---|---|
| Registry ID | `aptadb_2023_12_03` |
| Source | [AptaDB download page](https://lmmd.ecust.edu.cn/aptadb/about.php) |
| Related publication | [AptaDB: a comprehensive database integrating aptamer-target interactions](https://doi.org/10.1261/rna.079854.123) |
| Release shown | 2023-12-03 |
| Interactions reported | 1,350 experimentally validated aptamer-target interactions |
| Aptamers reported | 1,293 |
| Targets reported | 436 |
| Affinity values reported | 1,230 |
| Small-molecule interactions reported | 393 |
| Formats | CSV and TSV downloads by entity type |
| Current status | Metadata verified; not downloaded |
| License | Site and dataset reuse terms require verification |

Audit value:

- Provides standardized target identifiers, SMILES, affinity data, and experimental context.
- May contain verified steroid interactions suitable for retrospective evaluation.
- Steroid coverage, duplicates, parent families, assay comparability, and measured negatives remain unknown until profiling.

## Literature and supplementary sources

### Historical Aptafind curated steroid/small-molecule data

| Field | Value |
|---|---|
| Registry ID | `aptafind_historical_curated` |
| Location | Recovered local historical research assets; not currently redistributed in this repository |
| Known artifacts | `aptamers.json`, small-molecule CSV, target-feature table, and historical feature archive |
| Approximate historical counts | 437 JSON records; 168 observations in the recovered late-2023 feature archive |
| Evidence | Literature-derived candidate sequences, affinities, and target metadata |
| Current status | Recovered and checksummed; source-level provenance and redistribution audit incomplete |

Audit cautions:

- Historical records may overlap with UTexas, AptaDB, Apta-Index, or the same primary publications.
- The 168-row model archive is not the same as 168 independent aptamer families or experiments.
- Original-source identity and evidence strength must be reconstructed before model use.

### One-Pot SELEX steroid study

| Field | Value |
|---|---|
| Registry ID | `one_pot_selex_steroids_2019` |
| Targets | Estradiol, progesterone, and testosterone |
| Publication | [One-Pot SELEX: Identification of Specific Aptamers against Diverse Steroid Targets in One Selection](https://doi.org/10.1021/acsomega.9b02412) |
| Evidence | Candidate sequences, affinity measurements, counter-selection, and NGS-derived analysis |
| Public files found | Article and supplementary PDF |
| Raw read repository | Not found during initial search |
| Current status | Discovered; supplementary-table extraction and raw-data availability require audit |

Audit value:

- Contains multiple closely related steroid targets selected within one study.
- Counter-selection and cross-target design may provide unusually valuable specificity evidence.
- Without raw reads, it may contribute verified endpoints rather than full enrichment trajectories.

### Estradiol Capture-SELEX study

| Field | Value |
|---|---|
| Registry ID | `estradiol_capture_selex_2022` |
| Targets | 17-beta-estradiol with related estrogen and counter-target testing |
| Publication | [Capture-SELEX of DNA Aptamers for Estradiol Specifically and Estrogenic Compounds Collectively](https://doi.org/10.1021/acs.est.2c05808) |
| Repository copy | [University of Waterloo record](https://uwspace.uwaterloo.ca/items/06aafdbf-ff47-464b-b04c-05ae7310448f) |
| Evidence | Short candidate sequences, ITC affinity, and cross-reactivity/selectivity observations |
| Raw read repository | Not found during initial search |
| Current status | Discovered; supplementary data require extraction and evidence audit |

Audit value:

- Provides experimentally measured short aptamers.
- Related-estrogen and cortisol response data may support measured hard-negative or cross-reactivity labels.

### Progesterone high-throughput array study

| Field | Value |
|---|---|
| Registry ID | `progesterone_aptamer_array_2025` |
| Target | Progesterone with related-steroid specificity testing |
| Publication | [Discovery of high-specificity DNA aptamers for progesterone using a high-throughput array platform](https://doi.org/10.1101/2025.09.13.675901) |
| Evidence described | Measurements across millions of candidates and cross-reactivity testing |
| Current status | Preprint discovered; sequence-level data availability and reuse terms not verified |

Audit value:

- Could provide much denser sequence-to-binding supervision than conventional endpoint studies.
- Must not be counted as available training data until sequence-level files and permitted use are confirmed.

## Supporting or control datasets

### Rationally designed DNA screening libraries

| Field | Value |
|---|---|
| Registry ID | `rational_dna_libraries_2026` |
| Source | [Zenodo record 18719585](https://zenodo.org/records/18719585) |
| Version | v1 |
| Publication date | 2026-02-21 |
| Format | Paired compressed FASTQ |
| Approximate size | 128.4 MB |
| Evidence | Library composition; no target-binding selection evidence stated in the repository description |
| Current status | Metadata verified; not downloaded |

Audit value:

- May support sequence-quality controls, library-composition analysis, or generator pretraining comparisons.
- Must not be treated as positive or negative binding evidence.

## Explicit exclusions

### Steroid receptor DNA-binding SELEX

SELEX studies involving androgen, glucocorticoid, estrogen, or progesterone receptor proteins may describe DNA motifs recognized by those proteins. They do not necessarily contain ssDNA aptamers that bind the steroid molecules themselves.

These sources are excluded from the steroid-small-molecule training set unless a future protein-target research question explicitly requires them.

### APTANI islet dataset

The previously encountered APTANI/islet FASTQ data concern complex cellular targets and are outside the initial steroid research scope. They will not be downloaded or mixed into Gate 1.

## Fields still requiring verification

Before any download is approved, the registry must add or confirm:

- Exact license and reuse terms
- Exact file inventory and checksums
- Dataset version or publication snapshot
- Whether files are open, restricted, or request-only
- Sequencing layout and primer structure
- Round labels and experimental relationships
- Presence of negative or counter-selection pools
- Assay methods and measurement conditions
- Whether raw sequences may be redistributed
- Whether derived Parquet tables may be redistributed

## Initial download order

1. UTexas Aptamer Dataset
2. AptaDB entity tables
3. Existing historical Aptafind data for overlap/provenance comparison
4. One hydrocortisone FASTQ round as a pipeline proof of concept
5. Remaining hydrocortisone guided rounds
6. Testosterone rounds
7. Manual-library hydrocortisone experiment

Large sequencing files will not be downloaded until storage layout, checksum validation, licensing, and source manifests are ready.
