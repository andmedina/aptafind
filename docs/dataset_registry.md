# Aptafind ssDNA–Small-Molecule Source Catalog

## Purpose

This registry inventories candidate data sources for Aptafind's modern thesis-era sequence-generation pipeline: experimentally grounded single-stranded DNA aptamers that bind organic small-molecule targets. Steroids remain the downstream transfer and evaluation focus, but the upstream data search now covers small molecules broadly.

Inclusion in this document does not mean that a source has been approved for training, redistribution, or publication. Each source must pass provenance, licensing, schema, evidence, and leakage review before use.

Status terms:

- **Discovered**: a potentially relevant source has been identified.
- **Metadata verified**: core repository or publication metadata has been checked.
- **Downloaded**: source files have been acquired without modification.
- **Profiled**: contents, fields, record counts, targets, and evidence types have been measured.
- **Approved for analysis**: provenance and permitted use have been reviewed.
- **Approved for redistribution**: files or derived records may be included publicly under documented terms.

The machine-readable authority is [`manifests/datasets.yaml`](../manifests/datasets.yaml). This document explains why each source matters and how it may be used.

## What belongs in Git and what stays local

The repository uses a hybrid design:

| Location | Contents |
|---|---|
| Git | Source registry, DOI/accession/URL, license and access notes, expected file inventory, checksums, schemas, acquisition instructions, and small synthetic fixtures |
| `data_lake/bronze/` | Immutable third-party downloads such as CSV, XLSX, PDF, DOCX, FASTQ, and frozen external repositories |
| `data_lake/silver/` | Validated, standardized, provenance-preserving records |
| `data_lake/gold/` | Versioned model-ready datasets, family groups, and frozen splits |
| Remote-only | Multi-gigabyte files not yet approved, request-only sources, and sources with unresolved reuse terms |

All three data-lake layers are ignored by Git. Raw third-party data must not be committed merely because it is publicly downloadable.

## Inclusion boundary

The primary scope is unmodified ssDNA selected or tested against organic small molecules. Every retained record should eventually identify:

- the exact DNA sequence and whether primers or truncations are present;
- target identity, preferably with a canonical chemical identifier;
- experimental evidence and assay conditions;
- positive, negative, cross-reactive, counter-selection, or trajectory status;
- publication, database lineage, and parent-sequence relationships;
- access and reuse terms.

RNA-only aptamers, natural RNA–ligand interactions, protein/peptide/cell targets, steroid-receptor DNA motifs, and metal-ion targets are outside the primary training scope. Chemically modified X-aptamers are tracked only as external evidence because the current `A/C/G/T` tokenizer cannot represent their modifications faithfully.

## Source landscape

The catalog intentionally separates original experiments from aggregations and discovery indexes. A record appearing in three databases is still one experimental observation.

| Tier | Source | Registry ID | What it contributes | Acquisition state |
|---|---|---|---|---|
| Frozen benchmark | AptaBench | `aptabench_current_review_release` | 6,289 small-molecule interaction rows with measured positives/negatives and fixed splits | Downloaded and profiled |
| Primary experiment | Kynurenine N2A2 screen | `n2a2_kynurenine_specificity_2022` | 2.8 million ssDNA clusters tested across five related metabolites | Metadata verified |
| Primary experiment | Xiao-lab thermodynamics/specificity | `xiao_thermodynamics_specificity_2025` | 317 DNA aptamers with ITC data and about 6,000 cross-target tests | Metadata verified |
| Primary experiment | DL-SELEX endpoint supplement/ENA | `dl_selex_steroid_endpoints_2025` | 195 steroid–aptamer training pairs and six named CS/TES candidates | Metadata verified |
| Selection trajectory | DL-SELEX CS/TES/manual records | three `dl_selex_*` IDs | Paired reads from rounds 3/5/7 or 7/8; ~13.6 GB total | Metadata and checksums verified; remote-only |
| Primary experiment | High-affinity steroid receptors | `high_affinity_steroid_receptors_2017` | Steroid targets, counter-targets, sequences, affinities, and cross-reactivity | Metadata verified |
| Primary experiment | One-Pot steroid SELEX | `one_pot_selex_steroids_2019` | Estradiol/progesterone/testosterone candidates and NGS abundance tables | Downloaded, checksum-verified, and profiled |
| Primary experiment | Estradiol Capture-SELEX | `estradiol_capture_selex_2022` | Affinity, enrichment, and related-estrogen response evidence | Downloaded, checksum-verified, and profiled |
| Primary experiment | Progesterone aptamer array | `progesterone_aptamer_array_2025` | Millions of array measurements described; files/permission unresolved | Request/permission path only |
| Aggregate | UTexas Aptamer Dataset | `utexas_aptamer_database_v1_1_0` | 1,415 literature-curated sequences with target and assay context | Downloaded; checksums verified |
| Aggregate | AptaDB | `aptadb_2023_12_03` | 1,350 interactions including 393 small-molecule interactions | Downloaded locally; reuse terms unresolved |
| Historical aggregate | Aptamer Base archive | `aptamer_base_archive` | Recovered CSV/RDF records from the former Freebase database | Downloaded at a frozen commit; checksums verified |
| Local aggregate | Historical Aptafind | `aptafind_historical_curated` | Thesis-era steroid/small-molecule records | Recovered; provenance audit pending |
| Discovery index | Apta-Index | `apta_index_live` | Literature and target discovery | Do not bulk-scrape |
| Discovery index | Aptabase | `aptabase_iitg_live` | Literature and target discovery | Do not bulk-scrape |
| Derived overlap source | AptaCom | `aptacom_derived` | Cross-database lineage and duplicate auditing | No repository license detected |
| Derived overlap source | Mendeley Aptagen collection | `mendeley_aptagen_dna_2020` | 238 sequences for sequence-universe comparison | Secondary-only |

The legacy SELEX_DB, original Aptamer Database, HTPSELEX, and RiboaptDB remain useful clues but are inactive, unavailable, RNA-heavy, or outside scope. The UTexas database's [tips and resources page](https://sites.utexas.edu/aptamerdatabase/tips-and-resources/) is the maintained discovery map for these historical directories.

## Highest-value additions for the thesis pipeline

### Kynurenine-metabolite N2A2 specificity screen

The 2022 PNAS study [A system for multiplexed selection of aptamers with exquisite specificity without counterselection](https://doi.org/10.1073/pnas.2119945119) is unusually important for the positive-only problem.

It reports an ssDNA library with a 30-nt variable region, seven pooled-target selection rounds, and a modified MiSeq screen of 2.8 million clusters against five related metabolites: kynurenine, kynurenic acid, 3-hydroxykynurenine, xanthurenic acid, and 3-hydroxyanthranilic acid. About 87% of screened clusters represented unique sequences. The authors report 902 monospecific 3-hydroxykynurenine candidates, six for xanthurenic acid, and one for kynurenic acid.

Both supplements are now downloaded, checksum-verified, and profiled locally. The 164.3 MB workbook has 29 sheets. Its main screen contains 266,388 cluster rows representing 54,023 unique sequences with two or more clusters; a later enriched screen contains 59,714 unique sequence summaries. It also includes 108,334 rows for the high-copy HC-1 control and focused validation sheets. These are figure-underlying subsets, not a dump of all 2.8 million screened clusters. The 14-page PDF supplies the library, candidate, scramble, control, and displacement-strand sequences plus Figures S1-S11.

Why it matters:

- The same sequence is challenged against structurally related targets.
- Nonresponse and cross-reactivity are experimentally observed, not fabricated by random pairing.
- The target panel supports a realistic chemical-family discrimination task analogous to steroid selectivity.
- Family grouping can be reconstructed from the sequences instead of treating millions of clusters as independent experiments.

The article is CC BY-NC-ND 4.0. Analysis is possible, but publishing transformed records requires a specific rights review because of the no-derivatives term.

### DNA aptamer thermodynamics and specificity profiles

The 2025 Nucleic Acids Research study [Exploring the relationship between aptamer binding thermodynamics, affinity, and specificity](https://doi.org/10.1093/nar/gkaf219) provides two complementary DNA-only datasets:

- ITC-derived affinity and thermodynamic measurements for 317 small-molecule-binding DNA aptamers, comprising 319 measured aptamer–ligand pairs.
- Specificity profiles for 218 DNA aptamers tested against panels of 18–35 ligands, yielding approximately 6,000 aptamer–ligand observations.

The downloaded, checksum-verified 3.7 MB archive contains two source workbooks and two PDFs. The ITC workbook has 319 measurement records across 11 target groups in its `All ITC Data` sheet. The specificity workbook has 218 aptamer columns and exactly 6,033 target/panel cells: 218 target measurements and 5,815 non-target measurements across 11 family sheets. This is the primary source behind AptaBench's `Specificity` lineage, so it should strengthen provenance and supply richer quantitative fields, not be concatenated as independent rows on top of AptaBench.

The article and supplement are CC BY-NC 4.0. Two ingestion caveats are already visible: the specificity readme says negative cross-reactivity values were replaced with zero, and the paper's 317-aptamer claim does not map one-to-one to the workbook's 319 rows and 316 distinct ID strings. Sequence-level reconciliation must precede deduplication.

### DL-SELEX steroid endpoints and ENA candidates

The 2025 [DL-SELEX study](https://doi.org/10.1093/bib/bbaf680) is both a model precedent and a data source. Its supplement reports a literature-derived training set of 195 aptamer–target pairs across eight steroid-family molecules. This is close enough to the historical Aptafind problem that overlap should be assumed until disproven.

The 18.4 MB supplementary DOCX is downloaded, checksum-verified, and structurally profiled. It includes the literature table, guided library templates, ITC results, and experimental details. Table S1 reports 195 rows across eight steroid targets, but its score column mixes reported values, frequency-derived values, and assigned `0.99` values for known binders; it is not a uniform affinity label. Text extraction recovered 189 rows with sequences from 15 to 116 nt, while six layouts still need table-aware parsing. Table S6 includes explicit `No binding` cross-target outcomes, although its caption and extracted header disagree about units. Visual DOCX rendering remains pending because LibreOffice is unavailable in the environment.

ENA project [PRJEB89545](https://www.ebi.ac.uk/ena/browser/view/PRJEB89545) contains six named hydrocortisone/testosterone candidate sequences in project metadata. The project currently returns no read runs; the multi-round reads are instead in the three Zenodo records already cataloged below.

Important safeguards:

- Preserve author-provided full and truncated sequences as related representations, not separate discoveries.
- Keep the 195 literature pairs distinct from candidates generated or selected in the paper.
- Group all endpoint, ENA, and FASTQ material by study lineage before splitting.
- Compare the 195 pairs to historical Aptafind records, AptaBench, UTexas, and AptaDB before any evaluation set is frozen.

### High-affinity steroid receptors

The 2017 study [High-Affinity Nucleic-Acid-Based Receptors for Steroids](https://doi.org/10.1021/acschembio.7b00634) reports five stringent selections and a panel of 12 steroid targets/analogs, including deoxycorticosterone 21-glucoside, dehydroisoandrosterone sulfate, progesterone, testosterone, hydrocortisone, corticosterone, and aldosterone.

Its downloaded and checksum-verified 15-page supplement contains selection protocols for five target classes, 12 named sensor candidates, predicted structures, approximate affinities from 30 nM to 1.7 µM, and cross-reactivity across ten steroids. Table S2 also includes capture strands, minimal aptamers, anti-aptamers, and circular-dichroism constructs; these must remain related constructs rather than independent candidates. The source itself uses `CSS` candidate labels in the sequence table and `CCS` in the affinity text, so that discrepancy must be preserved in provenance. This study is likely a major upstream source for both the historical thesis data and the DL-SELEX 195-pair table.

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
- The DNA-only subset contains 4,721 records, 1,065 exact sequences, and 314 ligands, with 1,941 positive and 2,780 negative records.
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
| Current status | Metadata, file inventory, checksums, and license verified; not downloaded |
| License | CC BY 4.0, verified from the Zenodo API on 2026-08-27 |

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
| Rounds | 3, 5, and 7 |
| Format | Six paired compressed FASTQ files, approximately 5.2 GB total |
| Current status | Metadata, file inventory, checksums, and license verified; not downloaded |
| License | CC BY 4.0, verified from the Zenodo API on 2026-08-27 |

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
| Rounds | 7 and 8 |
| Format | Four paired compressed FASTQ files plus a 155-byte QC summary; approximately 3.5 GB total |
| Current status | Metadata, file inventory, checksums, and license verified; not downloaded |
| License | CC BY 4.0, verified from the Zenodo API on 2026-08-27 |

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
| Current status | Downloaded locally on 2026-08-27; both Zenodo MD5 checksums verified |
| License | CC BY 3.0 US, verified from the Zenodo API on 2026-08-27 |

Files:

| Filename | Size | MD5 |
|---|---:|---|
| `UTexas Aptamer Database dataset.xlsx` | 511.0 kB | `65b1430f6bad16bec09e72ebf202a824` |
| `UTexas Aptamer Database dataset_Sept2023.xlsx` | 579.8 kB | `39e4838d14d7837be021f24daa87021d` |

Reported fields include sequence, target, nucleic-acid type, affinity, buffer, original library, modifications, publication information, and parent-sequence identifiers.

The downloaded September 2023 workbook contains one 28-column worksheet with 1,495 rows. The difference from the publication's 1,415 reported aptamer sequences must be reconciled during profiling; likely explanations include repeated sequences, child variants, or non-sequence rows, but none should be assumed without measuring them.

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
| Current status | Three core tables downloaded locally on 2026-08-27; SHA-256 checksums verified |
| License | Site and dataset reuse terms require verification |

Audit value:

- Provides standardized target identifiers, SMILES, affinity data, and experimental context.
- May contain verified steroid interactions suitable for retrospective evaluation.
- Steroid coverage, duplicates, parent families, assay comparability, and measured negatives remain unknown until profiling.

Core verified download endpoints are `interaction.csv`, `aptamer.csv`, and `molecule.csv` on the AptaDB site. The local snapshot contains 1,350 interaction rows, 1,293 aptamer rows, and 131 molecule rows. The interaction and aptamer files require Latin-1 decoding; this is recorded so ingestion does not silently replace characters. The site exposes downloads but does not state a clear dataset reuse license, so files and derived exports must remain local pending clarification.

### Aptamer Base recovered archive

The [Aptamer Base GitHub repository](https://github.com/micheldumontier/aptamerbase) preserves data that formerly lived in Freebase. The public repository contains aptamer, experiment, and interaction CSV files plus an RDF extract, totaling roughly 12.7 MB. GitHub reports an MIT repository license. The four files were downloaded locally on 2026-08-27 from commit `e7a3eb43d24504fd5a7b6c3e470ab242bc471b3a`, and their SHA-256 hashes are frozen in the machine registry.

The local snapshot has 2,128 aptamer rows, 16,059 experiment rows, and 4,551 interaction rows. Target identities are represented through interaction participants, so the small-molecule DNA subset requires provenance-preserving joins rather than filtering the aptamer table alone.

This archive is useful for recovering old provenance and references. It is not an independent source of experimental evidence relative to AptaBench, which already contains an Aptamer Base lineage.

### Discovery-only directories

[Apta-Index](https://www.aptagen.com/apta-index/) and [Aptabase](https://www.iitg.ac.in/proj/aptabase/) are useful for finding target names, candidate sequences, and primary publications. Neither exposed a clearly licensed bulk download during this audit. They should be queried manually for discovery and cross-reference—not scraped into the training corpus.

### Derived overlap sources

[AptaCom](https://github.com/rpgv/AptaCom) consolidates six aptamer databases into a nonredundant representation. That makes it useful for testing database lineage and duplicate rules, but it is secondary evidence and GitHub reports no repository license.

The [Mendeley DNA/Aptamer dataset](https://doi.org/10.17632/76jgjbgndr.1) contains 238 unique sequences derived from Aptagen and is CC BY 4.0. It may help compare sequence coverage, but target/assay detail is not sufficient to promote it automatically to supervised training data.

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
| Supplement | [ACS Figshare record](https://acs.figshare.com/articles/journal_contribution/One-Pot_SELEX_Identification_of_Specific_Aptamers_against_Diverse_Steroid_Targets_in_One_Selection/10498772) |
| Evidence | Candidate sequences, affinity measurements, counter-selection, and NGS-derived analysis |
| Public files found | Article and 1.6 MB supplementary PDF |
| Raw read repository | Not found during initial search |
| Current status | Supplement downloaded, checksum-verified, text-extracted, and visually reviewed on 2026-08-27 |
| License | CC BY-NC 4.0, verified from the ACS Figshare API |

Audit value:

- Contains multiple closely related steroid targets selected within one study.
- Counter-selection and cross-target design may provide unusually valuable specificity evidence.
- Five pools were sequenced. The last-round estradiol, progesterone, and testosterone pools contain 322,105 total reads and 36,379 unique 80–100 bp sequences in aggregate.
- Supplement Tables S4–S6 report 55 target-preferential candidates: 30 estradiol, 18 progesterone, and 7 testosterone sequences. Table S7 reports their cross-pool rank/copy counts, and Table S8 identifies six final selected aptamers.
- The reported full sequences contain primer regions, which must be retained explicitly rather than silently trimmed.
- Without raw reads, it contributes candidates, measured endpoints, and summarized abundance rather than a reconstructable enrichment trajectory.

### Estradiol Capture-SELEX study

| Field | Value |
|---|---|
| Registry ID | `estradiol_capture_selex_2022` |
| Targets | 17-beta-estradiol with related estrogen and counter-target testing |
| Publication | [Capture-SELEX of DNA Aptamers for Estradiol Specifically and Estrogenic Compounds Collectively](https://doi.org/10.1021/acs.est.2c05808) |
| Supplement | [ACS Figshare record](https://acs.figshare.com/articles/journal_contribution/Capture-SELEX_of_DNA_Aptamers_for_Estradiol_Specifically_and_Estrogenic_Compounds_Collectively/21636826) |
| Repository copy | [University of Waterloo record](https://uwspace.uwaterloo.ca/items/06aafdbf-ff47-464b-b04c-05ae7310448f) |
| Evidence | Short candidate sequences, ITC affinity, and cross-reactivity/selectivity observations |
| Raw read repository | Not found during initial search |
| Current status | Supplement downloaded, checksum-verified, and visually reviewed on 2026-08-27 |
| License | CC BY-NC 4.0, verified from the ACS Figshare API |

Audit value:

- Table S1 reports the N30 library and primers, five full-length P5/P7 candidates, three unmodified CN-Es constructs, and assay-specific modified versions that must remain linked to their parent sequences.
- Table S3 profiles five enriched candidates across rounds 8, 9, 11, and 12. Round 12 contains 50,728 reported reads, with CN-Es1 at 26.1%.
- Figure S5 provides measured binding for related estrogens and an explicit non-binding result for CN-Es2 against EE2. This can become a measured negative label after target-name normalization.
- The source contributes endpoint sequences, summarized enrichment, affinity, and selectivity evidence; without raw reads, it does not provide a reconstructable selection trajectory.

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

These sources are excluded from the ssDNA-small-molecule training set unless a future protein-target research question explicitly requires them.

### APTANI islet dataset

The previously encountered APTANI/islet FASTQ data concern complex cellular targets and are outside the current research scope. They will not be downloaded or mixed into the thesis pipeline.

### RNA, natural-RNA, and ion data

RiboaptDB, RNA records from Ribocentre and RSAPred, R-BIND, and RNA-only SELEX benchmarks are not substitutes for ssDNA. Natural RNA–small-molecule databases describe the inverse biological problem and must not be mixed with selected ssDNA aptamers. Metal-ion aptamers are also held out because coordination chemistry differs materially from recognition of organic small molecules.

### Constructed negatives

An untested aptamer–target pairing is unlabeled, not a negative. Random cross-pairs may be used only in a clearly labeled contrastive ablation. The preferred negative hierarchy is:

1. Directly measured nonbinding under documented conditions
2. Measured cross-reactivity or lack of response in a target panel
3. Counter-selection pool evidence
4. Selection-trajectory depletion or non-enrichment, with appropriate caveats
5. Constructed mismatch used only as an explicit modeling control

## Fields still requiring verification

Before a source is promoted to analysis, the registry must add or confirm:

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

## Acquisition order

The complete first-wave public supplements and database snapshots listed below were acquired and profiled by 2026-08-28. Raw files remain in the ignored local Bronze lake; only source metadata, checksums, profiles, and rights caveats are committed to Git. The order is retained as the reproducible acquisition plan.

The first wave is small, high-value, and useful for schema/provenance work:

1. UTexas Aptamer Dataset (1.1 MB, CC BY 3.0 US)
2. Aptamer Base CSV/RDF archive (about 12.7 MB, MIT repository)
3. Xiao-lab thermodynamics/specificity supplement (3.7 MB, CC BY-NC 4.0)
4. DL-SELEX endpoint supplement and ENA candidate metadata (18.4 MB plus metadata, CC BY 4.0 article)
5. High-affinity steroid receptor supplement (2.0 MB; reuse review required)
6. One-Pot and estradiol supplementary material
7. AptaDB core tables, retained locally while reuse terms remain unclear
8. Kynurenine N2A2 spreadsheet (164.3 MB, CC BY-NC-ND 4.0; transformed-data rights review required)

The second wave is the ~13.6 GB of DL-SELEX FASTQ data. It remains remote-only until a trajectory-analysis milestone is active. If acquired, download one paired round first, validate primers/read merging/checksums, estimate expanded storage, and only then fetch the remaining records.

Request-only or no-reuse sources—especially the progesterone array data—must not enter the pipeline until permission and a concrete data-delivery path are documented.
