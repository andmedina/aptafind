# Gate 1 Dataset Sufficiency and Independence Audit

## Audit question

> Are the endpoint benchmark and planned HT-SELEX pretraining sources sufficiently independent to test whether trajectory supervision improves aptamer-small-molecule endpoint prediction beyond matched generic pretraining?

Gate 1A characterized the benchmark and is complete. Gate 1B now tests publication, sequence, family, assay, source, affinity, and ligand-identity independence across datasets.

This document defines how that question will be answered. Results will be added only after sources have been acquired, validated, standardized, and profiled.

## Unit of independence

Raw row count and sequencing-read count are not valid measures of independent supervision.

Aptafind will report at least the following units separately:

- Unique normalized sequence
- Sequence family or similarity cluster
- Unique target
- Unique aptamer-target pair
- Independent binding measurement
- Independent selection experiment
- Independent publication or laboratory source
- Sequence observed in a particular SELEX round
- Sequence trajectory across multiple rounds

For example, ten million reads from one hydrocortisone selection can provide detailed within-experiment enrichment evidence, but they do not create ten million independent target-level examples.

## Evidence classes

| Evidence class | Meaning | Primary permitted use |
|---|---|---|
| Measured affinity | Quantitative binding measurement with assay context | Affinity analysis, ranking, retrospective evaluation |
| Measured binding | Experimental evidence of binding without a directly comparable affinity value | Classification or supporting evaluation |
| Measured nonbinding | Experimental failure to bind under documented conditions | High-quality negative evidence |
| Measured cross-reactivity | Same aptamer tested against related targets | Target-specific ranking and hard-negative evaluation |
| Counter-selection | Sequence or pool exposed to a counter-target during selection | Specificity evidence with protocol caveats |
| Multi-round enrichment | Sequence-frequency trajectory across selection rounds | Within-experiment ranking and enrichment modeling |
| Late-round abundance | Sequence observed in an enriched pool without full trajectory | Weaker candidate evidence |
| Published candidate | Reported candidate without sufficient measurement detail | Retrieval, curation, or weak supervision |
| Constructed target mismatch | Aptamer paired with an unmeasured target | Contrastive experiment only; never a measured negative |

## Required profiles

### Target coverage

For each steroid:

- Unique sequences
- Unique sequence families
- Independent publications
- Independent experiments
- Measured affinity records
- Measured nonbinding records
- Cross-target measurements
- Multi-round experiments
- Assay types and conditions

### Sequence-family structure

The audit will cluster exact and near-duplicate sequences to estimate how many distinct families exist.

At minimum, report:

- Exact duplicates
- Reverse-complement duplicates where scientifically relevant
- Parent and truncated variants
- Edit-distance or identity-based clusters at documented thresholds
- Families spanning multiple publications or databases
- Families spanning multiple targets

Thresholds will be selected after inspecting sequence-length distributions and known parent-child annotations. Sensitivity to the threshold must be reported.

### Experimental independence

Records will be grouped by:

- Publication DOI
- Laboratory or study where known
- Selection experiment
- Initial library
- Target and counter-target design
- Assay method
- Buffer and temperature

Multiple measurements of the same sequence-target pair are retained, but they do not automatically count as independent aptamers.

### Label quality

For every prospective modeling label, report:

- Directly measured versus inferred status
- Missingness
- Assay comparability
- Unit normalization feasibility
- Replication
- Contradictions between sources
- Source confidence
- Whether the label can be used for training, evaluation, or descriptive analysis only

## Leakage audit

The following leakage paths must be tested before modeling:

1. Exact sequence appears in both training and evaluation data.
2. Truncated or extended variants cross the split boundary.
3. Near-identical sequence-family members cross the split boundary.
4. Same target appears in a purported target-level holdout.
5. Same selection experiment contributes related sequences to both sides.
6. Duplicate database records from the same publication cross the split boundary.
7. Preprocessing, structure fitting, or target-feature scaling uses evaluation data.
8. Candidate negatives are constructed using information unavailable at prediction time.

The cross-source audit must also determine, for every endpoint record where possible:

- Whether the aptamer originated from a planned pretraining SELEX experiment
- Whether the same sequence appears under multiple database identifiers
- Whether the same primary publication contributes to pretraining and evaluation
- Whether the same affinity measurement was copied across multiple databases
- Whether parent, truncated, extended, or point-mutated variants cross split boundaries
- Whether assay or source identity acts as a shortcut for the label

## Candidate evaluation designs

### Design 0: trajectory-information transfer ablation

Question:

> Does multi-round HT-SELEX trajectory pretraining improve aptamer-small-molecule endpoint prediction beyond matched generic pretraining?

Comparison:

```text
Endpoint-only ranker
        versus
Same ranker + steroid-SELEX trajectory pretraining
        versus
Same ranker + matched non-trajectory pretraining control
```

Minimum requirement:

- A leakage-audited endpoint interaction benchmark
- Identical downstream ranker architecture and evaluation folds
- Multi-round steroid SELEX data suitable for a clearly defined pretraining task
- A matched control for additional sequence exposure and pretraining compute
- Cross-source sequence-family overlap analysis
- A leakage-resistant endpoint protocol with steroid subgroup analysis where justified
- Repeated seeds or folds sufficient to characterize variation

Scientific scope:

- Tests whether selection trajectories contain transferable representation signal beyond endpoint labels.
- Separates trajectory-specific signal from generic sequence pretraining where possible.
- Does not establish that enrichment is equivalent to binding affinity.
- Requires special care because hydrocortisone and testosterone may also appear in the downstream benchmark.

### Design A: within-experiment enrichment ranking

Question:

> Can early-round sequence and structural information predict later-round enrichment within a steroid SELEX experiment?

Minimum requirement:

- At least three rounds from one experiment
- Reliable primer trimming and merged-read processing
- Unique-sequence counts per round
- A defensible treatment of zero counts and sampling depth

Scientific scope:

- Demonstrates enrichment prediction, not unseen-target binding prediction.

### Design B: cross-experiment same-target transfer

Question:

> Do sequence or structural signals transfer between independently designed libraries for the same target?

Minimum requirement:

- At least two sufficiently independent selection experiments for one steroid
- Documented differences in library design and protocol

Scientific scope:

- Tests robustness beyond a single library while holding target identity constant.

### Design C: cross-target specificity ranking

Question:

> Can the model rank an aptamer higher for its measured steroid target than for closely related steroids?

Minimum requirement:

- Multiple steroid targets
- Measured cross-reactivity, nonbinding, or strong counter-selection evidence
- Enough independent sequence families to prevent memorization

Scientific scope:

- Tests target specificity among related molecules.

### Design D: leave-one-steroid-out generalization

Question:

> Can a model trained without one steroid use its chemical representation to recover known binders for that target?

Minimum requirement:

- Multiple training steroids with sufficiently independent aptamer families
- A held-out steroid with independently measured binders and suitable controls
- Target chemical features calculated without label information
- Sequence-family and publication-aware splitting

Scientific scope:

- Primary test of the target-conditioned Aptafind hypothesis.

## Model-feasibility levels

The audit will assign the project to the highest justified level.

### Level 0: descriptive analysis only

Use when evidence is too sparse, inconsistent, or dependent for supervised modeling.

Permitted work:

- Data curation
- Sequence and target profiling
- Motif and structure exploration
- Similarity retrieval

### Level 1: within-experiment enrichment model

Use when multi-round pools are available but target diversity is insufficient.

Permitted claim:

- Predicts or ranks enrichment within specified SELEX experiments.

### Level 2: cross-experiment or specificity model

Use when independent experiments or measured cross-target evidence are available.

Permitted claim:

- Learns signals that transfer across libraries or discriminate among tested steroid targets.

### Level 3: held-out-target ranker

Use when multiple independent targets and families support target-level evaluation.

Permitted claim:

- Evaluates generalization to an unseen steroid under documented holdout conditions.

### Level 4: prospective candidate prioritization

Use only after retrospective held-out performance is credible and experimental collaborators can test novel candidates.

Permitted claim:

- Produces experimentally testable hypotheses for prospective validation.

## Initial decision thresholds

These are planning thresholds, not universal biological laws. They may be revised before looking at model test results, with justification recorded.

| Requirement | Minimum planning threshold | Preferred threshold |
|---|---:|---:|
| Independent steroid targets for leave-one-target-out work | 4 | 6 or more |
| Training targets in each held-out-target experiment | 3 | 5 or more |
| Independent sequence families per evaluated target | 10 | 30 or more |
| Independent publications/experiments per target | 2 where possible | 3 or more |
| Measured binders for held-out retrospective recovery | 5 | 20 or more |
| Measured related-steroid negatives/cross-reactivity observations | 5 | 20 or more |
| Multi-round observations per SELEX experiment | 3 rounds | 4 or more rounds |

Meeting a numeric threshold does not override severe confounding, inconsistent assays, or family leakage. Falling below a threshold does not invalidate descriptive or within-experiment analysis.

## Gate 1A benchmark scorecard

The frozen AptaBench profile produced the following benchmark-level result.

| Dimension | Result | Evidence | Decision |
|---|---|---|---|
| Steroid target coverage | 9 connectivity-level targets | Frozen profile | Limited subgroup only |
| Independent sequence-family coverage | 61 families at 90% identity | Frozen profile | Provisional; alignment-aware validation needed |
| Measured affinity coverage | 40 steroid records | Frozen profile | Sparse and uneven |
| Measured negative coverage | 10 records, one target connectivity | Frozen profile | Insufficient for broad steroid specificity |
| Cross-reactivity coverage | Not resolved as an independent evidence class | Frozen schema/source data | Gate 1B/manual curation needed |
| Multi-round enrichment coverage | Not present in AptaBench | Dataset schema | External pretraining data required |
| Assay comparability | Heterogeneous and incompletely encoded | Buffer/origin profile | Gate 1B needed |
| Provenance completeness | 1,157 benchmark records lack origin | Frozen profile | Incomplete |
| Licensing suitability | CC BY 4.0 declared | Frozen dataset card | Suitable subject to attribution |
| Sequence-family holdout feasibility | Feasible for general benchmark | Fixed split/family audit | Must freeze after Gate 1B |
| Steroid target-level holdout feasibility | Labels severely imbalanced by target | Frozen profile | Not rigorous with AptaBench alone |
| Independent statistical units | 9 targets, 61 families, 22 provisional origins | Frozen profile | Experimental independence unverified |
| Detectable effect/power feasibility | Not yet estimable | Independence unresolved | Gate 1B needed |
| Matched-control feasibility | Conceptually defined | Research charter | Freeze after Gate 1B |

## Gate 1B independence scorecard

| Dimension | Status |
|---|---|
| Publication overlap | In progress |
| Exact sequence overlap | Pending compact pretraining inventory |
| Sequence-family overlap | Pending compact pretraining inventory |
| Assay/experiment overlap | Pending publication audit |
| Source-database overlap | Pending |
| Affinity duplication | Pending |
| Ligand identity consistency | AptaBench profiled; cross-dataset audit pending |
| Pretraining/evaluation independence | Not yet established |

## Statistical analysis prerequisites

Before model training begins, the data profile must identify:

- The independent unit for each primary comparison
- The number of held-out targets, sequence families, and experiments
- Whether the same evaluation units can be paired across Models A, B, and C
- The primary performance metric
- A minimum scientifically relevant improvement
- A confidence-interval or resampling procedure appropriate to the evaluation design
- How repeated folds and random seeds will be summarized without treating seeds as biological replicates
- Whether the available units provide enough precision to distinguish a useful gain from noise

The primary Model C objective must be selected before final model evaluation. Test-set results must not be used to choose between alternative matched controls.

## Gate decision

Current decision: **Gate 1A complete; Gate 1B in progress**.

Model development, training-split freezing, and large-scale FASTQ processing are blocked until Gate 1B documents the proposed train, validation, and test units and establishes the independence needed to interpret Model B versus Model C.
