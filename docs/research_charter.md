# Aptafind Research Charter

## Vision

Aptafind is a research-oriented bioinformatics and machine-learning platform for target-conditioned single-stranded DNA (ssDNA) aptamer discovery.

Its long-term objective is to accept a molecular target and produce a ranked set of short aptamer candidates that can be tested experimentally for affinity and specificity. The first research domain is steroid molecules because they form a chemically related target family, published aptamer evidence is available, and domain expertise and potential experimental support are accessible.

Aptafind generates scientific hypotheses. A computationally generated or highly ranked sequence is not a confirmed aptamer until binding and specificity have been measured experimentally.

## Near-term research objective

Determine whether dynamic information in multi-round SELEX trajectories provides transferable supervision for aptamer-small-molecule interaction prediction beyond static endpoint binding labels alone.

The primary near-term question is intentionally broader than the first available pretraining corpus:

> Does multi-round HT-SELEX trajectory information provide transferable supervision that improves aptamer-small-molecule endpoint prediction beyond matched generic pretraining?

The initial trajectory-pretraining corpus consists of steroid HT-SELEX experiments. Steroids are therefore the motivating case and current data source, not a restriction on the scientific hypothesis or an unsupported claim of broad steroid-specific evaluation.

The central comparison will hold the downstream architecture and evaluation data constant:

```text
AptaBench endpoint interactions only
                  versus
AptaBench endpoint interactions + steroid-SELEX trajectory pretraining
```

An improvement by the trajectory-pretrained model, beyond matched pretraining controls, would provide evidence that selection trajectories contain transferable information not captured by endpoint binding labels or general sequence exposure alone.

Conditional sequence generation remains a later stage. The project must first demonstrate reproducible target-aware ranking and measure whether trajectory pretraining contributes independent signal.

## Primary scientific hypothesis

Under matched architecture, sequence exposure, compute, downstream training, and evaluation, intact multi-round HT-SELEX trajectory supervision improves aptamer-small-molecule endpoint prediction relative to generic sequence pretraining that removes the trajectory signal.

The steroid atom graphs, aptamer sequence and structure representations, and interaction mechanisms described below are supporting methodology for testing this information-transfer hypothesis.

Steroid molecules share a core chemical scaffold but differ in functional groups, geometry, polarity, and interaction surfaces. Aptamer sequences also differ in sequence composition, structural ensembles, accessible bases, and possible binding pockets.

A biologically motivated model may combine:

- target chemical representation,
- aptamer sequence representation,
- predicted aptamer structural representation,
- multi-round selection trajectories, and
- experimentally measured binding evidence

to learn target-dependent patterns that generalize beyond memorized target-sequence pairs. The study's novelty claim does not depend on proving that this particular architecture is uniquely optimal.

## Conditional steroid generalization experiment

If Gate 1B and additional evidence establish sufficient independent steroid supervision, a secondary experiment will hold out an entire steroid target during training.

1. Exclude the held-out steroid's aptamer interactions and sequence families from model training.
2. Provide the trained system with the held-out steroid's chemical representation.
3. Rank known binders, known nonbinders where available, and appropriate decoy candidates.
4. Compare performance with random ranking, chemical-similarity retrieval, sequence-similarity retrieval, abundance, enrichment, and other interpretable baselines.
5. Determine whether known binders for the unseen target are recovered near the top of the ranking.
6. Repeat across targets when sufficient data are available.

If steroid endpoint supervision remains sparse, the primary A/B/C comparison will use the broader leakage-audited AptaBench small-molecule task, while steroid performance is reported as a limited motivating subgroup. The strongest future validation remains prospective: generate previously untested candidates for a held-out target and measure their affinity and cross-reactivity experimentally.

## Central trajectory-information ablation

The primary ablation will compare models with identical downstream architecture, training labels, splits, and evaluation metrics.

### Model A: endpoint-only baseline

- Train the target-aptamer ranker using curated AptaBench interaction evidence.
- Evaluate on the frozen leakage-resistant endpoint protocol; use held-out steroid targets only if Gate 1B establishes feasibility.

### Model B: trajectory-pretrained model

- Pretrain compatible aptamer representations using hydrocortisone and testosterone multi-round SELEX trajectories.
- Fine-tune the same downstream ranker using the same AptaBench interaction evidence.
- Evaluate on the identical held-out examples.

### Model C: matched pretraining control

- Use the same pretraining sequences and similar compute without intact trajectory supervision.
- Candidate controls include shuffled round labels, disrupted trajectories, or a sequence-only self-supervised objective.
- Fine-tune and evaluate using the same downstream procedure.

The experiment must control model capacity, downstream optimization, split assignment, and evaluation procedure. Pretraining introduces additional data exposure and computation, so Model C is necessary to distinguish trajectory-specific information from generic pretraining benefits. Any claimed gain must persist across documented seeds or folds and must not result from duplicate sequence families crossing data sources.

### Primary comparisons

#### Model B versus Model A

Tests whether trajectory pretraining, including both additional sequence exposure and intact trajectory information, improves downstream endpoint prediction.

#### Model C versus Model A

Tests whether matched generic pretraining improves downstream endpoint prediction without intact trajectory information.

#### Model B versus Model C

This is the primary scientific comparison. It tests whether intact SELEX trajectory information contributes transferable signal beyond generic sequence pretraining and matched computational exposure.

### Statistical hypotheses

Null hypothesis:

> Under matched downstream training and evaluation, Model B does not outperform Model C beyond expected experimental variation.

Alternative hypothesis:

> Under matched downstream training and evaluation, Model B improves aptamer-ligand interaction prediction relative to Model C because intact multi-round trajectory information provides additional transferable supervision.

The exact estimand, metric, minimum scientifically relevant effect, uncertainty interval, and statistical procedure must be preregistered in the analysis plan after Gate 1B establishes the available evaluation units.

Random seeds are technical repetitions, not independent biological replicates. Statistical inference must respect the actual independent units, such as held-out targets, sequence families, experiments, or appropriately paired resampling units.

### Matched-control requirements

Models B and C should match in:

- Encoder architecture
- Parameter count and initialization policy
- Tokenizer and input representation
- Pretraining sequence membership and sampling policy
- Optimizer and learning-rate schedule
- Number of optimization steps
- Approximate compute budget
- Batch construction
- Augmentation policy
- Downstream fine-tuning procedure
- Evaluation folds and metrics

The intended experimental difference is whether intact round order, enrichment trajectory, or another formally specified selection-dynamics signal is available during pretraining.

One primary Model C control must be selected before final evaluation. Additional controls, such as shuffled round labels and sequence-only self-supervision, may be reported as secondary ablations but must not be chosen post hoc based on test performance.

### Interpretation rules

- If Model B outperforms both Models A and C, the results support transferable information in intact SELEX trajectories.
- If Models B and C both outperform Model A but remain similar to each other, the evidence supports generic pretraining benefits rather than trajectory-specific information.
- If Model B outperforms Model A but not Model C, trajectory-specific benefit is not established.
- If none outperform Model A, the selected pretraining strategies do not add demonstrated downstream value under the tested conditions.
- If results vary strongly by target, report target-dependent effects rather than one universal conclusion.

A null or inconclusive result remains scientifically informative when the evaluation is sufficiently powered, leakage-resistant, and honestly bounded to the tested data and methods.

## Evidence hierarchy

Aptafind will keep distinct forms of evidence separate rather than treating them as interchangeable labels.

From strongest to more indirect evidence:

1. Experimentally measured binding affinity under documented conditions
2. Experimentally measured specificity, nonbinding, or counter-target response
3. Replicated binding measurements or orthogonal assay confirmation
4. Multi-round target-selection enrichment and persistence
5. Late-round abundance without a complete selection trajectory
6. Published candidate designation without a quantitative measurement
7. Constructed negative or decoy pairs

SELEX enrichment is evidence of selection, not direct proof of binding affinity. PCR amplification, sequencing, nonspecific retention, library composition, and experimental conditions can influence observed abundance.

Detailed label definitions and permitted uses will be maintained in `docs/evidence_model.md`.

## Initial target and data scope

The first research scope includes:

- Verified ssDNA aptamer-steroid interactions from curated databases and primary literature
- Quantitative affinity measurements and assay conditions where available
- Positive-, negative-, and counter-selection observations
- Multi-round hydrocortisone and testosterone HT-SELEX sequencing data
- A separate hydrocortisone selection performed from a manually designed library
- Chemical structures and identifiers for steroid targets

Other small-molecule and non-steroid aptamers may later support pretraining, negative controls, or transfer-learning experiments, but they will not be mixed into the initial steroid analysis without an explicit scientific purpose.

## Modeling progression

Aptafind will advance through evidence-based stages:

1. Curate and validate steroid aptamer-target evidence.
2. Reconstruct sequence abundance and enrichment trajectories across SELEX rounds.
3. Establish sequence-only, structure-only, chemistry-only, and enrichment baselines.
4. Train target-aware candidate-ranking models.
5. Evaluate held-out-target and sequence-family generalization.
6. Add conditional candidate generation only after ranking demonstrates useful signal.
7. Apply structural and practical filters to generated candidates.
8. Experimentally test prioritized candidates.
9. Incorporate positive and negative experimental feedback into subsequent models.

## Aptamer design objectives

Candidate quality is multi-objective. Aptafind should consider:

- Predicted target binding or enrichment
- Specificity against closely related steroids
- Structural stability and confidence
- Sequence-family novelty and diversity
- Synthesis practicality
- Sequence length
- Experimental conditions and intended application

Shorter candidates are desirable when they reduce synthesis cost and preserve the essential binding structure. Short length is not itself proof of greater specificity: truncation can disrupt necessary stems, loops, or long-range interactions. The design goal is therefore the shortest sequence that preserves experimentally acceptable affinity, specificity, and stability.

## Data and evaluation safeguards

The project will enforce the following scientific safeguards:

- Raw source files remain immutable and checksummed.
- Every record retains its source, publication, license, and transformation history.
- Affinity values retain their original value, unit, assay, buffer, and conditions.
- Measurements produced by different assays are not assumed to be directly comparable.
- Aptamer families, truncated variants, and near-duplicate sequences are identified before splitting.
- Primary evaluation splits by target and, where possible, sequence family or publication source.
- Preprocessing and feature fitting use training data only.
- Constructed negatives are explicitly labeled and evaluated separately from measured nonbinding pairs.
- Baselines precede complex neural or generative models.
- Model output is reported with uncertainty and limitations.
- Generated sequences are never described as validated binders without experimental evidence.

## Engineering objective

Aptafind will also serve as an end-to-end scientific computing capstone demonstrating:

- Bioinformatics processing of FASTQ and aptamer sequence data
- Reproducible Python and SQL data pipelines
- A cloud-compatible Bronze, Silver, and Gold research lake
- Parquet-based analytical datasets and DuckDB/Athena querying
- Workflow orchestration and containerized execution
- Automated validation, testing, and continuous integration
- Chemical and nucleic-acid feature engineering
- Statistical analysis, machine learning, and generative modeling
- Experiment tracking, model evaluation, and scientific reporting

Technology choices must support the research. Infrastructure will not be added solely to make the system appear more complex.

## Research and commercialization pathways

### Research pathway

The project may support:

- A reproducible computational study
- Collaboration with steroid and aptamer laboratories
- Prospective experimental validation
- Peer-reviewed publication
- A future doctoral research program
- Expansion to other related small-molecule families

### Commercial pathway

The long-term commercial concept is a computationally guided aptamer-discovery service. A customer supplies a target and intended application; Aptafind prioritizes candidates, and laboratory partners validate affinity and specificity.

Commercial value would come from reducing the experimental search space, cost, and time required to obtain a useful aptamer. Unvalidated generated sequences alone are not the final commercial product.

Before commercialization, the project will require review of intellectual-property ownership, university policies, dataset and software licenses, sequence patentability, freedom to operate, customer confidentiality, and application-specific regulatory obligations.

## Success criteria

### Data foundation

- Public datasets are registered with provenance, license, version, size, and checksum information.
- Raw sources can be transformed reproducibly into validated analytical tables.
- Sequence, target, experiment, measurement, and selection-round identities remain traceable.

### Computational evidence

- Enrichment trajectories are reproduced from at least one multi-round steroid experiment.
- Models are compared with transparent and appropriately difficult baselines.
- Evaluation prevents target and sequence-family leakage.
- At least one held-out-target experiment is completed and reported honestly.

### Scientific evidence

- Known held-out binders are prioritized above suitable controls with reproducible results.
- Target specificity is evaluated using chemically related steroids rather than unrelated decoys alone.
- The limitations of enrichment-derived and constructed labels are quantified.

### Long-term validation

- Novel candidates are synthesized and tested.
- Affinity and cross-reactivity are measured under documented conditions.
- Experimental successes and failures are incorporated into a versioned learning loop.

## Non-goals for the initial release

The initial release will not claim:

- Reliable aptamer generation for arbitrary target classes
- Experimentally confirmed binding from computational prediction alone
- Therapeutic or diagnostic efficacy
- A production commercial service
- Complete molecular docking or atomically accurate binding-site prediction

These remain long-term research directions rather than current capabilities.
