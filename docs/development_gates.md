# Aptafind Development Gates

## Purpose

Aptafind will advance through explicit scientific gates. Each gate must produce documented evidence before work proceeds to the next stage.

The gates prevent implementation complexity from getting ahead of the available data and prevent the ranker and generator from being evaluated only against one another.

## Gate 1: Dataset sufficiency and independence audit

Gate 1 is divided into benchmark characterization and cross-dataset independence. This prevents a well-characterized endpoint benchmark from being mistaken for an independent evaluation set.

The scientific hypothesis is:

> Does multi-round HT-SELEX trajectory information provide transferable supervision that improves aptamer-small-molecule endpoint prediction beyond matched generic pretraining?

The planned initial pretraining corpus is steroid HT-SELEX. Steroids motivate the study but do not constrain the hypothesis to a steroid-only downstream benchmark.

### Gate 1A: Benchmark characterization — complete

Objectives:

- Freeze the exact AptaBench release and file hashes.
- Profile schema, contents, labels, missingness, and duplicates.
- Characterize the DNA and steroid subsets.
- Quantify endpoint-label and sequence-family coverage.
- Document benchmark limitations.

Result:

- AptaBench is suitable for general aptamer-small-molecule endpoint modeling.
- Its steroid subset contains 79 records, 9 connectivity-level targets, and 61 families at 90% identity.
- Only 10 steroid negatives are present, all for one target connectivity.
- AptaBench alone is insufficient for a rigorous steroid-only Model A/B/C comparison.

Evidence: [Frozen AptaBench Profile Report](aptabench_frozen_profile.md).

### Gate 1B: Cross-dataset independence audit — in progress

Determine whether every planned pretraining dataset is sufficiently independent of endpoint evaluation data.

The audit must evaluate:

- Publication overlap
- Exact sequence overlap
- Sequence-family and parent/variant overlap
- Assay and experimental-study overlap
- Source-database overlap
- Duplicate affinity measurements
- Ligand identity and stereochemistry consistency
- Pretraining/evaluation independence after documented exclusions
- Whether any remaining shortcut threatens a causal interpretation of Model B versus Model C

The audit will determine how ambitious Version 1 of the ranker can be.

### Gate 1B exit criteria

- Publication, sequence, family, assay, source, affinity, and ligand-identity overlap have been measured for every planned pretraining source.
- Exclusion rules and leakage-safe evaluation units have been documented.
- Pretraining and evaluation data are sufficiently independent to support a causal interpretation of Model B versus Model C, or the claim is narrowed explicitly.
- A matched non-trajectory pretraining control is feasible and defined before model evaluation.
- Final train, validation, and test protocols can be frozen without using model results.

### Current implementation status

- Model development: **blocked pending Gate 1B**.
- Training-split freeze: **blocked pending Gate 1B**.
- Large-scale HT-SELEX FASTQ acquisition and preprocessing: **deferred**.
- Permitted acquisition: supplementary sequence tables, published aptamer inventories, accession lists, and sequence identifiers needed for the independence audit.

## Gate 2: Ranking model

Develop and validate a target-aptamer ranker:

```text
R(a, t)
```

Where:

- `a` is an aptamer candidate.
- `t` is a molecular target.
- `R(a, t)` is a compatibility, enrichment, affinity, or specificity score whose interpretation is defined by its training evidence.

The ranker must be evaluated under:

- Sequence-family holdouts
- Target-level holdouts

It must also be compared with transparent baselines and ablations.

### Gate 2 exit criteria

- Data splitting prevents known target and sequence-family leakage.
- The ranker outperforms random and appropriate non-neural baselines on held-out evidence.
- Target-aware features improve performance beyond sequence-only and target-only alternatives.
- Performance uncertainty and variation across targets are reported.
- Results are reproduced from versioned data and configuration artifacts.

## Gate 3: Unguided target-conditioned diffusion

Train and evaluate a discrete sequence generator independently of the ranker.

The generator should demonstrate that it can produce:

- Valid ssDNA sequences
- Diverse candidates
- Novel aptamer-like sequences
- Target-conditioned differences that are not explained solely by memorization
- Configurable or biologically plausible sequence lengths

The ranker must not be used as the sole measure of Gate 3 success.

### Gate 3 exit criteria

- Generated sequences pass syntax, length, novelty, and duplication checks.
- Diversity and similarity to training families are quantified.
- Target conditioning is evaluated using independent distributional or held-out evidence.
- The generator is compared with random-library, empirical-frequency, and simpler sequence-generation baselines.
- Memorization and mode collapse are assessed.

## Gate 4: Generate, rank, and rerank

Evaluate whether the independently trained ranker improves candidate selection.

```text
Target
   |
   v
Discrete diffusion
   |
   v
Candidate library
   |
   v
Ranking model
   |
   v
Prioritized candidates
```

The primary comparison is ranked selection versus unguided or random selection from the same generated candidate pool.

### Gate 4 exit criteria

- Generated candidate libraries are fixed before comparative ranking.
- Ranker-based selection is compared with random and heuristic selection.
- Improvement is measured using evidence not used to train the ranker where possible.
- Ranker uncertainty, disagreement, and possible adversarial exploitation are evaluated.
- Selected candidates remain valid, diverse, and practically synthesizable.

## Gate 5: Ranker-guided discrete diffusion

Only after validating the ranker and generator independently may the ranker influence the generation process.

Guidance must be mathematically defined for a discrete state space. Possible approaches include:

- Reward-weighted transition probabilities
- Guidance through predicted clean-sequence probabilities
- Energy-based reweighting
- Candidate resampling or sequential Monte Carlo methods
- Discrete policy or reward optimization

The exact mechanism remains an implementation decision until the diffusion formulation and ranker behavior are established.

A continuous classifier-guidance expression must not be applied literally to discrete transition probabilities without a justified relaxation or discrete formulation.

### Gate 5 exit criteria

- The guidance rule is mathematically documented and tested.
- Guided generation is compared with the same unguided generator.
- Candidate validity and diversity do not collapse under guidance.
- Improvements persist on held-out targets or evidence.
- The generator does not merely exploit known ranker weaknesses.

## Gate 6: Experimental validation

Test prioritized candidates experimentally.

Evaluation should determine whether candidates:

- Bind the intended target
- Retain specificity against closely related steroid targets
- Achieve useful affinity under documented conditions
- Outperform unguided candidate selection
- Preserve predicted structural behavior where measurable
- Remain effective after any proposed truncation

### Gate 6 exit criteria

- Experimental protocols, controls, buffers, and conditions are documented.
- Affinity and cross-reactivity are measured using appropriate assays.
- Positive and negative outcomes are retained.
- Computational predictions are compared with experimental results.
- New evidence is versioned for subsequent model development.

## Current status

| Gate | Status |
|---|---|
| 1. Dataset sufficiency audit | In progress |
| 2. Ranking model | Not started |
| 3. Unguided target-conditioned diffusion | Not started |
| 4. Generate, rank, and rerank | Not started |
| 5. Ranker-guided diffusion | Not started |
| 6. Experimental validation | Not started |

No architecture hyperparameters, diffusion schedule, or guidance method will be finalized before Gate 1 establishes what supervision the data can realistically support.
