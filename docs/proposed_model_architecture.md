# Proposed Aptafind Model Architecture

## Status

This document defines the first serious model architecture proposed for the steroid-focused Aptafind research program. It is a research specification, not an implemented or validated result.

Implementation should begin only after the dataset sufficiency audit determines that the available positive, negative, enrichment, counter-selection, and cross-target evidence can support leakage-resistant training and evaluation.

## System overview

```text
Steroid molecular graph
        |
        v
Target graph encoder (E_t)

ssDNA sequence + predicted secondary-structure graph
        |
        v
Aptamer graph encoder (E_a)

          E_t             E_a
           |               |
           +---> Bidirectional <---+
                 cross-attention
                        |
                        v
          Compatibility/ranking score R(t, a)
```

Where:

- `E_t` is the learned target representation.
- `E_a` is the learned aptamer representation.
- `R(t, a)` is the learned compatibility or enrichment score for target `t` and aptamer `a`.

The ranker supplies the biological objective. A separate discrete sequence generator searches aptamer sequence space.

## Steroid target representation

Steroids are sufficiently small that the initial model can preserve atom-level chemistry without requiring an expensive atomistic representation of a large macromolecule.

### Nodes

Each atom node should initially encode:

- Element identity
- Formal charge
- Hybridization
- Aromaticity
- Hydrogen-bond donor status
- Hydrogen-bond acceptor status
- Optional stereochemical annotation

### Edges

Each chemical-bond edge should initially encode:

- Bond type
- Bond order
- Aromaticity or conjugation where applicable

Accurate 3D conformers will not be mandatory for the first ranker. Later experiments may add atom coordinates, interatomic distances, conformer ensembles, or spatial edges when those additions can be evaluated independently.

## Aptamer representation

The aptamer representation must preserve both primary-sequence topology and predicted folded secondary structure.

### Nucleotide nodes

Each nucleotide node may encode:

- Nucleotide identity
- Absolute and relative position
- Paired or unpaired state
- Base-pairing probability
- Predicted stem, loop, bulge, or junction membership
- Local accessibility
- Optional local structural-energy information

### Backbone edges

Backbone edges connect adjacent nucleotides and preserve the directional 5-prime-to-3-prime sequence relationship.

Candidate features include:

- Edge type: backbone
- Direction
- Sequential distance

### Predicted base-pair edges

Base-pair edges connect nucleotides predicted to interact in the folded aptamer, including pairs that are distant in the linear sequence.

Candidate features include:

- Edge type: predicted base pair
- Pair identity
- Pairing probability
- Structural confidence
- Separation in primary-sequence position

Where possible, pairing probabilities from a structural ensemble should supplement the single minimum-free-energy structure. A predicted fold is uncertain evidence rather than ground truth.

## Encoders and interaction module

The target and aptamer encoders should produce node-level representations rather than reduce each graph immediately to one fixed fingerprint.

Bidirectional cross-attention then models target-aptamer relationships:

- Aptamer nucleotides attend to steroid atoms and functional groups.
- Steroid atoms attend to aptamer nucleotides and structural regions.

This creates an interaction representation rather than merely concatenating independent target and aptamer embeddings.

Cross-attention weights may support exploratory interpretation, but they must not automatically be described as experimentally established contacts or binding sites.

## Ranking tasks

The exact meaning of `R(t, a)` depends on the evidence used for training. Separate outputs or models may be needed for:

- Relative SELEX enrichment
- Experimentally measured affinity
- Binary measured binding or nonbinding
- Target specificity and cross-reactivity
- General compatibility ranking

Evidence types should not be collapsed into one label without a justified statistical model. Multi-task learning may be considered when the dataset profile supports it.

Potential objectives include:

- Pairwise ranking: a stronger or more enriched candidate should outrank a weaker candidate for the same target.
- Affinity regression: predict a standardized affinity outcome while retaining assay context.
- Binding classification: distinguish measured binders from measured nonbinders.
- Contrastive target specificity: score an aptamer higher for its demonstrated target than for chemically similar counter-targets.

## Hard negatives and steroid specificity

Cortisol, testosterone, progesterone, estradiol, and related steroids share a core scaffold. This makes them scientifically useful hard-negative targets.

A target-specific model should learn relationships such as:

```text
R(cortisol, cortisol-specific aptamer)
    >
R(testosterone, cortisol-specific aptamer)
```

However, an unmeasured mismatched pair is not automatically a true nonbinder. The project will distinguish:

- Experimentally measured nonbinding
- Counter-selection evidence
- Measured cross-reactivity
- Constructed cross-target decoys

Constructed decoys may support contrastive training but must be evaluated separately from experimental negatives.

## Generation strategy

The proposed generator is a target-conditioned discrete diffusion or masked-sequence model operating directly in nucleotide sequence space.

### First generation experiment: post-generation reranking

```text
Target-conditioned or unguided discrete generator
                      |
                      v
             Large candidate library
                      |
                      v
          Independently evaluated ranker
                      |
                      v
              Prioritized candidates
```

This should be compared against unguided or random candidate selection. The experiment tests whether the ranker provides useful biological prioritization before allowing it to alter generation.

### Later generation experiment: guided generation

Only after the ranker generalizes to held-out evidence should Aptafind evaluate ranker-guided generation. The ranking signal may influence discrete denoising through a formally defined guidance, reward, energy-based, or candidate-resampling method.

The implementation must respect the mathematics of discrete sequence generation. A continuous gradient expression is only conceptual unless the selected discrete diffusion formulation provides a valid differentiable relaxation or guidance mechanism.

## Evaluation safeguards

The ranker and generator must not validate each other in a closed loop. Otherwise, the generator may learn to exploit ranker errors.

Required safeguards include:

- Entire-target holdouts
- Sequence-family or similarity-aware splits
- Independent affinity or enrichment evidence
- Comparison with transparent baselines
- Hard-negative steroid targets
- Ranker uncertainty or model disagreement
- Candidate validity, novelty, and diversity checks
- Structural and synthesis filters
- Prospective experimental validation

## Resolution hierarchy for future targets

Aptafind should use the lowest-cost representation that retains the relevant biological information.

```text
Steroid or other small molecule
    -> atom-level molecular graph

Protein target
    -> residue-level sequence/structure graph initially

ssDNA aptamer
    -> nucleotide graph with backbone and predicted base-pair edges

Final candidate reranking
    -> selected atom-level 3D modeling, docking, or molecular dynamics
```

## Required baselines

The proposed architecture must earn its complexity by outperforming appropriate baselines:

- Random ranking
- Round abundance
- Fold enrichment
- Sequence similarity
- Target chemical similarity
- Engineered-feature linear models
- Tree-based models
- Sequence-only encoder
- Structure-only encoder
- Target-only encoder
- Concatenation without cross-attention

Ablation experiments should determine whether target graphs, aptamer structural edges, and bidirectional cross-attention each contribute measurable held-out performance.

## Next research gate

Before model implementation, audit whether the available sources contain sufficient:

- Positive binders
- Measured negatives
- Counter-selection observations
- Cross-target measurements
- Quantitative affinity measurements
- Multi-round enrichment trajectories
- Independent steroid targets
- Sequence families and target coverage

The architecture remains provisional until this evidence profile is complete.
