# Historical CVAE Retrospective Run

## Status

This report records the first complete run of the modern PyTorch
target-conditioned sequence CVAE on the locally recovered late-2023 Aptafind
dataset. The run completed on August 27, 2026.

The result demonstrates a reproducible software workflow. It does not
demonstrate target-specific binding or establish that generated sequences are
functional aptamers.

## Source identity

The raw source files remain in the local Research archive and are not
redistributed through Git.

| Source | SHA-256 |
|---|---|
| Historical aptamer CSV | `a2ccdc64a9b73f013558780f2f22bbec028dc4564b4893b512d3f16a42d8d511` |
| Historical target-feature CSV | `46f12f11f10643777aba52d40b79d1d88bd87518f0805c7a6f606696d6722b0e` |

Validated dataset:

- 168 source rows
- 165 unique sequence–target pairs after removing 3 exact duplicates
- 154 unique DNA sequences
- 64 canonical target molecules
- Sequence lengths from 9 to 109 bases

## Evaluation design

The deterministic seed-42 splitter balances both rows and molecule counts while
keeping canonical target SMILES disjoint:

| Partition | Rows | Targets |
|---|---:|---:|
| Training | 133 | 52 |
| Validation | 16 | 6 |
| Test | 16 | 6 |

RDKit descriptor scaling was fitted only on the 52 training molecules. The test
targets were not used for model fitting, early stopping, or featurizer scaling.

Configuration: `configs/thesis_cvae.yaml`

Recorded software versions: Python 3.11.16, PyTorch 2.7.1, NumPy 2.4.6,
Pandas 3.0.5, RDKit 2026.03.5, and PyYAML 6.0.3.

- 32-dimensional DNA embedding
- 64-unit bidirectional GRU encoder
- 16-dimensional latent space
- 128-unit autoregressive GRU decoder
- 136 target-condition features: 8 standardized descriptors plus 128 Morgan bits
- Maximum 40 epochs with KL warmup and validation early stopping
- Best epoch: 7

Two independent runs with identical inputs and configuration produced
byte-identical training histories and split manifests, plus identical numerical
metrics after excluding the creation timestamp.

## Held-out results

| Model | Test NLL | Perplexity | Token accuracy |
|---|---:|---:|---:|
| Unigram baseline | 1.4543 | 4.2816 | 25.47% |
| Bigram baseline | 1.4491 | 4.2592 | 25.37% |
| Conditional sequence CVAE | **1.4358** | **4.2032** | **29.45%** |

Additional CVAE values:

- KL divergence: 0.1164
- Beta-weighted total loss: 1.4417
- Evaluated tokens excluding padding: 1,005
- Reconstruction-NLL improvement over the bigram baseline: 0.91%

The model modestly outperformed first-order nucleotide baselines on posterior
reconstruction. The margin is small. It does not establish meaningful
target-conditioned generation, and posterior reconstruction is not a binding
prediction task.

## Estradiol candidate-generation audit

Estradiol was wholly contained in the training partition under its canonical
SMILES. Its 14 recovered examples have an observed length range of 38–93 bases
and a median of 40 bases.

Using seed 42, temperature 0.9, top-k 5, and all 168 historical rows as the
authorized novelty reference:

- 40 sequences sampled for ranking
- 10 candidates passed and were retained
- 6 unique samples were rejected by sequence filters
- Retained lengths: 39–62 bases
- GC fraction: 0.359–0.590
- Maximum homopolymer: 2–4 bases
- Maximum normalized identity to a historical reference: 58.54%
- No retained candidate was an exact training-sequence match

The ranked candidate CSV, FASTA, checkpoint, run summary, training history,
split manifest, and generation metadata remain under the Git-ignored local
directory `artifacts/historical_cvae/`. The generation sidecar records the exact
checkpoint and reference-file hashes.

These are computational candidates selected for sequence novelty and basic
composition only. The ranking is not an affinity score.

## Interpretation

This run completes the intended train–evaluate–generate software loop and fixes
the major engineering failures of the historical prototypes. Scientifically,
the result should be treated as a reproducible baseline:

- The dataset remains too small for a strong deep-learning claim.
- Positive-only examples do not teach binder/non-binder discrimination.
- Target-disjoint evaluation is more honest than a random row split, but 16 test
  examples across 6 targets still yield high uncertainty.
- The 0.91% NLL improvement over a bigram model is modest.
- Candidate novelty and nucleotide composition do not imply binding.

The next evaluation should use repeated group-aware splits with uncertainty
intervals, sequence-family grouping, target-label permutation controls, and
simple non-neural baselines. Candidate advancement should add secondary-
structure analysis before any laboratory binding and counter-target assays.

## Reproduction

```bash
GRADUATE_PROJECT=/path/to/graduate_project

aptafind-generate train \
  --data "$GRADUATE_PROJECT/generativeModel/smallMolecule_aptamers_10172023.csv" \
  --legacy-target-features "$GRADUATE_PROJECT/generativeModel/targets_feature_vector.csv" \
  --config configs/thesis_cvae.yaml \
  --output-directory artifacts/historical_cvae
```

Run the documented `generate` command in
`docs/thesis_sequence_generation_pipeline.md` to reproduce target candidates.
