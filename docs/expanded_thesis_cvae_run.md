# Expanded Small-Molecule CVAE Run

## Status

This report freezes the first run of the modern PyTorch sequence CVAE on the
conflict-screened output of the Bronze-to-Silver thesis data pipeline. The run
completed on August 28, 2026.

It demonstrates that the modern software can harmonize multiple real sources,
train on a substantially larger small-molecule dataset, evaluate unseen target
partitions, and generate auditable computational candidates. It does not
demonstrate target-specific binding or establish that generated sequences are
functional aptamers.

Frozen benchmark identity:

- Benchmark: `thesis-cvae-expanded-v0.2.0`
- Implementation commit: `47fa787405ea19e333c8bac3b5b1ea4cd07ecf7e`
- Annotated Git tag: `thesis-cvae-expanded-v0.2.0`
- Machine-readable record:
  `benchmarks/thesis_cvae_expanded_v0.2.0.json`

The benchmark record contains local artifact hashes but does not publish raw
third-party data, Silver tables, the model checkpoint, or generated sequences.

## Dataset identity

The model input is the conflict-screened positive export produced by
`aptafind-harmonize`:

- Harmonization configuration SHA-256:
  `068e1f030828f0736e7a4cf180f5965a54d18a3ee0d48c8ece060547cb588acc`
- Harmonization report SHA-256:
  `3bd10b6c2b75963d962cf8e3dc1aca879c1ffce7ad963318e78c51a3f81bf0ab`
- Generator input SHA-256:
  `02c59ae2a2f6f2b5ad5304fe039b3600624b5e12a088a1459fafa83889c22532`

Validated model input:

- 1,835 unique positive sequence-target pairs
- 1,014 unique DNA sequences
- 288 canonical target structures
- Sequence lengths from 13 to 100 nt
- 0 duplicate rows removed by the generator loader

The underlying harmonization report records 282,371 source measurements. Only
the exact-deduplicated, canonical-SMILES positive pairs enter this generator
benchmark. Measured negatives remain available for a future interaction model;
they are not used as negative examples by this positive-only generator.

## Evaluation design

The seed-42 split keeps canonical target SMILES disjoint:

| Partition | Rows | Targets |
|---|---:|---:|
| Training | 1,467 | 230 |
| Validation | 184 | 29 |
| Test | 184 | 29 |

RDKit descriptor scaling was fitted on training targets only. The test targets
were not used for model fitting, early stopping, or featurizer scaling.

Configuration: `configs/expanded_thesis_cvae.yaml`

The architecture matches the frozen historical baseline: a 32-dimensional DNA
embedding, 64-unit bidirectional GRU encoder, 16-dimensional latent space,
128-unit autoregressive GRU decoder, and 136 target-condition features. Batch
size was increased to 64 for the larger dataset. Training used CPU and ended at
the configured 40-epoch cap; epoch 40 had the best validation loss.

## Held-out results

| Model | Test NLL | Perplexity | Token accuracy |
|---|---:|---:|---:|
| Unigram baseline | 1.4430 | 4.2332 | 32.78% |
| Bigram baseline | 1.4466 | 4.2484 | 32.78% |
| Conditional sequence CVAE | **1.3774** | **3.9644** | **40.13%** |

Additional CVAE values:

- KL divergence: 0.00527
- Beta-weighted total loss: 1.37762
- Evaluated tokens excluding padding: 8,857
- Reconstruction-NLL improvement over the bigram baseline: 4.78%

The model reconstructs held-out tokens better than the two simple token
baselines on this split. The near-zero KL divergence is nevertheless a strong
posterior-collapse warning: the decoder may be relying mainly on its
autoregressive history instead of the latent variable. No target-label
permutation or condition-ablation result yet establishes meaningful target
conditioning.

These figures cannot be described as a direct improvement over the historical
v0.1.0 run because the two benchmarks use different source data, folds, and test
examples. The correct future comparison is to rerun competing architectures on
the same frozen folds.

## Estradiol candidate-generation audit

Using seed 20260828, temperature 0.9, top-k 5, and all 1,835 generator-input
rows as the novelty reference:

- 80 sequences sampled for ranking
- 20 candidates passed and were retained
- 10 unique samples were rejected by sequence filters
- All retained candidates were 76 nt
- GC fraction: 0.395-0.579
- Maximum homopolymer: 4 nt
- Maximum normalized identity to a reference: 55.26%
- No retained candidate was an exact training or reference-sequence match

The fixed 76-nt length is not a learned biological conclusion. Estradiol has
only one example in this harmonized export, and its length is 76 nt; the CLI's
default target-observed length bounds therefore set both the minimum and maximum
to 76. A future candidate study must use explicitly justified length bounds and
report a sensitivity analysis.

The ranked candidate CSV, FASTA, checkpoint, run summary, history, split
manifest, and generation metadata remain under the ignored local directory
`artifacts/expanded_small_molecule_cvae/`. These are computational candidates
selected for sequence novelty and basic composition only. Their ranking is not
an affinity score.

## Interpretation and next controls

This run is a successful expanded-data engineering benchmark, not a completed
scientific model. Before making a target-conditioned generation claim, the next
iteration should:

1. Add sequence-family, publication, assay, and source-aware grouping.
2. Run repeated folds with uncertainty intervals.
3. Add target-label permutation and condition-ablation controls.
4. Address posterior collapse through architecture and training ablations.
5. Compare steroid-only training with broad-small-molecule pretraining followed
   by steroid fine-tuning on identical steroid folds.
6. Add secondary-structure screening and prospective laboratory binding and
   counter-target assays before advancing candidates.

## Reproduction

```bash
aptafind-harmonize \
  --bronze-root data_lake/bronze \
  --config configs/thesis_data_sources.yaml \
  --output-directory data_lake/silver/thesis_endpoints

aptafind-generate train \
  --data data_lake/silver/thesis_endpoints/generation_positive_pairs.csv \
  --config configs/expanded_thesis_cvae.yaml \
  --output-directory artifacts/expanded_small_molecule_cvae

aptafind-generate generate \
  --checkpoint artifacts/expanded_small_molecule_cvae/sequence_cvae.pt \
  --target-name estradiol \
  --target-smiles 'C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1CC[C@@H]2O' \
  --reference-data data_lake/silver/thesis_endpoints/generation_positive_pairs.csv \
  --output artifacts/expanded_small_molecule_cvae/estradiol_candidates.csv \
  --fasta-output artifacts/expanded_small_molecule_cvae/estradiol_candidates.fasta \
  --count 20 \
  --seed 20260828
```
