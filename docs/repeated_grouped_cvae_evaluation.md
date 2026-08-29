# Repeated Grouped CVAE Evaluation

## Purpose

The controlled v0.3.0 run used one target-disjoint fold and one training seed.
Its real-label point estimate beat the permuted-label control, but the
target-cluster interval crossed zero. This stage repeats the paired comparison
while also preventing sequence-family and publication overlap.

The evaluation remains a reconstruction and conditioning experiment. It does
not measure binding, affinity, specificity, or candidate efficacy.

## Why a conventional grouped split is not possible

The 1,835-pair generator table contains extensive dependency structure:

- 249 exact sequences are associated with multiple targets.
- A 90% normalized-Levenshtein single-linkage audit produces 792 sequence
  families from 1,014 exact sequences.
- Combining target, sequence-family, and publication links creates one
  connected component containing 1,279 rows and 132 targets.

That component makes a balanced five-fold split of the full table impossible.
Assigning its members to different folds would silently reintroduce the overlap
the grouped design is intended to prevent.

The strict repeated benchmark therefore excludes the AptaBench/Xiao
specificity lineage identified by `doi:10.1093/nar/gkaf219`. This removes 875
rows and 71 represented targets from this benchmark only. The source remains in
the immutable Bronze and derived Silver data and can be evaluated separately as
a lineage-specific dataset.

## Strict retained benchmark

The retained table contains:

- 960 sequence-target pairs
- 800 exact sequences
- 228 targets
- 647 sequence families at 90% identity
- 120 connected target/family/publication independence groups
- Largest independence group: 246 rows

Deterministic greedy assignment produces:

| Fold | Rows | Targets | Sequence families | Independence groups |
|---:|---:|---:|---:|---:|
| 0 | 246 | 39 | 142 | 1 |
| 1 | 178 | 48 | 125 | 23 |
| 2 | 179 | 40 | 113 | 32 |
| 3 | 178 | 55 | 133 | 31 |
| 4 | 179 | 46 | 134 | 33 |

Every target, 90%-identity sequence family, and nonempty publication identifier
occurs in exactly one fold. For each outer test fold, the next fold is used for
validation and the remaining three folds are used for training.

## Paired experiment

For every test fold and seed 42, 43, and 44, the runner trains:

1. The anti-collapse CVAE with correct training target conditions.
2. The same model with a zero-fixed-point derangement of the training targets.

The paired models share the source data, strict partitions, architecture,
initialization seed, batch order, optimizer, free bits, decoder-token dropout,
and evaluation examples. Fifteen fold-seed pairs produce 30 trained models.

The aggregate report includes:

- Per-run primary and control NLL
- Active latent units and wrong-target diagnostics
- Per-target out-of-fold reconstruction totals averaged across seeds
- A 5,000-replicate target-cluster bootstrap
- Hashes for every checkpoint and matched split manifest

## Frozen v0.4.0 result

All 30 models completed successfully. The token-weighted out-of-fold result was:

| Metric | Result |
|---|---:|
| Primary reconstruction NLL | 1.433576 |
| Permuted-label control reconstruction NLL | 1.430907 |
| Control minus primary NLL | -0.002669 |
| Relative NLL reduction versus control | -0.187% |
| Paired runs favoring correct labels | 5 of 15 |
| Target-cluster 95% bootstrap interval | [-0.006425, 0.000757] |
| Bootstrap fraction at or below zero | 0.9354 |
| Primary active latent units, mean | 11.33 of 16 |
| Control active latent units, mean | 11.47 of 16 |

Positive control-minus-primary values favor the correctly conditioned model;
negative values favor the permuted-label control. The mean paired difference by
test fold was +0.002903, -0.003457, +0.000365, -0.004436, and -0.009623 for
folds 0 through 4, respectively. The small mixed effects and interval spanning
zero do not support a reconstruction advantage from correct target labels.

The wrong-target diagnostics are consistent with the same conclusion. Mean
wrong-target NLL deltas were -0.001017 for the primary models and -0.000783 for
the controls, rather than a stable positive penalty for supplying an incorrect
target condition.

## Interpretation

The v0.3.0 one-split point estimate was not reproduced under stricter
independence and repeated training. The current model learns sequence structure
and uses multiple latent dimensions, but this experiment does not demonstrate
that its target descriptors provide generalizable target-specific information.
Target-labelled output from this model must therefore remain a computational
sequence sample, not a target-specific binding candidate.

This is a useful negative benchmark. The software pipeline now detects a weak
or absent conditioning signal instead of converting a favorable split into a
claim. A successor should be compared on these exact folds against both the
permuted-label control and simple non-neural baselines. With measured negatives
available in the broader data lake, the next modeling stage should separate
sequence generation from aptamer-target interaction ranking rather than asking
a positive-only reconstruction loss to supply evidence of binding.

## Reproduction and resumption

```bash
aptafind-generate repeated-evaluate \
  --data data_lake/silver/thesis_endpoints/generation_positive_pairs.csv \
  --config configs/repeated_controlled_cvae.yaml \
  --output-directory artifacts/repeated_controlled_cvae
```

Completed fold-seed model directories are detected and reused after verifying
the source hash and partition metadata. A single fold can be run or resumed
with `--fold-index 0`; repeat the option to select multiple folds. Partial
artifacts are never silently overwritten.

Generated grouping manifests, checkpoints, histories, comparisons, and the
aggregate summary remain under the ignored local `artifacts/` directory. The
[frozen v0.4.0 benchmark record](../benchmarks/thesis_cvae_repeated_v0.4.0.json)
stores their SHA-256 identities without publishing third-party sequences or
model weights.
