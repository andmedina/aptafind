# Controlled Anti-Collapse CVAE Run

## Status

This report freezes the first Aptafind sequence-CVAE experiment with explicit
latent-use measurements, wrong-target diagnostics, and a matched target-label
permutation control. The run completed on August 28, 2026.

Frozen benchmark identity:

- Benchmark: `thesis-cvae-controlled-v0.3.0`
- Implementation commit: `2a029b2abb5f3fdb4287eb625a94953d619f3524`
- Annotated Git tag: `thesis-cvae-controlled-v0.3.0`
- Machine-readable record:
  `benchmarks/thesis_cvae_controlled_v0.3.0.json`

The result resolves the v0.2.0 model's posterior collapse on this run while
retaining comparable held-out reconstruction. The target-label result is
promising but not conclusive under a target-cluster bootstrap. None of these
measurements establish binding, affinity, specificity, or efficacy.

## Matched design

The primary and permuted-label models use the same:

- 1,835-pair harmonized generator input
- Target-disjoint 1,467/184/184 train/validation/test split
- 230/29/29 target allocation
- Split-manifest SHA-256:
  `dcc39673577a74f10e529fc7bccee7b753d019c2ef99e5758bdeb48f7e3a8b49`
- Architecture, initialization seed, batch order, optimizer, and evaluation
- Free-bits allowance of 0.05 per latent dimension
- Teacher-forced decoder-token dropout of 0.30

The only designed difference is that the negative control maps every training
target to one different training target. The seed-42 derangement covers all 230
training targets with zero fixed points. Validation and test labels remain
correct.

## Posterior-collapse result

| Run | Raw test KL | Active units | Posterior-mean variance | Test NLL |
|---|---:|---:|---:|---:|
| Frozen v0.2.0 | 0.0053 | 0/16 | 0.00037 mean | 1.3774 |
| Anti-collapse v0.3.0 | 0.6344 | 16/16 | 0.06899 mean | 1.3734 |
| Permuted-label control | 0.8940 | 16/16 | 0.09667 mean | 1.4933 |

The anti-collapse model activates all 16 latent dimensions. Its test NLL is
0.0040 lower than v0.2.0, but the paired target-cluster 95% interval for that
difference is -0.0903 to 0.1191. The correct conclusion is that latent use was
restored without a demonstrated reconstruction penalty or improvement—not that
v0.3.0 significantly outperforms v0.2.0.

The anti-collapse model stopped at epoch 24 and restored epoch 14. The
permuted-label control stopped at epoch 22 and restored epoch 12. Neither run
reached the 60-epoch ceiling.

## Target-condition controls

Ten zero-fixed-point derangements of the 29 test-target conditions were applied
at the decoder while holding each sequence posterior fixed:

| Checkpoint | Wrong-target NLL delta, mean | Standard deviation |
|---|---:|---:|
| Frozen v0.2.0 | +0.2147 | 0.0806 |
| Anti-collapse v0.3.0 | +0.0454 | 0.0272 |
| Permuted-label control | -0.0516 | 0.0302 |

Wrong target conditions worsen the real-label models. For the control trained
on false pairings, wrong test conditions slightly improve reconstruction, and
zeroing its decoder condition improves NLL by 0.1215. This directionality is
consistent with the real pairing carrying useful reconstruction information.

On the exact same 184 test sequences, the real-label model has NLL 1.3734 and
the permutation-trained control has NLL 1.4933, an 8.03% relative reduction.
The control is worse for 21 of 29 test targets. However, the 5,000-replicate
target-cluster bootstrap interval for control minus primary is -0.0302 to
0.2881, with 6.04% of replicates at or below zero. The interval crosses zero,
so one fold and one training seed provide suggestive rather than conclusive
evidence.

## Token baselines

| Model | Test NLL | Perplexity | Token accuracy |
|---|---:|---:|---:|
| Unigram baseline | 1.4430 | 4.2332 | 32.78% |
| Bigram baseline | 1.4466 | 4.2484 | 32.78% |
| Anti-collapse CVAE | **1.3734** | **3.9486** | **37.15%** |

The primary CVAE improves reconstruction NLL over the bigram baseline by 5.06%
on this fold. Token accuracy is lower than v0.2.0's 40.13%, reflecting a metric
tradeoff after teacher-forced token dropout. Reconstruction remains a sequence
modeling task, not a binding-prediction task.

## Estradiol generation audit

Generation used explicit 40-90 nt bounds rather than the earlier default that
forced all candidates to the single observed estradiol example's 76-nt length:

- 80 unique samples considered
- 20 candidates retained and 9 unique samples rejected by filters
- Retained lengths: 64-87 nt; median 71 nt; 13 distinct lengths
- GC fraction: 0.359-0.569
- Maximum homopolymer: 4 nt
- Maximum normalized identity to a reference: 58.82%
- No retained candidate was an exact training or reference match

These are computational candidates ranked by sequence properties and novelty.
Their ranking is not an affinity score.

## Interpretation and next experiment

This experiment supplies evidence for two narrower statements:

1. Free bits plus decoder-token dropout restored use of the 16-dimensional
   posterior without materially changing reconstruction NLL on this fold.
2. Correct target labels outperform a fully permuted training-label control in
   the point estimate, but uncertainty across only 29 test targets is too wide
   for a publication-level conditioning claim.

The next model experiment should repeat this exact paired comparison across
multiple training seeds and target-group folds, then add sequence-family and
publication grouping. Only after that should the same controlled framework be
used to compare steroid-only training with broad-small-molecule pretraining and
steroid fine-tuning.
