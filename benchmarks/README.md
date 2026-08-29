# Aptafind Benchmarks

This directory contains compact, machine-readable records of frozen Aptafind
baselines. Large checkpoints, restricted source data, raw sequences, and
generated candidate files remain outside Git. Their SHA-256 hashes are recorded
when available so an authorized local archive can be verified.

## Frozen baselines

| Baseline | Purpose | Record | Git tag |
|---|---|---|---|
| Thesis CVAE v0.1.0 | First complete modern PyTorch train–evaluate–generate loop on the recovered late-2023 small-molecule dataset | `thesis_cvae_baseline_v0.1.0.json` | `thesis-cvae-baseline-v0.1.0` |
| Expanded thesis CVAE v0.2.0 | Same sequence-CVAE architecture on the first harmonized broad-small-molecule positive-pair export | `thesis_cvae_expanded_v0.2.0.json` | `thesis-cvae-expanded-v0.2.0` |
| Controlled thesis CVAE v0.3.0 | Anti-collapse training plus wrong-target diagnostics and a matched permuted-target-label control | `thesis_cvae_controlled_v0.3.0.json` | `thesis-cvae-controlled-v0.3.0` |
| Repeated thesis CVAE v0.4.0 | Five target/family/publication-independent folds, three seeds, and 15 paired target-label controls | `thesis_cvae_repeated_v0.4.0.json` | `thesis-cvae-repeated-v0.4.0` |

The thesis CVAE benchmarks measure held-out sequence reconstruction and
candidate-generation mechanics. None is a binding-prediction benchmark. The
v0.1.0 and v0.2.0 metrics use different datasets and therefore are not a direct
model comparison. The v0.2.0 and v0.3.0 runs use the same split; v0.3.0 adds
latent-use and target-label controls but represents only one fold and one
training seed. The v0.4.0 benchmark uses a stricter retained subset, so its NLL
is not directly comparable with v0.2.0 or v0.3.0. Its paired within-fold result
is the relevant finding: correct target labels did not beat matched permuted
labels across repeated leakage-resistant evaluation.

The v0.4.0 negative result is a completed benchmark, not a failed software run.
It prevents the earlier single-split point estimate from being mistaken for
generalizable target conditioning and establishes the threshold that any new
target-aware architecture must exceed on identical folds.

## Comparison policy

An apparent metric improvement is valid only when the competing models use the
same frozen source identities, preprocessing boundary, partition manifest,
evaluation code, and metric definition.

For the planned broad-small-molecule pretraining followed by steroid fine-tuning:

1. Keep this record as the historical software baseline.
2. Freeze the expanded steroid endpoint dataset before model development.
3. Rerun this baseline architecture and the new transfer model on the same new
   folds.
4. Compare steroid-only, broad-transfer, matched-pretraining-control, and simple
   non-neural baselines.
5. Report reconstruction, interaction ranking, enrichment, and candidate
   generation as separate tasks; do not treat one metric as evidence for
   another.

Results from different test datasets may be shown as separate experiments but
must not be described as direct performance improvements.
