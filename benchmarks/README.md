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

The thesis CVAE benchmarks measure held-out sequence reconstruction and
candidate-generation mechanics. None is a binding-prediction benchmark. The
v0.1.0 and v0.2.0 metrics use different datasets and therefore are not a direct
model comparison. The v0.2.0 and v0.3.0 runs use the same split; v0.3.0 adds
latent-use and target-label controls but still represents only one fold and one
training seed.

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
