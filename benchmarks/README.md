# Aptafind Benchmarks

This directory contains compact, machine-readable records of frozen Aptafind
baselines. Large checkpoints, restricted source data, raw sequences, and
generated candidate files remain outside Git. Their SHA-256 hashes are recorded
when available so an authorized local archive can be verified.

## Frozen baselines

| Baseline | Purpose | Record | Git tag |
|---|---|---|---|
| Thesis CVAE v0.1.0 | First complete modern PyTorch train–evaluate–generate loop on the recovered late-2023 small-molecule dataset | `thesis_cvae_baseline_v0.1.0.json` | `thesis-cvae-baseline-v0.1.0` |

The thesis CVAE baseline measures held-out sequence reconstruction and candidate
generation mechanics. It is not a binding-prediction benchmark.

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
