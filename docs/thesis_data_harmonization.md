# Thesis Data Harmonization

## Purpose

This stage converts immutable Bronze sources into provenance-preserving Silver
tables for the modernized thesis sequence-generation pipeline. It does not
rewrite source files, infer missing sequence mappings, or collapse different
assays into one artificial affinity label.

The implementation is configured by
`configs/thesis_data_sources.yaml` and run with:

```bash
aptafind-harmonize \
  --bronze-root data_lake/bronze \
  --config configs/thesis_data_sources.yaml \
  --output-directory data_lake/silver/thesis_endpoints
```

Silver outputs are generated locally and ignored by Git. Re-running against an
existing output requires the explicit `--overwrite` flag.

## Canonical tables

| Output | Meaning |
|---|---|
| `aptamers.csv` | Exact-sequence identities or explicitly provisional source-local aptamer identities |
| `targets.csv` | Canonical-SMILES, PubChem-CID, or explicitly provisional source-local target identities |
| `measurements.csv` | Source-level measurements with source file, sheet, row, publication, assay, evidence type, raw labels, and transformation notes |
| `model_interactions.csv` | DNA sequence–SMILES pairs carrying measured/published binary supervision; contradictory observations remain flagged |
| `generation_positive_pairs.csv` | Exact-deduplicated positive DNA sequence–SMILES pairs after contradictory pairs are excluded |
| `harmonization_report.json` | Input/output checksums, source audits, counts, coverage, conflicts, and known lineage-overlap warnings |

Common terminal `5'`/`3'` annotations and whitespace are removed from sequences.
Internal modification notation and ambiguous characters are never silently
deleted. Records without a valid plain DNA sequence retain their raw text and an
explicit status.

## First real run

The 2026-08-28 run harmonized:

| Source | Emitted measurements |
|---|---:|
| Frozen AptaBench DNA subset | 4,721 |
| UTexas unmodified ssDNA records | 890 |
| AptaDB DNA–molecule interactions | 293 |
| N2A2 five-target sequence screen | 270,115 |
| Xiao specificity panels and ITC pairs | 6,352 |
| **Total** | **282,371** |

The resulting entity layer contains 56,180 aptamer identities, including 55,837
with validated sequences, and 809 target identities, including 314 with
canonical SMILES. The 343 provisional aptamer identities are predominantly Xiao
source IDs whose sequence mappings have not yet been independently verified.

The binary model view currently contains 4,721 AptaBench DNA observations:
1,941 positive and 2,780 negative. Seven exact sequence–target pairs have
contradictory labels. Those observations remain in `model_interactions.csv` with
`pair_label_conflict=true` but are excluded from generator training data.

The initial generator-ready table therefore contains 1,835 exact-deduplicated
positive sequence–target pairs across 1,014 sequences and 288 canonical target
structures. Sequence lengths span 13–100 nt. This is a substantial expansion
over the recovered 165-pair historical table, but it remains heterogeneous
retrospective evidence rather than proof that a generator can design binders.

## Evidence safeguards

- AptaBench's `Specificity` lineage derives from the Xiao DOI. Both are retained
  for provenance, but they must not be treated as independent supervision.
- N2A2 z-score measurements are stored as continuous screen evidence. A
  `screen_positive` flag means only `z >= 2.576`; it does not apply the paper's
  separate specificity-ratio rule and is not a binary binding label.
- Xiao specificity zeros are preserved as author-processed measured
  nonresponses. The source readme states that negative cross-reactivity values
  were replaced with zero.
- UTexas and AptaDB rows without verified chemical structures remain useful for
  provenance and later target resolution but do not enter the current
  sequence–SMILES modeling view.
- Source-local IDs are not joined across datasets without explicit evidence.
- Affinity, cross-reactivity, z-score, and published-candidate evidence remain
  distinct measurement types.

## Next milestone

The next stage is deterministic target and publication reconciliation followed
by exact-sequence/source overlap analysis. After that audit, the expanded
positive-pair table can support a general small-molecule pretraining benchmark,
with a separately documented steroid subset for fine-tuning and evaluation.
