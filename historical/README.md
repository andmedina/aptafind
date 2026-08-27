# Historical Aptafind Prototypes

This directory preserves the research path from the 2023 master's thesis to the
current tested PyTorch implementation. Historical source is retained for
provenance and interpretation; it is not imported by the active package.

| Directory | Period | Role |
|---|---|---|
| `thesis_vae_2023/` | Spring 2023 | Original PyTorch VAE notebook recovered from the thesis-era Git commit |
| `feature_experiments_2023/` | Summer–Fall 2023 | Later notebook, sequence helpers, and expanded feature experiments |
| `tensorflow_cvae_2023/` | December 2023 | Target-conditioned TensorFlow/Keras prototype and its utilities |
| `tensorflow_cvae_2024_revision/` | February 2024 | Separately preserved local CVAE revision |
| `refactor_cleanup_2026/` | March 2026 | Unique work-in-progress file recovered from an unfinished local refactor |

Each directory explains what was preserved and the limitations of that phase.
Raw datasets, generated features, model weights, and candidate outputs are not
published here because their provenance, redistribution terms, or scientific
interpretation require separate review.

The supported implementation lives in `src/aptafind/`. Its sequence-generation
workflow is documented in `docs/thesis_sequence_generation_pipeline.md`.

## Preservation policy

- Historical files are kept unchanged unless their README explicitly says
  otherwise.
- Identical copies that once appeared at the repository root were consolidated
  into these directories after SHA-256 comparison.
- Unique files were moved rather than discarded; the original paths remain
  recoverable from Git history.
- Historical code is evidence of the project's development, not a supported or
  scientifically validated release.
