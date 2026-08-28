# Modern Thesis Sequence-Generation Pipeline

## Purpose

This workflow completes the software objective of the original Aptafind thesis:
train on known ssDNA aptamer–target examples and generate new DNA sequences for
a selected small-molecule target. It is designed to be reproducible, testable,
and explicit about what the available data cannot establish.

The output is a set of **computational candidates**, not predicted binders. The
historical data are small, predominantly positive-only, and heterogeneous.
Without well-matched negative examples or prospective experiments, the model
cannot demonstrate target-specific binding, affinity, or selectivity.

## What changed

| Historical implementation | Modern implementation |
|---|---|
| Reconstructed compressed heterogeneous feature vectors | Generates `A/C/G/T` tokens directly |
| TensorFlow and PyTorch experiments with different interfaces | One PyTorch package and CLI |
| Preprocessing fitted before splitting | Molecule descriptor scaling fitted on training targets only |
| Random row split | Target-disjoint split by default |
| Working-directory-relative inputs and outputs | Explicit CLI paths and versioned run artifacts |
| No recovered test evaluation | Validation early stopping plus held-out test metrics |
| Ad hoc generation and decoding | Autoregressive sampling with explicit sequence filters |
| No automated tests | Unit and end-to-end smoke tests |

The historical code remains preserved unchanged. The modern package does not
silently rewrite the historical experiment or claim that its earlier numerical
results have been reproduced.

## Architecture

```text
known ssDNA tokens ──> embedding ──> bidirectional GRU ──┐
                                                        ├─> posterior μ/log σ²
target SMILES ──> RDKit descriptors + Morgan FP ─> MLP ─┘          │
                                                                  v
                                                         sampled latent z
                                                                  │
target condition ──────────────────────────────────────────────────┤
                                                                  v
                                           autoregressive GRU DNA decoder
                                                                  │
                                                           A/C/G/T/EOS
```

The encoder is used during training to learn a conditional latent distribution.
At generation time, the decoder samples from a standard normal prior and receives
the selected target condition at every sequence step.

The condition vector contains:

- Molecular weight
- LogP
- Hydrogen-bond donor and acceptor counts
- Topological polar surface area
- Rotatable-bond count
- Ring count
- Fraction Csp3
- A configurable radius-2 Morgan fingerprint

Descriptor scaling is learned from unique training molecules. RDKit calculates
all features from canonical SMILES, replacing the fragile historical serialized
fingerprint columns.

## Data contract

The preferred CSV schema is one row per positive aptamer–target observation:

| Column | Required | Meaning |
|---|---:|---|
| `sequence` | Yes | ssDNA containing only `A`, `C`, `G`, and `T` |
| `target_smiles` | Yes | Parseable molecular SMILES |
| `target_name` | No | Human-readable label; `target` is also recognized |

Whitespace is removed from sequences, bases are uppercased, SMILES are
canonicalized, and exact sequence–target duplicates are removed. Invalid rows
cause the run to fail with row-level messages; they are never silently dropped.

The optional historical compatibility mode reads sequences and metadata from
`smallMolecule_aptamers_10172023.csv` and only the `Smiles` column from the
row-aligned `targets_feature_vector.csv`. Equal row counts are required and use
of this legacy alignment is recorded in the run summary.

The locally recovered files currently audit as:

- 168 source rows
- 165 unique sequence–target pairs after exact deduplication
- 154 unique sequences
- 64 canonical target molecules
- Sequence lengths from 9 to 109 nucleotides

These counts describe the recovered local files, not a redistributable dataset.
The source files remain outside Git pending provenance and licensing review.

An expanded, reproducible data path is now available through
`aptafind-harmonize`. The first conflict-screened Silver export contains 1,835
exact-deduplicated positive DNA sequence–target pairs, 1,014 unique sequences,
and 288 canonical target structures, with sequence lengths from 13 to 100 nt.
This general small-molecule table is intended for controlled pretraining and
benchmarking; steroid-specific fine-tuning and leakage-safe evaluation remain
separate stages. See [Thesis Data Harmonization](thesis_data_harmonization.md)
for the canonical schema, source counts, overlap warnings, and reproduction
command.

## Installation

```bash
conda env create -f environment.yml
conda activate aptafind
python -m pip install --no-deps -e .
python -m pytest -q
```

If the environment already exists, install any newly declared dependency with
Conda and rerun the editable install.

## Synthetic end-to-end demonstration

The checked-in `examples/synthetic_aptamers.csv` contains artificial sequences
with artificial target-associated motifs. It exists only to verify that the
software can train, checkpoint, evaluate, and generate from a clean checkout.

```bash
aptafind-generate inspect-data \
  --data examples/synthetic_aptamers.csv \
  --config configs/thesis_cvae.yaml

aptafind-generate train \
  --data examples/synthetic_aptamers.csv \
  --config configs/thesis_cvae.yaml \
  --output-directory artifacts/synthetic_demo
```

The demo dataset can also be regenerated deterministically:

```bash
aptafind-generate make-demo-data \
  --output examples/synthetic_aptamers.csv \
  --samples-per-target 12 \
  --sequence-length 40 \
  --seed 7
```

## Historical Aptafind training run

The historical source files remain in an authorized local research archive.
Set the variable to that archive's path before running the commands:

```bash
GRADUATE_PROJECT=/path/to/graduate_project

aptafind-generate inspect-data \
  --data "$GRADUATE_PROJECT/generativeModel/smallMolecule_aptamers_10172023.csv" \
  --legacy-target-features "$GRADUATE_PROJECT/generativeModel/targets_feature_vector.csv" \
  --config configs/thesis_cvae.yaml

aptafind-generate train \
  --data "$GRADUATE_PROJECT/generativeModel/smallMolecule_aptamers_10172023.csv" \
  --legacy-target-features "$GRADUATE_PROJECT/generativeModel/targets_feature_vector.csv" \
  --config configs/thesis_cvae.yaml \
  --output-directory artifacts/historical_cvae
```

The default split assigns each canonical molecule to exactly one partition. The
current historical audit produces 133 training rows, 16 validation rows, and 16
test rows. A random split is available in configuration only for explicitly
labeled exploratory comparisons.

## Training outputs

Each output directory contains:

| Artifact | Contents |
|---|---|
| `sequence_cvae.pt` | Model weights, architecture, tokenizer, fitted molecule featurizer, metadata, and hashes of training sequences |
| `run_summary.json` | Dataset audit, source hashes, software versions, split counts, configuration, baselines, best epoch, and test metrics |
| `training_history.csv` | Per-epoch training and validation loss, reconstruction loss, KL divergence, token accuracy, and perplexity |
| `split_manifest.csv` | Sequence hashes, target identities, and partition assignments without raw DNA sequences |

Raw training sequences are not embedded in the checkpoint. Their SHA-256 hashes
support exact novelty screening without silently redistributing the dataset.

## Candidate generation

```bash
aptafind-generate generate \
  --checkpoint artifacts/historical_cvae/sequence_cvae.pt \
  --target-name estradiol \
  --target-smiles 'C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@H]2O)CCC4=C3C=CC(=C4)O' \
  --reference-data /path/to/authorized_reference_sequences.csv \
  --output artifacts/historical_cvae/estradiol_candidates.csv \
  --fasta-output artifacts/historical_cvae/estradiol_candidates.fasta
```

The generator rejects sequences that violate configured length, GC-content, or
homopolymer constraints. It also rejects exact training-set matches and, when a
reference CSV is supplied, sequences above the configured normalized
Levenshtein-similarity threshold.

Candidates are ranked in the CSV, and generation automatically writes a
`*.metadata.json` sidecar containing the checkpoint and reference-file hashes,
target, random seed, sampling parameters, filters, and acceptance counts.
Ranking favors proximity to the target's training median length when that target
was observed, then lower reference similarity, balanced GC composition, and
higher mononucleotide entropy. These remain heuristic sequence properties, not
affinity scores.

When the requested molecule was represented in the training partition, default
sampling uses that target's observed sequence-length range, while retaining the
configured global minimum as a floor. For an unseen molecule, generation falls
back to the configured minimum and the complete dataset's observed maximum.
`--minimum-length` and `--maximum-length` provide explicit overrides for a
documented experimental design.

These filters measure sequence composition and novelty only. They do not score
binding. Temperature, top-k sampling, and every filter can be changed explicitly
through the CLI; their defaults are saved with the training run.

## Interpretation and evaluation

Reported test metrics are token reconstruction measurements:

- Cross-entropy reconstruction loss
- Token accuracy excluding padding
- Perplexity
- KL divergence
- Beta-weighted total loss

The run summary also evaluates smoothed training-set unigram and bigram token
baselines on the identical test partition and reports the CVAE reconstruction
improvement over the bigram negative log-likelihood. This prevents a superficially
non-random accuracy from being mistaken for meaningful learned structure.

They answer whether the model reconstructs held-out aptamer-like sequences under
unseen target conditions. They do **not** answer whether generated sequences bind
the target.

A defensible candidate funnel should subsequently include:

1. Exact and near-duplicate screening against all authorized aptamer sources
2. Secondary-structure prediction and structural-diversity analysis
3. Stability and synthesis-oriented sequence checks
4. Target-specific computational analyses whose assumptions are documented
5. Prospective laboratory binding, specificity, and counter-target assays

Because only laboratory measurements can establish binding, candidate tables
must retain the label "computational candidate" through every pre-assay stage.

## Known limitations and next work

- The historical dataset is far too small for strong deep-learning conclusions.
- Most targets have few examples, and many observations may share publication,
  scaffold, or assay provenance.
- Positive-only training does not teach the model to distinguish binders from
  non-binders.
- Target-disjoint splitting prevents direct target leakage but does not yet group
  related sequence families or publications.
- Sequence filters do not include secondary-structure folding in this first
  runnable version.
- Hyperparameters have not yet been selected through nested or repeated
  cross-validation.

The next scientific step is a retrospective historical run with repeated,
group-aware evaluation and comparisons against simple sequence baselines. The
next engineering step is a versioned candidate-screening stage that integrates
structure prediction without making unsupported affinity claims.
