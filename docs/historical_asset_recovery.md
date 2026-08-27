# Aptafind Historical Asset Recovery Inventory

## Purpose

This document records the historical Aptafind datasets, generated features, model artifacts, code versions, and external-tool outputs recovered from the local filesystem. It is an inventory, not a claim that the historical workflow is currently reproducible.

The recovered datasets and generated artifacts remain in their original archive
locations. Curated historical source copies are now organized under
`historical/`; source files were not rewritten during recovery.

## Primary locations

| Location | Contents |
|---|---|
| `historical workspace: generativeModel` | Original feature-engineering and generative-model workspace, datasets, artifacts, and outputs |
| `this repository` | Public Git repository clone |
| `historical/refactor_cleanup_2026` | Unique source recovered from a later incomplete modernization attempt |
| `historical workspace: secondaryStructureTools` | Secondary-structure and MEME-related tooling/output |
| `historical workspace: aptani` | APTANI installation, inputs, and outputs |
| `local thesis and workflow archive` | Thesis and workflow documentation |

## Core historical data

| File | Description | Historical role |
|---|---|---|
| `generativeModel/aptamers.json` | 437 scraped aptamer records with target, category, affinity, conditions, sequence, and related metadata | Original raw Aptagen-derived dataset; read by the thesis-era notebook and `features.py` |
| `generativeModel/aptamersList_080823.csv` | 399-line August 2023 aptamer list | Intermediate expanded/cleaned dataset |
| `generativeModel/small_molecule.csv` | 135 lines | Early small-molecule subset |
| `generativeModel/small_molecule_080823.csv` | 137 lines | August 2023 small-molecule revision; consumed by `features_v2.py` |
| `generativeModel/small_molecule_081423.csv` | 404 lines | Expanded August 2023 dataset; consumed by `features_v2.py` |
| `generativeModel/small_molecule_dataset.csv` | 153 lines | Another early small-molecule dataset version |
| `generativeModel/smallMolecule_aptamers_10172023.csv` | 168 lines | Final October 2023 model dataset; consumed by `targetFeature.py` and `features_v3.py` |
| `generativeModel/steroid_aptamers.csv` | 48 lines | Steroid-focused subset used in exploratory work |

Line counts include headers where present and should not automatically be interpreted as validated record counts.

## Generated feature artifacts

| File | Producer/consumer relationship |
|---|---|
| `targets_feature_vector.csv` | Produced by `targetFeature.py` from `smallMolecule_aptamers_10172023.csv`; consumed by `features_v3.py` |
| `features.npz` | Produced by `features_v3.py`; consumed by `cvae.py` |
| `pca_model_info.pkl` | Produced by `features_v3.py`; loaded by the historical model workflow to preserve dimensionality-reduction information |
| `features.csv` | Earlier thesis-era feature table |
| `sequence_based_features.csv` | Sequence-feature intermediate generated during later feature work |
| `secondaryStructure.csv` | Secondary-structure feature/output table |
| `aptamers_encoded.csv` | Large encoded-sequence intermediate from the earlier pipeline |
| `sequence_features_10_31_2023` | Saved sequence-feature snapshot |
| `target_features_10_31_2023` | Saved target-feature snapshot |
| `full_features_10_31_2023` | Saved combined-feature snapshot |

## Contents of `features.npz`

The recovered archive contains 168 samples:

| Array | Shape | Intended information |
|---|---:|---|
| `sequences` | `(168, 28)` | Reduced sequence representation |
| `kd` | `(168, 1)` | Binding-affinity value |
| `target_type` | `(168, 28)` | Target representation |
| `structures` | `(168, 28)` | Secondary-structure representation |
| `kmers` | `(168, 33)` | Sequence k-mer features |
| `sequence_embedding` | `(168, 12)` | Learned or reduced sequence embedding |
| `binding_energy` | `(168, 2)` | Binding/thermodynamic energy features |
| `fingerprint` | `(168, 28)` | Molecular fingerprint representation |
| `molecule_properties` | `(168, 6)` | Small-molecule physicochemical properties |

## Motif and structural-analysis assets

- `generativeModel/meme_clusters` contains target-specific FASTA inputs and MEME output directories for 28 target groups.
- Each completed target directory contains a `meme.xml` result used by `sequenceMotif.py`.
- `secondaryStructureTools/meme_out/meme.xml` is an additional MEME result.
- `secondaryStructureTools` contains secondary-structure tooling and outputs.
- `aptani/APTANI2_v1.0` contains APTANI scripts, example/input FASTQ data, motif results, structure scores, and aptamer scores.
- NUPACK installations and historical NUPACK-generated structure assets exist under the graduate-project and Documents workspaces.

## Code lineage

| Version | Primary files | Interpretation |
|---|---|---|
| Thesis-era, early 2023 | `features.ipynb`, `features.py`, `vae.py` | Original PyTorch VAE and feature pipeline described by the thesis |
| Expansion, mid-2023 | `features_working.py`, `features_v2.py`, motif/structure modules | Dataset expansion and additional feature experiments |
| Later prototype, late 2023 | `features_v3.py`, `targetFeature.py`, `cvae.py` | 168-sample integrated feature matrix and TensorFlow conditional VAE experiment |
| Modernization attempt | `historical/refactor_cleanup_2026/feature_pipeline.py` | Later package-structure refactor; incomplete and retained only as historical WIP |

The repository previously mixed files from more than one phase at its root.
They are now separated into dated historical directories so the thesis-era VAE,
later CVAE, and modern implementation are not presented as one uninterrupted
experiment.

## Compatibility and reproducibility notes

- `pca_model_info.pkl` is a Joblib-serialized dictionary containing `pca_model` and `original_shape`.
- Its `SparsePCA` estimator was serialized with scikit-learn 1.2.2. Loading it under scikit-learn 1.9.0 produces an incompatible-version warning.
- Historical scripts use working-directory-relative filenames, so they expect data beside the scripts.
- The workflow depends on external scientific tools including MEME, NUPACK, and APTANI/ViennaRNA components.
- Several intermediate datasets exist with similar names. Their chronology must be preserved until transformations and provenance are fully reconstructed.
- Model-ready artifacts exist, but their presence alone does not prove that the pipeline can be rerun from raw input in a clean environment.

## Publication and licensing caution

Do not commit the recovered raw Aptagen-derived dataset to a public repository until its source terms and redistribution permissions have been checked. A safer public-repository pattern may be:

1. Document the source and expected schema.
2. Provide a small legally redistributable example dataset.
3. Add instructions for authorized users to place the complete dataset locally.
4. Exclude private or restricted raw data and generated artifacts through `.gitignore` where appropriate.

## Recovery progress

Completed:

1. Preserved the historical workspace as the immutable data-and-artifact snapshot.
2. Calculated checksums for the core datasets, artifacts, and source versions.
3. Mapped every array in `features.npz` to the historical feature pipeline.
4. Separated the thesis VAE, feature experiments, TensorFlow CVAE, later
   revision, and unfinished refactor in the public documentation.
5. Created and tested a clean PyTorch environment and modern
   train–evaluate–generate workflow.

Remaining research work includes deeper raw-data provenance reconstruction,
repeated group-aware evaluation, and secondary-structure candidate screening.

## Recovery status

All filenames explicitly referenced by the current public Aptafind code were located:

- `aptamers.json`
- `small_molecule_081423.csv`
- `smallMolecule_aptamers_10172023.csv`
- `targets_feature_vector.csv`
- `features.npz`
- `pca_model_info.pkl`
- `meme_clusters` and its target-specific MEME XML outputs

The principal historical files are not lost. Repository organization and the
first reproducible train–evaluate–generate loop are complete; data provenance
and stronger scientific validation remain open research work.
