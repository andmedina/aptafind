# Aptafind `features_v3.py` Pipeline Map

## Scope

This document explains the recovered late-2023 feature-engineering prototype as it currently exists. It does not change the historical code or claim that the workflow is reproducible in a clean environment.

## Pipeline overview

```text
smallMolecule_aptamers_10172023.csv
                 |
                 +--> sequence cleaning and encoding
                 |      - nucleotide one-hot encoding
                 |      - 1-mer, 2-mer, and 3-mer frequencies
                 |      - Word2Vec sequence embeddings
                 |
                 +--> NUPACK-derived structureMotif processing
                 |      - dot-bracket structure
                 |      - minimum free energy (MFE)
                 |      - structure matrix
                 |      - stacking energy
                 |
                 +--> affinity and target-type processing
                        - standardized Kd
                        - one-hot target type

targets_feature_vector.csv
                 |
                 +--> molecular target processing
                        - molecular fingerprints
                        - Morgan fingerprints
                        - physicochemical properties
                        - missing-value imputation
                        - standardization

All feature groups
        |
        +--> PCA or SparsePCA dimensionality reduction
        |
        +--> features.npz
                 |
                 +--> cvae.py

SparsePCA fitted to sequence one-hot vectors
        |
        +--> pca_model_info.pkl
```

## Inputs

### Aptamer dataset

`smallMolecule_aptamers_10172023.csv` is the primary 168-line late-2023 dataset. The script expects fields including:

- `sequence`
- `target`
- `type`
- `kd`
- `cid`
- `cas`
- `reference`
- `length`

The script removes records with empty or missing sequences and strips leading/trailing whitespace from sequences and target names.

### Target feature dataset

`targets_feature_vector.csv` is generated separately by `targetFeature.py`. It contains PubChem-derived molecular descriptors and fingerprints aligned to the aptamer dataset.

## Sequence feature groups

### Nucleotide one-hot vectors

Each nucleotide is represented using four binary values:

- A: `[1, 0, 0, 0]`
- C: `[0, 1, 0, 0]`
- G: `[0, 0, 1, 0]`
- T: `[0, 0, 0, 1]`

The per-nucleotide vectors are flattened and zero-padded to the longest encoded sequence.

### K-mer frequencies

The script calculates normalized frequency vectors for:

- 1-mers: individual nucleotides
- 2-mers: all two-nucleotide combinations
- 3-mers: all three-nucleotide combinations

The three arrays are concatenated into one k-mer feature group.

### Word2Vec sequence embeddings

A Gensim Word2Vec model is trained directly on the available sequences, treating individual nucleotides as tokens. Each sequence's embeddings are flattened and padded to a common length.

This is an experimental nucleotide embedding—not a pretrained biological sequence model.

### GC content

GC percentage is calculated, but the recovered final output assembly does not include it in `features.npz`. It is therefore computed but apparently unused downstream.

## Structural feature groups

`structureMotif.compute_mfe_structures()` supplies NUPACK-derived values for each sequence:

- Dot-bracket secondary structure
- Gibbs/minimum free energy (`mfe`)
- NUPACK structure matrix
- Stacking energy

MFE and stacking energy are standardized. Dot-bracket symbols are one-hot encoded:

- `(`: opening paired base
- `)`: closing paired base
- `.`: unpaired base

The flattened dot-bracket encoding and flattened NUPACK matrix are padded and concatenated into the structural feature group.

Historical MEME motif code is present but commented out. Therefore MEME motifs are not part of the recovered `features.npz` output produced by this version.

## Target and affinity features

### Binding affinity

`kd` is standardized with `StandardScaler` and saved as a single-column feature group. Lower raw Kd generally represents stronger affinity, but the archive stores the standardized value rather than the original measurement.

### Target type

The categorical `type` field is one-hot encoded using categories discovered from the input dataset. The resulting matrix is later reduced with SparsePCA.

## Small-molecule target features

The script reads `targets_feature_vector.csv` and processes two broad groups.

### Fingerprints

- A hexadecimal PubChem fingerprint is converted to a padded binary vector.
- A Morgan fingerprint string is converted to a binary vector.
- Both are concatenated and reduced with SparsePCA.

### Physicochemical properties

Properties include molecular weight, logP, hydrogen-bond donor and acceptor counts, rotatable-bond count, polar surface area, formal charge/count fields, isotope count, stereocenter counts, and covalently bonded unit count.

Missing `xLogP3-AA` values are mean-imputed. Selected numeric properties are standardized, concatenated, and reduced with conventional PCA while retaining at least 95% cumulative explained variance.

## Dimensionality reduction

Two approaches are used:

| Feature group | Reduction method | Selection rule |
|---|---|---|
| Molecular properties | PCA | Minimum components exceeding 95% cumulative explained variance |
| Sequence embeddings | PCA | Minimum components exceeding 95% cumulative explained variance |
| K-mers | PCA | Minimum components exceeding 95% cumulative explained variance |
| Sequence one-hot vectors | SparsePCA | `number of samples // 6` components |
| Target type | SparsePCA | `number of samples // 6` components |
| Structural features | SparsePCA | `number of samples // 6` components |
| Fingerprints | SparsePCA | `number of samples // 6` components |

With 168 recovered samples, the SparsePCA rule results in 28 components.

Only the sequence SparsePCA model is saved in `pca_model_info.pkl`, along with the original sequence-array shape.

## Final archive

`features_v3.py` writes `features.npz`, which contains:

| Key | Recovered shape |
|---|---:|
| `sequences` | `(168, 28)` |
| `kd` | `(168, 1)` |
| `target_type` | `(168, 28)` |
| `structures` | `(168, 28)` |
| `kmers` | `(168, 33)` |
| `sequence_embedding` | `(168, 12)` |
| `binding_energy` | `(168, 2)` |
| `fingerprint` | `(168, 28)` |
| `molecule_properties` | `(168, 6)` |

`cvae.py` loads this archive to construct the later conditional-VAE experiment.

## Important limitations found during tracing

These findings should guide the modernization work; they are not retroactive criticism of the exploratory research prototype.

1. **The module executes immediately.** Data loading, external-tool processing, model fitting, and file writing occur at import time instead of through a controlled entry point.
2. **Paths depend on the current working directory.** Input and output filenames are not resolved relative to a project configuration or package root.
3. **Feature alignment is assumed.** The aptamer and target feature tables are combined by row order without an explicit join key or alignment assertion.
4. **Preprocessing is fitted before a train/test split.** Imputation, scaling, Word2Vec, PCA, and SparsePCA use the complete dataset. For predictive evaluation, this introduces data leakage.
5. **Most fitted transformers are not saved.** Only the sequence SparsePCA model is persisted, preventing consistent transformation of future observations.
6. **Word2Vec reproducibility is incomplete.** No explicit random seed or deterministic worker configuration is provided.
7. **A fingerprint reconstruction check appears incorrect.** The code inverse-transforms `aptamer_structures_reduced_data` through the fingerprint SparsePCA model instead of `fingerprints_reduced_data`. This affects the printed diagnostic and may cause a shape error; it does not define the saved fingerprint array itself.
8. **GC content is unused.** It is computed but omitted from the final saved archive.
9. **MEME features are inactive.** Motif-generation and extraction code is commented out in this version.
10. **Dependency compatibility matters.** The recovered SparsePCA artifact was created with scikit-learn 1.2.2 and produces a warning under newer scikit-learn versions.
11. **Model evaluation requires caution.** With only 168 samples and many derived dimensions, overfitting risk is substantial and biological validation is outside the scope of computational reconstruction alone.

## Modernization boundary

The historical script should remain unchanged as evidence of the original research process. A new implementation should reproduce its behavior through small, testable stages:

1. Load and validate source data.
2. Clean sequences and define stable record identifiers.
3. Join aptamer and target data explicitly.
4. Generate each feature family independently.
5. Split data using a biologically defensible grouping strategy.
6. Fit preprocessing only on training data.
7. Persist every fitted transformer and its metadata.
8. Assemble a versioned feature matrix.
9. Train and evaluate the model with reproducible seeds.
10. Compare regenerated artifacts with the recovered historical snapshot.
