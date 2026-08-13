# Aptafind Prototype Publication Manifest

## Purpose

This manifest proposes how to present Aptafind's research history without copying every temporary file into the public repository. It is a review document only; no historical files have been copied or reorganized yet.

## Recommended research timeline

### Prototype 1: Thesis VAE — Spring 2023

**Purpose:** Initial feature-engineering pipeline and PyTorch variational autoencoder developed for the master's thesis.

**Recommended public directory:**

```text
historical/thesis_vae_2023/
├── README.md
└── features.ipynb
```

**Proposed source:**

| Published file | Historical source | Reason |
|---|---|---|
| `features.ipynb` | Git commit `08f8b1d` from 2023-04-19 | Last thesis-era commit explicitly stating that the VAE was trained to produce sequences |
| `README.md` | New explanatory document | Records thesis citation, objective, dataset size, methods, results, and limitations |

Use the notebook from the April commit—not the current working-tree notebook, which was replaced or updated in November 2023.

**Document but do not automatically publish:**

- `aptamers.json`: original 437-record scraped source dataset
- `features.csv`: thesis-era generated feature table
- Final thesis document

The thesis can be linked through its DOI rather than duplicated if that is the preferred publication route. Dataset redistribution terms should be reviewed before raw data is committed.

### Prototype 2: Expanded feature experiments — Summer/Fall 2023

**Purpose:** Show the transition from the thesis dataset toward small-molecule conditioning, structural features, target descriptors, and sequence-generation experiments.

This phase is scientifically useful, but publishing every intermediate script would create clutter. Preserve it primarily through Git history and summarize it in a timeline document.

**Recommended public directory:**

```text
historical/feature_experiments_2023/
└── README.md
```

The README should discuss:

- Small-molecule dataset expansion
- NUPACK secondary-structure integration
- MEME target-cluster motif experiments
- Word2Vec sequence embeddings
- PubChem target descriptors and fingerprints
- Progression from `features_v2.py` to `features_v3.py`

**Optional representative files:**

| File | Include? | Reason |
|---|---:|---|
| `features_v2.py` | Optional | Representative intermediate feature revision already preserved in Git |
| `sequenceMotif.py` | Prefer shared utility | Documents MEME integration |
| `structureMotif.py` | Prefer shared utility | Documents NUPACK integration |
| `targetFeature.py` | Prefer CVAE prototype | Direct producer of target features used by the later model |
| `generateSequence.py` | Optional | Experimental generation helper; not part of the verified final execution path |

### Prototype 3: Conditional VAE — Late 2023/early 2024

**Purpose:** Integrate ssDNA features and small-molecule descriptors in a TensorFlow/Keras conditional VAE.

**Recommended public directory:**

```text
historical/tensorflow_cvae_2023/
├── README.md
├── cvae.py
├── features_v3.py
├── targetFeature.py
└── utilities/
    ├── structureMotif.py
    ├── sequenceMotif.py
    └── to_fasta_file.py
```

**Proposed sources:**

| Published file | Preferred source | Note |
|---|---|---|
| `cvae.py` | Public Git commit `c8fbc68` initially; compare with recovered February 2024 local version | The two versions differ and should not be silently combined |
| `features_v3.py` | Public Git commit `c8fbc68` or recovered December file after checksum comparison | Produces `features.npz` |
| `targetFeature.py` | Public tracked version | Produces `targets_feature_vector.csv` |
| `structureMotif.py` | Public tracked version | NUPACK interface |
| `sequenceMotif.py` | Public tracked version | MEME interface; feature use is commented out in `features_v3.py` |
| `to_fasta_file.py` | Public tracked version | FASTA conversion utility |
| `README.md` | New explanatory document | Documents architecture, data flow, execution status, results, and limitations |

The recovered February 2024 `cvae.py` should first be diffed against the December Git version. If it contains meaningful later work, publish it as a separately identified revision rather than overwriting the committed prototype.

### Current work: Reproducibility rebuild

**Purpose:** Build a tested, leakage-resistant, reproducible pipeline while preserving all historical prototypes.

**Recommended public directory:**

```text
src/aptafind/
```

This version should be called the **reproducibility rebuild** or **modern implementation**, rather than implying that it has already produced better scientific results.

## Shared documentation to retain

```text
docs/
├── historical_asset_recovery.md
├── historical_checksums.sha256
├── feature_pipeline_map.md
├── cvae_model_map.md
├── aptafind_v2_design.md
└── research_timeline.md          # create during reorganization
```

The recovery inventory contains local absolute paths and may be better retained on a private branch or rewritten with portable path labels before public release.

## Files and directories to exclude

### Never copy as project source

- `.DS_Store`
- `__pycache__/`
- `*.pyc`
- Excel lock files beginning with `~$`
- Build products and compiled dependencies
- Entire NUPACK, ViennaRNA, MEME, APTANI, or Taskflow installations
- Local environments and IDE settings
- Temporary notebooks and unnamed scratch outputs

### Keep out of Git until reviewed

- `aptamers.json`
- Historical CSV/XLSX datasets derived from external sources
- `features.npz`
- `pca_model_info.pkl`
- Large MEME/NUPACK/APTANI result directories
- Generated FASTA files
- Model binaries and checkpoints

Reasons include uncertain redistribution rights, artifact size, provenance, reproducibility, and the risk of publishing outputs without sufficient context.

### Summarize rather than duplicate

- `features_working.py`
- Repeated dated CSV/XLSX datasets
- `temp.ipynb`
- `workWithCompleteList.py`
- Exploratory sample files such as `aldosterone_sample.csv` and `core.txt`
- Repeated `model_results` figures unless directly discussed in a prototype README

These remain preserved in the historical workspace and checksummed core snapshot where applicable.

## README expectations for every prototype

Each prototype README should include:

1. Date and research phase
2. Scientific objective
3. Dataset used and approximate sample count
4. Input and output files
5. Feature families
6. Model architecture
7. What was successfully demonstrated
8. Evaluation results, if recoverable
9. Known technical and scientific limitations
10. How the prototype motivated the next phase
11. Environment and external-tool requirements
12. Explicit statement that computational candidates are not experimentally validated aptamers

## Git-history policy

The repository's existing Git history already preserves the evolution of several files. Reorganization should use normal commits and should not rewrite or squash the historical commits.

Recommended tags after the structure is reviewed:

- `thesis-vae-2023` pointing to the appropriate April 2023 thesis-era commit
- `tensorflow-cvae-prototype-2023` pointing to the December 2023 CVAE commit
- A future release tag for the first reproducible rebuild

A tag is a stable human-readable label attached to a historical commit; it does not copy or modify that commit.

## Proposed copy sequence

No copying should begin until this manifest is approved.

1. Export the April 2023 notebook from Git into `historical/thesis_vae_2023/`.
2. Add its explanatory README.
3. Create the feature-experiment timeline README without copying all intermediates.
4. Copy the selected tracked CVAE prototype files into its historical directory.
5. Compare the recovered February 2024 CVAE against the committed December version.
6. Add the CVAE prototype README with limitations.
7. Add protective `.gitignore` rules before introducing any data directories.
8. Verify that no raw datasets or large artifacts are staged.
9. Review the complete diff before committing.

## Recommendation

Publish the curated prototypes and their scientific narrative, not the entire recovered filesystem. This preserves authenticity while keeping the repository understandable to researchers, employers, and future contributors.
