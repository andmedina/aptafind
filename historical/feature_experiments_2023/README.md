# Expanded Feature Experiments (2023)

This directory represents the transition between the thesis VAE and the later conditional VAE.

## Purpose

After the thesis prototype, the feature pipeline was expanded to better represent both ssDNA aptamers and their small-molecule targets. The work explored whether richer biological and chemical representations could support target-conditioned modeling.

## Preserved artifacts

- `features.ipynb`: the later working notebook that replaced the thesis-era
  notebook in the repository during November 2023
- `features_v2.py`: representative intermediate feature revision
- `generateSequence.py`: experimental sequence-generation helper
- `utilities/cleanUpSequence.py`: sequence-cleaning helper
- `utilities/constructSqeuence.py`: historical sequence-construction helper;
  the original filename spelling is retained
- `utilities/extractCore.py`: motif-core extraction experiment
- `utilities/extractSequence.py`: sequence extraction helper

The distinct April 2023 thesis notebook is preserved in
`historical/thesis_vae_2023/`. Shared utilities used by the integrated CVAE
pipeline are preserved with that prototype in
`historical/tensorflow_cvae_2023/utilities/`.

## Areas explored

- Expanded small-molecule aptamer datasets
- Nucleotide one-hot encoding
- 1-mer, 2-mer, and 3-mer frequencies
- Word2Vec nucleotide embeddings
- NUPACK-derived secondary structures and energies
- MEME motif analysis grouped by target
- PubChem molecular descriptors and fingerprints
- PCA and SparsePCA dimensionality reduction

## Why this phase is summarized

This was an exploratory period with many dated datasets, temporary notebooks,
and overlapping script revisions. This directory retains every unique source
artifact that remained at the repository root, while Git history preserves the
earlier progression. The documentation explains how the work led to
`features_v3.py` and the TensorFlow CVAE.

## Limitations

- Several transformations were performed in monolithic scripts.
- Inputs were connected through relative filenames and assumed row order.
- External scientific tools were required.
- Experimental branches were sometimes commented in or out.
- The intermediate files were not all complete executable pipelines.

This phase should be understood as iterative research and feature discovery—not a finalized model release.
