# Expanded Feature Experiments (2023)

This directory represents the transition between the thesis VAE and the later conditional VAE.

## Purpose

After the thesis prototype, the feature pipeline was expanded to better represent both ssDNA aptamers and their small-molecule targets. The work explored whether richer biological and chemical representations could support target-conditioned modeling.

## Representative artifact

- `features_v2.py` is retained as a representative intermediate revision.

Other experiments remain preserved in Git history and the original local research workspace rather than being duplicated here.

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

This was an exploratory period with many dated datasets, temporary notebooks, and overlapping script revisions. Publishing every intermediate file would obscure the scientific progression. The representative script and Git history preserve the work, while the documentation explains how it led to `features_v3.py` and the TensorFlow CVAE.

## Limitations

- Several transformations were performed in monolithic scripts.
- Inputs were connected through relative filenames and assumed row order.
- External scientific tools were required.
- Experimental branches were sometimes commented in or out.
- The intermediate files were not all complete executable pipelines.

This phase should be understood as iterative research and feature discovery—not a finalized model release.
