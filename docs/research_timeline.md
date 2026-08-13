# Aptafind Research Timeline

## Spring 2023 — Thesis VAE

The original project collected and cleaned known aptamer records, engineered biological features, and trained a PyTorch variational autoencoder. The work culminated in the master's thesis *Deep Learning-Based Sequence Generation of Single-Stranded DNA Aptamers*.

The thesis-era notebook is preserved from Git commit `08f8b1d` dated April 19, 2023. The model results were exploratory: the feature-engineering workflow was substantial, while the reported reconstruction metrics did not demonstrate strong predictive performance or biological validity.

## Summer 2023 — Dataset and structural expansion

The project expanded beyond the initial thesis dataset toward small-molecule aptamers. Experiments incorporated:

- Larger and revised aptamer tables
- NUPACK secondary-structure calculations
- Minimum free energy and stacking energy
- MEME motif analysis by target group
- Sequence-generation and decoding utilities

This period produced multiple dated datasets and intermediate scripts typical of exploratory research.

## Fall 2023 — Integrated feature engineering

The later feature pipeline combined:

- Nucleotide one-hot representations
- 1-mer, 2-mer, and 3-mer frequencies
- Word2Vec sequence embeddings
- NUPACK-derived structural representations
- Binding affinity and target type
- PubChem molecular descriptors
- PubChem and Morgan fingerprints
- PCA and SparsePCA dimensionality reduction

The recovered final archive contains 168 observations across nine feature groups.

## December 2023 — TensorFlow conditional VAE

A TensorFlow/Keras conditional VAE was committed to Git. It encoded ssDNA and molecule features through separate branches, learned a 16-dimensional latent representation, and conditioned reconstruction on molecular features.

The prototype demonstrated heterogeneous feature integration and a conditional generative architecture. However, preprocessing leakage, random row splitting, output scaling, small sample size, and missing test evaluation limit conclusions about generalization.

## February 2024 — CVAE training revision

A later local revision added deeper layers, decoder L2 regularization, RMSE and MAE tracking, early stopping, and longer maximum training. It also disabled the active experimental sequence-decoding block.

This revision is preserved separately because it represents a meaningful iteration that was not committed to the original public Git history.

## Current — Reproducibility rebuild

The current effort is preserving the research history and rebuilding the workflow around:

- Canonical, validated data
- Stable identifiers and explicit joins
- Versioned feature artifacts
- Training-only preprocessing
- Target- and sequence-aware evaluation splits
- Interpretable baseline models
- Automated tests
- Reproducible model and results artifacts
- Honest separation between computational prioritization and laboratory validation

The generative model will be revisited only after the dataset, feature pipeline, and baseline evaluation are defensible.
