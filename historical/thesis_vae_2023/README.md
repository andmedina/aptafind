# Thesis VAE Prototype (Spring 2023)

This directory preserves the thesis-era Aptafind prototype as it existed at the end of the April 2023 Git history.

## Research objective

The prototype explored whether a variational autoencoder (VAE) could learn a compressed representation of known single-stranded DNA aptamers and reconstruct or generate sequence-related representations.

It supported the master's thesis:

> Andrew Medina, *Deep Learning-Based Sequence Generation of Single-Stranded DNA Aptamers*, 2023. DOI: 10.5281/zenodo.7922963.

## Preserved artifact

- `features.ipynb` was recovered from Git commit `08f8b1d` dated April 19, 2023. That commit was described as “VAE has been trained to produce sequence.”

The notebook is intentionally preserved as a historical research artifact. It has not been rewritten into modern package code.

## Data and methods

The thesis workflow began with an Aptagen-derived collection of aptamer records, cleaned the sequences, and engineered sequence, structural, motif, and target-related features. The thesis reports approximately 293 cleaned sequences from 437 scraped records and uses a PyTorch VAE.

Reported reconstruction metrics included:

- Mean squared reconstruction error: 0.0282
- R²: -0.0632
- Explained variance: -0.0436
- Mean absolute error: 0.1341

These results indicate exploratory reconstruction behavior, not validated biological performance.

## Limitations

- The dataset was small for deep generative modeling.
- The notebook combines exploration, preprocessing, modeling, and evaluation.
- Reproducibility depends on historical data and scientific-tool environments.
- Negative R² and explained-variance values show that reconstruction performance was limited.
- Generated sequences were not experimentally tested for binding.
- Raw historical data is not included here pending provenance and redistribution review.

## Contribution to later work

The strongest contribution of this phase was the integrated research pipeline: collecting aptamer data, constructing biologically motivated features, incorporating structural tools, and testing a generative model. Later prototypes expanded the small-molecule target features and introduced a conditional VAE.
