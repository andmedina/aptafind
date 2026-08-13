# TensorFlow Conditional VAE Prototype (December 2023)

This directory preserves the TensorFlow/Keras conditional variational autoencoder (CVAE) committed in December 2023 and the feature code associated with it.

## Research objective

The prototype explored whether known ssDNA aptamer features could be encoded into a latent space and reconstructed while conditioning the model on chemical features of a small-molecule target.

## Pipeline

```text
aptamer dataset + PubChem target descriptors
                |
                v
          features_v3.py
                |
                v
           features.npz
                |
                v
              cvae.py
```

`features_v3.py` assembles sequence, affinity, target-type, structural, k-mer, embedding, binding-energy, fingerprint, and molecular-property features. `cvae.py` separates these into ssDNA and molecule-conditioning inputs.

## Preserved files

- `cvae.py`: December 2023 TensorFlow/Keras CVAE
- `features_v3.py`: integrated late-2023 feature pipeline
- `targetFeature.py`: PubChem-related target feature generation
- `utilities/structureMotif.py`: NUPACK structure interface
- `utilities/sequenceMotif.py`: MEME motif interface
- `utilities/to_fasta_file.py`: FASTA conversion helper

Historical datasets and generated artifacts are intentionally not copied into this directory.

## Model summary

- Two encoder branches: ssDNA features and molecule features
- A 16-dimensional stochastic latent representation
- A molecule-conditioned decoder
- Reconstruction plus KL-divergence objective
- 80/10/10 random train/test/validation split
- Adam optimizer

## What it demonstrated

- Integration of heterogeneous biological and chemical features
- Target-conditioned generative-model architecture
- VAE reparameterization and latent sampling
- An attempted path from reduced model output back to nucleotides

## Important limitations

- Only 168 observations were available in the recovered feature archive.
- Preprocessing was fitted before the train/test split, creating leakage.
- The random split did not isolate targets or similar sequence scaffolds.
- The test subset was created but not evaluated.
- Sigmoid decoder outputs are incompatible with some standardized/PCA features.
- The executable path trained a reconstruction model; sequence generation and decoding were experimental.
- No generated aptamer was experimentally validated.

See `docs/feature_pipeline_map.md` and `docs/cvae_model_map.md` for detailed technical analysis.

## Successor

A distinct February 2024 revision added model-depth, regularization, metrics, and early stopping. It is preserved separately so the research history is not silently overwritten.
