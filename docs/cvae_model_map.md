# Aptafind `cvae.py` Model Map

## Scope

This document explains the recovered late-2023 conditional variational autoencoder (CVAE) prototype. The historical code remains unchanged.

## Intended question

The model explores whether a compressed representation of known ssDNA aptamers can be learned while conditioning on properties of their small-molecule targets. Conceptually, the decoder is intended to generate or reconstruct an aptamer-related feature vector for a selected molecular target.

This is a research prototype. It does not demonstrate experimentally validated aptamer generation or binding performance.

## Model data flow

```text
features.npz
    |
    +--> ssDNA-side features (132 columns)
    |      sequences             28
    |      Kd                     1
    |      target type           28
    |      structures            28
    |      k-mers                33
    |      sequence embedding    12
    |      binding energy         2
    |
    +--> molecule condition (34 columns)
           fingerprint           28
           molecular properties   6

ssDNA features + molecule condition
                |
                v
              Encoder
                |
       z_mean + z_log_variance
                |
        stochastic sampling
                |
          latent vector (16)
                |
                + molecule condition
                |
                v
              Decoder
                |
                v
    reconstructed ssDNA feature vector
```

The dimensions above are calculated from the recovered `features.npz` archive.

## Data preparation

The script loads `features.npz` and creates two matrices:

- **ssDNA features:** the sequence, affinity, target type, structure, k-mer, sequence-embedding, and binding-energy arrays concatenated horizontally.
- **Molecule features:** the molecular fingerprint and physicochemical-property arrays concatenated horizontally.

A single shuffled index is applied to both matrices so their row alignment is retained. The shuffled observations are then sliced into:

- 80% training
- 10% testing
- 10% validation

For 168 observations, integer slicing produces approximately 134 training, 16 testing, and 18 validation observations.

## Encoder

The encoder has two branches.

### ssDNA branch

```text
132 inputs -> Dense 128 -> Dense 64 -> Dense 32 -> Dense 16
```

### Molecule branch

```text
34 inputs -> Dense 128 -> Dense 64 -> Dense 32 -> Dense 16
```

Both branches use ReLU activations. Their 16-value outputs are concatenated, followed by:

```text
32 combined values -> Dense 128 -> Dense 64
```

Two parallel 16-value outputs are then produced:

- `z_mean`: center of the learned latent distribution
- `z_log_var`: log variance of that distribution

## Latent sampling

The sampling function applies the VAE reparameterization trick:

```text
z = z_mean + exp(0.5 * z_log_var) * epsilon
```

where `epsilon` is sampled from a standard normal distribution. This allows gradient-based training while representing each observation as a distribution rather than a single fixed point.

## Decoder

The 16-value latent vector is concatenated with the 34-value molecule condition. The combined vector passes through:

```text
Dense 128 -> Dense 256 -> Dense 512 -> Dense 256 -> Dense 128
```

These layers use ReLU activation and L2 regularization with a coefficient of `0.01`.

The final layer contains 132 sigmoid outputs, matching the combined ssDNA feature width.

## Objective function

The custom VAE objective combines:

1. **Reconstruction loss:** mean squared error between the original and reconstructed ssDNA feature vectors, multiplied by the original feature dimension.
2. **KL-divergence loss:** encourages the learned latent distribution to remain close to a standard normal distribution.

The intended total objective is:

```text
reconstruction loss + KL-divergence loss
```

The model is compiled with Adam using a learning rate of `0.001`. RMSE and MAE are registered as additional metrics.

## Training configuration

- Maximum epochs: 200
- Batch size: 5
- Early-stopping patience: 5 epochs
- Early-stopping monitor: validation loss
- Best validation weights restored: yes

The recovered script trains the model when the file is executed or imported.

## Generation code

The experimental generation section is commented out. Its intended workflow is:

1. Draw a random 16-value latent vector.
2. Select the molecular features for a target.
3. Pass both into the decoder.
4. Split the 132-value decoder output back into feature groups.
5. Inverse-transform the reduced sequence representation with the saved sequence SparsePCA model.
6. Threshold reconstructed values into binary groups.
7. Convert four-value binary groups back into A, C, G, or T.

Therefore, the recovered executable path trains a reconstruction model, but it does not currently execute or validate sequence generation.

## Important limitations found during tracing

1. **No fixed random seed.** NumPy shuffling, TensorFlow initialization, and latent sampling are nondeterministic.
2. **Preprocessing leakage already occurred upstream.** The feature transformations in `features_v3.py` were fitted using all observations before this split.
3. **The split is not biologically grouped.** A random row split can place related aptamers or identical target classes across training, validation, and testing sets, producing optimistic estimates of generalization.
4. **The test set is never evaluated.** It is constructed but not passed to an evaluation routine.
5. **Validation loss behavior is questionable.** The custom VAE loss is added only when `training=True`. During validation, the model call does not add reconstruction or KL loss, so `val_loss` may reflect only regularization losses rather than the intended VAE objective.
6. **The sigmoid output conflicts with some transformed inputs.** Several input features were standardized or PCA-transformed and can be negative or greater than one, while sigmoid outputs are restricted to `[0, 1]`.
7. **Feature groups use different scales and meanings.** A single unweighted reconstruction loss combines binary, standardized, PCA-derived, energy, affinity, and categorical representations. High-dimensional groups can dominate the objective.
8. **Kd and target type are reconstructed as outputs.** The model does not generate only a nucleotide sequence; it reconstructs a mixed 132-value representation containing biological and metadata-derived feature groups.
9. **Condition leakage is possible.** Target-related information appears in both the ssDNA-side target-type features and the molecule condition.
10. **The architecture is large for the dataset.** Dense layers reaching 512 units are fitted using only about 134 training observations, creating substantial overfitting risk despite decoder regularization.
11. **Generation is heuristic and inactive.** SparsePCA inversion followed by a fixed threshold does not guarantee valid one-hot nucleotide groups, valid sequence lengths, novelty, structural plausibility, or target binding.
12. **No trained model is saved.** The script does not persist the encoder, decoder, complete CVAE, training history, split indices, or evaluation results.
13. **The module executes immediately.** Loading, splitting, constructing, and training occur at module import time.
14. **The model does not establish biological efficacy.** Computational reconstruction metrics cannot replace binding assays or other experimental validation.

## What the prototype successfully demonstrates

Despite its limitations, the recovered work demonstrates several substantive ideas:

- Integration of heterogeneous sequence, structure, affinity, and chemical-target features
- Conditional generative-model architecture
- Separate ssDNA and target-molecule encoder branches
- VAE latent-variable sampling and KL regularization
- An attempted route from a reduced representation back to nucleotide sequences
- Awareness of overfitting controls through validation, early stopping, and L2 regularization

These are valuable research-engineering contributions when described as an exploratory prototype rather than a validated aptamer-design system.

## Recommended modernization boundary

Preserve `cvae.py` as the late-2023 historical experiment. A modern version should separately implement:

1. Deterministic configuration and saved split assignments.
2. Target- or scaffold-grouped evaluation.
3. Training-only preprocessing and persisted transformers.
4. Explicit per-feature-group losses and compatible output activations.
5. A custom `train_step` and `test_step` that calculate the same objective correctly.
6. Baseline models to determine whether the CVAE adds value.
7. Test-set evaluation and uncertainty reporting.
8. Model, configuration, and training-history persistence.
9. Sequence-validity, novelty, diversity, structure, and affinity-proxy evaluation.
10. Clear separation between computational candidates and experimentally validated aptamers.
