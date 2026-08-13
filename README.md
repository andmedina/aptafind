# Aptafind

Aptafind is a research software project exploring computational analysis and generative modeling of single-stranded DNA (ssDNA) aptamers for small-molecule targets.

The project began as a 2023 master's thesis and later expanded into richer sequence, secondary-structure, affinity, and target-chemistry feature experiments. This repository preserves that research history while preparing a cleaner, reproducible implementation.

> Aptafind produces computational analyses and candidate prioritizations. It does not establish experimental binding or biological efficacy.

## Research question

Can sequence, predicted secondary-structure, binding-affinity, and target-molecule features be integrated into a reproducible workflow that learns meaningful patterns from known aptamers and prioritizes plausible candidates for further study?

## Project evolution

| Phase | Period | Focus |
|---|---|---|
| Thesis VAE | Spring 2023 | Initial feature pipeline and PyTorch variational autoencoder |
| Feature expansion | Summer–Fall 2023 | NUPACK structures, MEME motifs, sequence embeddings, and PubChem target features |
| TensorFlow CVAE | Late 2023 | Small-molecule-conditioned generative-model prototype |
| CVAE revision | Early 2024 | Added regularization, metrics, early stopping, and deeper layers |
| Reproducibility rebuild | Current | Validated data, explicit joins, leakage-resistant evaluation, baselines, and testing |

## Research progression

The research developed through three connected questions:

1. **Thesis VAE — Can a model learn and generate general aptamer-like sequences?**
   The original PyTorch VAE learned from known ssDNA aptamers using sequence, motif, structure, binding, and target-related features. It explored whether samples from a learned latent space could be decoded into plausible DNA sequences.

2. **Conditional VAE — Can generation be influenced by a selected molecular target?**
   The later TensorFlow CVAE added small-molecule descriptors and fingerprints as conditioning information. This moved the project from general aptamer-like generation toward target-aware candidate generation.

3. **Reproducibility rebuild — Is there defensible predictive signal before generating candidates?**
   The modern implementation will first validate the data, prevent leakage, evaluate interpretable baseline models, and measure generalization to unseen targets or sequence groups. Conditional generation will be revisited only after that foundation is established.

The historical models generated computational candidates; they did not prove that those sequences bind to a target. Experimental assays would be required to establish binding affinity and biological utility.

See [the research timeline](docs/research_timeline.md) for more detail.

## Repository organization

```text
aptafind/
├── historical/   # curated research prototypes preserved as written
├── docs/         # recovery, architecture, limitations, and rebuild plans
├── src/          # planned modern implementation
└── tests/        # planned validation and feature tests
```

The legacy files currently visible at the repository root are retained from the original Git history. The curated versions under `historical/` provide the intended narrative while the modern implementation is built separately.

## Historical prototypes

- [Thesis VAE prototype](historical/thesis_vae_2023/README.md)
- [Expanded feature experiments](historical/feature_experiments_2023/README.md)
- [TensorFlow CVAE prototype](historical/tensorflow_cvae_2023/README.md)
- [February 2024 CVAE revision](historical/tensorflow_cvae_2024_revision/README.md)

## Recovered pipeline

The late-2023 prototype follows this general flow:

```text
aptamer sequences + affinity + target metadata
                    |
                    +--> sequence and k-mer features
                    +--> NUPACK structure and energy features
                    +--> PubChem descriptors and fingerprints
                    |
                    v
                features.npz
                    |
                    v
         TensorFlow conditional VAE
```

Detailed maps:

- [Feature pipeline](docs/feature_pipeline_map.md)
- [CVAE architecture and training](docs/cvae_model_map.md)
- [Recovered historical assets](docs/historical_asset_recovery.md)
- [Aptafind v2 design proposal](docs/aptafind_v2_design.md)

## Current status

Historical assets have been located, inventoried, and checksummed. Selected prototypes have been curated into dated directories with explanatory documentation.

The current code is not yet a one-command reproducible pipeline. The next implementation milestone is to create a canonical validated dataset, stable record identifiers, explicit target joins, reproducible sequence features, and automated tests before retraining any model.

## Scientific and technical limitations

- The recovered late-2023 model dataset contains only 168 observations.
- Historical preprocessing was fitted before data splitting, which introduces leakage.
- Random row splitting does not measure generalization to unseen targets or related sequence scaffolds.
- The historical test subset was not evaluated in the recovered CVAE script.
- Generated candidates were not experimentally tested.
- Historical data redistribution rights require review before raw datasets can be published.

These limitations motivate the reproducibility rebuild and are documented openly rather than hidden.

## Thesis

Andrew Medina, *Deep Learning-Based Sequence Generation of Single-Stranded DNA Aptamers* (2023). DOI: [10.5281/zenodo.7922963](https://doi.org/10.5281/zenodo.7922963).

## License

Source code is provided under the repository's [license](LICENSE). External datasets, scientific tools, and third-party outputs remain subject to their respective terms.
