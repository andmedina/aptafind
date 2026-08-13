# Aptafind v2 Design Proposal

## Project objective

Aptafind v2 will rebuild the historical thesis workflow as a reproducible research pipeline for analyzing known ssDNA aptamers and ranking computational candidate sequences for small-molecule targets.

The project will answer this question:

> Can sequence, predicted secondary-structure, binding-affinity, and target-molecule features be integrated into a reproducible computational workflow that learns meaningful patterns from known aptamers and prioritizes plausible candidate sequences for further study?

The output will be **computational candidates**, not experimentally validated aptamers. Binding claims will require laboratory validation.

## Why this scope is defensible

The recovered dataset contains only 168 observations in the later small-molecule experiment. That is valuable for a research prototype but too small to support strong claims about de novo therapeutic or diagnostic aptamer design.

Aptafind v2 will therefore emphasize:

- Data provenance and validation
- Reproducible feature engineering
- Leakage-resistant evaluation
- Interpretable baselines
- Honest uncertainty and limitations
- A clearly separated generative-model experiment

## Relationship to the historical project

### Preserve unchanged

- Original thesis and documentation
- Historical raw/intermediate datasets
- Thesis-era PyTorch VAE code
- Late-2023 TensorFlow CVAE code
- Recovered feature archives and PCA artifact
- MEME, NUPACK, and APTANI outputs
- Reconstruction-loss figures and historical results

These materials form the historical research snapshot and should not be silently rewritten.

### Reimplement cleanly

- Dataset loading and schema validation
- Sequence normalization
- Record identity and data joins
- Feature extraction
- Train/validation/test splitting
- Preprocessing and dimensionality reduction
- Baseline modeling
- Model evaluation
- Artifact persistence
- Candidate-ranking workflow

### Retain as optional experiments

- MEME motif features
- Word2Vec embeddings
- APTANI-derived features
- Conditional VAE sequence generation

These should not block the core reproducible pipeline.

## Proposed project stages

### Stage 1: Historical preservation

Deliverables:

- Recovery inventory
- SHA-256 manifest
- Feature-pipeline map
- CVAE architecture map
- Clear distinction between thesis-era and post-thesis experiments

Status: substantially complete.

### Stage 2: Dataset reconstruction

Create one canonical, versioned aptamer table with stable identifiers and documented provenance.

Minimum fields:

- `aptamer_id`
- `sequence`
- `target_name`
- `target_id` or PubChem CID where available
- `target_type`
- `kd_value`
- `kd_unit`
- `reference`
- `source_dataset`
- `record_version`

Validation should cover:

- DNA alphabet validity
- Missing and duplicated sequences
- Sequence length
- Missing targets
- Numeric affinity parsing and units
- Duplicate sequence-target pairs
- Target-feature join completeness

### Stage 3: Reproducible feature pipeline

Core feature families:

1. **Sequence composition**
   - Length
   - GC content
   - 1-mer, 2-mer, and 3-mer frequencies

2. **Predicted structure**
   - Dot-bracket encoding
   - Minimum free energy
   - Paired/unpaired fractions
   - Simple loop/stem summaries where reliable

3. **Target chemistry**
   - Molecular descriptors
   - Morgan fingerprint
   - Explicit missing-data indicators

4. **Experimental context**
   - Kd with normalized units
   - Selection/binding conditions only when sufficiently complete

Every produced feature table should retain `aptamer_id` and `target_id`. Feature groups must be joined explicitly rather than assumed to share row order.

### Stage 4: Leakage-resistant data splits

Primary evaluation should group observations by target so aptamers for the same target do not appear in both training and testing sets.

Recommended approaches:

- Grouped cross-validation by target
- Leave-one-target-out evaluation when target counts permit
- Sequence-similarity clustering before splitting to reduce scaffold leakage

A random row split may be retained only as a secondary comparison and must be labeled as optimistic.

### Stage 5: Baseline models

Before rebuilding the CVAE, implement simpler baselines appropriate to the final prediction task.

Potential supervised task:

> Predict standardized log-affinity or classify stronger versus weaker binders using sequence, structure, and target features.

Candidate models:

- Regularized linear/logistic regression
- Random forest
- Gradient-boosted trees

Baselines establish whether the engineered features contain useful signal and provide interpretable feature importance.

If the available labels cannot support defensible supervised prediction, the project should instead emphasize exploratory clustering, similarity retrieval, and candidate ranking.

### Stage 6: Candidate-ranking workflow

For a selected target:

1. Retrieve or calculate target chemistry features.
2. Compare against known target and aptamer representations.
3. Score candidate sequences with the validated baseline.
4. Apply sequence-validity and structural filters.
5. Rank candidates with uncertainty and similarity information.
6. Export a reviewable candidate report.

The ranking output must disclose that it is a computational prioritization tool.

### Stage 7: Conditional generative experiment

Rebuild the CVAE only after the baseline and evaluation pipeline work.

The modern experiment should:

- Use deterministic seeds
- Fit preprocessing on training folds only
- Use grouped evaluation
- Apply feature-appropriate output heads and losses
- Save all transformers and model metadata
- Separate conditioning features from reconstruction targets
- Generate nucleotide sequences through a sequence-aware representation
- Compare generated candidates against baseline and nearest-neighbor methods

The CVAE is an experiment within Aptafind v2—not the sole proof that the project succeeds.

## Evaluation framework

### Data quality

- Schema-validation pass rate
- Duplicate and invalid-sequence counts
- Target-feature join coverage
- Missingness by field

### Predictive baselines

Depending on the selected outcome:

- Regression: MAE, RMSE, R², and grouped-fold variability
- Classification: precision, recall, F1, ROC-AUC, and PR-AUC

Results should include simple comparison baselines such as predicting the training mean or majority class.

### Generated candidates

- Valid nucleotide percentage
- Unique-sequence percentage
- Novelty relative to training sequences
- Sequence-length distribution
- GC-content distribution
- Nearest-neighbor sequence similarity
- Predicted structure and energy distributions
- Target-conditioning consistency
- Diversity among top candidates

These metrics measure computational plausibility, not experimental binding.

## Proposed repository structure

```text
aptafind/
├── README.md
├── pyproject.toml
├── configs/
│   ├── data.yaml
│   ├── features.yaml
│   └── model.yaml
├── data/
│   ├── README.md
│   ├── sample/
│   ├── raw/                 # ignored when redistribution is restricted
│   ├── interim/             # ignored/generated
│   └── processed/           # ignored/generated
├── docs/
│   ├── historical_asset_recovery.md
│   ├── historical_checksums.sha256
│   ├── feature_pipeline_map.md
│   ├── cvae_model_map.md
│   ├── methodology.md
│   └── limitations.md
├── notebooks/
│   └── exploratory/         # analysis only; not the production pipeline
├── src/aptafind/
│   ├── data/
│   │   ├── load.py
│   │   ├── clean.py
│   │   └── validate.py
│   ├── features/
│   │   ├── sequence.py
│   │   ├── structure.py
│   │   ├── target.py
│   │   └── assemble.py
│   ├── models/
│   │   ├── baselines.py
│   │   ├── evaluate.py
│   │   └── cvae.py
│   ├── candidates/
│   │   ├── generate.py
│   │   ├── filter.py
│   │   └── rank.py
│   └── cli.py
├── tests/
│   ├── test_cleaning.py
│   ├── test_sequence_features.py
│   ├── test_feature_alignment.py
│   └── test_splitting.py
└── historical/              # add only artifacts approved for redistribution
    └── README.md
```

## Dependency strategy

Use a small core environment first:

- Python
- NumPy
- pandas
- scikit-learn
- Biopython if needed
- RDKit for molecular features
- pytest

NUPACK, MEME, ViennaRNA/APTANI, TensorFlow, and PyTorch should be optional dependency groups because they are heavier and may have separate installation or licensing constraints.

## Publication strategy

Before publishing historical data:

1. Verify Aptagen/source redistribution terms.
2. Remove local absolute paths and private metadata.
3. Publish an allowed sample dataset and schema when full redistribution is uncertain.
4. Provide download/preparation instructions rather than silently bundling restricted data.
5. Cite the thesis, original sources, and external scientific tools.

## Definition of done

Aptafind v2 can be called complete when a new user can:

1. Clone the repository.
2. Install a documented environment.
3. Obtain or use an allowed sample dataset.
4. Run validation and feature generation with one documented command.
5. Train and evaluate baseline models without leakage.
6. Reproduce a versioned results report.
7. Optionally run the generative experiment.
8. Understand the scientific assumptions and limitations from the documentation.

## First implementation milestone

The first implementation should stop before modeling. It should produce:

- A canonical cleaned dataset
- A validation report
- Stable record identifiers
- Explicit aptamer-to-target joins
- Reproducible sequence features
- Tests for the most important data and feature rules

This milestone creates the foundation needed to determine whether the later modeling questions are statistically supportable.
