# CVAE Latent and Target-Condition Controls

## Purpose

The expanded v0.2.0 benchmark reconstructed held-out sequences better than
simple token baselines, but its near-zero KL divergence did not establish that
the latent variable or target condition carried useful information. The next
experiment separates those questions:

1. Does the posterior use the latent dimensions?
2. Does reconstruction worsen when the decoder receives the wrong target?
3. Does a model trained on deliberately permuted target labels lose that
   condition sensitivity?

These remain software and reconstruction diagnostics. They are not binding,
affinity, or specificity measurements.

## Latent-use measurements

Every evaluation now records both raw and objective KL divergence. When free
bits are enabled, the objective clamps the batch-average KL for each dimension
to the configured allowance while the raw KL remains unchanged for honest
reporting.

An active latent unit is defined as a posterior-mean dimension whose variance
across the evaluated partition is greater than 0.01. The report also retains
the mean and maximum posterior-mean variance. A model with zero active units is
treated as collapsed even if its reconstruction score is competitive.

The anti-collapse configuration uses two reviewed changes:

- `free_bits_per_dimension: 0.05` removes optimization pressure to reduce each
  latent dimension below that allowance.
- `decoder_token_dropout: 0.30` masks entire teacher-forced token embeddings,
  excluding BOS and padding, during training so the decoder cannot rely as
  completely on the preceding true nucleotide.

The frozen v0.2.0 configuration and checkpoint are not modified.

## Condition diagnostics

`diagnose-condition` reconstructs the exact checkpoint test partition after
verifying the input SHA-256 and recorded split summary. It reports:

- Matched reconstruction using the correct target condition
- Zero-vector replacement in both encoder and decoder
- Zero-vector replacement at the decoder only
- Multiple seeded derangements of unique target conditions in both locations
- The same derangements at the decoder only, holding the sequence posterior
  and its matched encoder condition fixed

Each derangement maps every target group to one different target group; there
are no fixed points. The primary sensitivity value is the decoder-only
reconstruction-NLL delta relative to the matched condition.

```bash
aptafind-generate diagnose-condition \
  --checkpoint artifacts/expanded_small_molecule_cvae/sequence_cvae.pt \
  --data data_lake/silver/thesis_endpoints/generation_positive_pairs.csv \
  --output artifacts/expanded_small_molecule_cvae/condition_diagnostics.json \
  --permutations 10
```

## Target-label permutation control

The negative-control configuration constructs a deterministic derangement over
the 230 training targets. All sequences belonging to one target receive the
same wrong target condition. Validation and test records remain correctly
labeled, and the data split, architecture, initialization seed, batch order,
optimization settings, and test examples remain identical to the primary
anti-collapse run.

Only the mapping hash, seed, target count, and zero-fixed-point audit enter the
run summary; raw target mappings are not required for the checkpoint because
the mapping is reproducible from the frozen input and seed.

```bash
aptafind-generate train \
  --data data_lake/silver/thesis_endpoints/generation_positive_pairs.csv \
  --config configs/expanded_thesis_cvae_antcollapse.yaml \
  --output-directory artifacts/expanded_small_molecule_cvae_antcollapse

aptafind-generate train \
  --data data_lake/silver/thesis_endpoints/generation_positive_pairs.csv \
  --config configs/expanded_thesis_cvae_antcollapse_permuted.yaml \
  --output-directory artifacts/expanded_small_molecule_cvae_antcollapse_permuted
```

The real-label model should be compared with the permuted-label control on the
same test fold. A positive wrong-condition NLL penalty is evidence that the
decoder uses the supplied condition for reconstruction; it is still not proof
that prior-generated candidates bind that molecule.
