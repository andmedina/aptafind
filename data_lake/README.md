# Aptafind Research Data Lake

This directory is the local development root for Aptafind's research data lake.

```text
data_lake/
├── bronze/   # immutable source snapshots
├── silver/   # validated and standardized datasets
└── gold/     # versioned analysis- and model-ready datasets
```

The data layers are excluded from Git. The repository tracks source manifests, checksums, schemas, transformation code, validation rules, and small test fixtures instead of third-party or generated datasets.

## Bronze rules

- Preserve downloaded files without manual edits.
- Record the source URL, repository revision, retrieval timestamp, license, and file hashes.
- Treat a Git-backed dataset checkout as read-only after freezing its revision.
- Create a new versioned snapshot when the upstream source changes.

## Silver rules

- Read only from registered Bronze snapshots.
- Standardize fields without discarding original values or provenance.
- Write typed, validated Parquet datasets.
- Retain stable source and record identifiers.

## Gold rules

- Build each dataset for a documented research question.
- Record all Silver inputs, transformations, split definitions, and feature versions.
- Never overwrite a dataset used for a reported experiment.

The planned cloud equivalent uses the same paths beneath an object-storage URI such as an Amazon S3 bucket.
