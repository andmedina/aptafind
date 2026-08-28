# Aptafind Research Data Lake

This directory is the local development root for Aptafind's research data lake.

```text
data_lake/
├── bronze/   # immutable source snapshots
├── silver/   # validated and standardized datasets
└── gold/     # versioned analysis- and model-ready datasets
```

The data layers are excluded from Git. The repository tracks source manifests, checksums, schemas, transformation code, validation rules, and small test fixtures instead of third-party or generated datasets.

The authoritative acquisition registry is [`../manifests/datasets.yaml`](../manifests/datasets.yaml). Each local snapshot uses the registry's `local.bronze_uri` beneath this directory. A typical source is stored as:

```text
data_lake/bronze/<dataset_id>/
├── source/                 # files exactly as downloaded
├── retrieval.json         # local retrieval timestamp, resolved URLs, and tool version
└── checksums.sha256        # hashes calculated after retrieval
```

These local ledgers are intentionally ignored with the raw files. Stable expected metadata and upstream checksums belong in the Git-tracked registry.

## Bronze rules

- Preserve downloaded files without manual edits.
- Record the source URL, repository revision, retrieval timestamp, license, and file hashes.
- Treat a Git-backed dataset checkout as read-only after freezing its revision.
- Create a new versioned snapshot when the upstream source changes.
- Keep browser- or request-delivered files under the same dataset ID as their registry entry.
- Do not treat a successful download as permission to redistribute the file.
- Do not acquire multi-gigabyte records merely to prove that their URLs work; verify metadata first and download against an active milestone.

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

## Current acquisition boundary

Small public database exports and publication supplements may be acquired for local research after their registry entries are verified. The three DL-SELEX sequencing records total approximately 13.6 GB compressed and remain remote-only until the trajectory-analysis work resumes. Request-only or no-reuse sources remain catalog-only until permission is documented.
