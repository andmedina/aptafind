# March 2026 Refactor-Cleanup WIP

This directory preserves the only substantial unique source file recovered from
the unfinished `refactor-cleanup` working tree created on March 5, 2026.

The abandoned checkout was based on Git commit `c8fbc68` and contained no new
commits. Most of its staged changes were content-identical file moves. The file
`feature_pipeline.py` was untracked and reorganized the historical
`features_v3.py` script into functions.

`feature_pipeline.py` is preserved exactly as recovered. Its SHA-256 checksum is:

```text
8d1164ad5643dc52051cea68ba9a29cdaccb652c19e422dc918f38d6b7a6b4e2
```

This code is a historical work-in-progress, not the active Aptafind
implementation. It retains limitations from the late-2023 feature workflow,
including full-dataset preprocessing, working-directory-relative paths,
external scientific-tool dependencies, and no automated tests. Useful ideas
may be reimplemented selectively in the modern package with validation and
tests.
