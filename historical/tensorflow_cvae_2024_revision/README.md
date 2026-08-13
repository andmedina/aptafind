# TensorFlow CVAE Revision (February 2024)

This directory preserves a later local revision of `cvae.py` dated February 8, 2024. It was recovered from the original research workspace and was not present in the public repository's December 2023 commit.

## Changes from the December prototype

- Added 32- and 16-unit layers to both encoder branches
- Added additional 256- and 128-unit decoder layers
- Added L2 regularization to decoder hidden layers
- Added RMSE and MAE metrics
- Increased the maximum training duration from 50 to 200 epochs
- Changed batch size from 6 to 5
- Added early stopping with best-weight restoration
- Commented out the previously active experimental generation and decoding block

## Interpretation

This revision appears to focus on training control and overfitting mitigation. It should be treated as a separate iteration, not automatically considered a better validated model.

The deeper architecture also increases model capacity relative to the approximately 134 training observations, so overfitting remains a major concern. Early stopping and L2 regularization help manage that risk but do not eliminate it.

## Preservation status

- `cvae.py` is copied unchanged from the recovered historical workspace.
- Its SHA-256 checksum is recorded in `docs/historical_checksums.sha256`.
- No dataset, feature archive, model weights, or generated candidates are included here.

## Limitations

The data leakage, random-split, output-scaling, missing test evaluation, and biological-validation limitations documented for the December prototype still apply unless independently corrected and verified.
