# Machine learning experiment

This folder preserves the original binary model bundles, canonical data, experiment configuration, code and measured results. Both model files are fitted and include preprocessing; you can run inference without retraining.

From the repository root, after installing the Python 3.11 environment described in the top-level README:

```bash
.venv/bin/python ml/scripts/predict.py --input ml/data/simple_sample.csv --output .cache/xgb-predictions.csv
.venv/bin/python ml/scripts/predict.py --model ml/models/rf_depth24.joblib --input ml/data/simple_sample.csv --output .cache/rf-predictions.csv
```

The compact repository ships the two main Normal/Attack bundles. The original experiment's report, configuration and scripts also cover baseline and attack-category comparisons. Additional fitted binaries and prepared matrices are omitted to avoid redundant large files. Consequently, `scripts/verify.py` is the original **full seven-model** verifier, not the website's test command. Its recorded result is retained in `results/verification.json`.

To recreate the full original experiment in a new folder, including every model and prepared matrix:

```bash
.venv/bin/python ml/scripts/reproduce.py --output .cache/full-experiment
```

This runs training and takes substantially longer than prediction. Existing experiments are preserved. The original seed and environment are recorded; exact results can vary with platform or library versions. Website integrity and browser-inference checks are at the repository root: `npm test` and `node tests/model-parity.mjs` after asset export.

`data/source_manifest.json` preserves references to original source copies. Only the resolved canonical CSVs are included here; their hashes are unchanged.
