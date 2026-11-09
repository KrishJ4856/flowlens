**Model card: recorded-flow network intrusion detection**

**Identity and intended use**

Primary task: binary classification of UNSW-NB15-style flow measurements as Normal (0) or Attack (1). Intended use: a student benchmark demonstration and research pipeline. Selected bundle: models/xgb_depth7.joblib. Family: xgboost. Selection rule: grouped validation macro F1. Frozen score threshold: 0.50. The model is already trained; inference does not need an answer column or retraining.

**Training and selection**

Source: team's 175,341-row training attachment. Development fit/validation: 140,273 / 35,068 rows with exact-feature groups kept intact. Hyperparameters, weights, early-stopping tree counts and thresholds were chosen from training/validation only. The final bundle is refitted using the complete training partition. Full settings: results/selection_lock.json. Versions: requirements-lock.txt. Model-file hashes: models/model_index.json.

**Input contract**

All 42 named measurements in data/model_input_schema.json are required. Column order may vary. ID is optional and has no influence on predictions. label and attack_cat are answers and are ignored. proto, service and state are categorical. Other inputs are numeric; invalid numeric text raises an error. Missing numeric/categorical values use saved training-fitted imputation, and unknown categorical values are encoded using all-zero indicators for that categorical field. This tolerance avoids an input crash; it does not establish reliable prediction for unfamiliar protocols or shifted data.

The CSV contains precomputed flow measurements. A raw packet capture, arbitrary system log or Wi-Fi packet stream does not satisfy this contract without a compatible extraction process.

**Output contract**

The reusable helper produces a class prediction, readable label, uncalibrated attack_probability score and optional tracing ID. Evaluation exports call the same kind of score attack_score. The decision is score >= 0.50. Use src/ids.py predict_bundle or scripts/predict.py so the stored preparation and threshold are applied together. A direct call to the estimator's default predict method may use a different threshold.

**Measured outcomes**

All 82,332 separate test rows: accuracy 87.12%, attack recall 98.12%, attack precision 82.01%, macro F1 86.53%, false positive rate 26.36%. Missed attacks: 852. False alarms: 9,754.

The 73,791 test rows with no exact training feature match: accuracy 86.35%, attack recall 97.80%, macro F1 86.03%. The additional unique feature-plus-answer subset and every baseline are in results/test_metrics.csv. Conditional benchmark intervals are in results/intervals_xgb_depth7.json.

**Known limits**

The benchmark contains repeated patterns, conflicting identical-feature labels, class imbalance and a difference between training and test distributions. Exact-feature grouping does not prove separation by host, session, collection time or near-duplicate patterns. Real-network prevalence and costs can differ materially. Model scores are not calibrated operational risk estimates. No live deployment, intrusion blocking, novel-attack guarantee, future traffic guarantee or campus-network performance claim is established.

**Categorical extension**

Separate champion: multiclass_xgb_weighted. Ten known categories include Normal and nine attack families. Its class-specific mistakes are in REPORT.md and results/report_multiclass_xgb_weighted_official_test.json. It is not a replacement for the binary score and must not be presented using the binary accuracy figure.

**Verification and next interface step**

Stored hashes, answer/ID exclusion, column-order invariance, input validation, missing/unknown handling, train-only preparation statistics and fresh-process inference were verified. A future app should load the existing bundle once, validate the input schema, call predict_bundle and display scores as model scores. Keep the documented training/test caveats alongside any results page. Further validation is needed before any live operational use.
