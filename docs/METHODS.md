**Experiment protocol**

Written before viewing final model test scores. The current scope is recorded-flow network intrusion detection on UNSW-NB15. Binary normal/attack prediction is the primary task. Ten-class attack categorization (including Normal) is a separately assessed extension. An app, live packet capture, real-network deployment, road traffic prediction and new model architecture research are outside this run.

**Data identity and provenance**

The 175,341-row training attachment supplied by the team is the training source. The separate partition has 82,332 rows. The original dataset link on the UNSW page redirected to a Microsoft sign-in screen, so a public GitHub mirror was used to obtain the complementary partition. That mirror's two filenames are reversed relative to the counts on the UNSW page and the team's chosen training partition. Resolution uses row counts plus exact equality of all parsed values and row order between the team's attachment and the mirror's 175,341-row file. No datasets were merged or randomly reassigned between the two source partitions. This check establishes agreement with that mirror; it is not an author-signed authenticity certificate.

Original download filenames are retained, along with canonical train.csv and test.csv. data/source_manifest.json records source URLs, a pinned repository commit and SHA-256 checksums. The feature dictionary describes the wider original dataset; the reduced partitions in this experiment contain 45 columns: ID, 42 predictors and two answer columns. Do not describe these partition files as having 49 usable inputs.

**Audit before training**

Check schema, counts, target consistency, missing/nonfinite values, exact duplicate feature vectors, feature vectors with conflicting labels and train/test overlap. ID is a local row number, not a predictive feature. Both label and attack_cat are excluded from all inputs, including the categorical extension. This prevents one answer field from revealing the other.

Exact feature identity is computed using pandas 64-bit hashes over all 42 parsed feature columns. Numeric column dtypes were checked to agree between the two supplied partitions. This is an exact-feature audit, not a guarantee of independence between sessions, hosts or nearly identical flows. No session or host identifiers are available in these reduced CSVs for a stronger split.

**Development split**

Use the first fold of a five-fold StratifiedGroupKFold with shuffle=True and seed 42. Stratification uses attack_cat so rare categories remain represented. Every exact feature group is wholly in the fitting or validation subset, even when identical measurements have conflicting labels. The resulting split contains 140,273 fitting rows and 35,068 validation rows, with zero exact-feature overlap. Preserve natural record frequencies and ambiguous labels; report them rather than silently relabeling or removing difficult records.

Fit imputation and encoding only on the fitting subset during selection. Impute numeric missing values using the fitted training median and categorical missing values using the training mode. One-hot encode proto, service and state, ignoring unknown categories at inference. Preserve '-' in service as an observed category. Convert processed values to float32. Tree models use unscaled numeric values. The logistic baseline additionally standardizes numeric features with training-fitted statistics. No resampling or synthetic examples are used. Candidate class-weight settings are recorded explicitly.

**Model selection**

Candidate grids and random seeds are in config/experiment.json. Evaluate four XGBoost and three Random Forest configurations for the binary task, with majority-prior and logistic-regression baselines. For the categorical extension, evaluate two XGBoost and two Random Forest configurations plus a majority-prior baseline. Weighted XGBoost uses normalized square-root inverse class frequency from fitting labels only; balanced RF uses balanced_subsample.

For XGBoost, use CPU histogram trees and early stopping on development validation loss with patience 40. For binary candidates, choose the threshold from the predefined 0.10–0.90 grid using validation macro F1; the majority baseline retains 0.5. Multiclass uses the highest-scoring class. Select within each family by validation macro F1, then lower validation false positive rate for binary ties, then name. Choose the binary and multiclass champions among the respective tree family winners using the same rule. Macro F1 gives equal weight to class-level F1 scores; it does not imply any particular deployment cost trade-off.

Record every candidate, training duration, warnings, validation results, default-threshold results and selected settings. Save results/selection_lock.json before final evaluation. Refit each selected family model on all 175,341 training records. Use the validation-selected tree count and retain the selected threshold; do not use test examples for early stopping or fit preprocessing on them. Refitting may alter score distributions, so report both validation-tuned and default-threshold test results without retuning.

**Training-only stability diagnostics**

For the selected binary XGBoost and RF configurations, run a three-fold StratifiedGroupKFold with split seed 123, freshly fitted preprocessing in every fold and model seed 42. Use threshold 0.5 in all folds. Hyperparameters were selected using an earlier holdout from the same training partition: these are stability diagnostics, not nested CV or independent generalization estimates. They do not determine a replacement winner.

**Frozen test evaluation**

Evaluate the frozen models on the complete separate test partition. Report accuracy, balanced accuracy, precision, recall, F1, macro F1, false positive rate, ROC AUC, average precision, log loss, Brier score, confusion counts and per-category binary detection rates. Report macro/weighted F1 and full class-level reports for the categorical extension. Majority and logistic baselines remain visible even if weak.

The source partitions themselves contain exact feature overlap. Report three fixed views for every model:

1. All 82,332 rows of the original test partition, for direct benchmark reporting.
2. The 73,791 test rows whose 42-feature vectors have no exact match anywhere in training.
3. That same unseen-feature subset with repeated feature-plus-target records counted once. The target is label for binary, attack_cat for multiclass.

Do not drop overlapping records and then describe the resulting score as the complete official test score. Do not imply that exact-feature separation also excludes near duplicates or related network sessions. The unchanged official partition and additional subsets are all reported, so limitations are visible.

Compute descriptive 95% intervals for binary metrics using 400 Poisson bootstrap replications clustered by exact test feature vector. These intervals condition on this fitted model and this benchmark. They do not capture training/tuning variability, temporal drift, host correlation or performance on an actual campus network.

**Interpretation and artifacts**

Feature importances are training-derived model summaries, aggregated across one-hot columns. They are not causal explanations and can be affected by correlated inputs and high-cardinality features. Raw model scores are not calibrated estimates of real-world attack probability. The categorical model must identify a known class; it does not establish reliable detection of novel attack types.

Keep saved preprocessing, fitted models, native XGBoost files, raw model outputs, per-category metrics, plots, source checksums and version locks. Test fresh-process model loading; verify that changing answer columns or ID does not change predictions, missing required inputs are rejected, unknown categories/missing values are handled, stored preprocessing statistics match training only and confusion counts reconcile independently. Keep a small answer-free test demo with a separate answer key.

For score-invariance verification, require the prepared arrays to be exactly equal and class predictions to match. Compare floating-point scores with absolute tolerance 1e-12: parallel forest reductions can sum tree outputs in a different order and differ in their last floating-point bits even on unchanged inputs. This is a numerical verification tolerance, not a change to model settings or decision thresholds. Observed differences are recorded in results/verification.json.

**References used for method choices**

- [UNSW dataset description, partitions and required dataset citations](https://research.unsw.edu.au/projects/unsw-nb15-dataset).
- [Scikit-learn: avoiding inconsistent preprocessing and data leakage](https://scikit-learn.org/stable/common_pitfalls.html).
- [Scikit-learn: StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
- [XGBoost: estimator interface and early stopping](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html).
- [Scikit-learn: threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html).
- [Scikit-learn: evaluation metrics](https://scikit-learn.org/stable/modules/model_evaluation.html).
- Chen, T., & Guestrin, C. (2016). [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754).
- Breiman, L. (2001). [Random Forests](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf).

The project applies established methods with careful evaluation. It does not claim to invent XGBoost, Random Forest or a previously unknown intrusion-detection algorithm.
