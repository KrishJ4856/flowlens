**Poster content draft — for guide review**

**Suggested title:** Machine Learning-Based Network Intrusion Detection using XGBoost and Random Forest

**Problem:** Suspicious network communications can be difficult to identify from large volumes of recorded traffic measurements. A classifier can assist by flagging patterns associated with labelled attacks.

**Objective:** Build and evaluate a reproducible system that predicts normal versus attack from network-flow measurements, comparing XGBoost and Random Forest under a shared preparation and evaluation protocol.

**Dataset:** UNSW-NB15 reduced partition files; 175,341 training records, 82,332 separate test records; 42 predictors. Local ID and both answer fields excluded. Source: UNSW Canberra; original publication Moustafa & Slay (2015). The data retrieval mirror and source hashes are documented in the project package.

**Methodology:** Audit → grouped fit/validation split inside training → train-only preparation → candidate training → validation-only model/threshold selection → full-training refit → frozen separate-test evaluation.

**Key preparation:** Numeric imputation using training medians, categorical imputation using training modes, one-hot encoding of protocol/service/state, and preservation of saved preparation rules for inference. Exact duplicate feature vectors are grouped together during development validation.

**Results:** Validation-selected binary model: xgb_depth7; complete test accuracy 87.12%, attack precision 82.01%, attack recall 98.12%, macro F1 86.53%, false positive rate 26.36%. Test confusion counts: 44,480 attacks detected, 852 attacks missed, 27,246 normal records accepted, 9,754 normal records flagged.

**Evaluation limitation:** 8,541 test records have an exact training feature match. On the 73,791 test records without such a match, accuracy is 86.35% and attack recall is 97.80%. This additional subset is reported separately from the complete benchmark. Independence by host/session and near-duplicate separation are not established.

**Conclusion:** The project delivers trained offline classifiers, stored preprocessing, a reproducible comparison and explicit error analysis. Evaluation reveals useful detection capability and remaining missed attacks/false alarms on this benchmark. An app and live network integration remain future work.

**Suggested figures:** figures/02_binary_comparison.png and figures/03_binary_confusion.png; optionally figures/05_detection_by_category.png. PNG and SVG versions are available. Add the final approved guide and team details when designing the poster.

**Optional extension:** Ten-category classification is evaluated separately in REPORT.md. Keep its scores separate from normal/attack detection.

This is content for the group's review, not a designed or approved poster. Submit the poster soft copy to the guide by 19 November 2026; print the 3 ft × 2 ft flex only after approval. Presentation: 26 November 2026.
