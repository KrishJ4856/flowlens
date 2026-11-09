**Presentation notes: Network Intrusion Detection using Machine Learning**

These are rehearsal notes, not an announced college speaking-time requirement. Read START_HERE.md first, then practise running the answer-free sample in the notebook. Every number below comes from the actual frozen experiment.

**Suggested opening**

“Our project detects suspicious activity in recorded network communications. Each communication is described using measurements such as duration, packet counts and bytes transferred. We trained XGBoost and Random Forest to learn patterns from examples labelled normal or attack.”

**Explain what the model does**

“The model receives 42 network measurements. The correct-answer columns are kept separate. During training it sees examples and their answers. During testing it receives measurements only, makes predictions and is then scored against the hidden answers.”

**Explain the experiment**

“We used the group's 175,341-row training file and a separate 82,332-row test partition from UNSW-NB15. We chose model settings using a practice validation split inside training. Many records repeat, so identical measurement patterns were kept together during that practice split. After selection, we froze the settings and took the final test.”

**Present the real result**

“The validation-selected binary model was xgb_depth7. On the complete separate test partition it achieved 87.12% accuracy and detected 98.12% of attacks. It missed 852 attacks and incorrectly flagged 9,754 normal communications. We report these errors because accuracy alone does not describe a detector.”

“The original partitions contain exact repeated measurement patterns across training and test. We also evaluated the 73,791 test examples with no exact training match. On that subset, accuracy was 86.35% and attack recall was 97.80%.”

**Demonstrate it**

Open the walkthrough notebook at step 6 or run the README prediction command. Show data/demo_records.csv, point out that it has no answer columns, run the saved model and then open the separate answer key. Explain that the model is already trained, so prediction does not repeat the long teaching process. The 30 demo records are fixed, randomly chosen held-out examples, with an even mix for demonstration. Use the full test results when discussing performance.

**Suggested closing**

“We implemented a reproducible offline intrusion-detection pipeline and compared two established learning methods. The model files and preparation steps are saved. An interface and validation on live network data are future steps.”

**Likely questions and simple answers**

| Question | Answer |
| --- | --- |
| What exactly is network traffic? | Data exchanged between computers. Our records summarize individual communications rather than showing message contents. |
| What is a feature? | One measured detail, such as duration or packet count. |
| What is a label? | The known answer used for teaching or checking the model: normal or attack. |
| Where did the base model come from? | Python libraries supply the learning algorithms. Calling fit teaches a fresh model from our training records. |
| How are the methods different? | Random Forest combines many randomized decision trees. XGBoost builds trees progressively to improve the combined prediction. |
| What is preprocessing? | Preparing measurements in a consistent numeric format, including encoding categories and storing rules for missing values. |
| Why remove ID? | It is a file-local row number. It does not describe network behaviour. |
| Why remove attack_cat when predicting label? | It already reveals whether the record is an attack and would give away the answer. |
| Why have validation and test sets? | Validation helps choose settings. The final test checks the frozen choices on a separate partition. |
| What is a false positive? | A normal communication wrongly flagged as an attack. |
| What is a false negative? | An attack wrongly accepted as normal. |
| What is recall? | The proportion of actual attacks we catch. |
| Why isn't accuracy enough? | It can hide missed attacks, false alarms or poor results on rare categories. |
| Why mention duplicate patterns? | A model can score well when test patterns resemble training examples. We made exact repetition visible and reported a separate unseen-pattern view. |
| Does it detect attack types? | A separately tested extension predicts ten known categories. Its class-level results are in the report; binary detection remains the main task. |
| Does it detect brand-new attacks? | That has not been established by this benchmark. |
| Does it currently protect the college network? | The current deliverable classifies recorded benchmark flows. Live collection, integration and validation remain future work. |
| What is your contribution? | Implementing and comparing the complete preparation/training/testing pipeline, with careful duplicate handling, reproducible outputs and transparent error analysis. We use established algorithms. |
| Is a model score of 0.9 a guaranteed 90% risk? | Scores rank how strongly the model favours a class. Real-world probability calibration has not been established. |

**Charts to understand before presenting**

1. 02_binary_comparison.png: compare both learning methods and the baselines.
2. 03_binary_confusion.png: explain correct detections, missed attacks and false alarms.
3. 05_detection_by_category.png: explain why the same detector can work differently across attack families.
4. 07_multiclass_confusion.png: use only when discussing the optional attack-type extension.

**Claims supported by this run**

You can say the models were actually trained, tested on a separate partition, saved, reloaded and verified; the code and test predictions are included. You can explain the benchmark's duplicate-pattern limitations and measured errors. A poster should use these measured numbers and state the data/split clearly. The guide determines the final academic requirements.
