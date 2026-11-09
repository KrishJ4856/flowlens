"""Generate a beginner's input/code/output walkthrough with a real small run."""
from pathlib import Path
import sys, json, subprocess
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
import numpy as np
import joblib
from ids import read_data, feature_frame

ROOT = Path(__file__).resolve().parents[1]


def table(headers, rows):
    return '| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join(['---']*len(headers)) + ' |\n' + '\n'.join('| '+' | '.join(str(c) for c in row)+' |' for row in rows)


def main():
    index = json.loads((ROOT/'models/model_index.json').read_text())
    record = next(v for v in index if v['task']=='binary' and v['champion'])
    bundle = joblib.load(ROOT/record['path'])
    lock = json.loads((ROOT/'results/selection_lock.json').read_text())
    validation = lock['winners']['binary']['family_winners'][record['family']]['validation_metrics']
    all_metrics = pd.read_csv(ROOT/'results/test_metrics.csv')
    m = all_metrics[(all_metrics.model==record['name']) & (all_metrics.subset=='official_test')].iloc[0]
    schema = json.loads((ROOT/'data/model_input_schema.json').read_text())
    demo = read_data(ROOT/'data/demo_records.csv')
    answers = read_data(ROOT/'data/demo_answers.csv')
    joined = demo.merge(answers,on='id',validate='one_to_one')
    sample = pd.concat([joined[joined.label==0].head(3),joined[joined.label==1].head(2)],ignore_index=True)
    assert len(sample)==5 and sample.label.value_counts().to_dict()=={0:3,1:2}
    sample[['id']+schema['feature_columns']].to_csv(ROOT/'data/simple_sample.csv',index=False)
    sample[['id','label','attack_cat']].to_csv(ROOT/'data/simple_sample_answers.csv',index=False)
    commands = [
        [sys.executable,str(ROOT/'scripts/predict.py'),'--input',str(ROOT/'data/simple_sample.csv'),'--output',str(ROOT/'results/simple_sample_predictions.csv')],
        [sys.executable,str(ROOT/'scripts/check_predictions.py'),'--predictions',str(ROOT/'results/simple_sample_predictions.csv'),'--answers',str(ROOT/'data/simple_sample_answers.csv')],
    ]
    outputs=[]
    for command in commands:
        result=subprocess.run(command,check=True,capture_output=True,text=True)
        outputs.append(result.stdout)
    (ROOT/'results/simple_example_output.txt').write_text('\n'.join(outputs))
    p=pd.read_csv(ROOT/'results/simple_sample_predictions.csv')
    checked=p.merge(sample[['id','label','attack_cat']],left_on='record_id',right_on='id',validate='one_to_one')
    checked['correct']=checked.prediction.eq(checked.label)
    frozen=pd.read_csv(ROOT/'results/predictions'/f"{record['name']}.csv")
    false_flags=frozen[(frozen.actual_label==0) & (frozen.prediction==1) & (~frozen.exact_training_feature_match)].sort_values('attack_score',ascending=False).head(3)
    missed=frozen[(frozen.actual_label==1) & (frozen.prediction==0) & (~frozen.exact_training_feature_match)].sort_values('attack_score').head(2)
    error_ids=pd.concat([false_flags,missed]).record_id.to_list()
    raw_test=read_data(ROOT/'data/raw/test.csv').set_index('id',drop=False)
    errors=raw_test.loc[error_ids]
    errors[['id']+schema['feature_columns']].to_csv(ROOT/'data/error_examples.csv',index=False)
    errors[['id','label','attack_cat']].to_csv(ROOT/'data/error_examples_answers.csv',index=False)
    error_commands=[
        [sys.executable,str(ROOT/'scripts/predict.py'),'--input',str(ROOT/'data/error_examples.csv'),'--output',str(ROOT/'results/error_examples_predictions.csv')],
        [sys.executable,str(ROOT/'scripts/check_predictions.py'),'--predictions',str(ROOT/'results/error_examples_predictions.csv'),'--answers',str(ROOT/'data/error_examples_answers.csv')],
    ]
    error_outputs=[subprocess.run(cmd,check=True,capture_output=True,text=True).stdout for cmd in error_commands]
    (ROOT/'results/error_example_output.txt').write_text('\n'.join(error_outputs))
    input_table=table(['Record ID','Duration','Protocol','Source packets','Destination packets','Source bytes','Destination bytes'],[[r.id,r.dur,r.proto,r.spkts,r.dpkts,r.sbytes,r.dbytes] for _,r in sample.iterrows()])
    output_table=table(['Record ID','Attack score','Model prediction','Actual answer','Correct?'],[[int(r.record_id),f'{r.attack_probability:.6f}',r.predicted_label,'Normal' if r.label==0 else 'Attack','Yes' if r.correct else 'No'] for _,r in checked.iterrows()])
    example_matrix=bundle['preprocessor'].transform(feature_frame(sample.iloc[[0]],bundle['feature_columns']))
    cats=bundle['preprocessor'].named_transformers_['categorical'].named_steps['onehot'].categories_
    category_table=table(['Original text column','Training categories','Encoded indicator columns'],[[c,len(v),len(v)] for c,v in zip(['proto','service','state'],cats)])
    comparison=table(['Metric','Development validation','Separate final test'],[[k.replace('_',' '),f'{validation[k]*100:.2f}%',f'{m[k]*100:.2f}%'] for k in ['accuracy','attack_recall','macro_f1','false_positive_rate']])
    settings=json.dumps(lock['winners']['binary']['family_winners']['xgboost']['settings'],indent=2)
    guide=f'''**Learn and run the project: a guided walkthrough**

You can read this without knowing ML. We will follow one record from a CSV into the saved detector, inspect its prediction, then check the answer. After that, we will walk backwards through how the detector was trained and how the full result was verified. The full implementation is included, so each explanation points to the code that actually does the work.

**Step 1 — Know what we built**

The program receives measurements describing a recorded network communication. It predicts **Normal** or **Attack**. For example, measurements include communication duration, packet counts, bytes transferred and protocol. A packet is a small piece of transmitted data. A flow is a summary of a communication between endpoints.

A **model** is the saved result of learning patterns from examples. XGBoost and Random Forest are the learning methods supplied by Python libraries. We created fresh models, trained them with your dataset, compared them and saved the fitted results. The present project checks recorded data. Live packet collection and an app are future stages.

**Step 2 — See the small input we will use**

Open data/simple_sample.csv. It contains **five real held-out test records**, with no exact training-feature match. Each has the complete 42 required measurements plus an optional tracing ID. The correct-answer columns have been removed. Three records are normal and two are attacks according to the separate key; selection was based on those class counts, not on whether predictions would succeed.

Here are a few measurements from those five records; the file itself contains the rest:

{input_table}

The **ID** connects an output to its input row. It is excluded from model inputs. The detector needs all 42 named measurements listed in data/model_input_schema.json. This five-row file is a small, complete input example. DATA_GUIDE.md explains the measurements; you can also inspect every value of a record with `frame.iloc[0].to_frame('value')` in Python.

**Step 3 — Run it on this laptop**

I already installed a separate Python environment for this project. Open a terminal and run:

```bash
cd "{ROOT}"
bash run_example.sh
```

The launcher finds the existing environment, loads the saved binary model, predicts the five records and then checks its predictions against the separate key. No retraining is needed. Its generated file is results/my_simple_predictions.csv.

If you move the project to another location or machine, create the environment using README.md first. Then run the same launcher from the extracted project folder. The full source and data must remain together.

**Step 4 — Understand what the command actually does**

The launcher runs scripts/predict.py, then scripts/check_predictions.py. The central prediction operation in src/ids.py is:

```python
measurements = feature_frame(frame, bundle['feature_columns'])
prepared = bundle['preprocessor'].transform(measurements)
scores = bundle['model'].predict_proba(prepared)
predictions = (scores[:, 1] >= bundle['threshold']).astype(int)
```

Read this line by line:

1. `feature_frame` selects the 42 required input columns, puts them in the expected order, trims category text and checks numeric parsing. Extra ID/answer columns have no influence.
2. `transform` applies preparation rules already learned during training. It fills missing values using stored training statistics and encodes category words numerically.
3. `predict_proba` runs the fitted trees and produces class scores. For binary prediction, `scores[:, 1]` selects the score for Attack for every row. `:` means every row; `1` means the class numbered 1.
4. A score greater than or equal to the saved threshold becomes 1, meaning Attack. The other class is 0, Normal. Here the frozen threshold is **{bundle['threshold']:.2f}**.

The first sample record becomes a matrix of shape **{example_matrix.shape}**: one row and {example_matrix.shape[1]} numeric model inputs. All five sample records become five rows with the same number of numeric inputs. The same stored preparation object and model are loaded together.

**Step 5 — Read the actual output**

This table was generated by running the saved model on those exact five records:

{output_table}

`attack_probability` is the helper's column name for the raw Attack score. It is an uncalibrated model estimate. A score of 0.9 means the model strongly favours Attack; calibration as a 90% real-world probability has not been established. Full-evaluation files name this score `attack_score`.

The model supplies the prediction and score. The Actual answer and Correct columns are added afterwards by reading data/simple_sample_answers.csv. Opening an answer key afterwards lets us check the prediction without giving the answer to the detector. The example got **{int(checked.correct.sum())} out of 5** decisions right. A five-record demonstration is too small to establish overall accuracy; the full test below is the performance evidence.

**Step 6 — Separate prediction from checking it yourself**

After using the launcher once, you can run its two operations separately. In the commands below, replace `PYTHON` with your environment's Python executable. On this laptop it is:

```text
{Path(sys.executable)}
```

First predict, using the answer-free input:

```bash
PYTHON scripts/predict.py --input data/simple_sample.csv --output results/my_simple_predictions.csv
```

Then check, opening the answer key only after prediction:

```bash
PYTHON scripts/check_predictions.py --predictions results/my_simple_predictions.csv --answers data/simple_sample_answers.csv
```

The checker matches records by ID, rejects duplicate/missing IDs or mismatched answer coverage, and prints correct decisions, attacks caught and normal records falsely flagged. The actual output of my run is saved in results/simple_example_output.txt. You can open your prediction CSV and compare the score with {bundle['threshold']:.2f} manually for each record.

To compare the saved Random Forest, use:

```bash
PYTHON scripts/predict.py --model models/rf_depth24.joblib --input data/simple_sample.csv --output results/my_rf_predictions.csv
PYTHON scripts/check_predictions.py --predictions results/my_rf_predictions.csv --answers data/simple_sample_answers.csv
```

You can also use the larger 30-record demo_records.csv and its separate demo_answers.csv. For your own data, provide all 42 required measurements. It can be a single row. An answer is optional for prediction; checking correctness needs a reliable separate label. If you manually change a record's measurements, the original answer key only describes the original record.

**Step 7 — Understand how the original training file was separated**

Your file has 175,341 rows and 45 columns:

- 42 measurement columns: the model's questions.
- label: the binary answer, 0=Normal or 1=Attack.
- attack_cat: a more detailed answer naming the known category.
- id: a local tracing number.

Both answer columns and ID are excluded. For binary learning, label is supplied separately as y. A simplified version is:

```python
columns = [c for c in train.columns if c not in ('id', 'label', 'attack_cat')]
X = train[columns]
y = train['label']
```

`X` means the input measurements. `y` means the teaching/checking answers. The actual implementation adds strict parsing and schema checks in src/ids.py.

**Step 8 — Understand fit, validation and test**

We used three roles for the records:

| Role | Size | Purpose |
| --- | --- | --- |
| Practice fitting | 140,273 | Teach each candidate model and learn its preparation rules |
| Practice validation | 35,068 | Choose settings and the score threshold |
| Separate final test | 82,332 | Measure the frozen choices on a separate source partition |

The first two come only from the 175,341-row training file. The final test is the complementary partition, obtained and checked separately. After settings were chosen, the final models were refitted on all 175,341 training rows. The separate test never taught the model or chose its threshold.

Many records repeat measurements. scripts/audit.py groups identical 42-feature vectors and uses StratifiedGroupKFold so a group stays entirely in fitting or validation. “Stratified” means trying to preserve category proportions; “grouped” means related copies travel together. The split has zero exact-feature overlap. The saved row assignment is data/development_split.csv.

**Step 9 — Understand preparation technically**

The pipeline uses scikit-learn's ColumnTransformer to apply different operations to different columns:

- 39 numeric columns: SimpleImputer fills missing values with the fitting-data median. Values are converted to float32, a numeric representation that saves memory.
- proto, service, state: missing values use the fitting-data mode, then OneHotEncoder makes category indicator columns. handle_unknown='ignore' allows an unfamiliar value to be represented without crashing.
- Tree models use unscaled numeric values. The logistic baseline additionally uses StandardScaler learned from fitting data.

The supplied partitions had no missing or infinite numeric values; the saved missing-value rules still support future inputs. Preparation learns only from fitting rows during model selection. It is subsequently refitted on the complete training partition for final inference, then the test is transformed using those frozen settings.

For the final fitted preparation:

{category_table}

The 39 numeric inputs plus {sum(len(v) for v in cats)} indicator columns give **{schema['encoded_features']}** numeric model inputs. Development fitting had 193 encoded inputs; complete-training refitting has 194 because the category set grows by one. Each final model is freshly refitted alongside its matching complete-training preparation, so their input widths agree.

Prepared arrays are in data/prepared/train.npz and test.npz, with X and answers stored as distinct arrays. These complete-training-fitted arrays support final train/test use. For validation or cross-validation, refit preparation inside each fitting fold, as the provided scripts do.

**Step 10 — Understand how the model was taught**

The central instruction is:

```python
model.fit(prepared_fit_measurements, fitting_answers)
```

The model learns relationships between measurements and known labels. XGBoost progressively builds decision trees; later rounds improve the combined model. Random Forest builds randomized trees and combines their class scores. The code supplies the algorithm and settings; learning supplies the fitted decisions.

scripts/train.py compared four binary XGBoost settings, three Random Forest settings and two baselines. It selected using validation macro F1, which gives both classes equal weight through their F1 scores. The threshold came from the predefined 0.10–0.90 validation grid. XGBoost monitored validation loss with early-stopping patience 40. The selected binary model used 700 boosting rounds.

The selected XGBoost settings are:

```json
{settings}
```

`max_depth` limits how many decisions a tree can chain. `learning_rate` controls each round's contribution. `n_estimators` is the number of boosting rounds. `subsample` uses a fraction of fitting rows per round; `colsample_bytree` uses a fraction of encoded inputs per tree. `min_child_weight` and `reg_lambda` help constrain overly specific patterns. Random seeds are fixed for reproducibility, and learning uses two CPU threads.

The selected Random Forest uses 200 trees, depth at most 24, at least two training examples per leaf, and a randomized subset of encoded inputs at each split. All candidates, weights, training durations, warnings and validation metrics are in results/validation_candidates.json. Chosen settings were written to selection_lock.json before final testing.

scripts/stability.py also ran three grouped training-only folds with fresh preparation per fold. Those are stability diagnostics for already selected settings; they are not an independent final test.

**Step 11 — Understand the full test result**

XGBoost was selected using practice validation. The final test showed these results for that frozen model:

{comparison}

Its actual test decisions were:

| Actual answer | Predicted Normal | Predicted Attack |
| --- | --- | --- |
| Normal | {int(m.tn):,} correct normal decisions | {int(m.fp):,} false alarms |
| Attack | {int(m.fn):,} missed attacks | {int(m.tp):,} caught attacks |

The arithmetic is directly checkable:

- Accuracy = all correct decisions / all records = ({int(m.tp):,} + {int(m.tn):,}) / 82,332 = **{m.accuracy*100:.2f}%**.
- Attack recall = caught attacks / actual attacks = {int(m.tp):,} / {int(m.tp+m.fn):,} = **{m.attack_recall*100:.2f}%**.
- Attack precision = caught attacks / all flags = {int(m.tp):,} / {int(m.tp+m.fp):,} = **{m.attack_precision*100:.2f}%**.
- False positive rate = false alarms / actual normal records = {int(m.fp):,} / {int(m.tn+m.fp):,} = **{m.false_positive_rate*100:.2f}%**.

These rates have different denominators. Here, detecting roughly 98 of every 100 attacks comes with falsely flagging roughly 26 of every 100 normal records. That trade-off belongs in your explanation. Random Forest scored slightly higher on this final test; the report includes both and preserves the earlier validation selection.

To see concrete mistakes, data/error_examples.csv contains three false alarms and two missed attacks selected from the frozen test predictions, with answers in error_examples_answers.csv. These were deliberately selected to illustrate errors; they are not an accuracy sample. Their observed run is saved in results/error_example_output.txt. Run them yourself with:

```bash
PYTHON scripts/predict.py --input data/error_examples.csv --output results/my_error_predictions.csv
PYTHON scripts/check_predictions.py --predictions results/my_error_predictions.csv --answers data/error_examples_answers.csv
```

Some of these wrong predictions have strong scores. That is another reason to treat scores as learned estimates and check actual errors carefully.

Some test records repeat training patterns. REPORT.md also reports 73,791 test rows without an exact training-feature match, and a further version that counts repeated feature/answer pairs once. Keeping these views separate makes it harder to mistake repeated benchmark patterns for broad real-network ability.

**Step 12 — Verify the project independently**

For the full automated checks, run:

```bash
bash run_example.sh --verify
```

This also runs the small example. The verifier checks stored file/model hashes, separation of fitting/validation groups, exclusion of both answer fields and ID, reordered columns, missing/invalid inputs, unfamiliar categories, saved training-only preparation statistics, complete test prediction coverage, confusion counts and native XGBoost export equivalence. It starts a fresh process to check that prediction works after loading the saved model. The current run passed **102 checks**.

You can inspect results/verification.json for every check. Parallel Random Forest calculations can differ in the last floating-point bits, so score invariance uses tolerance 1e-12 while requiring identical prepared inputs and matching class decisions.

To manually cross-check the reported totals in a notebook or Python session:

```python
import pandas as pd
p = pd.read_csv('results/predictions/{record['name']}.csv')
tp = ((p.actual_label == 1) & (p.prediction == 1)).sum()
fn = ((p.actual_label == 1) & (p.prediction == 0)).sum()
fp = ((p.actual_label == 0) & (p.prediction == 1)).sum()
tn = ((p.actual_label == 0) & (p.prediction == 0)).sum()
print(tp, fn, fp, tn)
print('Accuracy:', (tp + tn) / len(p))
print('Attack recall:', tp / (tp + fn))
print('False positive rate:', fp / (fp + tn))
```

To repeat every training and evaluation step, use scripts/reproduce.py as described in README.md. It requires a new output folder so the delivered experiment remains intact. The original candidate search and full-data refits took about {json.loads((ROOT/'results/training_summary.json').read_text())['elapsed_seconds']/60:.1f} minutes on this laptop, followed by stability and verification. A complete fresh run takes longer than the five-record prediction.

**Step 13 — Know which files implement each operation**

| File | What to look for |
| --- | --- |
| src/ids.py | read_data, feature_frame, make_preprocessor, predict_bundle |
| scripts/audit.py | Source checks, duplicate/label audits and grouped development split |
| scripts/train.py | Candidate fitting, validation threshold search, selection lock and complete-training refit |
| scripts/stability.py | Three-fold training-only stability diagnostics |
| scripts/evaluate.py | Frozen final evaluation, extra overlap views, reports and test prediction exports |
| scripts/export_prepared.py | Saved numeric matrices, with answers separate |
| scripts/predict.py | Load a fitted bundle and predict a CSV |
| scripts/check_predictions.py | Compare predictions against a separate answer key |
| scripts/verify.py | Integrity and inference verification |
| config/experiment.json | Exact candidate settings, seeds and rules |

The executed walkthrough notebook provides the same progression with live Python cells and saved figures. REPORT.md contains the full metrics, MODEL_CARD.md defines the saved detector's input/output contract, and PRESENTATION_NOTES.md helps you explain it. The category-prediction extension has separate models and scores; normal/attack detection remains the main demonstration.
'''
    (ROOT/'LEARN_AND_RUN.md').write_text(guide)
    check_info={'records':5,'normal':3,'attack':2,'correct':int(checked.correct.sum()),'input_answer_columns_absent':not {'label','attack_cat'} & set(read_data(ROOT/'data/simple_sample.csv').columns),'commands_executed':4,'output':'results/simple_sample_predictions.csv','selected_error_examples':5,'error_example_output':'results/error_example_output.txt'}
    (ROOT/'results/simple_example_verification.json').write_text(json.dumps(check_info,indent=2)+'\n')
    print('Beginner walkthrough and checked five-record example created.',flush=True)


if __name__=='__main__':
    main()
