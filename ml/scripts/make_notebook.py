"""Build and genuinely execute a short review/inference notebook."""
from pathlib import Path
import sys, json
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    cells = []
    def md(s): cells.append(nbf.v4.new_markdown_cell(s))
    def code(s): cells.append(nbf.v4.new_code_cell(s))
    md("""# Network attack detection: a walkthrough of our completed experiment

This notebook reviews the actual UNSW-NB15 experiment and uses a saved detector.
It is genuinely executed after training, with outputs from the delivered files.
It does **not** repeat the long training search every time you open it.

The complete training and testing implementation is in `scripts/train.py`,
`scripts/stability.py` and `scripts/evaluate.py`. To repeat the experiment from
scratch while preserving the original, use `scripts/reproduce.py` as explained
in README.md.

Main question: **Is this recorded network communication normal or an attack?**
The optional second task predicts one of ten known categories. An app and live
network capture are outside the current scope.""")
    code("""from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
from IPython.display import display, Image

# This notebook lives in project/notebooks/. It also works when the notebook
# runner starts in the project folder.
candidates = [Path.cwd(), Path.cwd().parent, Path.cwd() / 'network-ids']
ROOT = next((p for p in candidates if (p / 'models/model_index.json').exists()), None)
if ROOT is None:
    raise RuntimeError('Extract the complete project and run from its folder or notebooks folder.')
sys.path.insert(0, str(ROOT / 'src'))
from ids import read_data, predict_bundle, feature_frame
import joblib
print('Project:', ROOT.name)
print('Python:', sys.version.split()[0])""")
    md("""## 1. What data do we have?

Every row describes one network flow. Columns describe measurements such as
duration, packets and bytes. `label` is the binary answer: 0 means normal and
1 means attack. `attack_cat` gives a known category. Neither answer is an input.

The supplied training partition has 175,341 rows; the separate test partition
has 82,332. The source manifest explains the reversed filenames found in the
retrieval mirror.""")
    code("""train = read_data(ROOT / 'data/raw/train.csv')
test = read_data(ROOT / 'data/raw/test.csv')
print('Training:', train.shape, 'Test:', test.shape)
display(train[['dur', 'proto', 'service', 'spkts', 'dpkts', 'label', 'attack_cat']].head())
display(train['attack_cat'].value_counts().rename('training_records').to_frame())""")
    code("display(Image(filename=str(ROOT / 'figures/01_class_distribution.png')))")
    md("""## 2. Why didn't we just mix and randomly split every row?

Many measurements repeat. A copy of the same example in both practice training
and practice testing can make results look stronger. We kept identical
measurement vectors together for validation. The original separate test file
also shares some feature patterns with training, so the report includes both
the full test and a view with no exact training-feature match.""")
    code("""audit = json.loads((ROOT / 'results/data_audit.json').read_text())
print('Practice fit rows:', audit['development_split']['fit_rows'])
print('Practice validation rows:', audit['development_split']['validation_rows'])
print('Identical feature overlap in development:', audit['development_split']['exact_feature_overlap'])
print('Test rows matching training features:', audit['cross_partition']['test_rows_with_exact_training_feature_match'])
print('Test rows with no exact match:', audit['cross_partition']['test_rows_without_exact_training_feature_match'])""")
    md("""## 3. Preparing the inputs

The saved preparation object fills missing numeric values using training
medians and converts protocol/service/state categories into numeric columns.
It learns those settings from fitting data only during model selection; after
selection it is refitted on the complete training file. The test file is only
transformed using already learned settings.

`id`, `label` and `attack_cat` are excluded from every model's inputs.""")
    code("""schema = json.loads((ROOT / 'data/model_input_schema.json').read_text())
print('Original predictors:', len(schema['feature_columns']))
print('Encoded numeric inputs:', schema['encoded_features'])
print('Excluded:', schema['excluded'])
assert not set(schema['feature_columns']) & {'id', 'label', 'attack_cat'}""")
    md("""## 4. What does teaching actually look like?

The central operations used in the training scripts are:

```python
model = XGBClassifier(**chosen_settings)
model.fit(prepared_training_measurements, training_answers)
scores = model.predict_proba(prepared_unseen_measurements)
```

XGBoost learns decision trees that progressively address mistakes. Random
Forest combines many randomized decision trees. The methods come from Python
libraries; the fitted model files contain what those methods learned here.

We tried predefined settings using development validation, then froze the
winner and threshold before evaluating the separate test file. This notebook
loads the completed artifacts rather than retraining the search.""")
    code("""candidates = json.loads((ROOT / 'results/validation_candidates.json').read_text())
comparison = pd.DataFrame([{'candidate': c['name'], 'task': c['task'],
                           'validation_macro_f1': c['validation_metrics']['macro_f1'],
                           'training_seconds': c['training_seconds']}
                          for c in candidates])
display(comparison.round(4))
lock = json.loads((ROOT / 'results/selection_lock.json').read_text())
print('Binary champion selected on validation:', lock['winners']['binary']['champion'])""")
    md("""## 5. How did the finished models do on the separate test?

Accuracy: all correct decisions. Attack recall: how many actual attacks we
caught. Attack precision: how many flags were truly attacks. False positive
rate: how many normal records were wrongly flagged. Macro F1 gives equal
weight to the two class-level F1 scores. These are distinct views of quality.""")
    code("""metrics = pd.read_csv(ROOT / 'results/test_metrics.csv')
binary = metrics[(metrics.task == 'binary') & (metrics.subset == 'official_test')]
display(binary[['model', 'accuracy', 'attack_precision', 'attack_recall',
                'macro_f1', 'false_positive_rate', 'tn', 'fp', 'fn', 'tp']].round(4))
display(Image(filename=str(ROOT / 'figures/03_binary_confusion.png')))""")
    code("""champion_name = lock['winners']['binary']['champion']
display(metrics[(metrics.model == champion_name)][['subset', 'records', 'accuracy',
                                                 'attack_recall', 'macro_f1',
                                                 'false_positive_rate']].round(4))""")
    md("""## 6. Run the saved detector yourself

These 30 demonstration records come from the held-out test file, with the
answer columns removed. The answer key is opened only after prediction. The
demo contains 15 normal and 15 attack records, so it is a demonstration rather
than an estimate of real-world attack frequency. The score is uncalibrated.""")
    code("""index = json.loads((ROOT / 'models/model_index.json').read_text())
chosen = next(i for i in index if i['task'] == 'binary' and i['champion'])
bundle = joblib.load(ROOT / chosen['path'])
demo = read_data(ROOT / 'data/demo_records.csv')
assert 'label' not in demo and 'attack_cat' not in demo
predictions = predict_bundle(bundle, demo)
display(predictions.head(10))
print('Flags:', int(predictions['prediction'].sum()), 'out of', len(predictions))""")
    code("""answers = read_data(ROOT / 'data/demo_answers.csv')
review = predictions.merge(answers, left_on='record_id', right_on='id', validate='one_to_one')
review['correct'] = review['prediction'] == review['label']
display(review[['record_id', 'predicted_label', 'label', 'attack_cat', 'correct']])
print('Demo correct decisions:', int(review.correct.sum()), '/', len(review))""")
    md("""## 6a. Follow five records through every stage

LEARN_AND_RUN.md explains the following steps line by line. The small file has
five complete records and no answer columns. Preparation turns 42 original
measurements into 194 numeric inputs for this saved model. We open the answer
key after predicting. The separate error_examples.csv deliberately contains
known mistakes for teaching; it is not a representative accuracy sample.

From a terminal in the project folder, `bash run_example.sh` predicts and checks
the five-record example. `bash run_example.sh --verify` also runs all 102 checks.""")
    code("""small = read_data(ROOT / 'data/simple_sample.csv')
print('CSV shape including tracing ID:', small.shape)
display(small[['id', 'dur', 'proto', 'spkts', 'dpkts', 'sbytes', 'dbytes']])
measurements = feature_frame(small, bundle['feature_columns'])
prepared = bundle['preprocessor'].transform(measurements)
print('Original measurement matrix:', measurements.shape)
print('Prepared numeric matrix:', prepared.shape, prepared.dtype)
small_predictions = predict_bundle(bundle, small)
display(small_predictions)
small_answers = read_data(ROOT / 'data/simple_sample_answers.csv')
checked = small_predictions.merge(small_answers, left_on='record_id', right_on='id', validate='one_to_one')
checked['correct'] = checked.prediction.eq(checked.label)
display(checked[['record_id', 'predicted_label', 'label', 'attack_cat', 'correct']])""")
    code("""mistakes = read_data(ROOT / 'data/error_examples.csv')
mistake_predictions = predict_bundle(bundle, mistakes)
mistake_answers = read_data(ROOT / 'data/error_examples_answers.csv')
mistake_review = mistake_predictions.merge(mistake_answers, left_on='record_id', right_on='id', validate='one_to_one')
mistake_review['correct'] = mistake_review.prediction.eq(mistake_review.label)
display(mistake_review[['record_id', 'attack_probability', 'predicted_label', 'label', 'attack_cat', 'correct']])
print('These rows were explicitly chosen to illustrate errors, not estimate accuracy.')""")
    md("""## 7. Optional extension: naming the attack category

This model predicts a specific known category. It is a separate task from
normal/attack detection. Class-level results matter, especially for rare
categories. Neither this extension nor the binary benchmark proves that the
system can reliably identify a completely new kind of attack.""")
    code("""display(metrics[(metrics.task == 'multiclass') &
                (metrics.subset == 'official_test')][['model', 'accuracy', 'macro_f1', 'weighted_f1']].round(4))
display(Image(filename=str(ROOT / 'figures/07_multiclass_confusion.png')))""")
    md("""## 8. What can we honestly say at the presentation?

We implemented, compared and tested ML classifiers for recorded network flows.
We kept answer columns out of the inputs, fitted preparation using training
data, checked repeated patterns and retained errors in the final report.

We apply established algorithms. We have not established live network
protection, novel-attack recognition or equal performance on college Wi-Fi.
The next step is an interface that uses these saved models.

Read PRESENTATION_NOTES.md for a speaking script and likely questions.
See METHODS.md and REPORT.md for sources, exact settings and limitations.""")
    notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python", "version": sys.version.split()[0]}})
    output = ROOT / "notebooks/Network_IDS_Walkthrough.ipynb"
    nbf.write(notebook, output)
    client = NotebookClient(notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(ROOT / "notebooks")}})
    # Use this experiment's interpreter, not an unrelated globally installed kernel.
    client.execute()
    nbf.write(notebook, output)
    executed = sum(c.cell_type == "code" and c.execution_count is not None for c in notebook.cells)
    errors = [o for c in notebook.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.output_type == "error"]
    assert not errors
    (ROOT / "results/notebook_execution.json").write_text(json.dumps({"executed_code_cells": executed, "errors": 0, "notebook": str(output.relative_to(ROOT)), "method": "Executed by nbclient with experiment Python interpreter; outputs saved."}, indent=2) + "\n")
    print(f"Notebook executed: {executed} code cells, zero errors.", flush=True)


if __name__ == "__main__":
    main()
