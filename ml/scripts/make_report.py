"""Create readable reports and publication-sized charts from frozen results."""
from __future__ import annotations
from pathlib import Path
import sys, json, textwrap, datetime
from zoneinfo import ZoneInfo
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
BLUE, TEAL, ORANGE, GREY = "#22577A", "#008A85", "#CC6B23", "#7B8493"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})


def load(name):
    return json.loads((RESULTS / name).read_text())


def save(fig, name, note=""):
    if note:
        fig.text(0.02, 0.01, note, fontsize=8.5, color="#46515E", va="bottom")
    fig.tight_layout(rect=(0, 0.06 if note else 0, 1, 0.96))
    fig.savefig(FIGURES / f"{name}.png", dpi=190, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def percentage(v):
    return f"{v * 100:.2f}%"


def table(headers, rows):
    return "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n" + "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in rows)


def main():
    FIGURES.mkdir(exist_ok=True)
    audit = load("data_audit.json")
    lock = load("selection_lock.json")
    verification = load("verification.json")
    training = load("training_summary.json")
    metrics = pd.read_csv(RESULTS / "test_metrics.csv")
    index = json.loads((ROOT / "models/model_index.json").read_text())
    binary = metrics[(metrics.task == "binary") & (metrics.subset == "official_test")].copy()
    order = ["dummy", "logistic_regression", "random_forest", "xgboost"]
    binary["order"] = binary.family.map({v: i for i, v in enumerate(order)})
    binary = binary.sort_values("order")
    champion = lock["winners"]["binary"]["champion"]
    multi_champion = lock["winners"]["multiclass"]["champion"]
    selected = binary[binary.model == champion].iloc[0]
    novel = metrics[(metrics.model == champion) & (metrics.subset == "unseen_features")].iloc[0]
    unique_novel = metrics[(metrics.model == champion) & (metrics.subset == "unique_unseen_feature_target")].iloc[0]
    multi_row = metrics[(metrics.model == multi_champion) & (metrics.subset == "official_test")].iloc[0]
    display = {"dummy": "Majority baseline", "logistic_regression": "Logistic regression", "random_forest": "Random Forest", "xgboost": "XGBoost"}
    colors = {"dummy": GREY, "logistic_regression": ORANGE, "random_forest": TEAL, "xgboost": BLUE}

    # Class imbalance with both partitions visible.
    counts = pd.read_csv(RESULTS / "class_distribution.csv")
    cats = ["Normal", "Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms"]
    fig, ax = plt.subplots(figsize=(12, 5.4))
    xx = np.arange(len(cats))
    for shift, part, color in [(-0.19, "train", BLUE), (0.19, "test", TEAL)]:
        vals = counts[counts.partition == part].set_index("category").loc[cats, "count"]
        bars = ax.bar(xx + shift, vals, 0.37, label=part.title(), color=color)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val*1.08, f"{val:,}", ha="center", fontsize=7.5, rotation=60)
    ax.set_yscale("log"); ax.set_ylim(20, 170000)
    ax.set_xticks(xx, cats, rotation=30, ha="right")
    ax.set_ylabel("Records (log scale)"); ax.set_title("UNSW-NB15 partitions: large differences in class frequency")
    ax.legend(frameon=False)
    save(fig, "01_class_distribution", "Training: 175,341 rows. Test: 82,332 rows. Rare categories require class-level evaluation.")

    fig, axes = plt.subplots(1, 3, figsize=(12, 5.4))
    for ax, key, title in zip(axes, ["macro_f1", "attack_recall", "false_positive_rate"], ["Macro F1 (higher is better)", "Attacks detected (higher is better)", "Normal records flagged (lower is better)"]):
        vals = binary[key].to_numpy() * 100
        ax.barh(np.arange(len(binary)), vals, color=[colors[f] for f in binary.family])
        ax.set_yticks(np.arange(len(binary)), [display[f] for f in binary.family] if ax is axes[0] else [])
        ax.set_xlim(0, 114); ax.set_xticks([0, 25, 50, 75, 100]); ax.xaxis.set_major_formatter(PercentFormatter(100))
        ax.set_title(title, fontsize=10)
        for yy, value in enumerate(vals): ax.text(value + 1.2, yy, f"{value:.1f}%", va="center", fontsize=9)
    fig.suptitle("Final binary detection results on all 82,332 test records", fontsize=14, weight="bold")
    save(fig, "02_binary_comparison", f"Models and thresholds selected on grouped validation only. Validation-selected champion: {champion}.")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.4))
    for ax, subset, title in zip(axes, ["official_test", "unseen_features"], ["Complete separate test partition", "No exact training-feature match"]):
        record = load(f"confusion_{champion}_{subset}.json")
        matrix = np.asarray(record["matrix"])
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i,j]:,}\n{normalized[i,j]*100:.1f}% of row", ha="center", va="center", color="white" if normalized[i,j] > 0.6 else "#172A3A", fontsize=12)
        ax.set_xticks([0, 1], ["Normal", "Attack"]); ax.set_yticks([0, 1], ["Normal", "Attack"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title, fontsize=11)
    fig.suptitle(f"Where the binary detector succeeds and fails: {champion}", fontsize=13, weight="bold")
    save(fig, "03_binary_confusion", "Off-diagonal counts are mistakes. Top-right: false alarms. Bottom-left: missed attacks.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.1))
    for _, row in binary[binary.family != "dummy"].iterrows():
        roc = pd.read_csv(RESULTS / f"roc_{row.model}.csv")
        pr = pd.read_csv(RESULTS / f"pr_{row.model}.csv")
        axes[0].plot(roc.false_positive_rate, roc.true_positive_rate, color=colors[row.family], label=f"{display[row.family]}: AUC {row.roc_auc:.3f}")
        axes[1].plot(pr.recall, pr.precision, color=colors[row.family], label=f"{display[row.family]}: AP {row.average_precision:.3f}")
    axes[0].plot([0,1],[0,1],"--",color=GREY,linewidth=1)
    prevalence = audit["partitions"]["test"]["binary_counts"]["1"] / 82332
    axes[1].axhline(prevalence, linestyle="--", color=GREY, linewidth=1, label="Attack prevalence")
    axes[0].set(xlabel="False positive rate", ylabel="Attack recall", title="ROC curves")
    axes[1].set(xlabel="Attack recall", ylabel="Attack precision", title="Precision-recall curves")
    for ax in axes:
        ax.set_xlim(0,1); ax.set_ylim(0,1.015); ax.legend(fontsize=8, loc="lower left", frameon=True)
    save(fig, "04_binary_curves", "Ranking curves use raw model scores; they do not establish probability calibration or real-network performance.")

    trees = binary[binary.family.isin(["xgboost", "random_forest"])]
    rates = np.vstack([pd.read_csv(RESULTS / f"category_detection_{row.model}.csv").set_index("category").loc[cats, "correct_rate"].to_numpy() for _, row in trees.iterrows()])
    fig, ax = plt.subplots(figsize=(12, 3.7))
    image = ax.imshow(rates, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(np.arange(len(trees)), [display[f] for f in trees.family]); ax.set_xticks(np.arange(len(cats)), cats, rotation=30, ha="right")
    for i in range(rates.shape[0]):
        for j in range(rates.shape[1]): ax.text(j,i,f"{rates[i,j]*100:.1f}%",ha="center",va="center",color="white" if rates[i,j] > .7 else "black",fontsize=9)
    ax.set_title("Binary detection across each known attack category")
    save(fig, "05_detection_by_category", "Normal column: normal records accepted. Other columns: attack records detected; this does not mean the attack type was identified.")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    threshold = pd.read_csv(RESULTS / f"thresholds_{champion}.csv")
    for key, title, color in [("attack_recall", "Attacks detected", TEAL), ("false_positive_rate", "False alarms", ORANGE), ("macro_f1", "Macro F1", BLUE)]:
        ax.plot(threshold.threshold, threshold[key]*100, marker="o", markersize=3, label=title, color=color)
    ax.axvline(float(selected.threshold), linestyle="--", color=GREY, label=f"Chosen: {selected.threshold:.2f}")
    ax.set(xlabel="Score threshold for flagging an attack", ylabel="Percentage", title="Threshold choice was made using validation data only")
    ax.legend(frameon=False); ax.set_ylim(0,105)
    save(fig, "06_validation_threshold", "This chart uses development validation records, not final test results.")

    record = load(f"confusion_{multi_champion}_official_test.json")
    matrix = np.asarray(record["matrix"]); normalized = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(11, 9.3))
    ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            if matrix[i,j]: ax.text(j,i,f"{normalized[i,j]*100:.0f}%\n({matrix[i,j]:,})",ha="center",va="center",fontsize=6.8,color="white" if normalized[i,j]>.6 else "#233445")
    ax.set_xticks(range(10), record["labels"], rotation=45, ha="right"); ax.set_yticks(range(10), record["labels"])
    ax.set_xlabel("Predicted category"); ax.set_ylabel("Actual category")
    ax.set_title(f"Attack-category extension: {multi_champion}\nRow-normalized confusion matrix; counts in parentheses")
    save(fig, "07_multiclass_confusion", "This harder task predicts a specific known category. Inspect rare-category errors before making claims.")

    fig, axes = plt.subplots(1,2,figsize=(12,5.3))
    for ax, (_, row) in zip(axes, trees.iterrows()):
        imp = pd.read_csv(RESULTS / f"feature_importance_{row.model}.csv").head(12).iloc[::-1]
        ax.barh(imp.feature, imp.importance, color=colors[row.family]); ax.set_title(display[row.family]); ax.set_xlabel("Aggregated built-in importance")
    fig.suptitle("Inputs used strongly by each fitted binary model",weight="bold")
    save(fig, "08_feature_importance", "Training-derived importance, aggregated across one-hot categories. These rankings are not causal explanations.")

    stability = pd.read_csv(RESULTS / "grouped_stability.csv")
    fig, ax = plt.subplots(figsize=(8.5, 4.7))
    for shift, family in [(-.17,"random_forest"),(.17,"xgboost")]:
        subset = stability[stability.family == family].sort_values("fold")
        ax.bar(subset.fold + shift, subset.macro_f1 * 100, .32, color=colors[family], label=display[family])
    ax.set_xticks([1,2,3],["Fold 1","Fold 2","Fold 3"]); ax.set_ylim(0,100); ax.set_ylabel("Macro F1 (%)"); ax.set_title("Three-fold grouped stability check within training data")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    save(fig, "09_grouped_stability", "Fresh preprocessing per fold; threshold 0.5. Selected settings reused: diagnostic only, not nested CV or a new test estimate.")

    # Reports use numerical outputs directly, so no manually copied scores drift.
    source = json.loads((ROOT / "data/source_manifest.json").read_text())
    when = datetime.datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %B %Y, %H:%M IST")
    champion_item = next(i for i in index if i["task"] == "binary" and i["champion"])
    model_table = table(["Model", "Accuracy", "Attack precision", "Attacks detected", "Macro F1", "False alarms"], [[display[r.family], percentage(r.accuracy), percentage(r.attack_precision), percentage(r.attack_recall), percentage(r.macro_f1), percentage(r.false_positive_rate)] for _, r in binary.iterrows()])
    subset_table = table(["View", "Records", "Accuracy", "Attack recall", "Macro F1", "False alarms"], [[label, f"{int(r.records):,}", percentage(r.accuracy), percentage(r.attack_recall), percentage(r.macro_f1), percentage(r.false_positive_rate)] for label, r in [("Entire test partition", selected), ("No exact feature match in training", novel), ("Unique unseen feature + answer", unique_novel)]])
    multi = metrics[(metrics.task == "multiclass") & (metrics.subset == "official_test")]
    multi_table = table(["Model", "Accuracy", "Macro F1", "Weighted F1"], [[r.model, percentage(r.accuracy), percentage(r.macro_f1), percentage(r.weighted_f1)] for _, r in multi.iterrows()])
    interval = load(f"intervals_{champion}.json")["intervals"]
    int_table = table(["Metric", "Estimate", "Descriptive 95% interval"], [[name.replace("_", " "), percentage(selected[name]), f"{percentage(bounds['lower'])}–{percentage(bounds['upper'])}"] for name,bounds in interval.items()])
    category = pd.read_csv(RESULTS / f"category_detection_{champion}.csv")
    cat_table = table(["Category", "Test records", "Correct binary decision", "Errors"], [[r.category, f"{r.records:,}", percentage(r.correct_rate), f"{r.errors:,}"] for _,r in category.iterrows()])
    validation = lock["winners"]["binary"]["family_winners"][champion_item["family"]]["validation_metrics"]
    warnings = [(r["name"], w) for r in load("validation_candidates.json") for w in r["warnings"]]
    report = f'''**Network Intrusion Detection with XGBoost and Random Forest**

Completed experiment report — {when}

This project trains software to classify recorded network flows as normal or attack. A separate extension predicts one of ten known categories. The deliverable includes fitted models, preprocessing, test predictions, source code, a walkthrough notebook and verified figures. The scope ends before building an app or connecting to live network traffic.

**What was built and the headline result**

The binary champion selected using grouped validation is **{champion}**, with score threshold **{float(selected.threshold):.2f}**. On the complete separate 82,332-row test partition it achieved **{percentage(selected.accuracy)} accuracy**, **{percentage(selected.attack_recall)} attack recall**, **{percentage(selected.attack_precision)} attack precision**, **{percentage(selected.macro_f1)} macro F1** and **{percentage(selected.false_positive_rate)} false positive rate**. These values are measured outputs, not targets or invented sample results.

It caught **{int(selected.tp):,}** attacks, missed **{int(selected.fn):,}**, correctly accepted **{int(selected.tn):,}** normal records and incorrectly flagged **{int(selected.fp):,}** normal records. The nonzero errors belong in the presentation. A benchmark score is not a claim of equal performance on a campus network.

**Dataset and provenance**

Canonical training data: the team's UNSW attachment, 175,341 rows. Canonical test data: a separate 82,332-row partition. Each contains 45 columns: a local ID, 42 predictors and two answers. ID, label and attack_cat are excluded from every model input. Original UNSW documentation and its dataset publications should be cited. The original SharePoint download redirected to sign-in; the complementary data was fetched from a public mirror pinned to commit `{source['mirror_commit']}`.

The mirror filenames were reversed relative to the official row counts and the team's selected training data. Its 175,341-row file was exactly equal to the team's attachment after parsing, including row order. Canonical names were resolved by contents. Hashes and original URLs are in data/source_manifest.json; row counts and audits are in results/data_audit.json. Agreement with a public mirror does not establish author-signed provenance.

![Class distribution](figures/01_class_distribution.png)

**Why the split matters**

Training contains {audit['partitions']['train']['duplicate_feature_rows']:,} repeated feature rows beyond the first occurrence, and {audit['partitions']['train']['feature_groups_with_conflicting_binary_labels']:,} feature groups have conflicting binary labels. A naive random row split can place identical examples in fitting and validation data. We grouped identical 42-feature records, then used a class-stratified approximately 80/20 development split. Fitting: 140,273 rows; validation: 35,068; exact-feature overlap: zero. Preprocessing is fitted on fitting rows only during selection. The full methodology and predefined grids are in METHODS.md and config/experiment.json.

**Preparation and model selection**

Numeric missing values use fitting-data medians; categorical missing values use fitting-data modes. proto, service and state are one-hot encoded; unknown categories are tolerated. Tree inputs remain unscaled, and the logistic baseline uses additional training-fitted scaling. No answer columns, local ID, oversampled test data or synthetic attacks enter the inputs. Existing frequencies and ambiguous labels are retained and documented.

Four binary XGBoost and three binary Random Forest candidates were compared with logistic and majority-prior baselines. Four tree candidates plus a majority baseline were tested for the categorical extension: {training['candidate_count']} candidate fits in total. Selection used validation macro F1; thresholds came from a predefined validation-only grid. XGBoost stopped according to validation loss, then its selected tree count was reused for a complete-training refit. The selection record was frozen before final scoring. Selected settings are in results/selection_lock.json. No configuration or threshold was changed to improve final test results.

The binary champion had validation macro F1 {percentage(validation['macro_f1'])}, versus test macro F1 {percentage(selected.macro_f1)}. This difference is part of the result and should not be hidden.

**Complete test comparison**

{model_table}

Attack precision means the proportion of flags that were actual attacks. Attack recall means the proportion of all attacks detected. False positive rate means the proportion of normal records wrongly flagged. Macro F1 averages class-level F1 equally, which makes both normal and attack performance matter. Selecting the champion used validation data; the table may show a different model looking stronger on an individual final test metric.

![Binary comparison](figures/02_binary_comparison.png)

![Binary confusion matrices](figures/03_binary_confusion.png)

**Dataset overlap and additional views**

Exactly {audit['cross_partition']['test_rows_with_exact_training_feature_match']:,} test rows have a 42-feature match somewhere in training. Therefore a source-file boundary alone does not guarantee every test pattern is new. All test rows remain in the complete benchmark result. Additional fixed views make the limitation visible:

{subset_table}

The unique unseen view counts each feature-plus-binary-answer combination once. It changes weighting and should be reported separately, not substituted silently for the complete test result. Exact-feature separation still cannot guarantee separation by real connection, time or host; these reduced CSVs lack the identifiers needed to prove that stronger independence.

**Detection by attack category**

{cat_table}

For Normal the percentage is normal acceptance. For every attack category it is binary attack recall. A binary model correctly flagging a DoS record has not necessarily identified that record as DoS.

![Category detection](figures/05_detection_by_category.png)

**Score curves, thresholds and uncertainty**

![Ranking curves](figures/04_binary_curves.png)

![Validation threshold trade-offs](figures/06_validation_threshold.png)

ROC AUC and average precision describe score ranking across thresholds. Scores are uncalibrated model estimates; a displayed 0.9 score is not a validated 90% real-world attack probability. Default-threshold results are retained alongside the selected-threshold results in results/default_threshold_*.json.

The following intervals use 400 Poisson bootstrap replications grouped by exact test feature vector:

{int_table}

They describe uncertainty within this fixed benchmark with a fixed model. They exclude model-training uncertainty, tuning uncertainty, temporal drift and host/session correlation.

**Training-only stability check**

![Grouped stability check](figures/09_grouped_stability.png)

Both selected binary tree configurations were re-fitted using fresh preprocessing in each of three grouped folds inside the training partition. Settings were already selected from the same training partition, so this is a diagnostic, not nested cross-validation or an independent test score. It used threshold 0.5 throughout and did not change the winner. Detailed fold outcomes are in results/grouped_stability.csv.

**Attack-category extension**

{multi_table}

The validation-selected categorical champion is **{multi_champion}**: test accuracy **{percentage(multi_row.accuracy)}**, macro F1 **{percentage(multi_row.macro_f1)}**. Accuracy and weighted F1 can mask poor rare-category identification; the full per-class precision/recall/F1 and confusion counts are included. This extension is an additional known-category classifier, not an open-set detector for previously unseen attack families.

![Categorical confusion matrix](figures/07_multiclass_confusion.png)

**Model interpretation**

![Feature importances](figures/08_feature_importance.png)

These built-in training importances are aggregated to original inputs. Correlated measurements and feature cardinality can affect their rankings. They help describe how the model uses the dataset; they do not establish causes or prove that a particular communication is malicious.

**What has been checked**

{verification['checks_passed']} automated integrity/inference checks passed. They cover frozen experiment and model checksums, split separation, answer/ID exclusion, invariance to column order, missing-column rejection, invalid numeric inputs, unknown categorical values, missing-value handling, training-only preprocessing statistics, complete prediction coverage, independent reconciliation of confusion counts, native XGBoost export equivalence and fresh-process command-line prediction. The walkthrough notebook's execution is checked separately. See results/verification.json.

Candidate warnings recorded during selection: {len(warnings)}. {('See validation_candidates.json for the exact warning text.' if warnings else 'No warnings were captured from candidate model fitting.')}

**Scope, limitations and next steps**

This is a working offline benchmark project with reusable model files. It does not inspect live Wi-Fi, capture packets, block attacks or establish deployment readiness. Dataset collection, duplicate patterns and train/test distribution differences limit real-network generalization. Conflicting identical-feature labels constrain what a deterministic classifier can reproduce from these measurements. Rare classes require particular caution. Final thresholds reflect a benchmark macro-F1 objective rather than an operational security cost budget.

The next product step is an interface that loads the saved binary bundle and applies the exact stored preprocessing to schema-compatible records. For the upcoming presentation, demonstrate answer-free sample prediction, explain the training/test separation and show real error counts. Obtain the guide's approval for the project scope and poster; no faculty approval is implied by this technical run.

**Sources and citations**

Dataset description and original publications: [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset). In the project bibliography include Moustafa & Slay (2015), *UNSW-NB15: A comprehensive data set for network intrusion detection systems*, and the additional dataset citations requested on the source page.

Algorithms: Chen & Guestrin (2016), [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754); Breiman (2001), [Random Forests](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf).

Method guidance: [Scikit-learn leakage prevention](https://scikit-learn.org/stable/common_pitfalls.html), [grouped stratification](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html), [threshold selection](https://scikit-learn.org/stable/modules/classification_threshold.html), [evaluation metrics](https://scikit-learn.org/stable/modules/model_evaluation.html), and [XGBoost early stopping](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html).

Data retrieval mirror: [{source['mirror_repository']}]({source['mirror_repository']}); the full pinned download URLs and checksums accompany the data. The mirror supplies files; the measured results in this report come from this local experiment.
'''
    (ROOT / "REPORT.md").write_text(report)

    start_here = f'''**Start here — your network attack detection project**

The work before building the app is complete. You now have trained detectors, a separate test evaluation, reusable preparation code, saved model files, an executed walkthrough notebook and charts for the presentation. The main task is deciding whether a recorded network communication is normal or an attack. Identifying the exact attack category is a separately tested extension.

For the hand-held explanation, read **LEARN_AND_RUN.md**. It follows actual inputs through the code and outputs, explains the training implementation and shows how to check the results yourself. To run a five-record example in this folder, use `bash run_example.sh`; add `--verify` to run all integrity checks too.

**The result in ordinary language**

The model selected during practice testing is **{champion}**. On the separate final test, it got **{percentage(selected.accuracy)}** of the records right. It caught **{int(selected.tp):,} of {int(selected.tp + selected.fn):,} attacks** and missed **{int(selected.fn):,}**. It wrongly flagged **{int(selected.fp):,} of {int(selected.tn + selected.fp):,} normal records**.

These are real measured results. The mistakes should be explained along with the successes. This experiment checks recorded benchmark data; performance on actual college Wi-Fi has not been established.

**What happened while training**

1. Verified your group's training file and obtained the complementary test partition.
2. Checked duplicate examples and kept identical examples together during the practice split.
3. Separated network measurements from their answers. Neither answer column was given to the model as input.
4. Prepared the measurements, using only practice training data to learn preparation settings.
5. Trained and compared XGBoost, Random Forest and simple baselines.
6. Chose settings using practice validation results, then froze those choices.
7. Trained the chosen configurations on the complete training file and took the final test.
8. Saved everything needed to run the detector again without retraining.

**An important dataset issue**

Some test records repeat patterns already present in training. I kept the complete test result and also checked the **{int(novel.records):,}** test records without an exact training match. On that view the selected model's accuracy was **{percentage(novel.accuracy)}**, with **{percentage(novel.attack_recall)}** of attacks detected. The full report explains this difference. This careful check is a useful part of the project story.

**What to open first**

- REPORT.md: real results and all charts, with explanations and limitations.
- notebooks/Network_IDS_Walkthrough.ipynb: step-by-step review of the actual data, saved results and model prediction.
- PRESENTATION_NOTES.md: a short speaking script, likely questions and suitable claims.
- figures/: nine charts, each in PNG and SVG form for a poster.
- README.md: commands for using and reproducing the work.

The trained model is the saved result of teaching. scripts/predict.py loads it, reads measurements and produces predictions. data/demo_records.csv contains 30 held-out examples with the answer columns removed; data/demo_answers.csv is the separate answer key. The demo sample is deliberately mixed (15 normal, 15 attack), so it is a demonstration, not an estimate of real-world attack prevalence.

**What is still left for your group**

Learn the explanation, run the sample yourselves, decide speaking roles, confirm the current mentor/approval contact and prepare the poster. The poster soft copy is due **19 November 2026**; the presentation is **26 November 2026**. Ask the guide about any separate submission format or demo requirement. App construction remains outside this handover, as requested.

Every method here is an established method implemented using Python libraries. The contribution is building, comparing and carefully evaluating the complete pipeline. No new ML algorithm is claimed.
'''
    (ROOT / "START_HERE.md").write_text(start_here)
    presentation = f'''**Presentation notes: Network Intrusion Detection using Machine Learning**

These are rehearsal notes, not an announced college speaking-time requirement. Read START_HERE.md first, then practise running the answer-free sample in the notebook. Every number below comes from the actual frozen experiment.

**Suggested opening**

“Our project detects suspicious activity in recorded network communications. Each communication is described using measurements such as duration, packet counts and bytes transferred. We trained XGBoost and Random Forest to learn patterns from examples labelled normal or attack.”

**Explain what the model does**

“The model receives 42 network measurements. The correct-answer columns are kept separate. During training it sees examples and their answers. During testing it receives measurements only, makes predictions and is then scored against the hidden answers.”

**Explain the experiment**

“We used the group's 175,341-row training file and a separate 82,332-row test partition from UNSW-NB15. We chose model settings using a practice validation split inside training. Many records repeat, so identical measurement patterns were kept together during that practice split. After selection, we froze the settings and took the final test.”

**Present the real result**

“The validation-selected binary model was {champion}. On the complete separate test partition it achieved {percentage(selected.accuracy)} accuracy and detected {percentage(selected.attack_recall)} of attacks. It missed {int(selected.fn):,} attacks and incorrectly flagged {int(selected.fp):,} normal communications. We report these errors because accuracy alone does not describe a detector.”

“The original partitions contain exact repeated measurement patterns across training and test. We also evaluated the {int(novel.records):,} test examples with no exact training match. On that subset, accuracy was {percentage(novel.accuracy)} and attack recall was {percentage(novel.attack_recall)}.”

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
'''
    (ROOT / "PRESENTATION_NOTES.md").write_text(presentation)

    poster = f'''**Poster content draft — for guide review**

**Suggested title:** Machine Learning-Based Network Intrusion Detection using XGBoost and Random Forest

**Problem:** Suspicious network communications can be difficult to identify from large volumes of recorded traffic measurements. A classifier can assist by flagging patterns associated with labelled attacks.

**Objective:** Build and evaluate a reproducible system that predicts normal versus attack from network-flow measurements, comparing XGBoost and Random Forest under a shared preparation and evaluation protocol.

**Dataset:** UNSW-NB15 reduced partition files; 175,341 training records, 82,332 separate test records; 42 predictors. Local ID and both answer fields excluded. Source: UNSW Canberra; original publication Moustafa & Slay (2015). The data retrieval mirror and source hashes are documented in the project package.

**Methodology:** Audit → grouped fit/validation split inside training → train-only preparation → candidate training → validation-only model/threshold selection → full-training refit → frozen separate-test evaluation.

**Key preparation:** Numeric imputation using training medians, categorical imputation using training modes, one-hot encoding of protocol/service/state, and preservation of saved preparation rules for inference. Exact duplicate feature vectors are grouped together during development validation.

**Results:** Validation-selected binary model: {champion}; complete test accuracy {percentage(selected.accuracy)}, attack precision {percentage(selected.attack_precision)}, attack recall {percentage(selected.attack_recall)}, macro F1 {percentage(selected.macro_f1)}, false positive rate {percentage(selected.false_positive_rate)}. Test confusion counts: {int(selected.tp):,} attacks detected, {int(selected.fn):,} attacks missed, {int(selected.tn):,} normal records accepted, {int(selected.fp):,} normal records flagged.

**Evaluation limitation:** {audit['cross_partition']['test_rows_with_exact_training_feature_match']:,} test records have an exact training feature match. On the {int(novel.records):,} test records without such a match, accuracy is {percentage(novel.accuracy)} and attack recall is {percentage(novel.attack_recall)}. This additional subset is reported separately from the complete benchmark. Independence by host/session and near-duplicate separation are not established.

**Conclusion:** The project delivers trained offline classifiers, stored preprocessing, a reproducible comparison and explicit error analysis. Evaluation reveals useful detection capability and remaining missed attacks/false alarms on this benchmark. An app and live network integration remain future work.

**Suggested figures:** figures/02_binary_comparison.png and figures/03_binary_confusion.png; optionally figures/05_detection_by_category.png. PNG and SVG versions are available. Add the final approved guide and team details when designing the poster.

**Optional extension:** Ten-category classification is evaluated separately in REPORT.md. Keep its scores separate from normal/attack detection.

This is content for the group's review, not a designed or approved poster. Submit the poster soft copy to the guide by 19 November 2026; print the 3 ft × 2 ft flex only after approval. Presentation: 26 November 2026.
'''
    (ROOT / "POSTER_CONTENT.md").write_text(poster)

    readme = f'''**Network IDS project package**

Start with START_HERE.md for the plain-language summary, **LEARN_AND_RUN.md** for the guided input/code/output walkthrough and REPORT.md for measured results. Run `bash run_example.sh` from this folder for a checked five-record demonstration. Presentation talking points are in PRESENTATION_NOTES.md. POSTER_CONTENT.md contains a draft based on real numbers. No app is included.

**What is included**

| Path | Purpose |
| --- | --- |
| data/raw/train.csv, test.csv | Canonical raw partitions, preserved separately |
| data/source_manifest.json | URLs, pinned mirror commit, original filename resolution and hashes |
| data/column_schema.csv | Every source column and whether it is input, answer or excluded ID |
| data/development_split.csv | Reproducible grouped fit/validation row assignment |
| data/prepared/train.npz, test.npz | Numeric inputs produced by frozen complete-training-fitted preparation |
| data/model_input_schema.json | Required original inputs and encoded names |
| data/demo_records.csv, demo_answers.csv | Answer-free 30-record demonstration and separate answer key |
| src/ids.py | Shared preparation, parsing and saved-bundle prediction |
| scripts/ | Auditing, model selection/refit, stability, evaluation, verification and reporting |
| config/experiment.json | Candidate grids, seeds, selection rules and test policy |
| models/ | Preprocessor, seven fitted bundles, native XGBoost exports and model index |
| results/ | Every candidate, selection lock, final metrics, predictions, errors and checks |
| figures/ | Nine chart sets in PNG and SVG |
| notebooks/Network_IDS_Walkthrough.ipynb | Genuinely executed review and inference notebook |
| METHODS.md | Protocol, caveats and primary documentation references |

The selected binary model is `{champion}`. It was selected using validation, before final test scoring. Seven bundles include two main binary models, logistic/majority baselines and three categorical-extension models. Choose a binary bundle for normal/attack prediction and a multiclass bundle for known category prediction.

**Environment**

Tested on Linux with Python {training['versions']['python']}, scikit-learn {training['versions']['scikit_learn']} and XGBoost {training['versions']['xgboost']}, using CPU and at most two learning-library threads. The exact environment is in requirements-lock.txt. This run used a CPU-only XGBoost package and does not require a GPU.

After extracting the whole folder, open a terminal in this folder. For a fresh environment with uv installed:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
```

Or create a Python 3.11 virtual environment and install the same lock file with pip. Keep the project folder intact: the model bundles need the helper module in src/ids.py, and the scripts set its import path automatically.

**Predict without retraining**

```bash
.venv/bin/python scripts/predict.py --input data/demo_records.csv --output results/my_demo_predictions.csv
```

The command automatically loads the validation-selected binary champion. To use another delivered bundle:

```bash
.venv/bin/python scripts/predict.py --model models/{champion}.joblib --input data/demo_records.csv --output results/my_demo_predictions.csv
```

The expected input contains all 42 original feature columns, in any order. ID is optional and is copied only for tracing. Answer columns are ignored; prediction needs only measurements. A missing required column or invalid numeric text causes a clear error. Previously unseen categorical values and missing numeric values use the stored preparation rules. Binary output uses the bundle's frozen threshold; calling the estimator's plain predict method directly would use its own default threshold and can give a different result.

Binary scores are named attack_probability in the reusable helper's output and attack_score in the evaluation exports; both are uncalibrated class scores. Do not describe them as proven real-world risk percentages.

**Inspect prepared inputs**

```python
import numpy as np
arrays = np.load('data/prepared/train.npz', allow_pickle=False)
X = arrays['X']
y = arrays['label']
print(X.shape, y.shape)
```

These prepared matrices use the final full-training-fitted preparation. For leakage-aware model selection or cross-validation, use the scripts that freshly fit preparation inside the fitting fold. Do not use the final prepared matrix to claim that CV preparation was fold-specific.

**Verify the saved work**

```bash
.venv/bin/python scripts/verify.py
```

This rechecks hashes, target exclusion, split separation, preprocessing statistics, model serialization, recorded confusion counts and fresh-process prediction. It does not retrain models or tune on the test data.

**Repeat the entire experiment**

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python scripts/reproduce.py --output ../network-ids-rerun
```

The destination must not already exist. The script preserves the delivered experiment, copies the included data/code/config and runs the full audit, training, three-fold stability, frozen test evaluation, prepared-data export, verification, reports and executed notebook. Allow time and disk space for training; the original measured training duration is in results/training_summary.json. Exact reproducibility is intended with the recorded environment and seeds; other versions/platforms may vary.

Final evaluation is protected by a completion marker so an accidental second run does not silently overwrite the locked test report. A deliberate new experiment should use a fresh destination.

**Use the walkthrough notebook**

Open notebooks/Network_IDS_Walkthrough.ipynb in Jupyter, VS Code or Colab after making the complete extracted project available. It already contains executed outputs. For a local Jupyter kernel, select the installed project environment. The notebook reviews saved results and demonstrates prediction; it does not automatically repeat the full long training search.

**Measured scope**

This is offline analysis of schema-compatible benchmark flow records. An app, raw-packet extraction and live-network validation are future work. Results include dataset overlap, conflicting labels, rare categories and distribution differences. All comparisons are evidence for this benchmark, not a claim of universal intrusion-detection performance. The project uses established algorithms rather than claiming a new algorithm.

**Citations**

The dataset's original publications and academic-use citation instructions are linked on the [official UNSW page](https://research.unsw.edu.au/projects/unsw-nb15-dataset). Algorithms: [Chen & Guestrin, 2016](https://arxiv.org/abs/1603.02754) and [Breiman, 2001](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf). Full methods and retrieval provenance accompany the package.
'''
    (ROOT / "README.md").write_text(readme)
    card = f'''**Model card: recorded-flow network intrusion detection**

**Identity and intended use**

Primary task: binary classification of UNSW-NB15-style flow measurements as Normal (0) or Attack (1). Intended use: a student benchmark demonstration and research pipeline. Selected bundle: models/{champion}.joblib. Family: {champion_item['family']}. Selection rule: grouped validation macro F1. Frozen score threshold: {float(selected.threshold):.2f}. The model is already trained; inference does not need an answer column or retraining.

**Training and selection**

Source: team's 175,341-row training attachment. Development fit/validation: 140,273 / 35,068 rows with exact-feature groups kept intact. Hyperparameters, weights, early-stopping tree counts and thresholds were chosen from training/validation only. The final bundle is refitted using the complete training partition. Full settings: results/selection_lock.json. Versions: requirements-lock.txt. Model-file hashes: models/model_index.json.

**Input contract**

All 42 named measurements in data/model_input_schema.json are required. Column order may vary. ID is optional and has no influence on predictions. label and attack_cat are answers and are ignored. proto, service and state are categorical. Other inputs are numeric; invalid numeric text raises an error. Missing numeric/categorical values use saved training-fitted imputation, and unknown categorical values are encoded using all-zero indicators for that categorical field. This tolerance avoids an input crash; it does not establish reliable prediction for unfamiliar protocols or shifted data.

The CSV contains precomputed flow measurements. A raw packet capture, arbitrary system log or Wi-Fi packet stream does not satisfy this contract without a compatible extraction process.

**Output contract**

The reusable helper produces a class prediction, readable label, uncalibrated attack_probability score and optional tracing ID. Evaluation exports call the same kind of score attack_score. The decision is score >= {float(selected.threshold):.2f}. Use src/ids.py predict_bundle or scripts/predict.py so the stored preparation and threshold are applied together. A direct call to the estimator's default predict method may use a different threshold.

**Measured outcomes**

All 82,332 separate test rows: accuracy {percentage(selected.accuracy)}, attack recall {percentage(selected.attack_recall)}, attack precision {percentage(selected.attack_precision)}, macro F1 {percentage(selected.macro_f1)}, false positive rate {percentage(selected.false_positive_rate)}. Missed attacks: {int(selected.fn):,}. False alarms: {int(selected.fp):,}.

The 73,791 test rows with no exact training feature match: accuracy {percentage(novel.accuracy)}, attack recall {percentage(novel.attack_recall)}, macro F1 {percentage(novel.macro_f1)}. The additional unique feature-plus-answer subset and every baseline are in results/test_metrics.csv. Conditional benchmark intervals are in results/intervals_{champion}.json.

**Known limits**

The benchmark contains repeated patterns, conflicting identical-feature labels, class imbalance and a difference between training and test distributions. Exact-feature grouping does not prove separation by host, session, collection time or near-duplicate patterns. Real-network prevalence and costs can differ materially. Model scores are not calibrated operational risk estimates. No live deployment, intrusion blocking, novel-attack guarantee, future traffic guarantee or campus-network performance claim is established.

**Categorical extension**

Separate champion: {multi_champion}. Ten known categories include Normal and nine attack families. Its class-specific mistakes are in REPORT.md and results/report_{multi_champion}_official_test.json. It is not a replacement for the binary score and must not be presented using the binary accuracy figure.

**Verification and next interface step**

Stored hashes, answer/ID exclusion, column-order invariance, input validation, missing/unknown handling, train-only preparation statistics and fresh-process inference were verified. A future app should load the existing bundle once, validate the input schema, call predict_bundle and display scores as model scores. Keep the documented training/test caveats alongside any results page. Further validation is needed before any live operational use.
'''
    (ROOT / "MODEL_CARD.md").write_text(card)
    print("Reports, presentation content and nine chart sets created.", flush=True)


if __name__ == "__main__":
    main()
