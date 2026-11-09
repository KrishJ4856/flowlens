"""Evaluate frozen models. Test outcomes never feed back into training."""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
from pathlib import Path
import sys, json, time, gc, hashlib
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve, brier_score_loss
from ids import read_data, feature_frame, feature_hashes, dump_json, predict_bundle
from train import binary_metrics, multiclass_metrics

ROOT = Path(__file__).resolve().parents[1]


def grouped_intervals(y, pred, hashes, repetitions=400, seed=42):
    """Poisson cluster bootstrap of test confusion counts; fixed trained model."""
    frame = pd.DataFrame({"g": hashes, "tn": ((y == 0) & (pred == 0)).astype(int), "fp": ((y == 0) & (pred == 1)).astype(int), "fn": ((y == 1) & (pred == 0)).astype(int), "tp": ((y == 1) & (pred == 1)).astype(int)})
    counts = frame.groupby("g")[["tn", "fp", "fn", "tp"]].sum().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        tn, fp, fn, tp = rng.poisson(1, len(counts)) @ counts
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1p = 2 * tp / max(2 * tp + fp + fn, 1)
        f1n = 2 * tn / max(2 * tn + fp + fn, 1)
        values.append([(tp + tn) / max(tn + fp + fn + tp, 1), precision, recall, f1p, (f1p + f1n) / 2, fp / max(fp + tn, 1)])
    array = np.asarray(values)
    return {name: {"lower": float(np.quantile(array[:, i], 0.025)), "upper": float(np.quantile(array[:, i], 0.975))} for i, name in enumerate(["accuracy", "attack_precision", "attack_recall", "attack_f1", "macro_f1", "false_positive_rate"])}


def main():
    marker = ROOT / "results/evaluation_complete.json"
    if marker.exists():
        raise RuntimeError("Final evaluation already exists. Use the frozen results; create a new explicitly named experiment for new evaluation.")
    lockpath = ROOT / "results/selection_lock.json"
    lock = json.loads(lockpath.read_text())
    index = json.loads((ROOT / "models/model_index.json").read_text())
    assert len(index) == 7 and sum(i["champion"] for i in index) == 2
    train = read_data(ROOT / "data/raw/train.csv")
    test = read_data(ROOT / "data/raw/test.csv")
    columns = json.loads((ROOT / "data/model_input_schema.json").read_text())["feature_columns"]
    train_hash = feature_hashes(train, columns)
    test_hash = feature_hashes(test, columns)
    unseen = ~np.isin(test_hash, train_hash)
    expected = test.label.to_numpy(dtype=int)
    summaries = []
    outputs = ROOT / "results/predictions"
    outputs.mkdir(exist_ok=True)
    started = time.perf_counter()
    for item in index:
        print("Evaluating frozen model", item["name"], flush=True)
        path = ROOT / item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        bundle = joblib.load(path)
        assert not set(bundle["feature_columns"]) & {"id", "label", "attack_cat"}
        start = time.perf_counter()
        probability = bundle["model"].predict_proba(bundle["preprocessor"].transform(feature_frame(test, columns)))
        seconds = time.perf_counter() - start
        task = item["task"]
        unique = ~pd.DataFrame({"h": test_hash, "target": expected if task == "binary" else test.attack_cat.to_numpy()}).duplicated().to_numpy()
        subsets = {"official_test": np.ones(len(test), dtype=bool), "unseen_features": unseen, "unique_unseen_feature_target": unseen & unique}
        if task == "binary":
            y = expected
            pred = (probability[:, 1] >= bundle["threshold"]).astype(int)
            output = pd.DataFrame({"record_id": test.id, "actual_label": y, "actual_category": test.attack_cat, "prediction": pred, "predicted_label": np.where(pred == 1, "Attack", "Normal"), "attack_score": probability[:, 1], "exact_training_feature_match": ~unseen})
            categories = []
            for category in sorted(test.attack_cat.unique()):
                mask = test.attack_cat.eq(category).to_numpy()
                correct = (pred[mask] == y[mask]).sum()
                categories.append({"model": item["name"], "category": category, "records": int(mask.sum()), "correct": int(correct), "errors": int(mask.sum() - correct), "correct_rate": correct / mask.sum(), "interpretation": "normal acceptance rate" if category == "Normal" else "attack detection recall"})
            pd.DataFrame(categories).to_csv(ROOT / "results" / f"category_detection_{item['name']}.csv", index=False)
            curve = roc_curve(y, probability[:, 1])
            pd.DataFrame({"false_positive_rate": curve[0], "true_positive_rate": curve[1], "threshold": curve[2]}).to_csv(ROOT / "results" / f"roc_{item['name']}.csv", index=False)
            precision, recall, threshold = precision_recall_curve(y, probability[:, 1])
            pd.DataFrame({"precision": precision, "recall": recall, "threshold": np.r_[threshold, np.nan]}).to_csv(ROOT / "results" / f"pr_{item['name']}.csv", index=False)
            if item["family"] != "dummy":
                intervals = grouped_intervals(y, pred, test_hash)
                dump_json(ROOT / "results" / f"intervals_{item['name']}.json", {"method": "400 Poisson exact-feature-group bootstrap replications", "seed": 42, "scope": "descriptive 95% intervals on official test benchmark, conditional on fixed model; excludes training, tuning and real-network uncertainty", "intervals": intervals})
        else:
            y = pd.Categorical(test.attack_cat, categories=bundle["classes"]).codes
            assert (y >= 0).all()
            pred = probability.argmax(axis=1)
            output = pd.DataFrame({"record_id": test.id, "actual_category": test.attack_cat, "predicted_category": np.asarray(bundle["classes"])[pred], "model_score": probability.max(axis=1), "correct": y == pred, "exact_training_feature_match": ~unseen})
        output.to_csv(outputs / f"{item['name']}.csv", index=False)
        for subset_name, mask in subsets.items():
            if task == "binary":
                metrics = binary_metrics(y[mask], probability[mask, 1], bundle["threshold"])
                metrics["brier_score"] = brier_score_loss(y[mask], probability[mask, 1])
                if subset_name == "official_test":
                    default = binary_metrics(y, probability[:, 1], 0.5)
                    dump_json(ROOT / "results" / f"default_threshold_{item['name']}.json", default)
            else:
                metrics = multiclass_metrics(y[mask], probability[mask])
            row = {"model": item["name"], "family": item["family"], "task": task, "subset": subset_name, "records": int(mask.sum()), "selected_on_validation": item["champion"], "predict_all_seconds": seconds, **metrics}
            summaries.append(row)
            labels = [0, 1] if task == "binary" else np.arange(len(bundle["classes"]))
            report = classification_report(y[mask], pred[mask], labels=labels, target_names=bundle["classes"], output_dict=True, zero_division=0)
            dump_json(ROOT / "results" / f"report_{item['name']}_{subset_name}.json", report)
            dump_json(ROOT / "results" / f"confusion_{item['name']}_{subset_name}.json", {"labels": bundle["classes"], "matrix": confusion_matrix(y[mask], pred[mask], labels=labels).tolist(), "rows": "actual", "columns": "predicted"})
            print(f"  {subset_name}: n={mask.sum()}, accuracy={metrics['accuracy']:.5f}, macro F1={metrics['macro_f1']:.5f}", flush=True)
        if item["family"] in ["xgboost", "random_forest"]:
            names = bundle["preprocessor"].get_feature_names_out()
            importance = pd.DataFrame({"encoded_feature": names, "importance": bundle["model"].feature_importances_})
            def original_feature(name):
                short = name.split("__", 1)[1]
                return next((c for c in ["proto", "service", "state"] if short.startswith(c + "_")), short)
            importance["feature"] = importance.encoded_feature.map(original_feature)
            importance.to_csv(ROOT / "results" / f"encoded_importance_{item['name']}.csv", index=False)
            importance.groupby("feature").importance.sum().sort_values(ascending=False).to_csv(ROOT / "results" / f"feature_importance_{item['name']}.csv")
        pd.DataFrame(summaries).to_csv(ROOT / "results/test_metrics.csv", index=False)
        del bundle, probability
        gc.collect()
    # A mixed, answer-free demo file and separate answer key; deterministic random
    # records from the test partition, explicitly marked as evaluation/demo data.
    rng = np.random.default_rng(42)
    sample_idx = np.sort(np.r_[rng.choice(np.flatnonzero((expected == 0) & unseen), 15, replace=False), rng.choice(np.flatnonzero((expected == 1) & unseen), 15, replace=False)])
    demo = test.iloc[sample_idx].copy()
    demo[["id"] + columns].to_csv(ROOT / "data/demo_records.csv", index=False)
    demo[["id", "label", "attack_cat"]].to_csv(ROOT / "data/demo_answers.csv", index=False)
    dump_json(marker, {"selection_lock_sha256": hashlib.sha256(lockpath.read_bytes()).hexdigest(), "elapsed_seconds": time.perf_counter() - started, "models_evaluated": [i["name"] for i in index], "test_rows": len(test), "unseen_feature_test_rows": int(unseen.sum()), "no_retuning_policy": "Models and thresholds were frozen before this evaluation. Report all results including failures.", "demo_policy": "30 fixed random held-out test records with no exact training-feature match, 15 normal and 15 attack; labels excluded from input CSV and stored separately."})
    print("FINAL EVALUATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
