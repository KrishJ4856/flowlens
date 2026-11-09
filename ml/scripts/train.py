"""Select candidates on grouped validation and refit before opening test data.

This script intentionally never reads data/raw/test.csv. The selection lock is
written before evaluation is allowed. Run from any working directory.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
from pathlib import Path
import sys, json, time, logging, gc, platform, hashlib, warnings
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
import joblib
import sklearn
import xgboost
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix, roc_auc_score, average_precision_score, log_loss
from sklearn.preprocessing import LabelEncoder
from ids import read_data, feature_columns, feature_frame, make_preprocessor, dump_json

ROOT = Path(__file__).resolve().parents[1]


def binary_metrics(y, score, threshold):
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
        "attack_precision": precision_score(y, pred, zero_division=0), "attack_recall": recall_score(y, pred, zero_division=0),
        "attack_f1": f1_score(y, pred, zero_division=0), "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "false_positive_rate": fp / (tn + fp), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "roc_auc": roc_auc_score(y, score), "average_precision": average_precision_score(y, score),
        "log_loss": log_loss(y, np.clip(score, 1e-7, 1 - 1e-7), labels=[0, 1]), "threshold": float(threshold),
    }


def multiclass_metrics(y, probability):
    pred = probability.argmax(axis=1)
    return {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
            "log_loss": log_loss(y, probability, labels=np.arange(probability.shape[1]))}


def fit_candidate(spec, family, task, xfit, yfit, xval, yval, seed, threads):
    settings = dict(spec)
    name = settings.pop("name")
    weighted = settings.pop("sample_weight_scheme", None)
    kwargs = {}
    if family == "xgboost":
        model = XGBClassifier(**settings, objective="binary:logistic" if task == "binary" else "multi:softprob",
                              tree_method="hist", n_jobs=threads, random_state=seed,
                              eval_metric="logloss" if task == "binary" else "mlogloss", early_stopping_rounds=40)
        kwargs = {"eval_set": [(xval, yval)], "verbose": False}
        if weighted:
            counts = np.bincount(yfit)
            weights = np.sqrt(len(yfit) / (len(counts) * counts[yfit]))
            kwargs["sample_weight"] = weights / weights.mean()
    elif family == "random_forest":
        model = RandomForestClassifier(**settings, random_state=seed, n_jobs=threads)
    elif family == "logistic_regression":
        model = LogisticRegression(C=0.1, max_iter=1200, solver="lbfgs", random_state=seed)
    elif family == "dummy":
        model = DummyClassifier(strategy="prior", random_state=seed)
    else:
        raise ValueError(family)
    start = time.perf_counter()
    model.fit(xfit, yfit, **kwargs)
    duration = time.perf_counter() - start
    probability = model.predict_proba(xval)
    selected_settings = dict(settings)
    if family == "xgboost":
        selected_settings["n_estimators"] = int(model.best_iteration + 1)
    if weighted:
        selected_settings["sample_weight_scheme"] = weighted
    return model, probability, duration, selected_settings


def refit_model(family, task, settings, x, y, seed, threads):
    settings = dict(settings)
    weighted = settings.pop("sample_weight_scheme", None)
    kwargs = {}
    if family == "xgboost":
        model = XGBClassifier(**settings, objective="binary:logistic" if task == "binary" else "multi:softprob", tree_method="hist", n_jobs=threads, random_state=seed, eval_metric="logloss" if task == "binary" else "mlogloss")
        if weighted:
            counts = np.bincount(y)
            weights = np.sqrt(len(y) / (len(counts) * counts[y]))
            kwargs["sample_weight"] = weights / weights.mean()
    elif family == "random_forest":
        model = RandomForestClassifier(**settings, random_state=seed, n_jobs=threads)
    elif family == "logistic_regression":
        model = LogisticRegression(C=0.1, max_iter=1200, solver="lbfgs", random_state=seed)
    else:
        model = DummyClassifier(strategy="prior", random_state=seed)
    model.fit(x, y, **kwargs)
    return model


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.FileHandler(ROOT / "results/training.log"), logging.StreamHandler(sys.stdout)])
    config = json.loads((ROOT / "config/experiment.json").read_text())
    seed, threads = config["seed"], config["threads"]
    start_all = time.perf_counter()
    train = read_data(ROOT / "data/raw/train.csv")
    columns = feature_columns(train)
    assert len(columns) == 42 and not set(columns) & {"id", "label", "attack_cat"}
    assignment = pd.read_csv(ROOT / "data/development_split.csv")
    assert np.array_equal(assignment.record_id.to_numpy(), train.id.to_numpy())
    fit_idx = np.flatnonzero(assignment.split.eq("fit"))
    val_idx = np.flatnonzero(assignment.split.eq("validation"))
    x = feature_frame(train, columns)
    encoder = LabelEncoder().fit(train.attack_cat)
    all_candidates = []
    winners = {}
    validation_data = {}
    preprocessor = make_preprocessor(columns)
    xfit = preprocessor.fit_transform(x.iloc[fit_idx])
    xval = preprocessor.transform(x.iloc[val_idx])
    logging.info("Development matrices: fit %s, validation %s; dtype=%s", xfit.shape, xval.shape, xfit.dtype)
    dump_json(ROOT / "data/development_encoding.json", {"feature_names": preprocessor.get_feature_names_out().tolist(), "encoded_features": xfit.shape[1], "fitted_on": "fit rows only"})
    for task in ["binary", "multiclass"]:
        y = train.label.to_numpy(dtype=int) if task == "binary" else encoder.transform(train.attack_cat)
        specs = [("dummy", {"name": "dummy_majority" if task == "binary" else "multiclass_dummy"})]
        if task == "binary":
            specs += [("logistic_regression", {"name": "logistic_baseline"})]
        specs += [("xgboost", s) for s in config[f"{task}_xgboost"]]
        specs += [("random_forest", s) for s in config[f"{task}_random_forest"]]
        task_candidates = []
        for family, spec in specs:
            logging.info("Training %s / %s", task, spec["name"])
            scaled = family == "logistic_regression"
            if scaled:
                prep_lr = make_preprocessor(columns, scale=True)
                xf = prep_lr.fit_transform(x.iloc[fit_idx]); xv = prep_lr.transform(x.iloc[val_idx])
            else:
                xf, xv = xfit, xval
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model, probability, seconds, settings = fit_candidate(spec, family, task, xf, y[fit_idx], xv, y[val_idx], seed, threads)
            warning_text = [str(w.message) for w in caught]
            if task == "binary":
                threshold_rows = [binary_metrics(y[val_idx], probability[:, 1], t) for t in ([0.5] if family == "dummy" else config["binary_threshold_grid"])]
                best = sorted(threshold_rows, key=lambda r: (-r["macro_f1"], r["false_positive_rate"], abs(r["threshold"] - 0.5)))[0]
                selected_threshold = best["threshold"]
                default = binary_metrics(y[val_idx], probability[:, 1], 0.5)
                for r in threshold_rows:
                    r.update({"candidate": spec["name"], "task": task})
                pd.DataFrame(threshold_rows).to_csv(ROOT / "results" / f"thresholds_{spec['name']}.csv", index=False)
            else:
                best = multiclass_metrics(y[val_idx], probability)
                default = dict(best)
                selected_threshold = None
            result = {"task": task, "name": spec["name"], "family": family, "settings": settings, "validation_metrics": best, "validation_default_metrics": default, "threshold": selected_threshold, "training_seconds": seconds, "warnings": warning_text}
            task_candidates.append(result); all_candidates.append(result)
            validation_data[spec["name"]] = probability
            logging.info("Finished %s: %.1fs; validation macro F1=%.5f; threshold=%s", spec["name"], seconds, best["macro_f1"], selected_threshold)
            dump_json(ROOT / "results/validation_candidates.json", all_candidates)
            del model
            if scaled:
                del xf, xv, prep_lr
            gc.collect()
        sortkey = lambda r: (-r["validation_metrics"]["macro_f1"], r["validation_metrics"].get("false_positive_rate", 0), r["name"])
        winner = sorted([r for r in task_candidates if r["family"] in ["xgboost", "random_forest"]], key=sortkey)[0]
        winners[task] = {"champion": winner["name"], "family_winners": {}}
        for family in sorted({r["family"] for r in task_candidates}):
            selected = sorted([r for r in task_candidates if r["family"] == family], key=sortkey)[0]
            winners[task]["family_winners"][family] = selected
            if task == "binary":
                pd.DataFrame({"row_index": val_idx, "label": y[val_idx], "attack_cat": train.iloc[val_idx].attack_cat.to_numpy(), "attack_probability": validation_data[selected["name"]][:, 1]}).to_csv(ROOT / "results" / f"validation_{selected['name']}.csv", index=False)
        logging.info("Selected %s champion on validation: %s", task, winner["name"])
    lock = {"selection_rule": config["selection_metric"], "config_sha256": hashlib.sha256((ROOT / "config/experiment.json").read_bytes()).hexdigest(), "train_sha256": hashlib.sha256((ROOT / "data/raw/train.csv").read_bytes()).hexdigest(), "test_read_by_this_script": False, "winners": winners, "classes": encoder.classes_.tolist()}
    dump_json(ROOT / "results/selection_lock.json", lock)
    logging.info("SELECTION LOCK SAVED. Refit selected configurations on complete training partition.")
    del xfit, xval, validation_data
    gc.collect()
    full_preprocessor = make_preprocessor(columns)
    xfull = full_preprocessor.fit_transform(x)
    joblib.dump(full_preprocessor, ROOT / "models/preprocessor.joblib", compress=3)
    dump_json(ROOT / "data/model_input_schema.json", {"feature_columns": columns, "excluded": ["id", "label", "attack_cat"], "categorical_features": ["proto", "service", "state"], "encoded_features": xfull.shape[1], "encoded_feature_names": full_preprocessor.get_feature_names_out().tolist(), "unseen_category_policy": "one-hot all-zero for unknown category", "numeric_missing_policy": "training median", "categorical_missing_policy": "training mode", "score_caveat": "Raw model scores are not calibrated real-world probabilities."})
    index = []
    for task, record in winners.items():
        y = train.label.to_numpy(dtype=int) if task == "binary" else encoder.transform(train.attack_cat)
        for family, winner in record["family_winners"].items():
            logging.info("Full-data refit: %s", winner["name"])
            if family == "logistic_regression":
                prep = make_preprocessor(columns, scale=True); xx = prep.fit_transform(x)
            else:
                prep, xx = full_preprocessor, xfull
            start = time.perf_counter()
            model = refit_model(family, task, winner["settings"], xx, y, seed, threads)
            elapsed = time.perf_counter() - start
            bundle = {"format_version": 1, "task": task, "name": winner["name"], "model": model, "preprocessor": prep, "threshold": winner["threshold"], "classes": ["Normal", "Attack"] if task == "binary" else encoder.classes_.tolist(), "feature_columns": columns, "selection": winner, "training_rows": len(train), "seed": seed, "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__, "joblib": joblib.__version__}}
            path = ROOT / "models" / f"{winner['name']}.joblib"
            joblib.dump(bundle, path, compress=3)
            if family == "xgboost":
                model.save_model(ROOT / "models" / f"{winner['name']}.ubj")
            index.append({"name": winner["name"], "task": task, "family": family, "path": str(path.relative_to(ROOT)), "champion": winner["name"] == record["champion"], "full_refit_seconds": elapsed, "threshold": winner["threshold"], "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
            dump_json(ROOT / "models/model_index.json", index)
            logging.info("Saved %s (%.1f MB), refit %.1fs", winner["name"], path.stat().st_size / 1e6, elapsed)
            del model, bundle
            if family == "logistic_regression":
                del xx, prep
            gc.collect()
    dump_json(ROOT / "results/training_summary.json", {"elapsed_seconds": time.perf_counter() - start_all, "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__}, "models": index, "candidate_count": len(all_candidates)})
    logging.info("TRAINING COMPLETE: %.1f seconds", time.perf_counter() - start_all)


if __name__ == "__main__":
    main()
