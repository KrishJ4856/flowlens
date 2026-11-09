"""Meaningful integrity and inference checks on the delivered artifacts."""
from pathlib import Path
import sys, json, subprocess, hashlib, tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from ids import read_data, feature_frame, feature_hashes, predict_bundle, dump_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    checks = []
    score_differences = {}
    def check(name, condition):
        assert condition, name
        checks.append({"check": name, "passed": True})
        print("PASS", name, flush=True)
    index = json.loads((ROOT / "models/model_index.json").read_text())
    lock = json.loads((ROOT / "results/selection_lock.json").read_text())
    marker = json.loads((ROOT / "results/evaluation_complete.json").read_text())
    check("Selection lock unchanged after test evaluation", hashlib.sha256((ROOT / "results/selection_lock.json").read_bytes()).hexdigest() == marker["selection_lock_sha256"])
    check("Experiment settings match frozen selection", hashlib.sha256((ROOT / "config/experiment.json").read_bytes()).hexdigest() == lock["config_sha256"])
    train = read_data(ROOT / "data/raw/train.csv")
    test = read_data(ROOT / "data/raw/test.csv")
    sources = json.loads((ROOT / "data/source_manifest.json").read_text())
    for partition in ["train", "test"]:
        check(f"Canonical {partition} file unchanged", hashlib.sha256((ROOT / f"data/raw/{partition}.csv").read_bytes()).hexdigest() == sources[f"canonical_{partition}"]["sha256"])
    demo = read_data(ROOT / "data/demo_records.csv")
    check("Demo input has no answer columns", not {"label", "attack_cat"} & set(demo))
    columns = [c for c in train if c not in ["id", "label", "attack_cat"]]
    split = pd.read_csv(ROOT / "data/development_split.csv")
    hashes = feature_hashes(train, columns)
    check("Demo records have no exact training-feature match", not np.isin(feature_hashes(demo, columns), hashes).any())
    fit = split.split.eq("fit").to_numpy(); val = ~fit
    check("No identical-feature groups cross fit/validation boundary", not set(hashes[fit]) & set(hashes[val]))
    metrics = pd.read_csv(ROOT / "results/test_metrics.csv")
    for item in index:
        bundle_path = ROOT / item["path"]
        check(f"Model checksum: {item['name']}", hashlib.sha256(bundle_path.read_bytes()).hexdigest() == item["sha256"])
        bundle = joblib.load(bundle_path)
        sample = test.iloc[:80].copy()
        reference = predict_bundle(bundle, sample)
        altered = sample.copy()
        altered["label"] = 1 - altered.label
        altered["attack_cat"] = "PURPOSELY_WRONG_ANSWER"
        altered["id"] = -999
        altered = altered.loc[:, list(reversed(altered.columns))]
        candidate = predict_bundle(bundle, altered)
        check(f"Excluded columns leave prepared inputs exactly unchanged: {item['name']}", np.array_equal(bundle['preprocessor'].transform(feature_frame(sample, columns)), bundle['preprocessor'].transform(feature_frame(altered, columns))))
        check(f"Answers, ID and column order cannot change predictions: {item['name']}", np.array_equal(reference.prediction, candidate.prediction))
        score_column = "attack_probability" if bundle["task"] == "binary" else "model_score"
        score_differences[item['name']] = float(np.abs(reference[score_column] - candidate[score_column]).max())
        check(f"Scores invariant within 1e-12 floating-point tolerance: {item['name']}", np.allclose(reference[score_column], candidate[score_column], rtol=0, atol=1e-12))
        try:
            predict_bundle(bundle, sample.drop(columns=[columns[0]]))
        except ValueError as exc:
            check(f"Missing input rejected: {item['name']}", "Missing required feature" in str(exc))
        else:
            raise AssertionError("Missing feature accepted")
        try:
            bad = sample.copy(); bad["dur"] = "invalid_number"; predict_bundle(bundle, bad)
        except (ValueError, TypeError):
            check(f"Invalid numeric input rejected: {item['name']}", True)
        else:
            raise AssertionError("Invalid numeric input accepted")
        unusual = sample.copy()
        unusual["proto"] = "new_protocol_not_in_training"
        unusual["dur"] = np.nan
        output = predict_bundle(bundle, unusual)
        check(f"Unknown categories and missing values handled: {item['name']}", np.isfinite(output[score_column]).all())
        categorical_missing = sample.copy()
        categorical_missing["proto"] = np.nan
        output = predict_bundle(bundle, categorical_missing)
        check(f"Missing categories use saved imputation: {item['name']}", np.isfinite(output[score_column]).all())
        numbers = [c for c in columns if c not in ["proto", "service", "state"]]
        imputer = bundle["preprocessor"].named_transformers_["numeric"].named_steps["imputer"]
        check(f"Saved imputation fitted on complete train only: {item['name']}", np.allclose(imputer.statistics_, train[numbers].median().to_numpy()))
        categories = bundle["preprocessor"].named_transformers_["categorical"].named_steps["onehot"].categories_
        check(f"Saved encoding fitted on train categories only: {item['name']}", all(set(values) == set(train[c].unique()) for c, values in zip(["proto", "service", "state"], categories)))
        score_field = "attack_score" if bundle["task"] == "binary" else "model_score"
        saved = pd.read_csv(ROOT / "results/predictions" / f"{item['name']}.csv", converters={score_field: float})
        check(f"Every official test row has a prediction: {item['name']}", len(saved) == len(test) and np.array_equal(saved.record_id, test.id))
        if bundle["task"] == "binary":
            actual = saved.actual_label.to_numpy(); prediction = saved.prediction.to_numpy()
            reported = metrics[(metrics.model == item["name"]) & (metrics.subset == "official_test")].iloc[0]
            check(f"Confusion counts independently reconcile: {item['name']}", all(int(reported[key]) == int(mask.sum()) for key, mask in {"tn": (actual == 0) & (prediction == 0), "fp": (actual == 0) & (prediction == 1), "fn": (actual == 1) & (prediction == 0), "tp": (actual == 1) & (prediction == 1)}.items()))
            check(f"Saved threshold reproduces predictions: {item['name']}", np.array_equal((saved.attack_score >= bundle["threshold"]).astype(int), prediction))
        if item["family"] == "xgboost":
            native = XGBClassifier()
            native.load_model(ROOT / "models" / f"{item['name']}.ubj")
            x = bundle["preprocessor"].transform(feature_frame(sample, columns))
            check(f"Native XGBoost serialization matches bundle: {item['name']}", np.allclose(native.predict_proba(x), bundle["model"].predict_proba(x), atol=1e-7))
        del bundle
    # Exercise the user-facing command in a fresh interpreter, testing imports
    # and saved-model loading outside this verifier's process.
    output_path = ROOT / "results/demo_predictions.csv"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/predict.py"), "--input", str(ROOT / "data/demo_records.csv"), "--output", str(output_path)], capture_output=True, text=True, check=True)
    check("Fresh-process command-line prediction succeeds", output_path.exists() and len(pd.read_csv(output_path)) == 30)
    check("Seven frozen models evaluated", len(index) == 7 and len(metrics) == 21)
    preprocessor = joblib.load(ROOT / "models/preprocessor.joblib")
    for partition, frame in [("train", train), ("test", test)]:
        with np.load(ROOT / f"data/prepared/{partition}.npz", allow_pickle=False) as data:
            prepared = data["X"]
            check(f"Prepared {partition} inputs and answers stay separate", prepared.shape[0] == len(frame) and len(data['feature_names']) == prepared.shape[1] and not any(name.endswith('__label') or name.endswith('__attack_cat') or name.endswith('__id') for name in data['feature_names']))
            check(f"Prepared {partition} answers match raw source", np.array_equal(data['label'], frame.label) and np.array_equal(data['attack_cat'], frame.attack_cat))
            check(f"Prepared {partition} transformation matches saved preparation", np.allclose(prepared[:80], preprocessor.transform(feature_frame(frame.iloc[:80], columns))))
    dump_json(ROOT / "results/verification.json", {"checks_passed": len(checks), "checks": checks, "prediction_command_output": result.stdout, "score_invariance_absolute_tolerance": 1e-12, "observed_max_score_differences": score_differences, "numerical_note": "Parallel forest prediction can sum trees in a different order and change scores in the last floating-point bits. Prepared input arrays must be exactly equal, class predictions must match, and scores must match within 1e-12."})
    print(f"VERIFIED: {len(checks)} checks passed.", flush=True)


if __name__ == "__main__":
    main()
