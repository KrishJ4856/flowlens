"""Three-fold grouped stability check of already-selected binary configurations.

The test partition is never opened. Hyperparameters were previously selected
using the development holdout, so these are stability diagnostics, not nested-CV
performance estimates. All folds use the conventional 0.5 threshold.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
from pathlib import Path
import sys, json, time, gc
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from ids import read_data, feature_frame, feature_hashes, make_preprocessor, dump_json
from train import refit_model, binary_metrics

ROOT = Path(__file__).resolve().parents[1]


def main():
    train = read_data(ROOT / "data/raw/train.csv")
    lock = json.loads((ROOT / "results/selection_lock.json").read_text())
    columns = [c for c in train if c not in ["id", "label", "attack_cat"]]
    hashes = feature_hashes(train, columns)
    x = feature_frame(train, columns)
    y = train.label.to_numpy()
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=123)
    records = []
    for fold, (fit, val) in enumerate(splitter.split(x, train.attack_cat, hashes), 1):
        assert not set(hashes[fit]) & set(hashes[val])
        prep = make_preprocessor(columns)
        xf = prep.fit_transform(x.iloc[fit]); xv = prep.transform(x.iloc[val])
        for family in ["xgboost", "random_forest"]:
            chosen = lock["winners"]["binary"]["family_winners"][family]
            print(f"Stability fold {fold}/3: {chosen['name']}", flush=True)
            start = time.perf_counter()
            model = refit_model(family, "binary", chosen["settings"], xf, y[fit], 42, 2)
            probability = model.predict_proba(xv)[:, 1]
            metrics = binary_metrics(y[val], probability, 0.5)
            records.append({"fold": fold, "family": family, "candidate": chosen["name"], "fit_rows": len(fit), "validation_rows": len(val), "exact_feature_overlap": 0, "training_seconds": time.perf_counter() - start, **metrics})
            pd.DataFrame(records).to_csv(ROOT / "results/grouped_stability.csv", index=False)
            print(f"  macro F1 {metrics['macro_f1']:.5f}, attack recall {metrics['attack_recall']:.5f}", flush=True)
            del model
            gc.collect()
        del xf, xv, prep
        gc.collect()
    dump_json(ROOT / "results/stability_method.json", {"folds": 3, "split_seed": 123, "model_seed": 42, "stratified_by": "attack_cat", "grouped_by": "exact 42-feature vector", "threshold": 0.5, "test_data_read": False, "interpretation": "Training-partition stability diagnostic for already selected hyperparameters; not nested CV, not an independent estimate and not used to change model selection."})


if __name__ == "__main__":
    main()
