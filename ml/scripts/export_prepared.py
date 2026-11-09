"""Export numeric matrices from the frozen train-fitted preprocessing object."""
from pathlib import Path
import sys, json, gc
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import joblib
from ids import read_data, feature_frame, dump_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    schema = json.loads((ROOT / "data/model_input_schema.json").read_text())
    columns = schema["feature_columns"]
    prep = joblib.load(ROOT / "models/preprocessor.joblib")
    target = ROOT / "data/prepared"
    target.mkdir(exist_ok=True)
    for partition in ["train", "test"]:
        df = read_data(ROOT / "data/raw" / f"{partition}.csv")
        x = prep.transform(feature_frame(df, columns))
        np.savez_compressed(target / f"{partition}.npz", X=x, label=df.label.to_numpy(dtype=np.int8), attack_cat=df.attack_cat.to_numpy(dtype=str), record_id=df.id.to_numpy(), feature_names=np.asarray(schema["encoded_feature_names"]))
        print(f"Saved {partition}: {x.shape}, {x.dtype}", flush=True)
        del df, x
        gc.collect()
    dump_json(target / "README.json", {"preprocessor_fitted_on": "all canonical training rows only", "numeric_inputs": "X", "binary_answer": "label", "multiclass_answer": "attack_cat", "excluded_from_X": ["id", "label", "attack_cat"], "warning": "These full-training-fitted matrices are suitable for final inference or fixed train/test model training. Hyperparameter selection or CV must refit preprocessing inside its training fold, as scripts/train.py and scripts/stability.py do; do not use this full-train-fitted train matrix for claiming leakage-free CV.", "loading": "np.load(path, allow_pickle=False)"})


if __name__ == "__main__":
    main()
