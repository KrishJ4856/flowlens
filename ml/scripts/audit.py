"""Audit raw partitions and create a training-only grouped development split."""
from pathlib import Path
import sys, hashlib, json
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from ids import read_data, feature_columns, feature_frame, feature_hashes, dump_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT / "config/experiment.json").read_text())
    train = read_data(ROOT / "data/raw/train.csv")
    test = read_data(ROOT / "data/raw/test.csv")
    columns = feature_columns(train)
    assert len(train) == 175341 and len(test) == 82332
    assert list(train.columns) == list(test.columns) and len(columns) == 42
    assert not set(columns) & {"id", "label", "attack_cat"}
    hashes = {"train": feature_hashes(train, columns), "test": feature_hashes(test, columns)}
    report = {"features": columns, "numeric_features": [c for c in columns if c not in ["proto", "service", "state"]], "partitions": {}}
    summaries = []
    for name, df in [("train", train), ("test", test)]:
        x = feature_frame(df, columns)
        groups = df[["label", "attack_cat"]].assign(feature_hash=hashes[name])
        conflicts = groups.groupby("feature_hash").agg(binary_labels=("label", "nunique"), attack_categories=("attack_cat", "nunique"))
        assert set(df.label.unique()) == {0, 1}
        assert ((df.attack_cat == "Normal") == (df.label == 0)).all(), "Contradictory target columns"
        info = {
            "rows": len(df), "columns": len(df.columns),
            "sha256": hashlib.sha256((ROOT / f"data/raw/{name}.csv").read_bytes()).hexdigest(),
            "null_fields": int(df.isna().sum().sum()),
            "nonfinite_numeric_fields": int((~np.isfinite(x.select_dtypes("number").to_numpy())).sum()),
            "duplicate_feature_rows": int(pd.Series(hashes[name]).duplicated().sum()),
            "duplicate_rows_excluding_id": int(df.drop(columns="id").duplicated().sum()),
            "unique_feature_groups": int(pd.Series(hashes[name]).nunique()),
            "feature_groups_with_conflicting_binary_labels": int((conflicts.binary_labels > 1).sum()),
            "feature_groups_with_conflicting_attack_categories": int((conflicts.attack_categories > 1).sum()),
            "binary_counts": df.label.value_counts().sort_index().to_dict(),
            "attack_category_counts": df.attack_cat.value_counts().to_dict(),
            "negative_numeric_fields": int((x.select_dtypes("number") < 0).sum().sum()),
        }
        report["partitions"][name] = info
        for category, count in df.attack_cat.value_counts().items():
            summaries.append({"partition": name, "category": category, "count": int(count), "proportion": count / len(df)})
    overlap = np.isin(hashes["test"], hashes["train"])
    report["cross_partition"] = {
        "test_rows_with_exact_training_feature_match": int(overlap.sum()),
        "test_rows_without_exact_training_feature_match": int((~overlap).sum()),
        "test_unique_features_with_training_match": int(pd.Series(hashes["test"][overlap]).nunique()),
        "caveat": "64-bit pandas row hashes after deterministic numeric parsing; exact-feature match, not connection/session independence or near-duplicate detection.",
    }
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config["seed"])
    fit, val = next(splitter.split(train[columns], train.attack_cat, hashes["train"]))
    assert not set(hashes["train"][fit]) & set(hashes["train"][val])
    assert set(train.iloc[fit].attack_cat) == set(train.attack_cat)
    assert set(train.iloc[val].attack_cat) == set(train.attack_cat)
    assignment = pd.DataFrame({"row_index": np.arange(len(train)), "record_id": train.id, "feature_group_hash": hashes["train"].astype(str), "split": "fit"})
    assignment.loc[val, "split"] = "validation"
    assignment.to_csv(ROOT / "data/development_split.csv", index=False)
    report["development_split"] = {
        "method": config["validation"], "seed": config["seed"], "fit_rows": len(fit), "validation_rows": len(val),
        "exact_feature_overlap": 0,
        "fit_categories": train.iloc[fit].attack_cat.value_counts().to_dict(),
        "validation_categories": train.iloc[val].attack_cat.value_counts().to_dict(),
        "fit_binary_counts": train.iloc[fit].label.value_counts().to_dict(),
        "validation_binary_counts": train.iloc[val].label.value_counts().to_dict(),
    }
    report["unseen_test_categories"] = {
        c: sorted(set(test[c]) - set(train[c])) for c in ["proto", "service", "state"]
    }
    dfcols = pd.DataFrame({"column": train.columns, "dtype": [str(v) for v in train.dtypes], "role": ["excluded_id" if c == "id" else "answer_excluded_from_inputs" if c in ["label", "attack_cat"] else "categorical_feature" if c in ["proto", "service", "state"] else "numeric_feature" for c in train.columns]})
    dfcols.to_csv(ROOT / "data/column_schema.csv", index=False)
    pd.DataFrame(summaries).to_csv(ROOT / "results/class_distribution.csv", index=False)
    dump_json(ROOT / "results/data_audit.json", report)
    print(json.dumps({"split": report["development_split"], "overlap": report["cross_partition"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
