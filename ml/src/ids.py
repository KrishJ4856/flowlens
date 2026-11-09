"""Reusable, leakage-aware data preparation and saved-model prediction."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

EXCLUDED = {"id", "label", "attack_cat"}
CATEGORICAL = ["proto", "service", "state"]


def read_data(path):
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    frame.columns = frame.columns.str.strip()
    for column in CATEGORICAL + ["attack_cat"]:
        if column in frame:
            frame[column] = frame[column].str.strip()
    return frame


def feature_columns(frame):
    return [c for c in frame.columns if c not in EXCLUDED]


def feature_frame(frame, columns):
    """Deterministic parsing only; learned imputation/encoding happens later."""
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names are not allowed.")
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    x = frame.loc[:, columns].copy()
    for c in columns:
        if c in CATEGORICAL:
            x[c] = x[c].astype(object)
            x[c] = x[c].where(x[c].notna(), np.nan)
            x[c] = x[c].map(lambda v: str(v).strip() if pd.notna(v) else np.nan)
        else:
            x[c] = pd.to_numeric(x[c], errors="raise").replace([np.inf, -np.inf], np.nan)
    return x


def as_float32(values):
    return np.asarray(values, dtype=np.float32)


def make_preprocessor(columns, scale=False):
    numbers = [c for c in columns if c not in CATEGORICAL]
    cats = [c for c in columns if c in CATEGORICAL]
    numeric = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        numeric.append(("scale", StandardScaler()))
    numeric.append(("float32", FunctionTransformer(as_float32, feature_names_out="one-to-one")))
    return ColumnTransformer([
        ("numeric", Pipeline(numeric), numbers),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)),
        ]), cats),
    ], sparse_threshold=0, verbose_feature_names_out=True)


def feature_hashes(frame, columns):
    return pd.util.hash_pandas_object(feature_frame(frame, columns), index=False).to_numpy()


def predict_bundle(bundle, frame):
    x = feature_frame(frame, bundle["feature_columns"])
    processed = bundle["preprocessor"].transform(x)
    probability = bundle["model"].predict_proba(processed)
    result = pd.DataFrame(index=frame.index)
    if "id" in frame:
        result["record_id"] = frame["id"].to_numpy()
    if bundle["task"] == "binary":
        result["attack_probability"] = probability[:, 1]
        result["prediction"] = (probability[:, 1] >= bundle["threshold"]).astype(int)
        result["predicted_label"] = result["prediction"].map({0: "Normal", 1: "Attack"})
    else:
        index = probability.argmax(axis=1)
        result["prediction"] = index
        result["predicted_label"] = np.asarray(bundle["classes"])[index]
        result["model_score"] = probability.max(axis=1)
    return result


def dump_json(path, value):
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + "\n")
