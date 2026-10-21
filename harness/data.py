"""Dtype-preserving input boundary. Query targets are not returned to adapters."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .common import digest, read_json, sha256


def load_split(config, dataset, split):
    spec = read_json(config.split_path(dataset, split))
    if (spec["repeat"], spec["fold"], spec["sample"], spec["validation_seed"]) != (split, 0, 0, 20261020 + split):
        raise ValueError("Split metadata differs from the frozen outer/validation protocol")
    actual = digest({k: spec[k] for k in ("fit", "validation", "test")})
    if actual != dataset["splits"][split]["indices_sha256"] or actual != spec["indices_sha256"]:
        raise ValueError("Frozen split hash mismatch")
    parts = [set(spec[k]) for k in ("fit", "validation", "test")]
    if any(parts[i] & parts[j] for i in range(3) for j in range(i)):
        raise ValueError("Overlapping fit/validation/test rows")
    if set.union(*parts) != set(range(dataset["n_rows"])):
        raise ValueError("Split does not partition the complete dataset")
    if any(len(spec[k]) != len(set(spec[k])) for k in ("fit", "validation", "test")):
        raise ValueError("Duplicate row IDs")
    return spec


@dataclass
class ModelInput:
    X_fit: pd.DataFrame
    y_fit: np.ndarray
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    row_ids: dict
    classes: list
    fit_target_mean: float | None


def read_frame(config, dataset, verify=True):
    path = config.data_path(dataset)
    if verify and sha256(path) != dataset["data_sha256"]:
        raise ValueError("Frozen parquet hash mismatch")
    df = pd.read_parquet(path)
    if len(df) != dataset["n_rows"] or dataset["target"] in dataset["feature_names"]:
        raise ValueError("Invalid data shape or target leakage in feature list")
    for column in dataset["categorical_columns"]:
        if not isinstance(df[column].dtype, pd.CategoricalDtype):
            raise ValueError(f"Categorical dtype lost: {column}")
    return df


def model_input(config, dataset, split):
    spec = load_split(config, dataset, split)
    df = read_frame(config, dataset)
    X = df[dataset["feature_names"]]
    frames = {k: X.iloc[spec[k]].copy() for k in ("fit", "validation", "test")}
    # Do not expose category levels learned from all parquet rows to fit preprocessing.
    # Query values remain strings, so official adapters can handle unseen values themselves.
    for column in dataset["categorical_columns"]:
        frames["fit"][column] = frames["fit"][column].cat.remove_unused_categories()
        for part in ("validation", "test"):
            frames[part][column] = frames[part][column].astype(object)
    y_fit = df[dataset["target"]].iloc[spec["fit"]].to_numpy()
    classes = []
    mean = None
    if dataset["problem_type"] != "regression":
        encoder = LabelEncoder().fit(y_fit)
        y_fit = encoder.transform(y_fit)
        classes = [v.item() if isinstance(v, np.generic) else v for v in encoder.classes_]
    else:
        y_fit = y_fit.astype(np.float64)
        mean = float(np.mean(y_fit))
    return ModelInput(frames["fit"], y_fit, frames["validation"], frames["test"],
                      {k: spec[k] for k in ("fit", "validation", "test")}, classes, mean)


def scoring_target(config, dataset, split, part, classes):
    """Only evaluation code calls this; model adapter APIs have no query-y argument."""
    if part not in ("validation", "test"):
        raise ValueError(part)
    spec = load_split(config, dataset, split)
    target = read_frame(config, dataset)[dataset["target"]].iloc[spec[part]].to_numpy()
    if dataset["problem_type"] == "regression":
        return target.astype(float)
    mapping = {label: i for i, label in enumerate(classes)}
    if any(label not in mapping for label in target):
        raise ValueError("Query class absent from fit target mapping")
    return np.array([mapping[label] for label in target], dtype=int)
