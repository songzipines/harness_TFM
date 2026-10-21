"""Tiny synthetic engineering check through the NEW adapters, never the six pilot datasets."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from .common import code_identity, environment, now, write_json
from .data import ModelInput
from .metrics import metrics
from .worker import execute


def synthetic_input(task):
    if task == "regression":
        X, y = make_regression(n_samples=160, n_features=6, n_informative=4, noise=0.2, random_state=42)
    else:
        X, y = make_classification(n_samples=160, n_features=6, n_informative=4, n_redundant=0,
                                   n_classes=2 if task == "binary" else 3, n_clusters_per_class=1, random_state=42)
    frame = pd.DataFrame(X.astype(np.float32), columns=[f"f{i}" for i in range(6)])
    frame["f0"] = pd.Categorical(np.where(X[:, 0] > 0, "positive", "nonpositive"))
    classes = [] if task == "regression" else list(range(int(y.max()) + 1))
    parts = {"fit": list(range(128)), "validation": list(range(128, 144)), "test": list(range(144, 160))}
    item = ModelInput(frame.iloc[:128].copy(), y[:128], frame.iloc[128:144].copy(), frame.iloc[144:].copy(),
                      parts, classes, float(np.mean(y[:128])) if task == "regression" else None)
    item.X_fit["f0"] = item.X_fit["f0"].cat.remove_unused_categories()
    item.X_validation["f0"] = item.X_validation["f0"].astype(object)
    item.X_test["f0"] = item.X_test["f0"].astype(object)
    return item, y[128:144], y[144:]


def run_smoke(config, model, task, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Smoke output already exists; choose a new --output so checks stay traceable")
    item, y_val, y_test = synthetic_input(task)
    unit = {"dataset": "SYNTHETIC_ENGINEERING_ONLY", "split": 0, "model": model, "problem_type": task}
    result = execute(config, unit, output, input_override=item)
    result["purpose"] = "Engineering smoke, not a formal pilot result"
    result["code"] = code_identity()
    result["environment"] = environment()
    result["created_at"] = now()
    if result["status"] == "success":
        for part, y in (("validation", y_val), ("test", y_test)):
            with np.load(output / f"{part}.npz", allow_pickle=False) as a:
                result[f"synthetic_{part}_metrics"] = metrics(task, y, a["prediction"])
    write_json(output / "smoke-result.json", result)
    return result
