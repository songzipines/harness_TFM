"""Frozen primary metrics and fit-only constant baseline."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, log_loss, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)


def metrics(task, y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    if task == "regression":
        rmse = float(np.sqrt(mean_squared_error(y, prediction)))
        return {"primary": rmse, "loss": rmse, "mae": float(mean_absolute_error(y, prediction)),
                "r2": float(r2_score(y, prediction))}
    probabilities = np.clip(prediction, 1e-7, 1 - 1e-7)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    loss = float(log_loss(y, probabilities, labels=np.arange(prediction.shape[1])))
    result = {"accuracy": float(accuracy_score(y, prediction.argmax(axis=1))), "log_loss": loss}
    if task == "binary":
        auc = float(roc_auc_score(y, prediction[:, 1]))
        result.update(primary=auc, loss=1 - auc)
    else:
        result.update(primary=loss, loss=loss, macro_f1=float(f1_score(
            y, prediction.argmax(axis=1), labels=np.arange(prediction.shape[1]), average="macro", zero_division=0)))
    return result


def constant_prediction(task, y_fit, n_query, n_classes=0):
    if task == "regression":
        return np.full(n_query, np.mean(y_fit))
    counts = np.bincount(np.asarray(y_fit, dtype=int), minlength=n_classes) + 1
    return np.tile(counts / counts.sum(), (n_query, 1))
