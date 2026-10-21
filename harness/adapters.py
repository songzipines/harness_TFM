"""The only module that knows each TFM's API. No routing or scoring here."""
from __future__ import annotations

from abc import ABC, abstractmethod
import json

import numpy as np


def precision_record(estimator):
    """Record resolved precision flags as well as the actual stored parameter dtypes."""
    flags = {name: repr(getattr(estimator, name)) for name in
             ("use_autocast_", "forced_inference_dtype_", "mix_precision", "inference_config_")
             if hasattr(estimator, name)}
    modules = list(getattr(estimator, "models_", []))
    for name in ("model_", "model"):
        model = getattr(estimator, name, None)
        if model is not None and hasattr(model, "parameters"):
            modules.append(model)
    flags["parameter_dtypes"] = sorted({str(p.dtype) for model in modules for p in model.parameters()})
    inference = getattr(estimator, "inference_config_", None)
    if inference is not None:
        flags["resolved_inference_config"] = {
            stage: {k: repr(v) for k, v in getattr(inference, stage).items()}
            for stage in ("COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG")
            if getattr(inference, stage, None) is not None}
    return flags


class Adapter(ABC):
    """Future models only implement these four operations; inputs stay DataFrames."""
    def __init__(self, config, task, device):
        self.config, self.task, self.device = config, task, device
        self.estimator = None

    @abstractmethod
    def load(self): ...

    @abstractmethod
    def fit(self, X, y): ...

    @abstractmethod
    def predict(self, X): ...

    @abstractmethod
    def resolved_config(self): ...

    def class_order(self):
        return np.asarray(self.estimator.classes_, dtype=int)


class TabPFNAdapter(Adapter):
    def load(self):
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion
        cls = TabPFNRegressor if self.task == "regression" else TabPFNClassifier
        self.estimator = cls.create_default_for_version(
            ModelVersion.V3_5, device=self.device, random_state=self.config.raw["model_seed"],
            model_path=str(self.config.runtime / self.config.assets["models"]["tabpfn"]["checkpoint"]))

    def fit(self, X, y):
        self.estimator.fit(X, y)

    def predict(self, X):
        if self.task == "regression":
            return self.estimator.predict(X)
        return self.estimator.predict_proba(X)

    def resolved_config(self):
        params = self.estimator.get_params(deep=False)
        return {"model_version": "3.5", "n_estimators": int(self.estimator.n_estimators_),
                "resolved_precision": precision_record(self.estimator),
                "parameters_repr": {k: repr(v) for k, v in params.items()},
                "inference_config_repr": repr(self.estimator.get_inference_config()),
                "load_timing_note": "Constructor is load_s; lazy checkpoint load is included in fit_s."}


class TabICLAdapter(Adapter):
    def load(self):
        from tabicl import TabICLClassifier, TabICLRegressor
        regression = self.task == "regression"
        cls = TabICLRegressor if regression else TabICLClassifier
        spec = self.config.assets["models"]["tabicl"]
        path = spec["regression_checkpoint" if regression else "classification_checkpoint"]
        self.estimator = cls(n_estimators=8, device=self.device, random_state=self.config.raw["model_seed"],
                             n_jobs=self.config.raw["cpu_threads"], model_path=str(self.config.runtime / path),
                             allow_auto_download=False)

    def fit(self, X, y):
        self.estimator.fit(X, y)

    def predict(self, X):
        if self.task == "regression":
            return self.estimator.predict(X)
        return self.estimator.predict_proba(X)

    def resolved_config(self):
        return {"model_version": "v2", "n_estimators": int(self.estimator.n_estimators),
                "resolved_precision": precision_record(self.estimator),
                "parameters_repr": {k: repr(v) for k, v in self.estimator.get_params(deep=False).items()},
                "load_timing_note": "Constructor is load_s; lazy checkpoint load is included in fit_s."}


class LimiXAdapter(Adapter):
    def load(self):
        import torch
        from inference.predictor import LimiXPredictor
        spec = self.config.assets["models"]["limix"]
        self.config_path = self.config.runtime / spec[
            "regression_config" if self.task == "regression" else "classification_config"]
        self.estimator = LimiXPredictor(
            device=torch.device(self.device), model_path=str(self.config.runtime / spec["checkpoint"]),
            inference_config=str(self.config_path), use_data_cache=False,
            seed=self.config.raw["model_seed"], preprocess_num_jobs=self.config.raw["cpu_threads"])

    def fit(self, X, y):
        # LimiX performs the actual context preprocessing in EACH predict call.
        self.X_fit, self.y_fit = X, y

    def predict(self, X):
        return self.estimator.predict(self.X_fit, self.y_fit, X,
                                      task_type="Regression" if self.task == "regression" else "Classification")

    def class_order(self):
        # Official implementation sorts np.unique(y_train); our targets are contiguous 0..K-1.
        return np.unique(self.y_fit)

    def resolved_config(self):
        return {"model_version": "LimiX-2", "n_estimators": int(self.estimator.n_estimators),
                "resolved_precision": precision_record(self.estimator),
                "inference_config": json.loads(self.config_path.read_text(encoding="utf-8")),
                "use_data_cache": False, "cache_dir": str(self.estimator.cache_manager.cache_dir),
                "preprocessing_note": "fit_s stores context only; val/test predict_s each include context processing",
                "transductive_note": "Official all_data transform tags can use current query X; partitions are separate"}


REGISTRY = {"tabpfn": TabPFNAdapter, "tabicl": TabICLAdapter, "limix": LimiXAdapter}


def create_adapter(model, config, task, device):
    return REGISTRY[model](config, task, device)


def checked_prediction(adapter, X, n_classes):
    prediction = np.asarray(adapter.predict(X), dtype=float)
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite predictions")
    if adapter.task == "regression":
        if prediction.shape != (len(X),):
            raise ValueError(f"Invalid regression shape {prediction.shape}")
        return prediction
    order = list(adapter.class_order())
    if len(order) != n_classes or set(order) != set(range(n_classes)):
        raise ValueError(f"Invalid probability class order {order}")
    if prediction.shape != (len(X), n_classes):
        raise ValueError(f"Invalid probability shape {prediction.shape}")
    prediction = prediction[:, [order.index(i) for i in range(n_classes)]]
    if (prediction < 0).any() or (prediction > 1).any():
        raise ValueError("Probability outside [0,1]")
    if not np.allclose(prediction.sum(axis=1), 1, atol=1e-4, rtol=0):
        raise ValueError("Probability rows do not sum to one")
    return prediction
