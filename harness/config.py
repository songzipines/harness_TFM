"""One replaceable experiment configuration, plus immutable asset identities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from .common import REPO, MODEL_ORDER, digest, read_json


@dataclass
class Config:
    path: Path
    raw: dict
    runtime: Path
    output: Path
    assets: dict

    @classmethod
    def load(cls, path=None, output=None):
        path = Path(path or REPO / "configs/pilot.json").resolve()
        raw = read_json(path)
        runtime = Path(raw["runtime_root"])
        if not runtime.is_absolute():
            runtime = (path.parent / runtime).resolve()
        out = Path(output) if output else runtime / raw["output_subdir"]
        if not out.is_absolute():
            out = (Path.cwd() / out).resolve()
        if os.name == "nt" and (runtime.drive.upper() != "D:" or out.drive.upper() != "D:"):
            raise ValueError("Runtime and outputs must stay on D: on this Windows machine")
        if REPO == out or REPO in out.parents:
            raise ValueError("Keep experimental outputs outside the source repository")
        if raw["models"] != list(MODEL_ORDER):
            raise ValueError("This frozen pilot requires all three models in tie-breaking order")
        if raw["model_seed"] != 42:
            raise ValueError("Pilot model seed must be 42; use a new protocol for a changed experiment")
        return cls(path, raw, runtime, out, read_json(REPO / "configs/assets.lock.json"))

    @property
    def fingerprint(self):
        return digest({"config": self.raw, "assets": self.assets})

    def dataset(self, name):
        return next(d for d in self.assets["datasets"] if d["name"] == name)

    def data_path(self, d):
        return self.runtime / "data" / d["relative_data_path"]

    def split_path(self, d, split):
        return self.runtime / "data" / d["splits"][split]["relative_path"]

    def units(self, stage="all"):
        if stage not in ("P1", "P2", "P3", "all"):
            raise ValueError("Only frozen core stages P1/P2/P3/all are implemented; P4 needs separate approval")
        units = []
        for d in self.assets["datasets"]:
            for s in range(3):
                assigned_stage = ("P1" if d["name"] in ("credit-g", "concrete_compressive_strength")
                                  else "P2") if s == 0 else "P3"
                if stage != "all" and stage != assigned_stage:
                    continue
                for model in MODEL_ORDER:
                    units.append({"dataset": d["name"], "split": s, "model": model,
                                  "stage": assigned_stage, "problem_type": d["problem_type"]})
        return units


def unit_key(unit):
    return f"{unit['dataset']}/s{unit['split']}/{unit['model']}"
