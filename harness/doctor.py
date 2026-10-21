"""Check frozen inputs and environment before committing GPU time."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import os
import platform

from .common import environment, git_head, read_json, sha256, digest


def source_identity(root):
    # Windows git checkouts use CRLF; ignore only line-ending differences across OSes.
    files = {str(p.relative_to(root)).replace("\\", "/"): content_hash(p)
             for p in sorted(Path(root).rglob("*.py")) if ".git" not in p.parts}
    return digest(files)


def content_hash(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def gpu_inventory():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    return [dict(zip(("index", "uuid", "name", "driver", "memory_mib"),
                     [v.strip() for v in line.split(",")])) for line in result.stdout.strip().splitlines()]


def check(config, verify_hashes=True):
    from .data import load_split, model_input, scoring_target
    current = environment()
    for package, expected in config.assets["required_packages"].items():
        if current["packages"].get(package) != expected:
            raise ValueError(f"Version mismatch: {package}: {current['packages'].get(package)} != {expected}")
    if current["python"] != config.assets["python"]:
        raise ValueError(f"Python version changed: {current['python']}")
    for entry in config.assets["weights"]:
        path = config.runtime / entry["relative_path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Missing/wrong-size weight: {path}")
        if verify_hashes and sha256(path) != entry["sha256"]:
            raise ValueError(f"Checkpoint SHA256 changed: {path}")
    limix = config.assets["models"]["limix"]
    source = config.runtime / "sources/LimiX"
    if git_head(source) != limix["source_commit"]:
        raise ValueError("LimiX source revision changed")
    if source_identity(source) != config.assets["limix_python_tree_sha256"]:
        raise ValueError("LimiX source content changed, including the registered cache patch")
    for kind in ("classification", "regression"):
        if content_hash(config.runtime / limix[f"{kind}_config"]) != limix[f"{kind}_config_content_sha256"]:
            raise ValueError("LimiX inference config changed")
    counts = []
    for d in config.assets["datasets"]:
        if sha256(config.data_path(d)) != d["data_sha256"]:
            raise ValueError(f"Data SHA256 changed: {d['name']}")
        for split in range(3):
            spec = load_split(config, d, split)
            item = model_input(config, d, split)
            for part in ("validation", "test"):
                scoring_target(config, d, split, part, item.classes)
            counts.append({"dataset": d["name"], "split": split, "counts": spec["counts"]})
    # This layer checks labels only for coverage, before any routing; it publishes no scores.
    import psutil
    return {"status": "passed", "environment": current, "gpus": gpu_inventory(),
            "cpu": {"name": platform.processor(), "logical_cores": os.cpu_count(),
                    "ram_mib": psutil.virtual_memory().total / 2**20},
            "checked_weight_hashes": verify_hashes, "checked_splits": counts,
            "formal_unit_count": len(config.units())}
