"""Small shared file/identity helpers; no GPU imports here."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
MODEL_ORDER = ("tabpfn", "tabicl", "limix")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def configure_runtime(root):
    """Process-local paths only. Call before importing numerical/model packages."""
    root = Path(root).resolve()
    if os.name == "nt" and root.drive.upper() != "D:":
        raise ValueError("Windows project runtime must be on D:, including caches and temporary files")
    paths = {
        "TEMP": "tmp", "TMP": "tmp", "TMPDIR": "tmp", "PIP_CACHE_DIR": "cache/pip",
        "HF_HOME": "cache/huggingface", "HF_HUB_CACHE": "cache/huggingface/hub",
        "HF_XET_CACHE": "cache/huggingface/xet", "HF_ASSETS_CACHE": "cache/huggingface/assets",
        "TORCH_HOME": "cache/torch", "TABPFN_MODEL_CACHE_DIR": "weights/tabpfn",
        "MPLCONFIGDIR": "cache/matplotlib", "NUMBA_CACHE_DIR": "cache/numba",
        "CUDA_CACHE_PATH": "cache/cuda", "PYTORCH_KERNEL_CACHE_PATH": "cache/torch-kernels",
        "TORCHINDUCTOR_CACHE_DIR": "cache/torch-inductor", "TORCH_EXTENSIONS_DIR": "cache/torch-extensions",
        "TRITON_CACHE_DIR": "cache/triton", "JOBLIB_TEMP_FOLDER": "tmp/joblib",
        "LIMIX_CACHE_DIR": "cache/limix", "XDG_CACHE_HOME": "cache/xdg",
        "XDG_CONFIG_HOME": "config", "PYTHONPYCACHEPREFIX": "cache/pycache",
    }
    for variable, relative in paths.items():
        target = root / relative
        target.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(target)
    os.environ.update({"TFM_RUNTIME_ROOT": str(root), "HF_HUB_OFFLINE": "1",
                       "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
                       "TABPFN_NO_BROWSER": "1", "PYTHONNOUSERSITE": "1",
                       "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
    # Changing this environment variable after Python startup is insufficient in this process.
    sys.pycache_prefix = str(root / "cache/pycache")
    import tempfile
    tempfile.tempdir = str(root / "tmp")


def environment():
    packages = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            packages[name] = dist.version
    packages = dict(sorted(packages.items(), key=lambda item: item[0].lower()))
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def code_identity():
    files = {str(p.relative_to(REPO)).replace("\\", "/"): sha256(p)
             for p in sorted((REPO / "harness").rglob("*.py"))}
    return {"sha256": digest(files), "files": files}


def git_head(path):
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True)
    return result.stdout.strip()
