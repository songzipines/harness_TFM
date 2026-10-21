"""Download only the four selected public checkpoints into this D-drive runtime."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.common import configure_runtime
ROOT = Path(os.environ.get("TFM_RUNTIME_ROOT", str(Path(__file__).resolve().parents[2] / "tfm_runtime"))).resolve()
configure_runtime(ROOT)

WEIGHTS = [
    {
        "model": "TabPFN-3.5", "repo": "Prior-Labs/tabpfn_3_5",
        "revision": "06bf2ba35c80a92a3b9abb436b99cf49e7a0365e",
        "file": "tabpfn-v3.5-20260909.safetensors", "directory": "tabpfn",
        "bytes": 876027932,
        "sha256": "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3",
    },
    {
        "model": "TabICLv2-classifier", "repo": "jingang/TabICL",
        "revision": "4dcd344ece2c00be9e831fdd35bed57b5ad83e19",
        "file": "tabicl-classifier-v2-20260212.ckpt", "directory": "tabicl",
        "bytes": 110368038,
        "sha256": "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0",
    },
    {
        "model": "TabICLv2-regressor", "repo": "jingang/TabICL",
        "revision": "4dcd344ece2c00be9e831fdd35bed57b5ad83e19",
        "file": "tabicl-regressor-v2-20260212.ckpt", "directory": "tabicl",
        "bytes": 114324594,
        "sha256": "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a",
    },
    {
        "model": "LimiX-2", "repo": "stable-ai/LimiX-2",
        "revision": "de07b679e74a41b50b9de18251a8fa245e537440",
        "file": "LimiX-2.ckpt", "directory": "limix",
        "bytes": 1625774719,
        "sha256": "ed01f7a0a18d25451bc9d1058b479ba6b68048aa1ebde06156c8fcfaf8b8fe81",
    },
]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(item: dict) -> dict:
    target = ROOT / "weights" / item["directory"] / item["file"]
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    url = f"https://huggingface.co/{item['repo']}/resolve/{item['revision']}/{item['file']}"
    if target.exists():
        if target.stat().st_size != item["bytes"] or sha256(target) != item["sha256"]:
            raise RuntimeError(f"Existing checkpoint fails verification: {target}")
        print(f"Already verified: {item['model']}", flush=True)
    else:
        for attempt in range(3):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                if offset < item["bytes"]:
                    headers = {"User-Agent": "TFM-research-setup/1.0"}
                    if offset:
                        headers["Range"] = f"bytes={offset}-"
                    request = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(request, timeout=120) as response:
                        resume = offset > 0 and response.status == 206
                        completed = offset if resume else 0
                        print(f"Downloading {item['model']} ({item['bytes'] / 2**20:.1f} MiB)", flush=True)
                        reported = time.monotonic()
                        with partial.open("ab" if resume else "wb") as stream:
                            while block := response.read(4 * 2**20):
                                stream.write(block)
                                completed += len(block)
                                if time.monotonic() - reported >= 25:
                                    print(f"  {item['model']}: {completed / item['bytes']:.0%}", flush=True)
                                    reported = time.monotonic()
                if partial.stat().st_size != item["bytes"]:
                    raise RuntimeError(f"Incomplete download: {partial}")
                if sha256(partial) != item["sha256"]:
                    raise RuntimeError(f"SHA256 mismatch: {partial}; remove this file before retrying")
                partial.replace(target)
                print(f"Verified: {item['model']}", flush=True)
                break
            except Exception as error:
                print(f"Attempt {attempt + 1} failed: {type(error).__name__}: {error}", flush=True)
                if attempt == 2 or "SHA256 mismatch" in str(error):
                    raise
                time.sleep(2)
    return {**item, "path": str(target), "relative_path": target.relative_to(ROOT).as_posix(), "verified": True}


def main() -> None:
    if os.name == "nt" and ROOT.drive.upper() != "D:":
        raise RuntimeError("This runtime must be on drive D.")
    os.environ.setdefault("TEMP", str(ROOT / "tmp"))
    os.environ.setdefault("TMP", str(ROOT / "tmp"))
    manifest = {"status": "in_progress", "weights": []}
    output = ROOT / "weights-manifest.json"
    for item in WEIGHTS:
        manifest["weights"].append(download(item))
        output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["status"] = "verified"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
