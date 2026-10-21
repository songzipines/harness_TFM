"""One disposable GPU process, one model/dataset/split; never scores query targets."""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import traceback

from .common import configure_runtime, now, read_json, sha256, write_json
from .config import Config


def execute(config, unit, output, device="cuda:0", input_override=None):
    started = time.perf_counter()
    result = {"schema_version": 1, "unit": unit, "status": "running", "started_at": now(),
              "device": device, "timings": {}, "test_targets_accessed": False}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "progress.json", result)
    try:
        import numpy as np
        import torch
        from .adapters import checked_prediction, create_adapter
        from .data import model_input

        torch.set_num_threads(config.raw["cpu_threads"])
        torch.manual_seed(config.raw["model_seed"])
        np.random.seed(config.raw["model_seed"])
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; CPU fallback would change the experiment")
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
            result["gpu"] = torch.cuda.get_device_name(device)

        def phase(name, action):
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            value = action()
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            result["timings"][name] = time.perf_counter() - t0
            write_json(output / "progress.json", result)
            return value

        dataset = config.dataset(unit["dataset"]) if input_override is None else None
        item = phase("prepare_s", lambda: model_input(config, dataset, unit["split"])) if dataset else input_override
        adapter = create_adapter(unit["model"], config, unit["problem_type"], device)
        phase("load_s", adapter.load)
        phase("fit_s", lambda: adapter.fit(item.X_fit, item.y_fit))
        result["classes"] = item.classes
        result["n_fit"], result["n_features"] = item.X_fit.shape
        result["input_dtypes"] = {c: str(t) for c, t in item.X_fit.dtypes.items()}
        result["artifacts"] = {}
        for part, X in (("validation", item.X_validation), ("test", item.X_test)):
            prediction = phase(f"{part}_predict_s", lambda: checked_prediction(adapter, X, len(item.classes)))
            path = output / f"{part}.npz"
            np.savez_compressed(path, row_ids=np.array(item.row_ids[part]), prediction=prediction)
            result["artifacts"][part] = {"file": path.name, "sha256": sha256(path), "n_rows": len(X)}
        result["resolved_config"] = adapter.resolved_config()
        if device.startswith("cuda"):
            result["peak_vram_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
            result["peak_vram_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2**20
        result["status"] = "success"
    except Exception as error:
        result["status"] = "oom" if "out of memory" in str(error).lower() else "error"
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        traceback.print_exc()
        try:
            if device.startswith("cuda") and torch.cuda.is_initialized():
                result["peak_vram_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
                result["peak_vram_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2**20
        except Exception:
            pass
    result["worker_total_s"] = time.perf_counter() - started
    result["finished_at"] = now()
    write_json(output / "result.json", result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--request", required=True)
    args = p.parse_args()
    request = read_json(args.request)
    configure_runtime(request["runtime"])
    config = Config.load(request["config"])
    result = execute(config, request["unit"], request["attempt_dir"])
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
