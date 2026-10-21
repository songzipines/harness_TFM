"""Bounded, isolated, resumable execution. Each group stays on one physical GPU."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time
import threading

from .common import REPO, code_identity, digest, environment, now, read_json, sha256, write_json
from .config import unit_key


@contextmanager
def run_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".run.lock"
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()} host={socket.gethostname()} time={now()}")
    except FileExistsError as error:
        raise RuntimeError(f"Output locked: {lock}. Check its PID/host; remove it only after confirming no run is active") from error
    try:
        yield
    finally:
        lock.unlink()


def reusable(record, fingerprint, root):
    if record.get("execution_fingerprint") != fingerprint or record.get("status") != "success":
        return False
    attempt = Path(root) / record["attempt_relative"]
    return all((attempt / a["file"]).is_file() and sha256(attempt / a["file"]) == a["sha256"]
               for a in record.get("artifacts", {}).values()) and set(record.get("artifacts", {})) == {"validation", "test"}


def terminate_tree(process):
    import psutil
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        parent.kill()
        psutil.wait_procs(children + [parent], timeout=5)
    except psutil.NoSuchProcess:
        pass
    process.wait(timeout=10)


def monitor(command, env, log, timeout, stop_event=None):
    """Timeout includes process startup; peak RSS is sampled for the whole process tree."""
    import psutil
    started, peak = time.perf_counter(), 0
    with Path(log).open("wb") as stream:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
        timed_out, cancelled = False, False
        try:
            while process.poll() is None:
                if stop_event is not None and stop_event.is_set():
                    cancelled = True
                    terminate_tree(process)
                    break
                if time.perf_counter() - started >= timeout:
                    timed_out = True
                    terminate_tree(process)
                    break
                try:
                    parent = psutil.Process(process.pid)
                    processes = [parent] + parent.children(recursive=True)
                    rss = 0
                    for p in processes:
                        try:
                            rss += p.memory_info().rss
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    peak = max(peak, rss)
                except psutil.NoSuchProcess:
                    pass
                time.sleep(0.2)
        except BaseException:
            terminate_tree(process)
            raise
        code = process.wait()
    return {"observed_wall_s": time.perf_counter() - started, "peak_process_tree_rss_mib": peak / 2**20,
            "exit_code": code, "timed_out": timed_out, "cancelled": cancelled}


def run(config, stage, devices, retry_failed=False):
    from .doctor import check
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("Provide distinct physical GPU indices, for example --gpus 0,1,2")
    with run_lock(config.output):
        print("Checking frozen weights, source, data, splits and versions ...", flush=True)
        preflight = check(config)
        by_id = {g["index"]: g for g in preflight["gpus"]}
        if any(g not in by_id for g in devices):
            raise ValueError(f"GPU does not exist; available indices: {list(by_id)}")
        payload = {"config_fingerprint": config.fingerprint, "code": code_identity(),
                   "environment": environment(), "hardware": [by_id[g] for g in devices],
                   "cpu": preflight.get("cpu"),
                   "host": socket.gethostname(), "python_executable": sys.executable}
        fingerprint = digest(payload)
        manifest_path = config.output / "run-manifest.json"
        if manifest_path.exists():
            old = read_json(manifest_path)
            if old["execution_fingerprint"] != fingerprint:
                raise RuntimeError("Code/config/environment/hardware changed. Use a NEW --output directory; old evidence stays intact")
        else:
            write_json(manifest_path, {"created_at": now(), "execution_fingerprint": fingerprint,
                                      "identity": payload, "config": config.raw, "assets": config.assets,
                                      "preflight": preflight, "planned_core_units": 54})
        units = config.units(stage)
        groups = {}
        for u in units:
            groups.setdefault((u["dataset"], u["split"]), []).append(u)
        lanes = [[] for _ in devices]
        group_positions = {key: i for i, key in enumerate(sorted({(u["dataset"], u["split"]) for u in config.units()}))}
        for key, values in sorted(groups.items()):
            random.Random(config.raw["order_seed"] + int(digest(key)[:8], 16)).shuffle(values)
            lanes[group_positions[key] % len(devices)].append(values)
        stop_event = threading.Event()

        def lane(device, jobs):
            for group in jobs:
                for unit in group:
                    if stop_event.is_set():
                        return
                    key = unit_key(unit)
                    base = config.output / "units" / key
                    latest = base / "result.json"
                    if latest.exists():
                        old = read_json(latest)
                        if reusable(old, fingerprint, config.output):
                            print(f"REUSE {key}", flush=True)
                            continue
                        if old.get("status") == "success":
                            raise RuntimeError(f"Successful artifacts corrupted: {key}; use a new output after diagnosis")
                        if not retry_failed:
                            print(f"KEEP FAILURE {key} ({old.get('status')}); explicit --retry-failed required", flush=True)
                            continue
                    base.mkdir(parents=True, exist_ok=True)
                    numbers = [int(p.name.split("-")[-1]) for p in base.glob("attempt-*") if p.is_dir()]
                    attempt = base / f"attempt-{max(numbers, default=0) + 1:03d}"
                    attempt.mkdir(exist_ok=False)
                    request = {"config": str(config.path), "runtime": str(config.runtime), "unit": unit,
                               "attempt_dir": str(attempt)}
                    write_json(attempt / "request.json", request)
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, PYTHONUNBUFFERED="1")
                    print(f"START gpu={device} {key}", flush=True)
                    try:
                        observed = monitor([sys.executable, "-m", "harness.worker", "--request", str(attempt / "request.json")],
                                           env, attempt / "worker.log", config.raw["timeout_s"], stop_event)
                    except BaseException:
                        write_json(latest, {"unit": unit, "status": "interrupted", "execution_fingerprint": fingerprint,
                                            "attempt_relative": str(attempt.relative_to(config.output)).replace("\\", "/")})
                        raise
                    partial = attempt / ("result.json" if (attempt / "result.json").exists() else "progress.json")
                    record = read_json(partial) if partial.exists() else {"unit": unit}
                    if observed["timed_out"]:
                        record.update(status="timeout", error_type="Timeout", error=f"Exceeded {config.raw['timeout_s']} seconds")
                    elif observed.get("cancelled"):
                        record.update(status="interrupted", error_type="Cancelled")
                    elif record.get("status") == "oom":
                        pass  # Preserve the worker's OOM classification even when exit_code is nonzero.
                    elif observed["exit_code"] != 0 and record.get("status") != "oom":
                        record.update(status="error", error_type=record.get("error_type", "WorkerExit"))
                    elif record.get("status") != "success":
                        record.update(status="error", error_type="MissingCompletedResult")
                    record.update(observed)
                    record.update(execution_fingerprint=fingerprint, physical_gpu=by_id[device],
                                  attempt_relative=str(attempt.relative_to(config.output)).replace("\\", "/"))
                    write_json(attempt / "result.json", record)
                    write_json(latest, record)
                    print(f"{record['status'].upper()} {key} wall={observed['observed_wall_s']:.1f}s", flush=True)
        pool = ThreadPoolExecutor(max_workers=len(devices))
        try:
            futures = [pool.submit(lane, d, jobs) for d, jobs in zip(devices, lanes)]
            for future in futures:
                future.result()
        except BaseException:
            stop_event.set()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        statuses = []
        for unit in units:
            path = config.output / "units" / unit_key(unit) / "result.json"
            statuses.append(read_json(path)["status"] if path.exists() else "missing")
        return {"stage": stage, "planned": len(units), "success": statuses.count("success"),
                "failed_or_missing": sum(s != "success" for s in statuses)}
