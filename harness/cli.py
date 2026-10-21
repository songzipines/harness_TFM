"""Run with python -m harness. Heavy imports occur after D-drive paths are set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import MODEL_ORDER, configure_runtime, write_json
from .config import Config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Three-TFM selection-space pilot (no LLM calls)")
    parser.add_argument("--config", help="Experiment JSON; default configs/pilot.json")
    parser.add_argument("--output", help="Override output root; use a new directory after any code/config change")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Show units without loading models")
    plan.add_argument("--stage", choices=["P1", "P2", "P3", "all"], default="all")
    commands.add_parser("doctor", help="Verify frozen environment, weights, data and all 18 splits")
    run = commands.add_parser("run", help="FORMAL experiment; do not use for an engineering check")
    run.add_argument("--stage", choices=["P1", "P2", "P3", "all"], required=True)
    run.add_argument("--gpus", default="0", help="Physical nvidia-smi indices, comma-separated")
    run.add_argument("--retry-failed", action="store_true", help="Explicit new attempts, preserving old logs and compute")
    commands.add_parser("summarize", help="Freeze validation choices, then score and export reports")
    smoke = commands.add_parser("smoke", help="Synthetic engineering check, not formal data")
    smoke.add_argument("--model", choices=MODEL_ORDER, required=True)
    smoke.add_argument("--task", choices=["binary", "multiclass", "regression"], required=True)
    args = parser.parse_args(argv)
    try:
        config = Config.load(args.config, args.output)
        configure_runtime(config.runtime)
        if args.command == "plan":
            units = config.units(args.stage)
            result = {"stage": args.stage, "units": len(units), "runtime_root": str(config.runtime),
                      "output_root": str(config.output), "timeout_s_per_unit": config.raw["timeout_s"],
                      "gpu_hours_max_no_retries": len(units) * config.raw["timeout_s"] / 3600,
                      "formal_jobs_started": 0, "plan": units}
        elif args.command == "doctor":
            from .doctor import check
            result = check(config)
            path = config.runtime / "verification/framework-doctor.json"
            write_json(path, result)
            print(f"Doctor report: {path}", flush=True)
        elif args.command == "run":
            from .runner import run
            result = run(config, args.stage, [v.strip() for v in args.gpus.split(",")], args.retry_failed)
        elif args.command == "summarize":
            from .evaluation import summarize
            result = summarize(config)
        else:
            from .smoke import run_smoke
            if not args.output:
                raise ValueError("Provide a NEW --output directory for each synthetic check (before the smoke subcommand)")
            result = run_smoke(config, args.model, args.task, config.output)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
        if args.command == "run":
            return int(result["failed_or_missing"] > 0)
        return int(result.get("status") in ("error", "oom"))
    except Exception as error:
        print(f"ERROR {type(error).__name__}: {error}", flush=True)
        return 1
