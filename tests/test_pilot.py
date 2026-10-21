"""CPU tests of research-critical boundaries, not assertions about TFM quality."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from harness.adapters import checked_prediction, precision_record
from harness.common import MODEL_ORDER, REPO, digest, read_json, sha256, write_json
from harness.config import Config
from harness.data import load_split, model_input
from harness.evaluation import selection_space, summarize
from harness.metrics import constant_prediction, metrics
from harness.routing import DecisionContext, ValidationSelector
from harness.runner import monitor, reusable, run, run_lock


class PilotTests(unittest.TestCase):
    def test_stage_counts_and_disjoint_units(self):
        c = Config.load()
        counts = [len(c.units(s)) for s in ("P1", "P2", "P3", "all")]
        self.assertEqual(counts, [6, 12, 36, 54])
        self.assertEqual(len({(u['dataset'], u['split'], u['model']) for u in c.units()}), 54)

    def test_validation_choice_and_tie_order(self):
        router = ValidationSelector()
        self.assertEqual(router.select(DecisionContext("x", 0, "binary", dict.fromkeys(MODEL_ORDER, 0.2))), "tabpfn")
        self.assertEqual(router.select(DecisionContext("x", 0, "binary", {"tabpfn": .3, "tabicl": .2, "limix": .25})), "tabicl")
        with self.assertRaises(ValueError):
            router.select(DecisionContext("x", 0, "binary", {"tabpfn": .1}))
        self.assertNotIn("test", str(DecisionContext.__annotations__))

    def test_laplace_baseline_and_metric_units(self):
        p = constant_prediction("multiclass", np.array([0, 0, 1]), 2, 3)
        np.testing.assert_allclose(p[0], [3/6, 2/6, 1/6])
        b = metrics("binary", [0, 1, 0, 1], np.full((4, 2), .5))
        self.assertEqual(b["loss"], .5)
        r = metrics("regression", [1, 3], [0, 3])
        self.assertAlmostEqual(r["loss"], np.sqrt(.5))
        self.assertTrue(np.isfinite(metrics("multiclass", [0, 1], np.array([[1., 0], [1., 0]]))["loss"]))

    def test_probability_alignment_and_invalid_output(self):
        class Fake:
            task = "multiclass"
            def predict(self, X): return np.array([[.8, .2], [.1, .9]])
            def class_order(self): return [1, 0]
        p = checked_prediction(Fake(), pd.DataFrame({"x": [0, 1]}), 2)
        np.testing.assert_allclose(p, [[.2, .8], [.9, .1]])
        with patch.object(Fake, "predict", return_value=np.array([[np.nan, 0], [0, 1]])):
            with self.assertRaises(ValueError): checked_prediction(Fake(), [0, 1], 2)
        with patch.object(Fake, "predict", return_value=np.array([[.8, .3], [.1, .9]])):
            with self.assertRaises(ValueError): checked_prediction(Fake(), [0, 1], 2)

    def test_resolved_precision_records_values_not_just_object_addresses(self):
        from types import SimpleNamespace
        estimator = SimpleNamespace(inference_config_=SimpleNamespace(COL_CONFIG={"use_amp": True}, ROW_CONFIG={"use_amp": False}))
        record = precision_record(estimator)
        self.assertEqual(record["resolved_inference_config"]["COL_CONFIG"]["use_amp"], "True")
        json.dumps(record,allow_nan=False)

    def test_selection_space_averages_splits_before_oracle(self):
        rows = []
        # Each type has two tasks with distinct winners; same fixed model is best in aggregate.
        for d, values in (("a", [.2, .8, .7]), ("b", [.8, .2, .9])):
            rows.extend(dict(dataset=d, problem_type="binary", model=m, normalized_loss=v)
                        for m, v in zip(MODEL_ORDER, values))
        result = selection_space(pd.DataFrame(rows))
        self.assertAlmostEqual(result["SBS"], .5)
        self.assertAlmostEqual(result["VBS"], .2)
        self.assertAlmostEqual(result["G_type"], .6)

    def test_lock_prevents_two_writers(self):
        with tempfile.TemporaryDirectory() as d:
            with run_lock(d):
                with self.assertRaises(RuntimeError):
                    with run_lock(d): pass
            self.assertFalse((Path(d) / ".run.lock").exists())

    def test_timeout_stops_process(self):
        with tempfile.TemporaryDirectory() as d:
            result = monitor([sys.executable, "-c", "import time; time.sleep(30)"], dict(os.environ), Path(d) / "log", .5)
            self.assertTrue(result["timed_out"])
            self.assertLess(result["observed_wall_s"], 10)


class SyntheticWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = copy.deepcopy(Config.load(output=self.root / "run"))
        self.config.runtime = self.root
        self.config.raw = dict(self.config.raw, runtime_root=str(self.root))
        self.config.assets = copy.deepcopy(self.config.assets)
        # Six tiny local fixture tables. These ARE NOT the actual benchmark datasets.
        for d in self.config.assets["datasets"]:
            n = 30
            target = np.arange(n) % (2 if d["problem_type"] == "binary" else 3)
            if d["problem_type"] == "regression": target = np.arange(n, dtype=float)
            df = pd.DataFrame({"number": np.arange(n, dtype=float),
                               "category": pd.Categorical(["a", "b"] * 12 + ["unseen"] * 6), "y": target})
            path = self.config.data_path(d)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
            d.update(n_rows=n, n_features=2, target="y", feature_names=["number", "category"],
                     categorical_columns=["category"], data_sha256=sha256(path))
            for s in range(3):
                spec = {"fit": list(range(18)), "validation": list(range(18,24)), "test": list(range(24,30)),
                        "repeat": s, "fold": 0, "sample": 0, "validation_seed": 20261020+s,
                        "counts": {"fit": 18, "validation": 6, "test": 6}}
                spec["indices_sha256"] = digest({k: spec[k] for k in ("fit", "validation", "test")})
                d["splits"][s]["indices_sha256"] = spec["indices_sha256"]
                write_json(self.config.split_path(d,s), spec)

    def tearDown(self): self.temp.cleanup()

    def test_dtype_target_boundary_and_fit_only_categories(self):
        item = model_input(self.config, self.config.assets["datasets"][0], 0)
        self.assertEqual(item.X_fit.number.dtype, np.dtype("float64"))
        self.assertIsInstance(item.X_fit.category.dtype, pd.CategoricalDtype)
        self.assertEqual(set(item.X_fit.category.cat.categories), {"a", "b"})
        self.assertTrue((item.X_test.category == "unseen").all())
        self.assertNotIn("y", item.X_fit)
        self.assertFalse(hasattr(item, "y_test"))
        self.assertFalse(hasattr(item, "y_validation"))

    def test_modified_split_and_data_rejected(self):
        d = self.config.assets["datasets"][0]
        path = self.config.split_path(d, 0)
        spec = read_json(path); spec["test"][0] = spec["fit"][0]; write_json(path,spec)
        with self.assertRaises(ValueError): load_split(self.config,d,0)
        self.config.data_path(d).write_bytes(b"tampered")
        with self.assertRaises(ValueError): model_input(self.config,d,1)

    def fake_monitor(self, command, env, log, timeout, stop_event=None):
        req = read_json(command[-1]); unit = req["unit"]
        d = self.config.dataset(unit["dataset"])
        item = model_input(self.config,d,unit["split"])
        out = Path(req["attempt_dir"])
        artifacts = {}
        # Validation prefers PFN; test prefers LIMIX. This detects test-based choice leakage.
        for part in ("validation", "test"):
            df = pd.read_parquet(self.config.data_path(d))
            y = df.y.iloc[item.row_ids[part]].to_numpy()
            rank = MODEL_ORDER.index(unit["model"])
            error = rank + 1 if part == "validation" else 3 - rank
            if d["problem_type"] == "regression": p = y + error
            else:
                k=len(item.classes)
                p = np.full((len(y),k), .1)
                p[np.arange(len(y)),y.astype(int)] = .9
                if rank > 0 and part == "validation" or rank < 2 and part == "test":
                    p = np.roll(p, 1, axis=0)
                p /= p.sum(axis=1,keepdims=True)
            file = out / f"{part}.npz"
            np.savez_compressed(file, row_ids=np.array(item.row_ids[part]), prediction=p)
            artifacts[part] = {"file": file.name, "sha256":sha256(file),"n_rows":len(y)}
        result = {"unit":unit,"status":"success","classes":item.classes,"artifacts":artifacts,
                  "resolved_config":{"n_estimators":8}, "peak_vram_allocated_mib":1,
                  "timings":dict(load_s=1,fit_s=2,validation_predict_s=3,test_predict_s=4)}
        write_json(out/'result.json',result)
        return dict(observed_wall_s=11,peak_process_tree_rss_mib=20,exit_code=0,timed_out=False)

    def test_entire_scheduler_resume_selection_and_report(self):
        gpu = {"index":"0","uuid":"fixture","name":"FAKE_NOT_A_GPU","driver":"test","memory_mib":"1"}
        with patch("harness.doctor.check",return_value={"status":"passed","gpus":[gpu]}), patch("harness.runner.monitor",side_effect=self.fake_monitor) as worker:
            first = run(self.config,"all",["0"])
            self.assertEqual(first["success"],54)
            second = run(self.config,"all",["0"])
            self.assertEqual(second["success"],54)
            self.assertEqual(worker.call_count,54)
        summary = summarize(self.config)
        self.assertTrue(summary["formal_complete"])
        self.assertEqual(summary["common_success_groups"],18)
        choices = pd.read_csv(self.config.output/'analysis/selection_decisions.csv')
        self.assertEqual(set(choices.selected),{"tabpfn"})
        report = self.config.output/'analysis/report.md'
        self.assertTrue(report.exists())
        # A failed unit removes only that complete group, while remaining visible in coverage.
        latest = self.config.output/'units/credit-g/s0/tabpfn/result.json'
        result = read_json(latest); result['status']='oom'; write_json(latest,result)
        summary = summarize(self.config)
        self.assertFalse(summary['formal_complete'])
        self.assertEqual(summary['common_success_groups'],17)
        self.assertEqual(summary['statuses']['oom'],1)

    def test_resume_rejects_corrupt_artifacts_and_changed_identity(self):
        directory = self.root/'attempt'; directory.mkdir()
        artifacts={}
        for part in ('validation','test'):
            p=directory/f'{part}.npz'; p.write_bytes(b'predictions')
            artifacts[part]={'file':p.name,'sha256':sha256(p)}
        record=dict(status='success',execution_fingerprint='a',attempt_relative='attempt',artifacts=artifacts)
        self.assertTrue(reusable(record,'a',self.root))
        self.assertFalse(reusable(record,'b',self.root))
        (directory/'test.npz').write_bytes(b'corrupt')
        self.assertFalse(reusable(record,'a',self.root))

    def test_failed_units_require_explicit_retry_and_preserve_attempts(self):
        gpu = {"index":"0","uuid":"fixture","name":"FAKE_NOT_A_GPU","driver":"test","memory_mib":"1"}
        def fail_one(command, env, log, timeout, stop_event=None):
            req=read_json(command[-1])
            if req['unit']['model']=='limix' and req['unit']['dataset']=='credit-g':
                write_json(Path(req['attempt_dir'])/'result.json',dict(unit=req['unit'],status='oom',error_type='SyntheticOOM'))
                return dict(observed_wall_s=1,peak_process_tree_rss_mib=1,exit_code=1,timed_out=False)
            return self.fake_monitor(command,env,log,timeout,stop_event)
        with patch('harness.doctor.check',return_value={'status':'passed','gpus':[gpu]}), patch('harness.runner.monitor',side_effect=fail_one) as worker:
            self.assertEqual(run(self.config,'P1',['0'])['success'],5)
            run(self.config,'P1',['0'])
            self.assertEqual(worker.call_count,6)
        with patch('harness.doctor.check',return_value={'status':'passed','gpus':[gpu]}), patch('harness.runner.monitor',side_effect=self.fake_monitor) as worker:
            self.assertEqual(run(self.config,'P1',['0'],retry_failed=True)['success'],6)
            self.assertEqual(worker.call_count,1)
        base=self.config.output/'units/credit-g/s0/limix'
        self.assertEqual(read_json(base/'attempt-001/result.json')['status'],'oom')
        self.assertEqual(read_json(base/'attempt-002/result.json')['status'],'success')
        self.config.raw['cpu_threads']=3
        with patch('harness.doctor.check',return_value={'status':'passed','gpus':[gpu]}):
            with self.assertRaisesRegex(RuntimeError,'changed'):
                run(self.config,'P1',['0'])


if __name__ == "__main__": unittest.main()
