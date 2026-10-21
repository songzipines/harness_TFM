"""Independently audit the prepared data against the downloaded official split files."""
import hashlib
import json
from pathlib import Path


import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.common import configure_runtime
ROOT = Path(os.environ.get("TFM_RUNTIME_ROOT", str(Path(__file__).resolve().parents[2] / "tfm_runtime"))).resolve()
configure_runtime(ROOT)

import arff
import pandas as pd

DATA = ROOT / "data"


def main():
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["datasets"]) == 6
    checks = []
    for dataset in manifest["datasets"]:
        name = dataset["name"]
        folder = DATA / name
        parquet = folder / "data.parquet"
        with parquet.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == dataset["data_sha256"]
        frame = pd.read_parquet(parquet)
        assert len(frame) == dataset["n_rows"]
        assert dataset["target"] not in dataset["feature_names"]
        assert not set(dataset["ignored_columns"]) & set(dataset["feature_names"])
        assert not frame[dataset["target"]].isna().any()
        with (folder / "raw" / "splits.arff").open(encoding="utf-8") as stream:
            raw = arff.load(stream)
        official = pd.DataFrame(raw["data"], columns=[column[0] for column in raw["attributes"]])
        for repeat, split_meta in enumerate(dataset["splits"]):
            split = json.loads((folder / "splits" / f"s{repeat}.json").read_text(encoding="utf-8"))
            fit, val, test = (set(split[key]) for key in ("fit", "validation", "test"))
            assert not (fit & val or fit & test or val & test)
            assert fit | val | test == set(range(len(frame)))
            subset = official[(official["repeat"].astype(int) == repeat) &
                              (official["fold"].astype(int) == 0)]
            train_ids = set(subset.loc[subset["type"] == "TRAIN", "rowid"].astype(int))
            test_ids = set(subset.loc[subset["type"] == "TEST", "rowid"].astype(int))
            assert fit | val == train_ids and test == test_ids
            canonical = json.dumps({key: split[key] for key in ("fit", "validation", "test")},
                                   sort_keys=True, separators=(",", ":")).encode()
            assert hashlib.sha256(canonical).hexdigest() == split_meta["indices_sha256"]
            if dataset["problem_type"] != "regression":
                y = frame[dataset["target"]]
                assert set(y.iloc[sorted(fit)]) == set(y)
                assert set(y.iloc[sorted(val)]) == set(y)
                assert set(y.iloc[sorted(test)]) == set(y)
            checks.append({"dataset": name, "split": f"s{repeat}", "status": "passed",
                           "counts": split["counts"], "indices_sha256": split["indices_sha256"]})
    output = {"status": "passed", "datasets": 6, "splits_checked": len(checks),
              "formal_model_runs": 0, "checks": checks}
    (ROOT / "verification" / "data_checks.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
