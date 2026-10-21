"""Prepare the six agreed OpenML tasks, keeping official row order and outer splits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.common import configure_runtime
ROOT = Path(os.environ.get("TFM_RUNTIME_ROOT", str(Path(__file__).resolve().parents[2] / "tfm_runtime"))).resolve()
configure_runtime(ROOT)

DATA = ROOT / "data"
TASKS = [
    ("credit-g", 46918, 363626, 1000, "binary", 2),
    ("qsar-biodeg", 46952, 363696, 1054, "binary", 2),
    ("website_phishing", 46963, 363707, 1353, "multiclass", 3),
    ("splice", 46958, 363702, 3190, "multiclass", 3),
    ("concrete_compressive_strength", 46917, 363625, 1030, "regression", None),
    ("healthcare_insurance_expenses", 46931, 363675, 1338, "regression", None),
]


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def digest(path: Path, algorithm: str = "sha256") -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def fetch(url: str, target: Path) -> None:
    if target.exists() and target.stat().st_size:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "TFM-pilot-setup/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as stream:
                while block := response.read(2**20):
                    stream.write(block)
            partial.replace(target)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def download_raw(task: tuple) -> tuple[Path, dict, dict, dict]:
    name, dataset_id, task_id, *_ = task
    folder = DATA / name
    raw = folder / "raw"
    for endpoint, filename in [(f"task/{task_id}", "task.json"),
                               (f"data/{dataset_id}", "dataset.json"),
                               (f"data/features/{dataset_id}", "features.json")]:
        fetch(f"https://www.openml.org/api/v1/json/{endpoint}", raw / filename)
    task_meta = json.loads((raw / "task.json").read_text(encoding="utf-8"))["task"]
    dataset = json.loads((raw / "dataset.json").read_text(encoding="utf-8"))["data_set_description"]
    features = json.loads((raw / "features.json").read_text(encoding="utf-8"))["data_features"]
    inputs = {item["name"]: item for item in task_meta["input"]}
    assert int(inputs["source_data"]["data_set"]["data_set_id"]) == dataset_id
    assert int(dataset["id"]) == dataset_id
    fetch(dataset["url"], raw / "dataset.arff")
    fetch(inputs["estimation_procedure"]["estimation_procedure"]["data_splits_url"], raw / "splits.arff")
    expected_md5 = dataset.get("md5_checksum")
    if expected_md5 and digest(raw / "dataset.arff", "md5") != expected_md5:
        raise RuntimeError(f"Official MD5 mismatch for {name}")
    print(f"Raw data downloaded and identity checked: {name}", flush=True)
    return folder, task_meta, dataset, features


def build(task: tuple, downloaded: tuple) -> dict:
    import arff
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    name, dataset_id, task_id, expected_rows, problem_type, expected_classes = task
    folder, task_meta, dataset, features = downloaded
    raw = folder / "raw"
    with (raw / "dataset.arff").open(encoding="utf-8") as stream:
        table = arff.load(stream)
    columns = [attribute[0] for attribute in table["attributes"]]
    frame = pd.DataFrame(table["data"], columns=columns)
    assert len(frame) == expected_rows, (name, len(frame), expected_rows)
    assert len(columns) == len(set(columns)), "Duplicate column names"
    source = next(item["data_set"] for item in task_meta["input"] if item["name"] == "source_data")
    target = source["target_feature"]
    assert target in frame and not frame[target].isna().any()
    feature_meta = features["feature"]
    ignored = [item["name"] for item in feature_meta
               if item.get("is_ignore") == "true" or item.get("is_row_identifier") == "true"]
    feature_names = [column for column in columns if column != target and column not in ignored]
    categorical = []
    for column, kind in table["attributes"]:
        if isinstance(kind, list):
            frame[column] = pd.Categorical(frame[column], categories=kind)
            if column in feature_names:
                categorical.append(column)
        elif str(kind).upper() in {"NUMERIC", "REAL", "INTEGER"}:
            frame[column] = pd.to_numeric(frame[column])
    y = frame[target]
    is_classification = problem_type != "regression"
    if is_classification:
        assert y.nunique() == expected_classes, (name, y.nunique())
    else:
        assert np.isfinite(y.to_numpy(dtype=float)).all()
    with (raw / "splits.arff").open(encoding="utf-8") as stream:
        split_arff = arff.load(stream)
    split_frame = pd.DataFrame(split_arff["data"], columns=[a[0] for a in split_arff["attributes"]])
    prepared_splits = []
    for repeat in range(3):
        selected = split_frame[(split_frame["repeat"].astype(int) == repeat) &
                               (split_frame["fold"].astype(int) == 0)]
        kind = selected["type"].astype(str).str.upper()
        outer_train = selected.loc[kind == "TRAIN", "rowid"].astype(int).to_numpy()
        test = selected.loc[kind == "TEST", "rowid"].astype(int).to_numpy()
        if not len(outer_train) or not len(test):
            raise RuntimeError(f"Missing official repeat {repeat}, fold 0 for {name}")
        fit, val = train_test_split(outer_train, test_size=0.2, random_state=20261020 + repeat,
                                   stratify=y.iloc[outer_train].to_numpy() if is_classification else None)
        fit, val, test = (sorted(map(int, index)) for index in (fit, val, test))
        groups = [set(fit), set(val), set(test)]
        assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        assert len(set.union(*groups)) == len(frame)
        assert set(fit) | set(val) == set(map(int, outer_train))
        if is_classification:
            assert set(y.iloc[fit]) == set(y), (name, "fit misses a class")
        split = {"repeat": repeat, "fold": 0, "sample": 0,
                 "validation_seed": 20261020 + repeat,
                 "fit": fit, "validation": val, "test": test,
                 "counts": {"fit": len(fit), "validation": len(val), "test": len(test)}}
        canonical = json.dumps({key: split[key] for key in ("fit", "validation", "test")},
                               separators=(",", ":"), sort_keys=True).encode()
        split["indices_sha256"] = hashlib.sha256(canonical).hexdigest()
        path = folder / "splits" / f"s{repeat}.json"
        save_json(path, split)
        prepared_splits.append({"path": str(path), "relative_path": path.relative_to(DATA).as_posix(),
                                "indices_sha256": split["indices_sha256"],
                                "counts": split["counts"]})
    parquet = folder / "data.parquet"
    frame.to_parquet(parquet, index=False)
    result = {
        "name": name, "openml_name": dataset["name"], "dataset_id": dataset_id, "task_id": task_id,
        "problem_type": problem_type, "n_rows": len(frame), "n_features": len(feature_names),
        "target": target, "feature_names": feature_names, "ignored_columns": ignored,
        "categorical_columns": categorical,
        "data_path": str(parquet), "relative_data_path": parquet.relative_to(DATA).as_posix(),
        "dtypes": {column: str(frame[column].dtype) for column in frame.columns},
        "data_sha256": digest(parquet),
        "raw_arff_sha256": digest(raw / "dataset.arff"), "raw_arff_md5": digest(raw / "dataset.arff", "md5"),
        "official_splits_sha256": digest(raw / "splits.arff"),
        "splits": prepared_splits,
        "note": "Development pilot; full-table metadata is for auditing, not a router input.",
    }
    save_json(folder / "manifest.json", result)
    print(f"Prepared {name}: rows={len(frame)}, features={len(feature_names)}, splits=3", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    if os.name == "nt" and ROOT.drive.upper() != "D:":
        raise RuntimeError("This runtime must be on drive D.")
    datasets = []
    for task in TASKS:
        raw = download_raw(task)
        if not args.download_only:
            datasets.append(build(task, raw))
    if datasets:
        save_json(DATA / "manifest.json", {"status": "prepared", "datasets": datasets,
                                          "dataset_count": len(datasets), "outer_split_count": 18,
                                          "formal_model_runs": 0})


if __name__ == "__main__":
    main()
