"""Validation decisions are frozen before this module loads any test targets."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .common import MODEL_ORDER, code_identity, digest, now, read_json, sha256, write_json
from .config import unit_key
from .data import load_split, model_input, scoring_target
from .metrics import constant_prediction, metrics
from .routing import DecisionContext, ValidationSelector
from .runner import reusable, run_lock


def csv_out(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_prediction(root, record, part, expected_rows):
    meta = record["artifacts"][part]
    path = root / record["attempt_relative"] / meta["file"]
    if sha256(path) != meta["sha256"]:
        raise ValueError(f"Prediction SHA256 changed: {path}")
    with np.load(path, allow_pickle=False) as a:
        if not np.array_equal(a["row_ids"], expected_rows):
            raise ValueError("Prediction row IDs do not match frozen split")
        return a["prediction"].copy()


def selection_space(frame):
    """Input has one row per dataset/model AFTER averaging common-success splits."""
    if frame.empty:
        return {"status": "no_common_success"}
    matrix = frame.pivot(index="dataset", columns="model", values="normalized_loss")[list(MODEL_ORDER)]
    sbs_losses = matrix.mean()
    sbs = float(sbs_losses.min())
    vbs = float(matrix.min(axis=1).mean())
    by_type = {}
    types = frame.drop_duplicates("dataset").set_index("dataset")["problem_type"]
    type_sum = 0.0
    for task in types.unique():
        subset = matrix.loc[types[types == task].index]
        losses = subset.mean()
        best = min(MODEL_ORDER, key=lambda m: (losses[m], MODEL_ORDER.index(m)))
        value = float(losses[best])
        by_type[task] = {"best_fixed_posthoc": best, "sbs": value, "n_datasets": len(subset)}
        type_sum += value * len(subset)
    sbs_type = type_sum / len(matrix)
    return {"status": "exploratory", "n_datasets": len(matrix), "SBS": sbs, "VBS": vbs,
            "SBS_model_posthoc": min(MODEL_ORDER, key=lambda m: (sbs_losses[m], MODEL_ORDER.index(m))),
            "SBS_type": sbs_type, "G_all": (sbs - vbs) / max(sbs, 1e-12),
            "G_type": (sbs_type - vbs) / max(sbs_type, 1e-12), "by_type": by_type,
            "VBS_without": {m: float(matrix.drop(columns=m).min(axis=1).mean()) for m in MODEL_ORDER},
            "warning": "Posthoc descriptive bounds; six datasets, not 54 independent tasks"}


def table(rows, columns):
    if not rows:
        return "暂无共同成功结果。\n"
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.6g}"
        return str(value).replace("|", "\\|")
    return ("| " + " | ".join(columns) + " |\n| " + " | ".join("---" for _ in columns) + " |\n" +
            "\n".join("| " + " | ".join(cell(r.get(c, "")) for c in columns) + " |" for r in rows) + "\n")


def summarize(config):
    with run_lock(config.output):
        return _summarize(config)


def _summarize(config):
    root = config.output
    manifest = read_json(root / "run-manifest.json")
    if manifest["identity"]["config_fingerprint"] != config.fingerprint:
        raise ValueError("Analysis config/assets differ from the frozen run")
    if manifest["identity"]["code"]["sha256"] != code_identity()["sha256"]:
        raise ValueError("Analysis code changed since execution; preserve the original checkout or create a separately versioned re-analysis")
    output = root / "analysis"
    output.mkdir(exist_ok=True)
    records, coverage = {}, []
    for u in config.units():
        path = root / "units" / unit_key(u) / "result.json"
        r = read_json(path) if path.exists() else {"unit": u, "status": "missing"}
        if r["status"] == "success" and not reusable(r, manifest["execution_fingerprint"], root):
            raise ValueError(f"Invalid completed evidence: {unit_key(u)}")
        records[(u["dataset"], u["split"], u["model"])] = r
        coverage.append({**u, "status": r["status"], "error_type": r.get("error_type", ""),
                         "observed_wall_s": r.get("observed_wall_s", "")})
    csv_out(output / "coverage.csv", coverage)

    # PHASE 1: only validation predictions/labels and model costs enter the selector.
    val_scores, decisions = {}, []
    for d in config.assets["datasets"]:
        for s in range(3):
            group = {m: records[(d["name"], s, m)] for m in MODEL_ORDER}
            successful = {m: r for m, r in group.items() if r["status"] == "success"}
            if not successful:
                continue
            item = model_input(config, d, s)
            y = scoring_target(config, d, s, "validation", item.classes)
            for m, r in successful.items():
                if r["classes"] != item.classes:
                    raise ValueError("Prediction class mapping differs from fit mapping")
                prediction = load_prediction(root, r, "validation", item.row_ids["validation"])
                score = metrics(d["problem_type"], y, prediction)
                dummy = metrics(d["problem_type"], y, constant_prediction(d["problem_type"], item.y_fit, len(y), len(item.classes)))
                val_scores[(d["name"], s, m)] = dict(score, normalized_loss=score["loss"] / max(dummy["loss"], 1e-12))
            if len(successful) == 3:
                losses = {m: val_scores[(d["name"], s, m)]["loss"] for m in MODEL_ORDER}
                selected = ValidationSelector().select(DecisionContext(d["name"], s, d["problem_type"], losses))
                decisions.append({"dataset": d["name"], "split": s, "selected": selected,
                                  "validation_losses": losses,
                                  "validation_prediction_hashes": {m: group[m]["artifacts"]["validation"]["sha256"] for m in MODEL_ORDER}})
    frozen = {"execution_fingerprint": manifest["execution_fingerprint"], "decisions": decisions}
    decision_hash = digest(frozen)
    selection_dir = output / "selections"
    selection_dir.mkdir(exist_ok=True)
    decision_path = selection_dir / f"{decision_hash}.json"
    if not decision_path.exists():
        write_json(decision_path, dict(frozen, decision_hash=decision_hash, frozen_before_test_scoring_at=now()))
    selected_by = {(r["dataset"], r["split"]): r["selected"] for r in decisions}

    # PHASE 2: choices now exist on disk. Only now load test targets and score.
    rows, cost_rows = [], []
    for d in config.assets["datasets"]:
        for s in range(3):
            item = None
            for m in MODEL_ORDER:
                r = records[(d["name"], s, m)]
                if r["status"] != "success":
                    continue
                if item is None:
                    item = model_input(config, d, s)
                    y = scoring_target(config, d, s, "test", item.classes)
                    dummy = metrics(d["problem_type"], y, constant_prediction(d["problem_type"], item.y_fit, len(y), len(item.classes)))
                prediction = load_prediction(root, r, "test", item.row_ids["test"])
                score = metrics(d["problem_type"], y, prediction)
                val = val_scores[(d["name"], s, m)]
                t = r["timings"]
                row = {"dataset": d["name"], "dataset_id": d["dataset_id"], "task_id": d["task_id"],
                       "split": s, "model": m, "problem_type": d["problem_type"],
                       "config_hash": manifest["identity"]["config_fingerprint"],
                       "n_fit": len(item.X_fit), "n_val": len(item.X_validation), "n_test": len(item.X_test),
                       "n_features": len(d["feature_names"]), "val_primary": val["primary"], "val_loss": val["loss"],
                       "test_primary": score["primary"], "test_loss": score["loss"], "dummy_test_loss": dummy["loss"],
                       "normalized_loss": score["loss"] / max(dummy["loss"], 1e-12), "status": "success",
                       **{f"test_{k}": v for k, v in score.items() if k not in ("primary", "loss")},
                       **t, "observed_wall_s": r["observed_wall_s"],
                       "peak_vram_allocated_mib": r.get("peak_vram_allocated_mib"),
                       "peak_vram_reserved_mib": r.get("peak_vram_reserved_mib"),
                       "peak_process_tree_rss_mib": r.get("peak_process_tree_rss_mib"),
                       "gpu_uuid": r["physical_gpu"]["uuid"],
                       "resolved_estimators": r["resolved_config"]["n_estimators"]}
                rows.append(row)
                cost_rows.append({"dataset": d["name"], "split": s, "model": m,
                                  "cold_test_estimate_s": t["load_s"] + t["fit_s"] + t["test_predict_s"],
                                  "loaded_test_estimate_s": t["fit_s"] + t["test_predict_s"],
                                  "actual_unit_wall_s": r["observed_wall_s"]})
    csv_out(output / "metrics.csv", rows)
    csv_out(output / "fixed_costs.csv", cost_rows)
    csv_out(output / "selection_decisions.csv", [
        {"dataset": r["dataset"], "split": r["split"], "selected": r["selected"], "decision_hash": decision_hash,
         **{f"val_loss_{m}": r["validation_losses"][m] for m in MODEL_ORDER}} for r in decisions])

    # All quality comparisons use exactly the same complete three-model dataset/split groups.
    frame = pd.DataFrame(rows)
    common = pd.DataFrame()
    if len(frame):
        complete_keys = set(selected_by)
        common = frame[[((d, int(s)) in complete_keys) for d, s in zip(frame.dataset, frame.split)]].copy()
    quality, quality_raw, methods, search_costs, reversals = [], [], [], [], []
    space = {"status": "no_common_success"}
    if len(common):
        means = common.groupby(["dataset", "problem_type", "model"], as_index=False).agg(normalized_loss=("normalized_loss", "mean"))
        space = selection_space(means)
        for d, group in common.groupby("dataset", sort=False):
            row = {"dataset": d, "task": group.problem_type.iloc[0], "common_splits": group.split.nunique()}
            for m in MODEL_ORDER:
                values = group[group.model == m].test_primary.to_numpy()
                mean, std = float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else None
                row[m] = f"{mean:.6g} ± {std:.3g}" if std is not None else f"{mean:.6g} (n=1)"
                quality_raw.append({"dataset": d, "model": m, "mean": mean, "sd_across_splits": std, "n_splits": len(values)})
            quality.append(row)
            # Stable pairwise advantages: predefined practical threshold and >=2 splits.
            task = group.problem_type.iloc[0]
            for a in MODEL_ORDER:
                for b in MODEL_ORDER:
                    if a == b:
                        continue
                    pa = group[group.model == a].set_index("split").test_primary
                    pb = group[group.model == b].set_index("split").test_primary
                    margin = pa - pb if task == "binary" else (pb - pa) / np.maximum(pb, 1e-12)
                    average_margin = float(pa.mean() - pb.mean()) if task == "binary" else float((pb.mean() - pa.mean()) / max(pb.mean(), 1e-12))
                    threshold = 0.005 if task == "binary" else 0.02
                    reversals.append({"dataset": d, "winner": a, "reference": b, "mean_improvement": average_margin,
                                      "positive_splits": int((margin > 0).sum()), "n_splits": len(margin),
                                      "practical_threshold": threshold,
                                      "stable_practical_advantage": len(margin) == 3 and (margin > 0).sum() >= 2 and average_margin >= threshold})
        raw_methods = []
        for (d, s), group in common.groupby(["dataset", "split"]):
            index = group.set_index("model")
            for m in MODEL_ORDER:
                raw_methods.append({"dataset": d, "split": int(s), "method": f"Fixed-{m}", "loss": float(index.loc[m].normalized_loss)})
            selected = selected_by[(d, int(s))]
            for method, value in (("RandomSelect-expectation", index.normalized_loss.mean()),
                                  ("ValSelect", index.loc[selected].normalized_loss),
                                  ("Oracle-per-split-POSTHOC", index.normalized_loss.min())):
                raw_methods.append({"dataset": d, "split": int(s), "method": method, "loss": float(value)})
            costs = sum(float(index.loc[m].load_s + index.loc[m].fit_s + index.loc[m].validation_predict_s) for m in MODEL_ORDER)
            search_costs.append({"dataset": d, "split": int(s), "selected": selected,
                                 "valselect_flow_estimate_s": costs + float(index.loc[selected].test_predict_s),
                                 "all_three_actual_unit_wall_s": float(index.observed_wall_s.sum()),
                                 "note": "Phase-sum replay estimate, not an online cold-start measurement"})
        mf = pd.DataFrame(raw_methods).groupby(["method", "dataset"], as_index=False).loss.mean()
        fixed_pfn = float(mf[mf.method == "Fixed-tabpfn"].loss.mean())
        best_fixed = min(float(mf[mf.method == f"Fixed-{m}"].loss.mean()) for m in MODEL_ORDER)
        for method, group in mf.groupby("method", sort=False):
            value = float(group.loss.mean())
            methods.append({"method": method, "normalized_test_loss": value, "n_datasets": len(group),
                            "relative_gain_vs_PFN": (fixed_pfn - value) / max(fixed_pfn, 1e-12),
                            "relative_gain_vs_best_fixed_posthoc": (best_fixed - value) / max(best_fixed, 1e-12)})
        csv_out(output / "method_losses_per_split.csv", raw_methods)
    csv_out(output / "quality.csv", quality_raw)
    csv_out(output / "method_summary.csv", methods)
    csv_out(output / "search_costs.csv", search_costs)
    csv_out(output / "pairwise_advantages.csv", reversals)
    write_json(output / "selection_space.json", space)
    statuses = {status: sum(r["status"] == status for r in coverage) for status in sorted(set(r["status"] for r in coverage))}
    # Include every attempt, including failures and retries, in actual research compute.
    attempts = [read_json(p) for p in root.glob("units/*/s*/*/attempt-*/result.json")]
    actual_wall = sum(r.get("observed_wall_s", r.get("worker_total_s", 0)) for r in attempts)
    cost_summary = []
    for m in MODEL_ORDER:
        valid = [r for r in rows if r["model"] == m]
        statuses_m = [r for r in coverage if r["model"] == m]
        cost_summary.append({"model": m, "success": len(valid), "planned": len(statuses_m),
                             "mean_success_wall_s": float(np.mean([r["observed_wall_s"] for r in valid])) if valid else "",
                             "max_allocated_vram_mib": max((r["peak_vram_allocated_mib"] or 0 for r in valid), default="")})
    csv_out(output / "cost_summary.csv", cost_summary)
    summary = {"created_at": now(), "core_planned": 54, "statuses": statuses,
               "common_success_groups": len(decisions), "formal_complete": statuses.get("success", 0) == 54,
               "recorded_attempts": len(attempts), "actual_research_gpu_wall_s": actual_wall,
               "decision_hash": decision_hash, "selection_space": space}
    write_json(output / "summary.json", summary)
    report = f"""# TFM 路由选择空间 pilot 报告

生成时间：{summary['created_at']}。核心计划 54 单元；状态：{statuses}。三模型共同成功组 {len(decisions)}/18。
{'完整核心覆盖已达成。' if summary['formal_complete'] else '核心覆盖尚未完成，以下仅为当前覆盖的探索性汇总，不能代替完整协议结果。'}
有效单元、失败及缺失见 coverage.csv；失败未填补成绩。共同成功范围可能有幸存者偏差。

## 1. 逐任务质量

二分类 AUC 越大越好；多分类 log-loss、回归 RMSE 越小越好。± 为划分间样本标准差，不是置信区间。

{table(quality, ['dataset', 'task', 'common_splits', *MODEL_ORDER])}
## 2. 选择收益

损失先除以同分区常量基线，再先平均 split、后平均 dataset。RandomSelect 是损失的期望，未平均预测。
ValSelect 在读取测试标签前已冻结选择：selections/{decision_hash}.json。
Oracle-per-split 是看过测试答案的逐划分上限；下节 VBS 则先平均划分、后选模型，二者不同。

{table(methods, ['method', 'normalized_test_loss', 'relative_gain_vs_PFN', 'relative_gain_vs_best_fixed_posthoc'])}
## 3. 选择空间

SBS/VBS/G_all/G_type 与移除模型后的 VBS 见 selection_space.json。当前结果：

```json
{__import__('json').dumps(space, indent=2, ensure_ascii=False)}
```

各数据集两两比较、每个划分的反例及预设实用阈值见 pairwise_advantages.csv、metrics.csv。
稳定优势必须有完整三个划分，且至少两次方向相同；不等于统计显著。

## 4. 成本与失败

{table(cost_summary, ['model', 'success', 'planned', 'mean_success_wall_s', 'max_allocated_vram_mib'])}
已记录 {len(attempts)} 次尝试的实际进程墙钟累计 {actual_wall:.1f} GPU秒（并行时不是人等待的秒数）。
包含失败和重跑；异常断电未落盘的尝试可能低估，不能据此声称成本完整。
固定模型冷/热流程估算见 fixed_costs.csv，验证搜索估算见 search_costs.csv；均由同步后的阶段时间求和，
不是已经实测的在线部署时延。构造器可能延迟到 fit 才加载权重，load_s 不能单独解释为完整权重加载耗时。
LimiX 每次查询的上下文处理计入对应 predict_s。显存为 PyTorch 分配/保留峰值，不是整卡占用；RAM 是进程树采样峰值。

## 解释边界

这里只检验三个固定 TFM 配置的互补性与验证反馈。尚未证明 LLM 路由、ensemble、自迭代有效。
六个数据集是开发筛查；不同 split 并非独立任务。预算判断还需检查具体硬件一致性、失败覆盖和跨任务重复性。
P0 官方缓存分析未由本程序执行；P4 扩展集不自动运行。test 成绩不回写 ToolCard 或全局 memory。
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    if len(common):
        make_plots(common, output)
    return summary


def make_plots(frame, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mean = frame.groupby(["dataset", "model"]).normalized_loss.mean().unstack()[list(MODEL_ORDER)]
    delta = mean.sub(mean.min(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    plot = ax.imshow(delta.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(3), MODEL_ORDER)
    ax.set_yticks(range(len(delta)), delta.index, fontsize=8)
    fig.colorbar(plot, ax=ax, label="Normalized loss above dataset best (common splits)")
    fig.tight_layout()
    fig.savefig(output / "relative_gap.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in MODEL_ORDER:
        subset = frame[frame.model == m].groupby("dataset").agg(wall=("observed_wall_s", "mean"), loss=("normalized_loss", "mean"))
        ax.scatter(subset.wall, subset.loss, label=m)
    ax.set(xlabel="Mean actual unit wall time (s; includes val + test)", ylabel="Normalized test loss (lower is better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "quality_cost.png", dpi=160)
    plt.close(fig)
