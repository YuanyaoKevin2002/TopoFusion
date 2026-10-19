from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

TASK_ORDER = ("cms", "msi", "pn", "cin", "os", "dfs")
TASK_SCORE_KEYS = {
    "cms": "auc_ovr_macro",
    "msi": "auroc",
    "pn": "auroc",
    "cin": "auroc",
    "os": "c_index",
    "dfs": "c_index",
}
CSV_METRIC_KEYS = (
    "n",
    "acc",
    "balanced_acc",
    "macro_f1",
    "auc_ovr_macro",
    "auroc",
    "sensitivity",
    "specificity",
    "events",
    "c_index",
)


def _finite_float(value: object) -> float | None:
    try:
        x = float(value)  # type: ignore[arg-type]
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def extract_task_scores(metrics: Mapping[str, object]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for task in TASK_ORDER:
        task_metrics = metrics.get(task)
        if not isinstance(task_metrics, Mapping):
            continue
        value = _finite_float(task_metrics.get(TASK_SCORE_KEYS[task]))
        if value is not None:
            scores[task] = value
    return scores


def arithmetic_score(task_scores: Mapping[str, float]) -> float:
    vals = [_finite_float(v) for v in task_scores.values()]
    vals = [v for v in vals if v is not None]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def standardized_score(task_scores: Mapping[str, float], history_scores: Sequence[Mapping[str, float]]) -> float:
    z_values = []
    for task, value in task_scores.items():
        value = float(value)
        vals = []
        for item in history_scores:
            hist_val = _finite_float(item.get(task))
            if hist_val is not None:
                vals.append(hist_val)
        vals.append(value)
        if len(vals) < 2:
            z_values.append(0.0)
            continue
        mean = sum(vals) / len(vals)
        var = sum((x - mean) ** 2 for x in vals) / len(vals)
        std = math.sqrt(var)
        z_values.append(0.0 if std < 1e-8 else (value - mean) / std)
    if not z_values:
        return float("nan")
    return float(sum(z_values) / len(z_values))


def compute_score_bundle(metrics: Mapping[str, object], history_scores: Sequence[Mapping[str, float]]) -> Dict[str, object]:
    task_scores = extract_task_scores(metrics)
    return {
        "score": arithmetic_score(task_scores),
        "score_standardized": standardized_score(task_scores, history_scores),
        "available_tasks": list(task_scores.keys()),
        "task_scores": task_scores,
    }


def task_valid_counts_from_rows(rows: Iterable[Mapping[str, object]]) -> Dict[str, int]:
    counts = {task: 0 for task in TASK_ORDER}
    for row in rows:
        labels = row.get("_labels", {})
        if not isinstance(labels, Mapping):
            continue
        for task in ("cms", "msi", "pn", "cin"):
            if labels.get(task) is not None:
                counts[task] += 1
        for task in ("os", "dfs"):
            if labels.get(f"{task}_time") is not None and labels.get(f"{task}_event") is not None:
                counts[task] += 1
    return counts


def task_counts_from_metrics(metrics: Mapping[str, object]) -> Dict[str, int]:
    counts = {task: 0 for task in TASK_ORDER}
    for task in TASK_ORDER:
        task_metrics = metrics.get(task)
        if isinstance(task_metrics, Mapping):
            value = task_metrics.get("n", 0)
            try:
                counts[task] = int(value)  # type: ignore[arg-type]
            except Exception:
                counts[task] = 0
    return counts


def assert_fixed_task_counts(observed: Mapping[str, int], expected: Mapping[str, int], split_name: str = "val") -> None:
    obs = {task: int(observed.get(task, 0)) for task in TASK_ORDER}
    exp = {task: int(expected.get(task, 0)) for task in TASK_ORDER}
    if obs != exp:
        raise RuntimeError(f"{split_name} task count changed: expected={exp} observed={obs}")


def _json_or_blank(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def metrics_log_fieldnames() -> list[str]:
    fields = ["epoch", "split", "score", "score_standardized", "available_tasks", "task_scores"]
    for task in TASK_ORDER:
        fields.extend([f"{task}_{key}" for key in CSV_METRIC_KEYS])
        fields.append(f"{task}_monitor")
    return fields


def metrics_log_row(epoch: int, split: str, metrics: Mapping[str, object], score_bundle: Mapping[str, object]) -> Dict[str, object]:
    row: Dict[str, object] = {
        "epoch": int(epoch),
        "split": split,
        "score": score_bundle.get("score", ""),
        "score_standardized": score_bundle.get("score_standardized", ""),
        "available_tasks": json.dumps(score_bundle.get("available_tasks", [])),
        "task_scores": json.dumps(score_bundle.get("task_scores", {}), sort_keys=True),
    }
    for task in TASK_ORDER:
        task_metrics = metrics.get(task, {})
        if not isinstance(task_metrics, Mapping):
            task_metrics = {}
        for key in CSV_METRIC_KEYS:
            row[f"{task}_{key}"] = task_metrics.get(key, "")
        row[f"{task}_monitor"] = task_metrics.get(TASK_SCORE_KEYS[task], "")
    return row


def append_metrics_log(path: str | Path, epoch: int, split: str, metrics: Mapping[str, object], score_bundle: Mapping[str, object]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = metrics_log_fieldnames()
    exists = out_path.exists() and out_path.stat().st_size > 0
    row = metrics_log_row(epoch, split, metrics, score_bundle)
    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: _json_or_blank(row.get(k, "")) for k in fieldnames})
