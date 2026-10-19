#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping


def fmt(x) -> str:
    try:
        return f"{float(x):.4f}"
    except Exception:
        return ""


def per_class(metrics: Mapping[str, object]) -> str:
    vals = metrics.get("per_class_f1", [])
    if not isinstance(vals, list):
        return ""
    return "[" + ", ".join(fmt(v) for v in vals) + "]"


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="${PROJECT_ROOT}")
    parser.add_argument("--out-md", default="")
    args = parser.parse_args()
    project = Path(args.project)
    rows = []

    gnn_runs = {
        "GNN Exp5a CMS-only CE meanraw": project / "runs/gnn_cms_single_5a_ce_meanraw_20260705_0546/summary_cms_single.json",
        "GNN Exp5b CMS-only focal attn": project / "runs/gnn_cms_single_5b_focal_attn_20260705_0600/summary_cms_single.json",
    }
    for name, path in gnn_runs.items():
        if not path.exists():
            continue
        summary = load_summary(path)
        metrics = summary.get("best_loaded_external_metrics", {})
        rows.append({
            "model": name,
            "epoch": summary.get("best_loaded_epoch", summary.get("best_epoch", "")),
            "val_auc": summary.get("best_val_cms_auc", ""),
            "auc": metrics.get("auc_ovr_macro", ""),
            "acc": metrics.get("acc", ""),
            "macro_f1": metrics.get("macro_f1", ""),
            "per_class_f1": per_class(metrics),
            "run_dir": str(path.parent),
        })

    base_root = project / "runs/baseline_cms_single_40x"
    for path in sorted(base_root.glob("*/summary_baseline_cms_single.json")):
        summary = load_summary(path)
        metrics = summary.get("best_loaded_external_metrics", {})
        rows.append({
            "model": summary.get("model", path.parent.name),
            "epoch": summary.get("best_loaded_epoch", summary.get("best_epoch", "")),
            "val_auc": summary.get("best_val_cms_auc", ""),
            "auc": metrics.get("auc_ovr_macro", ""),
            "acc": metrics.get("acc", ""),
            "macro_f1": metrics.get("macro_f1", ""),
            "per_class_f1": per_class(metrics),
            "run_dir": str(path.parent),
        })

    rows.sort(key=lambda r: str(r["model"]))
    lines = [
        "# CMS-only External EXTERNAL_COHORT_A Results",
        "",
        "All models are selected by validation CMS macro-AUC. EXTERNAL_COHORT_A metrics are computed from the single selected checkpoint.",
        "",
        "| Model | Best epoch | Val CMS AUC | EXTERNAL_COHORT_A CMS AUC | EXTERNAL_COHORT_A Acc | EXTERNAL_COHORT_A macro-F1 | EXTERNAL_COHORT_A F1 [CMS1/CMS2/CMS3/CMS4] |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['epoch']} | {fmt(row['val_auc'])} | {fmt(row['auc'])} | {fmt(row['acc'])} | {fmt(row['macro_f1'])} | {row['per_class_f1']} |"
        )
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out_md:
        out = Path(args.out_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
