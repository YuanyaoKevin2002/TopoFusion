#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(ROOT / "src"))

from topofusion_v3.baselines.multitask_baseline_adapters import make_multitask_baseline, model_uses_graph
from topofusion_v3.baselines.multitask_utils import (
    DEFAULT_TASK_WEIGHTS,
    class_weights_for_task,
    compute_losses,
    dataset_summary,
    evaluate_multitask,
    make_loader,
    move_labels,
    select_manifest_rows,
    split_rows,
    MultiTaskSlideDataset,
)
from topofusion_v3.multitask.composite import (
    append_metrics_log,
    assert_fixed_task_counts,
    compute_score_bundle,
    task_counts_from_metrics,
    task_valid_counts_from_rows,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_expected_counts(raw: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        key, val = item.split(":")
        out[key.strip()] = int(val)
    return out


def parse_task_weights(raw: str | None) -> Dict[str, float]:
    weights = dict(DEFAULT_TASK_WEIGHTS)
    if not raw:
        return weights
    for key, value in json.loads(raw).items():
        weights[key] = float(value)
    return weights


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def filter_task_counts(counts: Dict[str, int], tasks: Sequence[str]) -> Dict[str, int]:
    enabled = set(tasks)
    return {task: int(value) for task, value in counts.items() if task in enabled}


def role_rows(rows: Sequence[dict], splits: Sequence[str]) -> list[dict]:
    split_set = set(splits)
    return [row for row in rows if row.get("split", "") in split_set]


def assert_exact_counts(rows_by_split: Mapping[str, Sequence[dict]], expected: Mapping[str, int]) -> None:
    actual = {k: len(v) for k, v in rows_by_split.items()}
    if dict(actual) != dict(expected):
        raise RuntimeError(f"split count mismatch: expected={dict(expected)} actual={actual}")


def write_split_manifest(rows_by_split: Mapping[str, Sequence[dict]], out_csv: Path) -> None:
    fieldnames = ["baseline_split", "dataset", "split", "slide_id", "sample_id", "patient_id", "cms_label", "label_id", "h5_path", "cache_path"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for split in ("train", "val", "EXTERNAL_COHORT_A_external"):
            for row in rows_by_split[split]:
                out = {k: row.get(k, "") for k in fieldnames if k != "baseline_split"}
                out["baseline_split"] = split
                writer.writerow(out)


def write_metrics_csv(metrics: Mapping[str, object], out_csv: Path, split_name: str) -> None:
    rows = []
    for task, vals in metrics.items():
        if not isinstance(vals, Mapping):
            continue
        flat = {"split": split_name, "task": task}
        for key, value in vals.items():
            flat[key] = json.dumps(value) if key == "confusion_matrix" else value
        rows.append(flat)
    keys = sorted({k for row in rows for k in row.keys()})
    if not keys:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def limit_rows(rows_by_split: Dict[str, list[dict]], limit: int) -> Dict[str, list[dict]]:
    if int(limit) <= 0:
        return rows_by_split
    return {k: list(v)[: int(limit)] for k, v in rows_by_split.items()}


def build_class_weights(train_ds: MultiTaskSlideDataset, device: torch.device, enabled: bool) -> Dict[str, torch.Tensor]:
    if not enabled:
        return {}
    return {
        "cms": class_weights_for_task(train_ds, "cms", 4, device),
        "msi": class_weights_for_task(train_ds, "msi", 2, device),
        "pn": class_weights_for_task(train_ds, "pn", 2, device),
        "cin": class_weights_for_task(train_ds, "cin", 2, device),
    }


def train_one_epoch(model, loader, device, optimizer, scaler, task_weights, class_weights, grad_clip_norm, use_amp: bool, accum_steps: int, tasks: Sequence[str]) -> Dict[str, object]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sums: Dict[str, float] = {}
    n_sums: Dict[str, int] = {}
    steps = 0
    opt_steps = 0
    accum_steps = max(int(accum_steps), 1)
    for batch_idx, batch in enumerate(loader):
        labels = move_labels(batch.labels, device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            preds = model(batch.items, device)
        loss, loss_items, n_items = compute_losses(preds, labels, task_weights=task_weights, class_weights=class_weights, tasks=tasks)
        loss_for_backward = loss / float(accum_steps)
        if use_amp:
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()
        should_step = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == len(loader))
        if should_step:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            opt_steps += 1
        for key, value in loss_items.items():
            loss_sums[key] = loss_sums.get(key, 0.0) + float(value)
        for key, value in n_items.items():
            n_sums[key] = n_sums.get(key, 0) + int(value)
        steps += 1
    return {"loss": {k: v / max(steps, 1) for k, v in loss_sums.items()}, "label_counts_seen": n_sums, "steps": steps, "optimizer_steps": opt_steps}


@torch.no_grad()
def debug_first_batch(model, loader, device, task_weights, class_weights, use_amp: bool, tasks: Sequence[str]) -> Dict[str, object]:
    model.train()
    batch = next(iter(loader))
    labels = move_labels(batch.labels, device)
    with torch.cuda.amp.autocast(enabled=use_amp):
        preds = model(batch.items, device)
    loss, loss_items, n_items = compute_losses(preds, labels, task_weights=task_weights, class_weights=class_weights, tasks=tasks)
    item_sizes = []
    for item in batch.items:
        item_sizes.append(int(item.shape[0]) if torch.is_tensor(item) else int(item.x.shape[0]))
    return {
        "batch_size": len(batch.items),
        "item_sizes": item_sizes,
        "head_shapes": {k: list(v.shape) for k, v in preds.items()},
        "label_counts_in_batch": n_items,
        "losses": loss_items,
        "total_loss_requires_grad": bool(loss.requires_grad),
        "first_slides": [m.get("slide_id", "") for m in batch.metas[:3]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["abmil", "meanpool", "transmil", "clam_sb", "clam_mb", "dsmil", "patch_gcn", "heat_pyg", "hact_pyg"])
    parser.add_argument("--manifest", default="${PROJECT_ROOT}/manifests/topofusion_v3_connectfix_manifest.csv")
    parser.add_argument("--clinical-dir", default="${DATA_ROOT}/gnn-crc/data/clinical")
    parser.add_argument("--feature-cache-dir", default="${PROJECT_ROOT}/cache/baseline_features_40x")
    parser.add_argument("--out-root", default="${PROJECT_ROOT}/runs/baseline_multitask_40x")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--expected-counts", default="train:385,val:96,EXTERNAL_COHORT_A_external:98")
    parser.add_argument("--train-splits", default="train", help="Comma-separated manifest splits used for training.")
    parser.add_argument("--val-splits", default="val", help="Comma-separated manifest splits used for validation/model selection.")
    parser.add_argument("--external-splits", default="EXTERNAL_COHORT_A_external", help="Comma-separated manifest splits logged as external.")
    parser.add_argument("--skip-count-check", action="store_true")
    parser.add_argument("--limit-per-split", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--input-dim", type=int, default=1536)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--attn-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--projection-dim", type=int, default=0)
    parser.add_argument("--clam-mb-pool", default="flatten", choices=["flatten", "mean"])
    parser.add_argument("--dsmil-pool", default="flatten", choices=["flatten", "mean"])
    parser.add_argument("--patch-gcn-hidden-dim", type=int, default=128)
    parser.add_argument("--patch-gcn-layers", type=int, default=4)
    parser.add_argument("--heat-hidden-dim", type=int, default=256)
    parser.add_argument("--heat-layers", type=int, default=2)
    parser.add_argument("--hact-hidden-dim", type=int, default=256)
    parser.add_argument("--hact-layers", type=int, default=2)
    parser.add_argument("--max-patches", type=int, default=0, help="0 uses all H5 features; smoke jobs can set 512/1024.")
    parser.add_argument("--task-weights-json", default="")
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--debug-first-batch", action="store_true")
    parser.add_argument("--selection-score", choices=["score", "score_standardized"], default="score")
    parser.add_argument("--tasks", default="cms,msi,pn,cin,os,dfs", help="Comma-separated tasks to train/evaluate, e.g. os,dfs.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(device.type == "cuda" and not args.no_amp)
    task_weights = parse_task_weights(args.task_weights_json)
    tasks = parse_csv(args.tasks)
    valid_tasks = {"cms", "msi", "pn", "cin", "os", "dfs"}
    unknown_tasks = sorted(set(tasks) - valid_tasks)
    if unknown_tasks:
        raise ValueError(f"Unknown tasks: {unknown_tasks}")
    if not tasks:
        raise ValueError("--tasks must enable at least one task")
    mode = "graph" if model_uses_graph(args.model) else "bag"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.model}_multitask_40x_{stamp}"
    run_dir = Path(args.out_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_splits = parse_csv(args.train_splits)
    val_splits = parse_csv(args.val_splits)
    external_splits = parse_csv(args.external_splits)
    requested_splits = sorted(set(train_splits + val_splits + external_splits))
    rows = select_manifest_rows(
        args.manifest,
        requested_splits,
        args.clinical_dir,
        require_h5=True,
        require_graph=(mode == "graph"),
        deduplicate_patients=True,
    )
    rows_by_split = {
        "train": role_rows(rows, train_splits),
        "val": role_rows(rows, val_splits),
        "EXTERNAL_COHORT_A_external": role_rows(rows, external_splits),
    }
    if args.limit_per_split > 0:
        rows_by_split = limit_rows(rows_by_split, args.limit_per_split)
    if not args.skip_count_check and args.limit_per_split <= 0:
        assert_exact_counts(rows_by_split, parse_expected_counts(args.expected_counts))
    write_split_manifest(rows_by_split, run_dir / "baseline_multitask_split_manifest.csv")
    metrics_log_path = run_dir / "metrics_log.csv"
    val_expected_counts = filter_task_counts(task_valid_counts_from_rows(rows_by_split["val"]), tasks)
    score_histories = {"val": [], "external": []}

    feature_cache_dir = args.feature_cache_dir if mode == "bag" and args.feature_cache_dir else None
    train_ds = MultiTaskSlideDataset(rows_by_split["train"], mode=mode, max_patches=args.max_patches, seed=args.seed, feature_cache_dir=feature_cache_dir)
    val_ds = MultiTaskSlideDataset(rows_by_split["val"], mode=mode, max_patches=args.max_patches, seed=args.seed, feature_cache_dir=feature_cache_dir)
    ext_ds = MultiTaskSlideDataset(rows_by_split["EXTERNAL_COHORT_A_external"], mode=mode, max_patches=args.max_patches, seed=args.seed, feature_cache_dir=feature_cache_dir)
    train_loader = make_loader(train_ds, batch_size=args.batch_size, workers=args.num_workers, weighted=args.weighted_sampler, shuffle=True)
    val_loader = make_loader(val_ds, batch_size=args.batch_size, workers=args.num_workers, weighted=False, shuffle=False)
    ext_loader = make_loader(ext_ds, batch_size=args.batch_size, workers=args.num_workers, weighted=False, shuffle=False)

    model = make_multitask_baseline(
        args.model,
        input_dim=args.input_dim,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        attn_heads=args.attn_heads,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        projection_dim=args.projection_dim,
        clam_mb_pool=args.clam_mb_pool,
        dsmil_pool=args.dsmil_pool,
        patch_gcn_hidden_dim=args.patch_gcn_hidden_dim,
        patch_gcn_layers=args.patch_gcn_layers,
        heat_hidden_dim=args.heat_hidden_dim,
        heat_layers=args.heat_layers,
        hact_hidden_dim=args.hact_hidden_dim,
        hact_layers=args.hact_layers,
    ).to(device)
    class_weights = build_class_weights(train_ds, device, enabled=args.use_class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_event = {
        "event": "baseline_multitask_start",
        "model": args.model,
        "mode": mode,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "num_workers": args.num_workers,
        "max_patches": args.max_patches,
        "feature_cache_dir": str(feature_cache_dir) if feature_cache_dir else "",
        "task_weights": task_weights,
        "tasks": tasks,
        "train_splits": train_splits,
        "val_splits": val_splits,
        "external_splits": external_splits,
        "use_class_weights": bool(args.use_class_weights),
        "model_params": int(sum(p.numel() for p in model.parameters())),
        "embedding_dim": int(model.embedding_dim),
        "run_dir": str(run_dir),
        "selection_score": args.selection_score,
        "val_expected_task_counts": val_expected_counts,
        "train": dataset_summary(rows_by_split["train"]),
        "val": dataset_summary(rows_by_split["val"]),
        "external": dataset_summary(rows_by_split["EXTERNAL_COHORT_A_external"]),
    }
    print(json.dumps(start_event, sort_keys=True), flush=True)
    (run_dir / "config_baseline_multitask.json").write_text(json.dumps({"args": vars(args), "start_event": start_event}, indent=2), encoding="utf-8")

    if args.debug_first_batch:
        print(json.dumps({"event": "debug_first_batch", **debug_first_batch(model, train_loader, device, task_weights, class_weights, use_amp, tasks)}, sort_keys=True), flush=True)

    history = []
    best_score = -1.0
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        print(json.dumps({"event": "validation_effective_counts", "model": args.model, "epoch": epoch, "split": "val", "expected": val_expected_counts}, sort_keys=True), flush=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        train_stats = train_one_epoch(model, train_loader, device, optimizer, scaler, task_weights, class_weights, args.grad_clip_norm, use_amp, args.accum_steps, tasks)
        val_metrics = evaluate_multitask(model, val_loader, device, use_amp=use_amp, tasks=tasks)
        observed_val_counts = task_counts_from_metrics(val_metrics)
        assert_fixed_task_counts(observed_val_counts, val_expected_counts, "val")
        val_score = compute_score_bundle(val_metrics, score_histories["val"])
        score_histories["val"].append(val_score["task_scores"])
        ext_metrics = evaluate_multitask(model, ext_loader, device, use_amp=use_amp, tasks=tasks)
        ext_score = compute_score_bundle(ext_metrics, score_histories["external"])
        score_histories["external"].append(ext_score["task_scores"])
        append_metrics_log(metrics_log_path, epoch, "val", val_metrics, val_score)
        append_metrics_log(metrics_log_path, epoch, "EXTERNAL_COHORT_A_external", ext_metrics, ext_score)
        row = {"event": "epoch", "model": args.model, "epoch": epoch, "train": train_stats, "val": val_metrics, "external": ext_metrics, "scores": {"val": val_score, "external": ext_score}}
        if torch.cuda.is_available():
            row["cuda_memory_mb"] = {
                "allocated": round(torch.cuda.memory_allocated() / 1024**2, 1),
                "reserved": round(torch.cuda.memory_reserved() / 1024**2, 1),
                "max_allocated": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
                "max_reserved": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
            }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        (run_dir / "history_baseline_multitask.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        score = float(val_score[args.selection_score])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if use_amp else None,
                    "epoch": epoch,
                    "best_metric": best_score,
                    "selection_score": args.selection_score,
                    "args": vars(args),
                    "val_score": val_score,
                    "external_score": ext_score,
                    "val_metrics": val_metrics,
                    "external_metrics": ext_metrics,
                },
                run_dir / "best.pt",
            )
            write_metrics_csv(val_metrics, run_dir / "best_val_metrics.csv", "val")
            write_metrics_csv(ext_metrics, run_dir / "best_external_metrics.csv", "EXTERNAL_COHORT_A_external")
        if epoch >= args.min_epochs and (epoch - best_epoch) >= args.early_stop_patience:
            print(json.dumps({"event": "early_stop", "epoch": epoch, "best_epoch": best_epoch, "best_val_composite_score": best_score, "selection_score": args.selection_score}, sort_keys=True), flush=True)
            break

    best_loaded = {}
    best_path = run_dir / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        best_val_metrics = evaluate_multitask(model, val_loader, device, use_amp=use_amp, tasks=tasks)
        best_ext_metrics = evaluate_multitask(model, ext_loader, device, use_amp=use_amp, tasks=tasks)
        best_loaded = {
            "best_loaded_epoch": int(ckpt.get("epoch", best_epoch)),
            "best_loaded_val_metrics": best_val_metrics,
            "best_loaded_external_metrics": best_ext_metrics,
        }
        (run_dir / "best_loaded_metrics.json").write_text(json.dumps(best_loaded, indent=2), encoding="utf-8")
    final = {"event": "baseline_multitask_done", "model": args.model, "best_epoch": best_epoch, "best_val_composite_score": best_score, "selection_score": args.selection_score, "history_last": history[-1] if history else {}, "run_dir": str(run_dir), **best_loaded}
    (run_dir / "summary_baseline_multitask.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
