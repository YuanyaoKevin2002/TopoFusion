from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader

from topofusion_v3.config import deep_get, load_config
from topofusion_v3.multitask.data_utils import MultiTaskGraphCacheDataset
from topofusion_v3.multitask.losses import DEFAULT_TASK_WEIGHTS, compute_losses, labels_from_batch
from topofusion_v3.multitask.metrics import evaluate_multitask
from topofusion_v3.multitask.multitask_heads import MultiTaskCMSGraphModel
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


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def filter_task_counts(counts: Dict[str, int], tasks: list[str]) -> Dict[str, int]:
    enabled = set(tasks)
    return {task: int(value) for task, value in counts.items() if task in enabled}


def parse_task_weights(raw: str | None, base: Dict[str, float] | None = None) -> Dict[str, float]:
    weights = dict(DEFAULT_TASK_WEIGHTS if base is None else base)
    if not raw:
        return weights
    update = json.loads(raw)
    for key, value in update.items():
        weights[key] = float(value)
    return weights


def task_weights_from_config(tr_cfg: Dict[str, object]) -> Dict[str, float]:
    weights = dict(DEFAULT_TASK_WEIGHTS)
    raw = tr_cfg.get("task_weights", {})
    if isinstance(raw, dict):
        for key, value in raw.items():
            weights[str(key)] = float(value)
    return weights


def class_weights_for_task(ds: MultiTaskGraphCacheDataset, task: str, n_classes: int, device: torch.device) -> torch.Tensor:
    labels = ds.label_values(task)
    if not labels:
        return torch.ones(n_classes, dtype=torch.float32, device=device)
    counts = Counter(labels)
    total = float(sum(counts.values()))
    vals = []
    for cls in range(n_classes):
        frac = float(counts.get(cls, 0)) / total
        vals.append(1.0 / np.sqrt(max(frac, 1e-6)) if counts.get(cls, 0) > 0 else 1.0)
    w = torch.tensor(vals, dtype=torch.float32, device=device)
    return w / w.mean()


def make_loader(ds: MultiTaskGraphCacheDataset, batch_size: int, workers: int, weighted: bool) -> DataLoader:
    if len(ds) == 0:
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    if weighted:
        labels = [int(row["_labels"].get("cms", 0)) for row in ds.rows]  # type: ignore[index]
        counts = Counter(labels)
        sample_weights = [1.0 / float(counts[y]) for y in labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=workers)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers)


def dataset_summary(ds: MultiTaskGraphCacheDataset) -> Dict[str, object]:
    s = ds.summary()
    return {"n": s.n_rows, "n_patients": s.n_patients, "task_counts": s.task_counts}


def split_patient_keys(ds: MultiTaskGraphCacheDataset) -> set[tuple[str, str]]:
    return {(r.get("dataset", ""), r.get("patient_id") or r.get("sample_id") or r.get("slide_id", "")) for r in ds.rows}


def debug_first_batch(
    model,
    loader,
    device: torch.device,
    task_weights,
    class_weights,
    cms_entropy_lambda: float,
    survival_risk_scale: float,
    cms_loss_type: str,
    cms_focal_gamma: float,
    cms_focal_alpha,
    tasks: list[str],
) -> Dict[str, object]:
    model.train()
    data = next(iter(loader))
    data = data.to(device)
    preds = model(data)
    labels = labels_from_batch(data)
    total, loss_items, n_items = compute_losses(
        preds,
        labels,
        task_weights=task_weights,
        class_weights=class_weights,
        tasks=tasks,
        cms_entropy_lambda=cms_entropy_lambda,
        survival_risk_scale=survival_risk_scale,
        cms_loss_type=cms_loss_type,
        cms_focal_gamma=cms_focal_gamma,
        cms_focal_alpha=cms_focal_alpha,
    )
    backbone_debug = getattr(model, "_last_embedding_debug", {})
    return {
        "batch_graphs": int(preds["cms"].shape[0]),
        "head_shapes": {k: list(v.shape) for k, v in preds.items()},
        "slide_embedding_shape": backbone_debug.get("slide_embedding_shape", []),
        "backbone_debug": backbone_debug,
        "label_counts_in_batch": n_items,
        "losses": loss_items,
        "total_loss_requires_grad": bool(total.requires_grad),
    }


def train_one_epoch(
    model,
    loader,
    device,
    opt,
    grad_accum,
    grad_clip_norm,
    task_weights,
    class_weights,
    cms_entropy_lambda: float,
    survival_risk_scale: float,
    cms_loss_type: str,
    cms_focal_gamma: float,
    cms_focal_alpha,
    tasks: list[str],
) -> Dict[str, object]:
    model.train()
    opt.zero_grad(set_to_none=True)
    loss_sums: Dict[str, float] = {}
    n_sums: Dict[str, int] = {}
    steps = 0
    for step, data in enumerate(loader, start=1):
        data = data.to(device)
        preds = model(data)
        labels = labels_from_batch(data)
        loss, loss_items, n_items = compute_losses(
            preds,
            labels,
            task_weights=task_weights,
            class_weights=class_weights,
            tasks=tasks,
            cms_entropy_lambda=cms_entropy_lambda,
            survival_risk_scale=survival_risk_scale,
            cms_loss_type=cms_loss_type,
            cms_focal_gamma=cms_focal_gamma,
            cms_focal_alpha=cms_focal_alpha,
        )
        (loss / grad_accum).backward()
        if step % grad_accum == 0 or step == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            opt.step()
            opt.zero_grad(set_to_none=True)
        for k, v in loss_items.items():
            loss_sums[k] = loss_sums.get(k, 0.0) + float(v)
        for k, v in n_items.items():
            n_sums[k] = n_sums.get(k, 0) + int(v)
        steps += 1
    mean_losses = {k: v / max(steps, 1) for k, v in loss_sums.items()}
    return {"loss": mean_losses, "label_counts_seen": n_sums, "steps": steps}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--clinical-dir", default="${DATA_ROOT}/gnn-crc/data/clinical")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", "--accum-steps", dest="grad_accum_steps", type=int, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--head-projection-dim", type=int, default=0)
    parser.add_argument("--task-weights-json", default="")
    parser.add_argument("--train-splits", default="train")
    parser.add_argument("--val-splits", default="val")
    parser.add_argument("--external-splits", default="EXTERNAL_COHORT_A_external")
    parser.add_argument("--EXTERNAL_COHORT_B-splits", default="EXTERNAL_COHORT_B_external")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--limit-external", type=int, default=0)
    parser.add_argument("--limit-EXTERNAL_COHORT_B", type=int, default=0)
    parser.add_argument("--min-epochs", type=int, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--no-dedup-patients", action="store_true")
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--debug-first-batch", action="store_true")
    parser.add_argument("--selection-score", choices=["score", "score_standardized"], default="score")
    parser.add_argument("--tasks", default="cms,msi,pn,cin,os,dfs", help="Comma-separated tasks to train/evaluate, e.g. os,dfs.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tr_cfg = dict(deep_get(cfg, "training", {}))
    model_cfg = dict(deep_get(cfg, "model", {}))
    set_seed(int(tr_cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_csv = deep_get(cfg, "manifest.out_csv")
    dedup = not args.no_dedup_patients

    train_ds = MultiTaskGraphCacheDataset(manifest_csv, parse_csv(args.train_splits), args.clinical_dir, limit=args.limit_train, deduplicate_patients=dedup)
    val_ds = MultiTaskGraphCacheDataset(manifest_csv, parse_csv(args.val_splits), args.clinical_dir, limit=args.limit_val, deduplicate_patients=dedup)
    external_ds = MultiTaskGraphCacheDataset(manifest_csv, parse_csv(args.external_splits), args.clinical_dir, limit=args.limit_external, deduplicate_patients=dedup, allow_empty=True)
    EXTERNAL_COHORT_B_ds = MultiTaskGraphCacheDataset(manifest_csv, parse_csv(args.EXTERNAL_COHORT_B_splits), args.clinical_dir, limit=args.limit_EXTERNAL_COHORT_B, deduplicate_patients=dedup, allow_empty=True)

    batch_size = int(args.batch_size if args.batch_size is not None else tr_cfg.get("batch_size", 2))
    workers = int(args.num_workers if args.num_workers is not None else tr_cfg.get("num_workers", 2))
    grad_accum = int(args.grad_accum_steps if args.grad_accum_steps is not None else tr_cfg.get("grad_accum_steps", 4))
    grad_clip_norm = float(args.grad_clip_norm if args.grad_clip_norm is not None else tr_cfg.get("grad_clip_norm", 1.0))
    lr = float(args.lr if args.lr is not None else tr_cfg.get("lr", 1e-4))
    task_weights = parse_task_weights(args.task_weights_json, base=task_weights_from_config(tr_cfg))
    tasks = parse_csv(args.tasks)
    valid_tasks = {"cms", "msi", "pn", "cin", "os", "dfs"}
    unknown_tasks = sorted(set(tasks) - valid_tasks)
    if unknown_tasks:
        raise ValueError(f"Unknown tasks: {unknown_tasks}")
    if not tasks:
        raise ValueError("--tasks must enable at least one task")
    cms_entropy_lambda = float(tr_cfg.get("cms_entropy_lambda", 0.0))
    survival_risk_scale = float(tr_cfg.get("survival_risk_scale", 1.0))
    cms_loss_type = str(tr_cfg.get("cms_loss_type", "ce"))
    cms_focal_gamma = float(tr_cfg.get("cms_focal_gamma", 2.0))
    cms_focal_alpha_raw = tr_cfg.get("cms_focal_alpha", None)
    cms_focal_alpha = None if cms_focal_alpha_raw is None else float(cms_focal_alpha_raw)
    min_epochs = int(args.min_epochs if args.min_epochs is not None else tr_cfg.get("min_epochs", 20))
    early_stop_patience = int(args.early_stop_patience if args.early_stop_patience is not None else tr_cfg.get("early_stop_patience", 20))

    train_loader = make_loader(train_ds, batch_size, workers, weighted=not args.no_weighted_sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    external_loader = DataLoader(external_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    EXTERNAL_COHORT_B_loader = DataLoader(EXTERNAL_COHORT_B_ds, batch_size=batch_size, shuffle=False, num_workers=workers)

    model = MultiTaskCMSGraphModel(model_cfg, projection_dim=args.head_projection_dim).to(device)
    class_weights = {
        "cms": class_weights_for_task(train_ds, "cms", 4, device),
        "msi": class_weights_for_task(train_ds, "msi", 2, device),
        "pn": class_weights_for_task(train_ds, "pn", 2, device),
        "cin": class_weights_for_task(train_ds, "cin", 2, device),
    }
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(tr_cfg.get("weight_decay", 1e-4)))

    run_name = args.run_name or "topofusion_v3_multitask_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(tr_cfg.get("run_root")) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_multitask.json").write_text(json.dumps({"base_config": cfg, "args": vars(args), "task_weights": task_weights}, indent=2), encoding="utf-8")
    metrics_log_path = run_dir / "metrics_log.csv"
    val_expected_counts = filter_task_counts(task_valid_counts_from_rows(val_ds.rows), tasks)
    score_histories = {"val": [], "external": [], "EXTERNAL_COHORT_B": []}

    overlap = {
        "train_val": len(split_patient_keys(train_ds) & split_patient_keys(val_ds)),
        "train_external": len(split_patient_keys(train_ds) & split_patient_keys(external_ds)),
        "train_EXTERNAL_COHORT_B": len(split_patient_keys(train_ds) & split_patient_keys(EXTERNAL_COHORT_B_ds)),
    }
    start_event = {
        "event": "multitask_train_start",
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum,
        "effective_batch_size": batch_size * grad_accum,
        "lr": lr,
        "deduplicate_patients": int(dedup),
        "task_weights": task_weights,
        "tasks": tasks,
        "cms_loss_type": cms_loss_type,
        "cms_focal_gamma": cms_focal_gamma,
        "cms_focal_alpha": cms_focal_alpha,
        "cms_entropy_lambda": cms_entropy_lambda,
        "survival_risk_scale": survival_risk_scale,
        "model_cfg": model_cfg,
        "weight_decay": float(tr_cfg.get("weight_decay", 1e-4)),
        "min_epochs": min_epochs,
        "early_stop_patience": early_stop_patience,
        "model_params": int(sum(p.numel() for p in model.parameters())),
        "slide_embed_dim": int(model.slide_embed_dim),
        "slide_embed_dims": getattr(model, "slide_embed_dims", {}),
        "selection_score": args.selection_score,
        "val_expected_task_counts": val_expected_counts,
        "train": dataset_summary(train_ds),
        "val": dataset_summary(val_ds),
        "external": dataset_summary(external_ds),
        "EXTERNAL_COHORT_B": dataset_summary(EXTERNAL_COHORT_B_ds),
        "patient_overlap": overlap,
    }
    print(json.dumps(start_event, sort_keys=True), flush=True)

    if args.debug_first_batch:
        print(json.dumps({"event": "debug_first_batch", **debug_first_batch(model, train_loader, device, task_weights, class_weights, cms_entropy_lambda, survival_risk_scale, cms_loss_type, cms_focal_gamma, cms_focal_alpha, tasks)}, sort_keys=True), flush=True)

    history = []
    best_score = -1.0
    best_epoch = -1
    for epoch in range(1, int(args.epochs) + 1):
        print(json.dumps({"event": "validation_effective_counts", "epoch": epoch, "split": "val", "expected": val_expected_counts}, sort_keys=True), flush=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        train_stats = train_one_epoch(model, train_loader, device, opt, grad_accum=max(grad_accum, 1), grad_clip_norm=grad_clip_norm, task_weights=task_weights, class_weights=class_weights, cms_entropy_lambda=cms_entropy_lambda, survival_risk_scale=survival_risk_scale, cms_loss_type=cms_loss_type, cms_focal_gamma=cms_focal_gamma, cms_focal_alpha=cms_focal_alpha, tasks=tasks)
        val_metrics = evaluate_multitask(model, val_loader, device, tasks=tasks)
        observed_val_counts = task_counts_from_metrics(val_metrics)
        assert_fixed_task_counts(observed_val_counts, val_expected_counts, "val")
        val_score = compute_score_bundle(val_metrics, score_histories["val"])
        score_histories["val"].append(val_score["task_scores"])
        external_metrics = evaluate_multitask(model, external_loader, device, tasks=tasks) if len(external_ds) else {"skipped": 1}
        external_score = compute_score_bundle(external_metrics, score_histories["external"])
        if len(external_ds):
            score_histories["external"].append(external_score["task_scores"])
        EXTERNAL_COHORT_B_metrics = evaluate_multitask(model, EXTERNAL_COHORT_B_loader, device, tasks=tasks) if len(EXTERNAL_COHORT_B_ds) else {"skipped": 1}
        EXTERNAL_COHORT_B_score = compute_score_bundle(EXTERNAL_COHORT_B_metrics, score_histories["EXTERNAL_COHORT_B"])
        if len(EXTERNAL_COHORT_B_ds):
            score_histories["EXTERNAL_COHORT_B"].append(EXTERNAL_COHORT_B_score["task_scores"])
        append_metrics_log(metrics_log_path, epoch, "val", val_metrics, val_score)
        if len(external_ds):
            append_metrics_log(metrics_log_path, epoch, "EXTERNAL_COHORT_A_external", external_metrics, external_score)
        if len(EXTERNAL_COHORT_B_ds):
            append_metrics_log(metrics_log_path, epoch, "EXTERNAL_COHORT_B", EXTERNAL_COHORT_B_metrics, EXTERNAL_COHORT_B_score)
        row = {"event": "epoch", "epoch": epoch, "train": train_stats, "val": val_metrics, "external": external_metrics, "EXTERNAL_COHORT_B": EXTERNAL_COHORT_B_metrics, "scores": {"val": val_score, "external": external_score, "EXTERNAL_COHORT_B": EXTERNAL_COHORT_B_score}}
        if torch.cuda.is_available():
            row["cuda_memory_mb"] = {
                "allocated": round(torch.cuda.memory_allocated() / 1024**2, 1),
                "reserved": round(torch.cuda.memory_reserved() / 1024**2, 1),
                "max_allocated": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
                "max_reserved": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
            }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        (run_dir / "history_multitask.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        score = float(val_score[args.selection_score])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "best_metric": best_score,
                "selection_score": args.selection_score,
                "val_score": val_score,
                "external_score": external_score,
                "EXTERNAL_COHORT_B_score": EXTERNAL_COHORT_B_score,
                "val_metrics": val_metrics,
                "external_metrics": external_metrics,
                "EXTERNAL_COHORT_B_metrics": EXTERNAL_COHORT_B_metrics,
            }, run_dir / "best_multitask.pt")
        if epoch >= min_epochs and (epoch - best_epoch) >= early_stop_patience:
            print(json.dumps({"event": "early_stop", "epoch": epoch, "best_epoch": best_epoch, "best_val_composite_score": best_score, "selection_score": args.selection_score}, sort_keys=True), flush=True)
            break

    best_loaded = {}
    best_path = run_dir / "best_multitask.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        best_val_metrics = evaluate_multitask(model, val_loader, device, tasks=tasks)
        best_external_metrics = evaluate_multitask(model, external_loader, device, tasks=tasks) if len(external_ds) else {"skipped": 1}
        best_EXTERNAL_COHORT_B_metrics = evaluate_multitask(model, EXTERNAL_COHORT_B_loader, device, tasks=tasks) if len(EXTERNAL_COHORT_B_ds) else {"skipped": 1}
        best_loaded = {
            "best_loaded_epoch": int(ckpt.get("epoch", best_epoch)),
            "best_loaded_val_metrics": best_val_metrics,
            "best_loaded_external_metrics": best_external_metrics,
            "best_loaded_EXTERNAL_COHORT_B_metrics": best_EXTERNAL_COHORT_B_metrics,
        }
        (run_dir / "best_loaded_metrics.json").write_text(json.dumps(best_loaded, indent=2), encoding="utf-8")
    final = {"best_epoch": best_epoch, "best_val_composite_score": best_score, "selection_score": args.selection_score, "history_last": history[-1] if history else {}, **best_loaded}
    (run_dir / "summary_multitask.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps({"event": "multitask_train_done", "run_dir": str(run_dir), **final}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
