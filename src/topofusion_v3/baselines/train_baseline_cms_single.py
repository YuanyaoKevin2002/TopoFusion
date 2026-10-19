#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
import sys

sys.path.insert(0, str(ROOT / "src"))

from topofusion_v3.baselines.multitask_baseline_adapters import make_multitask_baseline, model_uses_graph
from topofusion_v3.baselines.multitask_utils import (
    MultiTaskSlideDataset,
    class_weights_for_task,
    dataset_summary,
    make_loader,
    move_labels,
    select_manifest_rows,
    split_rows,
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


def assert_exact_counts(rows_by_split: Mapping[str, Sequence[dict]], expected: Mapping[str, int]) -> None:
    actual = {k: len(v) for k, v in rows_by_split.items()}
    if dict(actual) != dict(expected):
        raise RuntimeError(f"split count mismatch: expected={dict(expected)} actual={actual}")


def filter_cms_rows(rows_by_split: Dict[str, list[dict]]) -> Dict[str, list[dict]]:
    return {
        split: [row for row in rows if row.get("_labels", {}).get("cms") is not None]
        for split, rows in rows_by_split.items()
    }


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


class CMSOnlyBaselineModel(nn.Module):
    def __init__(self, multitask_model, dropout: float = 0.1):
        super().__init__()
        self.encoder = multitask_model.encoder
        self.embedding_dim = int(multitask_model.embedding_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(float(dropout)),
            nn.Linear(self.embedding_dim, 4),
        )

    def forward_embedding(self, items: Sequence, device: torch.device) -> torch.Tensor:
        embeds = []
        for item in items:
            obj = item.to(device, non_blocking=True) if torch.is_tensor(item) else item.to(device)
            embeds.append(self.encoder.forward_one(obj))
        return torch.stack(embeds, dim=0)

    def forward(self, items: Sequence, device: torch.device) -> torch.Tensor:
        return self.head(self.forward_embedding(items, device))


def make_cms_model(args) -> CMSOnlyBaselineModel:
    base = make_multitask_baseline(
        args.model,
        input_dim=args.input_dim,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        attn_heads=args.attn_heads,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        projection_dim=0,
        clam_mb_pool=args.clam_mb_pool,
        dsmil_pool=args.dsmil_pool,
        patch_gcn_hidden_dim=args.patch_gcn_hidden_dim,
        patch_gcn_layers=args.patch_gcn_layers,
        heat_hidden_dim=args.heat_hidden_dim,
        heat_layers=args.heat_layers,
        hact_hidden_dim=args.hact_hidden_dim,
        hact_layers=args.hact_layers,
    )
    return CMSOnlyBaselineModel(base, dropout=args.head_dropout)


def cms_metrics(y_true: np.ndarray, logits: np.ndarray) -> Dict[str, object]:
    if y_true.size == 0:
        return {"n": 0}
    logits = np.asarray(logits, dtype=np.float64)
    prob = np.exp(logits - logits.max(axis=1, keepdims=True))
    prob = prob / prob.sum(axis=1, keepdims=True)
    pred = prob.argmax(axis=1)
    out: Dict[str, object] = {
        "n": int(y_true.size),
        "acc": float(accuracy_score(y_true, pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "per_class_f1": [float(x) for x in f1_score(y_true, pred, labels=[0, 1, 2, 3], average=None, zero_division=0)],
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1, 2, 3]).astype(int).tolist(),
    }
    try:
        out["auc_ovr_macro"] = float(roc_auc_score(y_true, prob, multi_class="ovr", average="macro"))
    except Exception:
        out["auc_ovr_macro"] = float("nan")
    return out


def train_one_epoch(model, loader, device, optimizer, scaler, class_weights, grad_clip_norm: float, use_amp: bool, accum_steps: int) -> Dict[str, object]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    steps = 0
    opt_steps = 0
    seen = 0
    accum_steps = max(int(accum_steps), 1)
    for batch_idx, batch in enumerate(loader):
        labels = move_labels(batch.labels, device)
        mask = labels["cms_mask"].bool()
        if not mask.any():
            continue
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(batch.items, device)
            loss = F.cross_entropy(logits.float()[mask], labels["cms"][mask].long(), weight=class_weights)
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
        loss_sum += float(loss.detach().cpu())
        seen += int(mask.sum().detach().cpu())
        steps += 1
    return {"loss": loss_sum / max(steps, 1), "cms_labels_seen": seen, "steps": steps, "optimizer_steps": opt_steps}


@torch.no_grad()
def evaluate_cms(model, loader, device, use_amp: bool = False) -> Dict[str, object]:
    model.eval()
    labels_all = []
    logits_all = []
    for batch in loader:
        labels = move_labels(batch.labels, device)
        mask = labels["cms_mask"].bool()
        if not mask.any():
            continue
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(batch.items, device)
        labels_all.append(labels["cms"][mask].detach().cpu().numpy())
        logits_all.append(logits[mask].detach().float().cpu().numpy())
    if not labels_all:
        return {"n": 0}
    return cms_metrics(np.concatenate(labels_all).astype(np.int64), np.concatenate(logits_all, axis=0))


@torch.no_grad()
def debug_first_batch(model, loader, device, class_weights, use_amp: bool) -> Dict[str, object]:
    model.train()
    batch = next(iter(loader))
    labels = move_labels(batch.labels, device)
    with torch.cuda.amp.autocast(enabled=use_amp):
        logits = model(batch.items, device)
        loss = F.cross_entropy(logits.float()[labels["cms_mask"].bool()], labels["cms"][labels["cms_mask"].bool()].long(), weight=class_weights)
    item_sizes = []
    for item in batch.items:
        item_sizes.append(int(item.shape[0]) if torch.is_tensor(item) else int(item.x.shape[0]))
    return {
        "batch_size": len(batch.items),
        "item_sizes": item_sizes,
        "logits_shape": list(logits.shape),
        "cms_labels": int(labels["cms_mask"].sum().detach().cpu()),
        "loss": float(loss.detach().cpu()),
        "loss_requires_grad": bool(loss.requires_grad),
        "first_slides": [m.get("slide_id", "") for m in batch.metas[:3]],
    }


def append_metrics_log(path: Path, epoch: int, split: str, metrics: Mapping[str, object], monitor_auc: float) -> None:
    fields = ["epoch", "split", "monitor_auc", "n", "acc", "balanced_acc", "macro_f1", "auc_ovr_macro", "per_class_f1", "confusion_matrix"]
    exists = path.exists() and path.stat().st_size > 0
    row = {
        "epoch": int(epoch),
        "split": split,
        "monitor_auc": monitor_auc,
        "n": metrics.get("n", ""),
        "acc": metrics.get("acc", ""),
        "balanced_acc": metrics.get("balanced_acc", ""),
        "macro_f1": metrics.get("macro_f1", ""),
        "auc_ovr_macro": metrics.get("auc_ovr_macro", ""),
        "per_class_f1": json.dumps(metrics.get("per_class_f1", [])),
        "confusion_matrix": json.dumps(metrics.get("confusion_matrix", [])),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def finite_auc(metrics: Mapping[str, object]) -> float:
    try:
        out = float(metrics.get("auc_ovr_macro"))  # type: ignore[arg-type]
    except Exception:
        return -1.0
    return out if np.isfinite(out) else -1.0


def cms_counts(rows: Sequence[dict]) -> Dict[str, int]:
    counts = Counter(int(row["_labels"]["cms"]) for row in rows if row["_labels"].get("cms") is not None)
    return {f"CMS{i + 1}": int(counts.get(i, 0)) for i in range(4)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["abmil", "meanpool", "transmil", "clam_sb", "clam_mb", "dsmil", "patch_gcn", "heat_pyg", "hact_pyg"])
    parser.add_argument("--manifest", default="${PROJECT_ROOT}/manifests/topofusion_v3_downsample_v1_manifest.csv")
    parser.add_argument("--clinical-dir", default="${DATA_ROOT}/gnn-crc/data/clinical")
    parser.add_argument("--feature-cache-dir", default="${PROJECT_ROOT}/cache/baseline_features_40x")
    parser.add_argument("--out-root", default="${PROJECT_ROOT}/runs/baseline_cms_single_40x")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--expected-counts", default="train:385,val:96,EXTERNAL_COHORT_A_external:98")
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
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--input-dim", type=int, default=1536)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--attn-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--clam-mb-pool", default="flatten", choices=["flatten", "mean"])
    parser.add_argument("--dsmil-pool", default="flatten", choices=["flatten", "mean"])
    parser.add_argument("--patch-gcn-hidden-dim", type=int, default=128)
    parser.add_argument("--patch-gcn-layers", type=int, default=4)
    parser.add_argument("--heat-hidden-dim", type=int, default=256)
    parser.add_argument("--heat-layers", type=int, default=2)
    parser.add_argument("--hact-hidden-dim", type=int, default=256)
    parser.add_argument("--hact-layers", type=int, default=2)
    parser.add_argument("--max-patches", type=int, default=0)
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--debug-first-batch", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(device.type == "cuda" and not args.no_amp)
    mode = "graph" if model_uses_graph(args.model) else "bag"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.model}_cms_single_40x_{stamp}"
    run_dir = Path(args.out_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = select_manifest_rows(
        args.manifest,
        ("train", "val", "EXTERNAL_COHORT_A_external"),
        args.clinical_dir,
        require_h5=True,
        require_graph=(mode == "graph"),
        deduplicate_patients=True,
    )
    rows_by_split = filter_cms_rows(split_rows(rows))
    if args.limit_per_split > 0:
        rows_by_split = {k: list(v)[: int(args.limit_per_split)] for k, v in rows_by_split.items()}
    if not args.skip_count_check and args.limit_per_split <= 0:
        assert_exact_counts(rows_by_split, parse_expected_counts(args.expected_counts))
    write_split_manifest(rows_by_split, run_dir / "baseline_cms_single_split_manifest.csv")

    feature_cache_dir = args.feature_cache_dir if mode == "bag" and args.feature_cache_dir else None
    train_ds = MultiTaskSlideDataset(rows_by_split["train"], mode=mode, max_patches=args.max_patches, seed=args.seed, feature_cache_dir=feature_cache_dir)
    val_ds = MultiTaskSlideDataset(rows_by_split["val"], mode=mode, max_patches=args.max_patches, seed=args.seed, feature_cache_dir=feature_cache_dir)
    ext_ds = MultiTaskSlideDataset(rows_by_split["EXTERNAL_COHORT_A_external"], mode=mode, max_patches=args.max_patches, seed=args.seed, feature_cache_dir=feature_cache_dir)
    train_loader = make_loader(train_ds, batch_size=args.batch_size, workers=args.num_workers, weighted=args.weighted_sampler, shuffle=True)
    val_loader = make_loader(val_ds, batch_size=args.batch_size, workers=args.num_workers, weighted=False, shuffle=False)
    ext_loader = make_loader(ext_ds, batch_size=args.batch_size, workers=args.num_workers, weighted=False, shuffle=False)

    model = make_cms_model(args).to(device)
    class_weights = None if args.no_class_weights else class_weights_for_task(train_ds, "cms", 4, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    metrics_log_path = run_dir / "cms_metrics_log.csv"

    start_event = {
        "event": "baseline_cms_single_start",
        "model": args.model,
        "mode": mode,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "effective_batch_size": args.batch_size * args.accum_steps,
        "num_workers": args.num_workers,
        "max_patches": args.max_patches,
        "feature_cache_dir": str(feature_cache_dir) if feature_cache_dir else "",
        "use_class_weights": not bool(args.no_class_weights),
        "class_weights_cms": [float(x) for x in class_weights.detach().cpu().tolist()] if class_weights is not None else [],
        "model_params": int(sum(p.numel() for p in model.parameters())),
        "embedding_dim": int(model.embedding_dim),
        "run_dir": str(run_dir),
        "monitor": "val_cms_auc",
        "train": {**dataset_summary(rows_by_split["train"]), "cms_counts": cms_counts(rows_by_split["train"])},
        "val": {**dataset_summary(rows_by_split["val"]), "cms_counts": cms_counts(rows_by_split["val"])},
        "external": {**dataset_summary(rows_by_split["EXTERNAL_COHORT_A_external"]), "cms_counts": cms_counts(rows_by_split["EXTERNAL_COHORT_A_external"])},
    }
    print(json.dumps(start_event, sort_keys=True), flush=True)
    (run_dir / "config_baseline_cms_single.json").write_text(json.dumps({"args": vars(args), "start_event": start_event}, indent=2), encoding="utf-8")

    if args.debug_first_batch:
        print(json.dumps({"event": "debug_first_batch", **debug_first_batch(model, train_loader, device, class_weights, use_amp)}, sort_keys=True), flush=True)

    history = []
    best_auc = -1.0
    best_epoch = -1
    expected_val_n = len(rows_by_split["val"])
    for epoch in range(1, args.epochs + 1):
        print(json.dumps({"event": "validation_effective_counts", "model": args.model, "epoch": epoch, "split": "val", "cms": expected_val_n}, sort_keys=True), flush=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        train_stats = train_one_epoch(model, train_loader, device, optimizer, scaler, class_weights, args.grad_clip_norm, use_amp, args.accum_steps)
        val_metrics = evaluate_cms(model, val_loader, device, use_amp=use_amp)
        if int(val_metrics.get("n", 0)) != expected_val_n:
            raise RuntimeError(f"val CMS count changed: expected={expected_val_n} observed={val_metrics.get('n', 0)}")
        ext_metrics = evaluate_cms(model, ext_loader, device, use_amp=use_amp)
        val_auc = finite_auc(val_metrics)
        append_metrics_log(metrics_log_path, epoch, "val", val_metrics, val_auc)
        append_metrics_log(metrics_log_path, epoch, "EXTERNAL_COHORT_A_external", ext_metrics, finite_auc(ext_metrics))
        row: Dict[str, object] = {
            "event": "epoch",
            "model": args.model,
            "epoch": epoch,
            "train": train_stats,
            "val": val_metrics,
            "external": ext_metrics,
            "monitor": {"val_cms_auc": val_auc},
        }
        if torch.cuda.is_available():
            row["cuda_memory_mb"] = {
                "allocated": round(torch.cuda.memory_allocated() / 1024**2, 1),
                "reserved": round(torch.cuda.memory_reserved() / 1024**2, 1),
                "max_allocated": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
                "max_reserved": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
            }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        (run_dir / "history_baseline_cms_single.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if use_amp else None,
                    "epoch": epoch,
                    "best_metric": best_auc,
                    "args": vars(args),
                    "val_metrics": val_metrics,
                    "external_metrics": ext_metrics,
                },
                run_dir / "best_cms_single.pt",
            )
        if epoch >= args.min_epochs and (epoch - best_epoch) >= args.early_stop_patience:
            print(json.dumps({"event": "early_stop", "epoch": epoch, "best_epoch": best_epoch, "best_val_cms_auc": best_auc}, sort_keys=True), flush=True)
            break

    best_loaded = {}
    best_path = run_dir / "best_cms_single.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        best_loaded = {
            "best_loaded_epoch": int(ckpt.get("epoch", best_epoch)),
            "best_loaded_val_metrics": evaluate_cms(model, val_loader, device, use_amp=use_amp),
            "best_loaded_external_metrics": evaluate_cms(model, ext_loader, device, use_amp=use_amp),
        }
        (run_dir / "best_loaded_cms_metrics.json").write_text(json.dumps(best_loaded, indent=2), encoding="utf-8")

    final = {
        "event": "baseline_cms_single_done",
        "model": args.model,
        "best_epoch": int(best_epoch),
        "best_val_cms_auc": float(best_auc),
        "history_last": history[-1] if history else {},
        "run_dir": str(run_dir),
        **best_loaded,
    }
    (run_dir / "summary_baseline_cms_single.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
