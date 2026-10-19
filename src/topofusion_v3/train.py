from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader

from topofusion_v3.config import deep_get, load_config
from topofusion_v3.dataset import GraphCacheDataset
from topofusion_v3.metrics import classification_metrics
from topofusion_v3.model import CMSGraphModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights(labels, device: torch.device) -> torch.Tensor:
    counts = Counter(labels)
    total = float(sum(counts.values()))
    weights = []
    for cls in range(4):
        frac = float(counts.get(cls, 1)) / total
        weights.append(1.0 / np.sqrt(max(frac, 1e-6)))
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    return w / w.mean()


def margin_loss_cms1_cms4(logits: torch.Tensor, y: torch.Tensor, margin: float) -> torch.Tensor:
    mask = (y == 0) | (y == 3)
    if not mask.any():
        return logits.sum() * 0.0
    sub = logits[mask]
    yy = y[mask]
    true_score = sub.gather(1, yy.view(-1, 1)).squeeze(1)
    opposite = torch.where(yy == 0, torch.full_like(yy, 3), torch.zeros_like(yy))
    opp_score = sub.gather(1, opposite.view(-1, 1)).squeeze(1)
    return F.relu(float(margin) - (true_score - opp_score)).mean()


def make_loader(ds: GraphCacheDataset, batch_size: int, workers: int, weighted: bool) -> DataLoader:
    if weighted:
        counts = Counter(ds.labels)
        sample_weights = [1.0 / float(counts[y]) for y in ds.labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=workers)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers)


@torch.no_grad()
def evaluate(model: CMSGraphModel, loader: DataLoader, device: torch.device, ce_weight: torch.Tensor) -> Tuple[float, Dict[str, float]]:
    model.eval()
    losses, logits_all, y_all = [], [], []
    for data in loader:
        data = data.to(device)
        logits = model(data)
        y = data.y.view(-1)
        loss = F.cross_entropy(logits, y, weight=ce_weight)
        losses.append(float(loss.detach().cpu()))
        logits_all.append(logits.detach().cpu().numpy())
        y_all.append(y.detach().cpu().numpy())
    logits_np = np.concatenate(logits_all, axis=0)
    y_np = np.concatenate(y_all, axis=0)
    metrics = classification_metrics(y_np, logits_np)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics["loss"], metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--grad-accum-steps", "--accum-steps", "--accum_steps",
        dest="grad_accum_steps", type=int, default=None,
        help="Gradient accumulation steps (effective batch_size = batch_size * grad_accum_steps)",
    )
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--eval-EXTERNAL_COHORT_B-cached", type=int, default=None)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--limit-external", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tr_cfg = dict(deep_get(cfg, "training", {}))
    model_cfg = dict(deep_get(cfg, "model", {}))
    seed = int(tr_cfg.get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_csv = deep_get(cfg, "manifest.out_csv")
    train_ds = GraphCacheDataset(manifest_csv, "train", limit=args.limit_train)
    val_ds = GraphCacheDataset(manifest_csv, "val", limit=args.limit_val)
    EXTERNAL_COHORT_A_ds = GraphCacheDataset(manifest_csv, "EXTERNAL_COHORT_A_external", limit=args.limit_external)
    eval_EXTERNAL_COHORT_B_cached = bool(
        args.eval_EXTERNAL_COHORT_B_cached
        if args.eval_EXTERNAL_COHORT_B_cached is not None
        else tr_cfg.get("eval_EXTERNAL_COHORT_B_cached", False)
    )
    EXTERNAL_COHORT_B_ds = GraphCacheDataset(manifest_csv, "EXTERNAL_COHORT_B_external", limit=args.limit_external) if eval_EXTERNAL_COHORT_B_cached else None

    batch_size = int(args.batch_size if args.batch_size is not None else tr_cfg.get("batch_size", 1))
    workers = int(args.num_workers if args.num_workers is not None else tr_cfg.get("num_workers", 2))
    train_loader = make_loader(train_ds, batch_size, workers, bool(tr_cfg.get("use_weighted_sampler", True)))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    EXTERNAL_COHORT_A_loader = DataLoader(EXTERNAL_COHORT_A_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    EXTERNAL_COHORT_B_loader = (
        DataLoader(EXTERNAL_COHORT_B_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
        if EXTERNAL_COHORT_B_ds is not None
        else None
    )

    model = CMSGraphModel(model_cfg).to(device)
    ce_weight = class_weights(train_ds.labels, device)
    lr = float(args.lr if args.lr is not None else tr_cfg.get("lr", 1e-4))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(tr_cfg.get("weight_decay", 1e-4)))
    epochs = int(args.epochs if args.epochs is not None else tr_cfg.get("epochs", 80))
    grad_accum = int(args.grad_accum_steps if args.grad_accum_steps is not None else tr_cfg.get("grad_accum_steps", 4))
    if grad_accum < 1:
        raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum}")
    grad_clip_norm = float(
        args.grad_clip_norm if args.grad_clip_norm is not None else tr_cfg.get("grad_clip_norm", 1.0)
    )
    margin_lambda = float(tr_cfg.get("margin_lambda", 0.1))
    margin_value = float(tr_cfg.get("margin_value", 0.2))
    patience = int(tr_cfg.get("early_stop_patience", 20))
    min_epochs = int(tr_cfg.get("min_epochs", 20))

    run_name = args.run_name or f"topofusion_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(tr_cfg.get("run_root")) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "train_start",
                "device": str(device),
                "cuda_visible_devices": str(__import__("os").environ.get("CUDA_VISIBLE_DEVICES", "")),
                "batch_size": batch_size,
                "grad_accum_steps": grad_accum,
                "effective_batch_size": batch_size * grad_accum,
                "grad_clip_norm": grad_clip_norm,
                "num_workers": workers,
                "lr": lr,
                "n_train": len(train_ds),
                "n_val": len(val_ds),
                "n_EXTERNAL_COHORT_A": len(EXTERNAL_COHORT_A_ds),
                "n_EXTERNAL_COHORT_B": len(EXTERNAL_COHORT_B_ds) if EXTERNAL_COHORT_B_ds is not None else 0,
                "eval_EXTERNAL_COHORT_B_cached": int(eval_EXTERNAL_COHORT_B_cached),
                "model_params": int(sum(p.numel() for p in model.parameters())),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    best_metric = -1.0
    best_epoch = -1
    history = []
    for epoch in range(1, epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model.train()
        opt.zero_grad(set_to_none=True)
        train_losses = []
        for step, data in enumerate(train_loader, start=1):
            data = data.to(device)
            logits = model(data)
            y = data.y.view(-1)
            loss_ce = F.cross_entropy(logits, y, weight=ce_weight)
            loss_margin = margin_loss_cms1_cms4(logits, y, margin=margin_value)
            loss = loss_ce + margin_lambda * loss_margin
            (loss / grad_accum).backward()
            if step % grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)
            train_losses.append(float(loss.detach().cpu()))

        _, val_metrics = evaluate(model, val_loader, device, ce_weight)
        _, EXTERNAL_COHORT_A_metrics = evaluate(model, EXTERNAL_COHORT_A_loader, device, ce_weight)
        if EXTERNAL_COHORT_B_loader is not None:
            _, EXTERNAL_COHORT_B_metrics = evaluate(model, EXTERNAL_COHORT_B_loader, device, ce_weight)
        else:
            EXTERNAL_COHORT_B_metrics = {"skipped": 1.0}
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val": val_metrics,
            "EXTERNAL_COHORT_A": EXTERNAL_COHORT_A_metrics,
            "EXTERNAL_COHORT_B": EXTERNAL_COHORT_B_metrics,
        }
        if torch.cuda.is_available():
            row["cuda_memory_mb"] = {
                "allocated": round(torch.cuda.memory_allocated() / 1024**2, 1),
                "reserved": round(torch.cuda.memory_reserved() / 1024**2, 1),
                "max_allocated": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
                "max_reserved": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
            }
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        score = float(val_metrics.get("macro_f1", 0.0))
        if score > best_metric:
            best_metric = score
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "best_metric": best_metric}, run_dir / "best.pt")
        if epoch >= min_epochs and (epoch - best_epoch) >= patience:
            print(f"early_stop epoch={epoch} best_epoch={best_epoch} best_val_macro_f1={best_metric:.4f}")
            break

    final = {"best_epoch": best_epoch, "best_val_macro_f1": best_metric, "history_last": history[-1] if history else {}}
    (run_dir / "summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print("run_dir", run_dir)


if __name__ == "__main__":
    main()
