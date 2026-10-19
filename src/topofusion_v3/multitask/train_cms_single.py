from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader

from topofusion_v3.config import deep_get, load_config
from topofusion_v3.model import CMSGraphModel
from topofusion_v3.multitask.data_utils import MultiTaskGraphCacheDataset
from topofusion_v3.multitask.metrics import multiclass_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def finite_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except Exception:
        return default
    return out if np.isfinite(out) else default


class SingleTaskCMSGraphModel(CMSGraphModel):
    """Downsampled TopoFusion GNN backbone with a CMS-only head.

    The embedding follows Exp5a: cat([global_mean_pool(x_out), raw_mean_pool]).
    Exp5b can reuse the same class by switching raw_pool_mode to attention.
    """

    def __init__(self, cfg: Dict[str, object], raw_pool_mode: str = "mean"):
        cfg = dict(cfg)
        cfg["decoupled_raw_pool_mode"] = raw_pool_mode
        super().__init__(cfg)
        self.fused_dim = int(cfg.get("fused_dim", 128))
        self.raw_pool_mode = raw_pool_mode
        self.cms_include_raw = bool(cfg.get("cms_slide_include_raw", cfg.get("slide_pool_include_raw", True)))
        self.cms_embed_dim = int(self.gnn_out_dim + (self.fused_dim if self.cms_include_raw else 0))
        self.slide_embed_dim = self.cms_embed_dim
        self.cms_head = nn.Linear(self.cms_embed_dim, 4)

    def forward_embedding(self, data) -> torch.Tensor:
        cls_embedding = self.compute_decoupled_embeddings(data)["cls"]
        raw = cls_embedding[:, : self.fused_dim]
        gnn_mean = cls_embedding[:, self.fused_dim :]
        if self.cms_include_raw:
            out = torch.cat([gnn_mean, raw], dim=1)
        else:
            out = gnn_mean
        self._last_embedding_debug["cms_include_raw"] = int(self.cms_include_raw)
        self._last_embedding_debug["cms_embedding_shape"] = list(out.shape)
        return out

    def forward(self, data) -> torch.Tensor:
        return self.cms_head(self.forward_embedding(data))


def filter_cms_rows(ds: MultiTaskGraphCacheDataset) -> None:
    ds.rows = [row for row in ds.rows if row.get("_labels", {}).get("cms") is not None]  # type: ignore[union-attr]


def class_weights_for_cms(ds: MultiTaskGraphCacheDataset, device: torch.device) -> torch.Tensor:
    labels = [int(row["_labels"]["cms"]) for row in ds.rows if row["_labels"].get("cms") is not None]  # type: ignore[index]
    counts = Counter(labels)
    total = float(sum(counts.values()))
    values = []
    for cls in range(4):
        n = counts.get(cls, 0)
        frac = float(n) / total if total > 0 else 0.0
        values.append(1.0 / np.sqrt(max(frac, 1e-6)) if n > 0 else 1.0)
    weights = torch.tensor(values, dtype=torch.float32, device=device)
    return weights / weights.mean()


def make_train_loader(ds: MultiTaskGraphCacheDataset, batch_size: int, workers: int, weighted: bool) -> DataLoader:
    if len(ds) == 0:
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    if weighted:
        labels = [int(row["_labels"]["cms"]) for row in ds.rows]  # type: ignore[index]
        counts = Counter(labels)
        sample_weights = [1.0 / float(counts[y]) for y in labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=workers)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers)


def cms_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor,
    loss_type: str = "ce",
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    loss_type = str(loss_type).lower()
    if loss_type == "ce":
        return F.cross_entropy(logits, target, weight=class_weights)
    if loss_type == "focal":
        log_prob = F.log_softmax(logits, dim=1)
        prob_t = log_prob.gather(1, target.view(-1, 1)).exp().view(-1)
        ce = F.nll_loss(log_prob, target, weight=class_weights, reduction="none")
        return (float(focal_alpha) * (1.0 - prob_t).pow(float(focal_gamma)) * ce).mean()
    raise ValueError(f"Unsupported CMS loss type: {loss_type}")


def labels_from_data(data) -> tuple[torch.Tensor, torch.Tensor]:
    labels = data.cms.view(-1).long()
    mask = data.cms_mask.view(-1).bool()
    return labels, mask


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    grad_accum: int,
    grad_clip_norm: float,
    class_weights: torch.Tensor,
    loss_type: str,
    focal_alpha: float,
    focal_gamma: float,
) -> Dict[str, object]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    n_batches = 0
    n_labels = 0
    for step, data in enumerate(loader, start=1):
        data = data.to(device)
        logits = model(data)
        labels, mask = labels_from_data(data)
        if not mask.any():
            continue
        loss = cms_loss(
            logits[mask],
            labels[mask],
            class_weights=class_weights,
            loss_type=loss_type,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )
        (loss / max(int(grad_accum), 1)).backward()
        if step % max(int(grad_accum), 1) == 0 or step == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        loss_sum += float(loss.detach().cpu())
        n_batches += 1
        n_labels += int(mask.sum().detach().cpu())
    return {
        "loss": loss_sum / max(n_batches, 1),
        "steps": n_batches,
        "cms_labels_seen": n_labels,
    }


@torch.no_grad()
def evaluate_cms(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, object]:
    model.eval()
    logits_list = []
    labels_list = []
    for data in loader:
        data = data.to(device)
        logits = model(data)
        labels, mask = labels_from_data(data)
        if mask.any():
            logits_list.append(logits[mask].detach().cpu().numpy())
            labels_list.append(labels[mask].detach().cpu().numpy())
    if not labels_list:
        return {"n": 0}
    y = np.concatenate(labels_list).astype(np.int64)
    z = np.concatenate(logits_list, axis=0)
    return multiclass_metrics(y, z, n_classes=4)


def cms_counts(ds: MultiTaskGraphCacheDataset) -> Dict[str, int]:
    counts = Counter(int(row["_labels"]["cms"]) for row in ds.rows if row["_labels"].get("cms") is not None)  # type: ignore[index]
    return {f"CMS{cls + 1}": int(counts.get(cls, 0)) for cls in range(4)}


def append_log(path: Path, epoch: int, split: str, metrics: Mapping[str, object], monitor: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "epoch",
        "split",
        "monitor_auc",
        "n",
        "acc",
        "balanced_acc",
        "macro_f1",
        "auc_ovr_macro",
        "per_class_f1",
        "confusion_matrix",
    ]
    exists = path.exists() and path.stat().st_size > 0
    row = {
        "epoch": int(epoch),
        "split": split,
        "monitor_auc": monitor,
        "n": metrics.get("n", ""),
        "acc": metrics.get("acc", ""),
        "balanced_acc": metrics.get("balanced_acc", ""),
        "macro_f1": metrics.get("macro_f1", ""),
        "auc_ovr_macro": metrics.get("auc_ovr_macro", ""),
        "per_class_f1": json.dumps(metrics.get("per_class_f1", [])),
        "confusion_matrix": json.dumps(metrics.get("confusion_matrix", [])),
    }
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def dataset_summary(ds: MultiTaskGraphCacheDataset) -> Dict[str, object]:
    patients = {(row.get("dataset", ""), row.get("patient_id") or row.get("sample_id") or row.get("slide_id", "")) for row in ds.rows}
    return {"n": len(ds), "n_patients": len(patients), "cms_counts": cms_counts(ds)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--clinical-dir", default="${DATA_ROOT}/gnn-crc/data/clinical")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", "--accum-steps", dest="grad_accum_steps", type=int, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--train-splits", default="train")
    parser.add_argument("--val-splits", default="val")
    parser.add_argument("--external-splits", default="EXTERNAL_COHORT_A_external")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--limit-external", type=int, default=0)
    parser.add_argument("--min-epochs", type=int, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--no-dedup-patients", action="store_true")
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--loss-type", choices=["ce", "focal"], default=None)
    parser.add_argument("--raw-pool-mode", choices=["mean", "attention"], default=None)
    parser.add_argument("--focal-alpha", type=float, default=None)
    parser.add_argument("--focal-gamma", type=float, default=None)
    parser.add_argument("--debug-first-batch", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tr_cfg = dict(deep_get(cfg, "training", {}))
    model_cfg = dict(deep_get(cfg, "model", {}))
    single_cfg = dict(deep_get(cfg, "single_task", {}))
    set_seed(int(tr_cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_csv = deep_get(cfg, "manifest.out_csv")
    dedup = not args.no_dedup_patients

    train_ds = MultiTaskGraphCacheDataset(manifest_csv, parse_csv(args.train_splits), args.clinical_dir, limit=args.limit_train, deduplicate_patients=dedup)
    val_ds = MultiTaskGraphCacheDataset(manifest_csv, parse_csv(args.val_splits), args.clinical_dir, limit=args.limit_val, deduplicate_patients=dedup)
    external_ds = MultiTaskGraphCacheDataset(manifest_csv, parse_csv(args.external_splits), args.clinical_dir, limit=args.limit_external, deduplicate_patients=dedup, allow_empty=True)
    for ds in (train_ds, val_ds, external_ds):
        filter_cms_rows(ds)

    batch_size = int(args.batch_size if args.batch_size is not None else tr_cfg.get("batch_size", 2))
    workers = int(args.num_workers if args.num_workers is not None else tr_cfg.get("num_workers", 2))
    grad_accum = int(args.grad_accum_steps if args.grad_accum_steps is not None else tr_cfg.get("grad_accum_steps", 4))
    grad_clip_norm = float(args.grad_clip_norm if args.grad_clip_norm is not None else tr_cfg.get("grad_clip_norm", 1.0))
    lr = float(args.lr if args.lr is not None else tr_cfg.get("lr", 1e-4))
    min_epochs = int(args.min_epochs if args.min_epochs is not None else tr_cfg.get("min_epochs", 20))
    early_stop_patience = int(args.early_stop_patience if args.early_stop_patience is not None else tr_cfg.get("early_stop_patience", 10))
    loss_type = str(args.loss_type or single_cfg.get("cms_loss_type", "ce")).lower()
    raw_pool_mode = str(args.raw_pool_mode or single_cfg.get("raw_pool_mode", "mean")).lower()
    focal_alpha = float(args.focal_alpha if args.focal_alpha is not None else single_cfg.get("cms_focal_alpha", 0.25))
    focal_gamma = float(args.focal_gamma if args.focal_gamma is not None else single_cfg.get("cms_focal_gamma", 2.0))

    train_loader = make_train_loader(train_ds, batch_size, workers, weighted=not args.no_weighted_sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    external_loader = DataLoader(external_ds, batch_size=batch_size, shuffle=False, num_workers=workers)

    model = SingleTaskCMSGraphModel(model_cfg, raw_pool_mode=raw_pool_mode).to(device)
    class_weights = class_weights_for_cms(train_ds, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(tr_cfg.get("weight_decay", 5e-4)))

    run_name = args.run_name or "gnn_cms_single_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(tr_cfg.get("run_root")) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_log = run_dir / "cms_metrics_log.csv"
    start_event = {
        "event": "cms_single_train_start",
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum,
        "effective_batch_size": batch_size * grad_accum,
        "lr": lr,
        "weight_decay": float(tr_cfg.get("weight_decay", 5e-4)),
        "min_epochs": min_epochs,
        "early_stop_patience": early_stop_patience,
        "loss_type": loss_type,
        "raw_pool_mode": raw_pool_mode,
        "focal_alpha": focal_alpha,
        "focal_gamma": focal_gamma,
        "model_params": int(sum(p.numel() for p in model.parameters())),
        "slide_embed_dim": int(model.slide_embed_dim),
        "class_weights_cms": [float(x) for x in class_weights.detach().cpu().tolist()],
        "train": dataset_summary(train_ds),
        "val": dataset_summary(val_ds),
        "external": dataset_summary(external_ds),
    }
    (run_dir / "config_cms_single.json").write_text(json.dumps({"base_config": cfg, "args": vars(args), "start_event": start_event}, indent=2), encoding="utf-8")
    print(json.dumps(start_event, sort_keys=True), flush=True)

    if args.debug_first_batch:
        model.train()
        data = next(iter(train_loader)).to(device)
        logits = model(data)
        labels, mask = labels_from_data(data)
        loss = cms_loss(logits[mask], labels[mask], class_weights, loss_type=loss_type, focal_alpha=focal_alpha, focal_gamma=focal_gamma)
        print(json.dumps({
            "event": "debug_first_batch",
            "logits_shape": list(logits.shape),
            "embedding_debug": getattr(model, "_last_embedding_debug", {}),
            "cms_labels": int(mask.sum().detach().cpu()),
            "loss": float(loss.detach().cpu()),
            "loss_requires_grad": bool(loss.requires_grad),
        }, sort_keys=True), flush=True)

    history = []
    best_auc = -1.0
    best_epoch = -1
    expected_val_n = len(val_ds)
    for epoch in range(1, int(args.epochs) + 1):
        print(json.dumps({"event": "validation_effective_counts", "epoch": epoch, "split": "val", "cms": expected_val_n}, sort_keys=True), flush=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        train_stats = train_one_epoch(
            model,
            train_loader,
            device,
            optimizer,
            grad_accum=grad_accum,
            grad_clip_norm=grad_clip_norm,
            class_weights=class_weights,
            loss_type=loss_type,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )
        val_metrics = evaluate_cms(model, val_loader, device)
        if int(val_metrics.get("n", 0)) != expected_val_n:
            raise RuntimeError(f"val CMS count changed: expected={expected_val_n} observed={val_metrics.get('n', 0)}")
        external_metrics = evaluate_cms(model, external_loader, device) if len(external_ds) else {"n": 0}
        val_auc = finite_float(val_metrics.get("auc_ovr_macro"), default=-1.0)
        append_log(metrics_log, epoch, "val", val_metrics, val_auc)
        if len(external_ds):
            append_log(metrics_log, epoch, "EXTERNAL_COHORT_A_external", external_metrics, finite_float(external_metrics.get("auc_ovr_macro"), default=float("nan")))
        row: Dict[str, object] = {
            "event": "epoch",
            "epoch": epoch,
            "train": train_stats,
            "val": val_metrics,
            "external": external_metrics,
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
        (run_dir / "history_cms_single.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(row, sort_keys=True), flush=True)
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "best_metric": best_auc,
                "val_metrics": val_metrics,
                "external_metrics": external_metrics,
                "loss_type": loss_type,
                "raw_pool_mode": raw_pool_mode,
                "class_weights_cms": class_weights.detach().cpu(),
            }, run_dir / "best_cms_single.pt")
        if epoch >= min_epochs and (epoch - best_epoch) >= early_stop_patience:
            print(json.dumps({"event": "early_stop", "epoch": epoch, "best_epoch": best_epoch, "best_val_cms_auc": best_auc}, sort_keys=True), flush=True)
            break

    best_loaded = {}
    best_path = run_dir / "best_cms_single.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        best_loaded = {
            "best_loaded_epoch": int(ckpt.get("epoch", best_epoch)),
            "best_loaded_val_metrics": evaluate_cms(model, val_loader, device),
            "best_loaded_external_metrics": evaluate_cms(model, external_loader, device) if len(external_ds) else {"n": 0},
        }
        (run_dir / "best_loaded_cms_metrics.json").write_text(json.dumps(best_loaded, indent=2), encoding="utf-8")
    final = {
        "best_epoch": int(best_epoch),
        "best_val_cms_auc": float(best_auc),
        "history_last": history[-1] if history else {},
        **best_loaded,
    }
    (run_dir / "summary_cms_single.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps({"event": "cms_single_train_done", "run_dir": str(run_dir), **final}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
