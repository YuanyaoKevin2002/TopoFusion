from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


CMS_NAMES = ["CMS1", "CMS2", "CMS3", "CMS4"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_manifest_rows(manifest_csv: str | Path, splits: Iterable[str]) -> List[dict]:
    split_set = set(splits)
    rows = []
    with Path(manifest_csv).open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("split") not in split_set:
                continue
            if not Path(row["cache_path"]).exists():
                continue
            if not Path(row["h5_path"]).exists():
                continue
            rows.append(row)
    return rows


def split_rows(rows: List[dict]) -> Dict[str, List[dict]]:
    out = {"train": [], "val": [], "EXTERNAL_COHORT_A_external": []}
    for row in rows:
        if row["split"] in out:
            out[row["split"]].append(row)
    return out


def write_split_manifest(rows_by_split: Dict[str, List[dict]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "abmil_split",
        "dataset",
        "split",
        "slide_id",
        "sample_id",
        "patient_id",
        "cms_label",
        "label_id",
        "h5_path",
        "cache_path",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for split_name in ["train", "val", "EXTERNAL_COHORT_A_external"]:
            for row in rows_by_split[split_name]:
                writer.writerow({k: row.get(k, "") for k in fieldnames if k != "abmil_split"} | {"abmil_split": split_name})


def label_counter(rows: List[dict]) -> Dict[str, int]:
    c = Counter(int(r["label_id"]) for r in rows)
    return {CMS_NAMES[i]: int(c.get(i, 0)) for i in range(4)}


def assert_exact_counts(rows_by_split: Dict[str, List[dict]], expected: Dict[str, int]) -> None:
    actual = {k: len(v) for k, v in rows_by_split.items()}
    if actual != expected:
        raise RuntimeError(f"ABMIL split count mismatch. expected={expected} actual={actual}")


class H5FeatureBagDataset(Dataset):
    def __init__(self, rows: List[dict], max_patches: int = 0, seed: int = 42):
        self.rows = rows
        self.max_patches = int(max_patches)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        h5_path = row["h5_path"]
        with h5py.File(h5_path, "r") as f:
            feats = f["features"][:]
        if self.max_patches > 0 and feats.shape[0] > self.max_patches:
            rng = np.random.default_rng(self.seed + idx)
            keep = np.sort(rng.choice(feats.shape[0], size=self.max_patches, replace=False))
            feats = feats[keep]
        x = torch.from_numpy(feats.astype(np.float32, copy=False))
        y = torch.tensor(int(row["label_id"]), dtype=torch.long)
        meta = {
            "slide_id": row["slide_id"],
            "sample_id": row.get("sample_id", ""),
            "patient_id": row.get("patient_id", ""),
            "dataset": row.get("dataset", ""),
            "split": row.get("split", ""),
            "n_patches": int(x.shape[0]),
        }
        return x, y, meta


def collate_bags(batch):
    xs, ys, metas = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    feat_dim = xs[0].shape[1]
    x_pad = torch.zeros(len(xs), max_len, feat_dim, dtype=torch.float32)
    mask = torch.zeros(len(xs), max_len, dtype=torch.bool)
    for i, x in enumerate(xs):
        n = x.shape[0]
        x_pad[i, :n] = x
        mask[i, :n] = True
    return x_pad, torch.stack(list(ys), dim=0), mask, list(metas)


class GatedABMIL(nn.Module):
    def __init__(
        self,
        in_dim: int = 1536,
        embed_dim: int = 512,
        attn_dim: int = 256,
        n_classes: int = 4,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.attn_v = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Tanh())
        self.attn_u = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Sigmoid())
        self.attn_w = nn.Linear(attn_dim, 1)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        a = self.attn_w(self.attn_v(h) * self.attn_u(h)).squeeze(-1)
        a = a.masked_fill(~mask, torch.finfo(a.dtype).min)
        attn = torch.softmax(a, dim=1)
        bag = torch.sum(h * attn.unsqueeze(-1), dim=1)
        logits = self.classifier(bag)
        return logits, attn


def class_weights(labels: List[int], device: torch.device) -> torch.Tensor:
    counts = Counter(labels)
    total = float(sum(counts.values()))
    weights = []
    for cls in range(4):
        frac = float(counts.get(cls, 1)) / total
        weights.append(1.0 / math.sqrt(max(frac, 1e-6)))
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    return w / w.mean()


def margin_loss_cms1_cms4(logits: torch.Tensor, y: torch.Tensor, margin: float) -> torch.Tensor:
    mask = (y == 0) | (y == 3)
    if not bool(mask.any()):
        return logits.sum() * 0.0
    sub = logits[mask]
    yy = y[mask]
    true_score = sub.gather(1, yy.view(-1, 1)).squeeze(1)
    opposite = torch.where(yy == 0, torch.full_like(yy, 3), torch.zeros_like(yy))
    opp_score = sub.gather(1, opposite.view(-1, 1)).squeeze(1)
    return F.relu(float(margin) - (true_score - opp_score)).mean()


def compute_metrics(y_true: np.ndarray, logits: np.ndarray) -> Dict[str, object]:
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    preds = logits.argmax(axis=1)
    try:
        auc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(y_true, preds, labels=[0, 1, 2, 3])
    return {
        "acc": float(accuracy_score(y_true, preds)),
        "balanced_acc": float(balanced_accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, average="macro", labels=[0, 1, 2, 3], zero_division=0)),
        "auc_ovr_macro": auc,
        "confusion_matrix": cm.tolist(),
        "per_class_f1": {
            CMS_NAMES[i]: float(f1_score((y_true == i).astype(int), (preds == i).astype(int), zero_division=0))
            for i in range(4)
        },
    }


def make_loader(ds: H5FeatureBagDataset, batch_size: int, workers: int, shuffle: bool, weighted: bool = False) -> DataLoader:
    if weighted:
        labels = [int(r["label_id"]) for r in ds.rows]
        counts = Counter(labels)
        sample_weights = [1.0 / float(counts[y]) for y in labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        return DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=workers,
            collate_fn=collate_bags,
            pin_memory=True,
        )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_bags,
        pin_memory=True,
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    ce_weight,
    grad_accum: int,
    grad_clip_norm: float,
    margin_lambda: float,
    margin_value: float,
    use_amp: bool,
) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    correct = 0
    seen = 0
    for step, (x, y, mask, _) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits, _ = model(x, mask)
            loss_ce = F.cross_entropy(logits, y, weight=ce_weight)
            loss_margin = margin_loss_cms1_cms4(logits, y, margin=margin_value)
            loss = loss_ce + float(margin_lambda) * loss_margin
            loss_scaled = loss / max(grad_accum, 1)
        if use_amp:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()
        if step % grad_accum == 0 or step == len(loader):
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        correct += int((logits.argmax(dim=1) == y).sum().detach().cpu())
        seen += int(y.numel())
    return {"loss": float(np.mean(losses)) if losses else float("nan"), "acc": correct / max(seen, 1)}


@torch.no_grad()
def evaluate(model, loader, device, ce_weight, margin_lambda: float, margin_value: float, out_csv: Path | None = None) -> Dict[str, object]:
    model.eval()
    losses = []
    y_all, logits_all, pred_rows = [], [], []
    for x, y, mask, metas in loader:
        x = x.to(device, non_blocking=True)
        y_dev = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        logits, _ = model(x, mask)
        loss_ce = F.cross_entropy(logits, y_dev, weight=ce_weight)
        loss = loss_ce + float(margin_lambda) * margin_loss_cms1_cms4(logits, y_dev, margin=margin_value)
        losses.append(float(loss.detach().cpu()))
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        logits_np = logits.detach().cpu().numpy()
        preds = logits_np.argmax(axis=1)
        y_np = y.numpy()
        y_all.append(y_np)
        logits_all.append(logits_np)
        for i, meta in enumerate(metas):
            row = {
                **meta,
                "true_label_id": int(y_np[i]),
                "true_label": CMS_NAMES[int(y_np[i])],
                "pred_label_id": int(preds[i]),
                "pred_label": CMS_NAMES[int(preds[i])],
            }
            for cls_idx, cls_name in enumerate(CMS_NAMES):
                row[f"prob_{cls_name}"] = float(probs[i, cls_idx])
            pred_rows.append(row)
    y_cat = np.concatenate(y_all, axis=0) if y_all else np.array([], dtype=int)
    logits_cat = np.concatenate(logits_all, axis=0) if logits_all else np.zeros((0, 4), dtype=np.float32)
    metrics = compute_metrics(y_cat, logits_cat) if len(y_cat) else {}
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    metrics["n"] = int(len(y_cat))
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "dataset",
            "split",
            "slide_id",
            "sample_id",
            "patient_id",
            "n_patches",
            "true_label_id",
            "true_label",
            "pred_label_id",
            "pred_label",
            "prob_CMS1",
            "prob_CMS2",
            "prob_CMS3",
            "prob_CMS4",
        ]
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in pred_rows:
                writer.writerow(row)
    return metrics


def parse_expected_counts(raw: str) -> Dict[str, int]:
    out = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        key, value = item.split(":")
        out[key.strip()] = int(value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="${PROJECT_ROOT}/manifests/topofusion_v3_connectfix_manifest.csv")
    parser.add_argument("--out-root", default="${PROJECT_ROOT}/runs")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--expected-counts", default="train:388,val:99,EXTERNAL_COHORT_A_external:106")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--margin-lambda", type=float, default=0.1)
    parser.add_argument("--margin-value", type=float, default=0.2)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--attn-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--max-patches", type=int, default=0, help="0 means use all patches.")
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or f"abmil_40x_gnnmatched_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.out_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest_rows(args.manifest, ["train", "val", "EXTERNAL_COHORT_A_external"])
    rows_by_split = split_rows(rows)
    expected = parse_expected_counts(args.expected_counts)
    assert_exact_counts(rows_by_split, expected)
    write_split_manifest(rows_by_split, run_dir / "abmil_split_manifest.csv")

    config = vars(args) | {
        "run_dir": str(run_dir),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "counts": {k: len(v) for k, v in rows_by_split.items()},
        "label_counts": {k: label_counter(v) for k, v in rows_by_split.items()},
    }
    (run_dir / "config_abmil_40x.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps({"event": "abmil_40x_start", **config}, sort_keys=True), flush=True)

    train_ds = H5FeatureBagDataset(rows_by_split["train"], max_patches=args.max_patches, seed=args.seed)
    val_ds = H5FeatureBagDataset(rows_by_split["val"], max_patches=args.max_patches, seed=args.seed)
    EXTERNAL_COHORT_A_ds = H5FeatureBagDataset(rows_by_split["EXTERNAL_COHORT_A_external"], max_patches=args.max_patches, seed=args.seed)

    train_loader = make_loader(
        train_ds,
        batch_size=args.batch_size,
        workers=args.num_workers,
        shuffle=True,
        weighted=not args.no_weighted_sampler,
    )
    val_loader = make_loader(val_ds, batch_size=args.batch_size, workers=args.num_workers, shuffle=False)
    EXTERNAL_COHORT_A_loader = make_loader(EXTERNAL_COHORT_A_ds, batch_size=args.batch_size, workers=args.num_workers, shuffle=False)

    model = GatedABMIL(embed_dim=args.embed_dim, attn_dim=args.attn_dim, dropout=args.dropout).to(device)
    ce_weight = class_weights([int(r["label_id"]) for r in rows_by_split["train"]], device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and not args.no_amp))
    use_amp = bool(device.type == "cuda" and not args.no_amp)
    print(json.dumps({"event": "model_ready", "params": int(sum(p.numel() for p in model.parameters())), "class_weights": [float(x) for x in ce_weight.detach().cpu()]}, sort_keys=True), flush=True)

    best_metric = -1.0
    best_epoch = -1
    history = []
    for epoch in range(1, args.epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            ce_weight,
            args.grad_accum_steps,
            args.grad_clip_norm,
            args.margin_lambda,
            args.margin_value,
            use_amp,
        )
        val_metrics = evaluate(model, val_loader, device, ce_weight, args.margin_lambda, args.margin_value)
        EXTERNAL_COHORT_A_metrics = evaluate(model, EXTERNAL_COHORT_A_loader, device, ce_weight, args.margin_lambda, args.margin_value)
        row = {"epoch": epoch, "train": train_stats, "val": val_metrics, "EXTERNAL_COHORT_A": EXTERNAL_COHORT_A_metrics}
        if torch.cuda.is_available():
            row["cuda_memory_mb"] = {
                "allocated": round(torch.cuda.memory_allocated() / 1024**2, 1),
                "reserved": round(torch.cuda.memory_reserved() / 1024**2, 1),
                "max_allocated": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
                "max_reserved": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
            }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        (run_dir / "history_abmil_40x.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        score = float(val_metrics.get("macro_f1", 0.0))
        if score > best_metric:
            best_metric = score
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "best_metric": best_metric}, run_dir / "best.pt")
        if epoch >= args.min_epochs and (epoch - best_epoch) >= args.patience:
            print(json.dumps({"event": "early_stop", "epoch": epoch, "best_epoch": best_epoch, "best_val_macro_f1": best_metric}, sort_keys=True), flush=True)
            break

    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    final = {
        "best_epoch": int(ckpt["epoch"]),
        "best_val_macro_f1": float(ckpt["best_metric"]),
        "train": evaluate(model, train_loader, device, ce_weight, args.margin_lambda, args.margin_value, run_dir / "predictions_train.csv"),
        "val": evaluate(model, val_loader, device, ce_weight, args.margin_lambda, args.margin_value, run_dir / "predictions_val.csv"),
        "EXTERNAL_COHORT_A": evaluate(model, EXTERNAL_COHORT_A_loader, device, ce_weight, args.margin_lambda, args.margin_value, run_dir / "predictions_EXTERNAL_COHORT_A.csv"),
    }
    (run_dir / "summary_abmil_40x.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps({"event": "abmil_40x_complete", "run_dir": str(run_dir), **final}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
