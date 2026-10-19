from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

CLASSIFICATION_TASKS = ("cms", "msi", "pn", "cin")
SURVIVAL_TASKS = ("os", "dfs")
TASK_ORDER = ("cms", "msi", "pn", "os", "dfs", "cin")
CMS_TO_ID = {"CMS1": 0, "CMS2": 1, "CMS3": 2, "CMS4": 3}
DEFAULT_TASK_WEIGHTS = {"cms": 1.0, "msi": 1.0, "pn": 1.0, "os": 0.1, "dfs": 0.1, "cin": 1.0}


def clean_value(value: object) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    if value.lower() in {"", "nan", "none", "na", "n/a", "null", "not reported", "unknown"}:
        return ""
    return value


def safe_cache_token(value: object) -> str:
    value = clean_value(value)
    out = []
    for ch in value:
        out.append(ch if ch.isalnum() or ch in {".", "_", "-"} else "_")
    return "".join(out).strip("_") or "unknown"


def feature_cache_path_for_row(cache_dir: str | Path, row: Mapping[str, str]) -> Path:
    root = Path(cache_dir)
    dataset = safe_cache_token(row.get("dataset", "unknown"))
    split = safe_cache_token(row.get("split", "unknown"))
    slide_id = safe_cache_token(row.get("slide_id") or row.get("sample_id") or row.get("patient_id"))
    return root / dataset / split / f"{slide_id}.pt"


def to_float(value: object) -> Optional[float]:
    value = clean_value(value)
    if not value:
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def to_binary_int(value: object) -> Optional[int]:
    x = to_float(value)
    if x is None:
        return None
    if int(x) == 1:
        return 1
    if int(x) == 0:
        return 0
    return None


def cms_label(row: Mapping[str, str]) -> Optional[int]:
    for key in ("label", "cms_from_rna", "cms_raw", "cms_label"):
        value = clean_value(row.get(key)).upper()
        if value in CMS_TO_ID:
            return CMS_TO_ID[value]
    return None


def msi_label(dataset: str, row: Mapping[str, str]) -> Optional[int]:
    dataset = dataset.upper()
    if dataset == "TRAIN_COHORT":
        status = clean_value(row.get("MSI_status") or row.get("msi_status")).upper()
        if status:
            if "MSI-H" in status or status in {"MSI_H", "MSIH"}:
                return 1
            if "MSS" in status or "MSI-L" in status or "MSIL" in status:
                return 0
        sensor = to_float(row.get("msisensor_score"))
        if sensor is not None:
            return int(sensor >= 10.0)
        mantis = to_float(row.get("msi_mantis_score"))
        if mantis is not None:
            return int(mantis >= 0.4)
        return None
    if dataset == "EXTERNAL_COHORT_A":
        status = clean_value(row.get("msi_status")).upper()
        if "MSI-H" in status or status in {"MSIH", "MSI_H"}:
            return 1
        if "MSS" in status or "MSI-L" in status or "MSIL" in status:
            return 0
        return None
    if dataset == "EXTERNAL_COHORT_B":
        status = clean_value(row.get("mmr_status")).lower()
        if not status:
            return None
        if status in {"1", "1.0"} or "dmmr" in status or "deficient" in status:
            return 1
        if status in {"0", "0.0"} or "pmmr" in status or "proficient" in status:
            return 0
    return None


def pn_label(row: Mapping[str, str]) -> Optional[int]:
    value = clean_value(row.get("pN") or row.get("pathologic_N") or row.get("pathologic_n")).upper()
    if not value or value in {"NX", "N X"}:
        return None
    value = value.replace(" ", "")
    if value.startswith("N0"):
        return 0
    if value.startswith(("N1", "N2", "N3")):
        return 1
    return None


def survival_pair(time_value: object, event_value: object) -> Tuple[Optional[float], Optional[int]]:
    time = to_float(time_value)
    event = to_binary_int(event_value)
    if time is None or time <= 0 or event is None:
        return None, None
    return float(time), int(event)


def os_label(dataset: str, row: Mapping[str, str]) -> Tuple[Optional[float], Optional[int]]:
    dataset = dataset.upper()
    if dataset == "TRAIN_COHORT":
        return survival_pair(row.get("OS.time"), row.get("OS"))
    if dataset == "EXTERNAL_COHORT_A":
        return survival_pair(row.get("os_days"), row.get("os_event"))
    if dataset == "EXTERNAL_COHORT_B":
        return survival_pair(row.get("os"), row.get("os.event"))
    return None, None


def dfs_label(dataset: str, row: Mapping[str, str]) -> Tuple[Optional[float], Optional[int]]:
    dataset = dataset.upper()
    if dataset == "TRAIN_COHORT":
        return survival_pair(row.get("PFI.time"), row.get("PFI"))
    if dataset == "EXTERNAL_COHORT_A":
        return survival_pair(row.get("dfs_days"), row.get("dfs_event"))
    if dataset == "EXTERNAL_COHORT_B":
        return survival_pair(row.get("dfs"), row.get("dfs.event"))
    return None, None


def cin_label(dataset: str, row: Mapping[str, str]) -> Optional[int]:
    dataset = dataset.upper()
    if dataset == "TRAIN_COHORT":
        subtype = clean_value(row.get("molecular_subtype")).upper()
        if not subtype:
            return None
        if "CIN" in subtype:
            return 1
        if any(token in subtype for token in ("MSI", "GS", "POLE")):
            return 0
        return None
    if dataset == "EXTERNAL_COHORT_A":
        cin = clean_value(row.get("CIN")).lower()
        if "high" in cin:
            return 1
        if "low" in cin:
            return 0
    return None


def labels_from_clinical_row(dataset: str, row: Mapping[str, str]) -> Dict[str, Optional[float | int]]:
    os_time, os_event = os_label(dataset, row)
    dfs_time, dfs_event = dfs_label(dataset, row)
    return {
        "cms": cms_label(row),
        "msi": msi_label(dataset, row),
        "pn": pn_label(row) if dataset.upper() in {"TRAIN_COHORT", "EXTERNAL_COHORT_A"} else None,
        "os_time": os_time,
        "os_event": os_event,
        "dfs_time": dfs_time,
        "dfs_event": dfs_event,
        "cin": cin_label(dataset, row),
    }


def clinical_files(clinical_dir: str | Path) -> Dict[str, Path]:
    root = Path(clinical_dir)
    return {
        "TRAIN_COHORT": root / "TRAIN_COHORT_clinical_aligned.csv",
        "EXTERNAL_COHORT_A": root / "EXTERNAL_COHORT_A_clinical_aligned.csv",
        "EXTERNAL_COHORT_B": root / "EXTERNAL_COHORT_B_clinical_aligned.csv",
    }


def load_multitask_label_index(clinical_dir: str | Path) -> Dict[tuple[str, str, str], Dict[str, Optional[float | int]]]:
    index: Dict[tuple[str, str, str], Dict[str, Optional[float | int]]] = {}
    for dataset, path in clinical_files(clinical_dir).items():
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                labels = labels_from_clinical_row(dataset, row)
                for key_type, col in (
                    ("sample_id", "sample_id"),
                    ("patient_id", "patient_id"),
                    ("patient_id", "hospital_id"),
                    ("slide_id", "slide_id"),
                    ("path_id", "path_id"),
                ):
                    key = clean_value(row.get(col))
                    if key:
                        index[(dataset, key_type, key)] = labels
    return index


def labels_for_manifest_row(row: Mapping[str, str], label_index: Mapping[tuple[str, str, str], Mapping[str, Optional[float | int]]]) -> Dict[str, Optional[float | int]]:
    dataset = clean_value(row.get("dataset")).upper()
    labels: Dict[str, Optional[float | int]] = {
        "cms": int(row["label_id"]) if clean_value(row.get("label_id")) else None,
        "msi": None,
        "pn": None,
        "os_time": None,
        "os_event": None,
        "dfs_time": None,
        "dfs_event": None,
        "cin": None,
    }
    for key_type, col in (("sample_id", "sample_id"), ("patient_id", "patient_id"), ("slide_id", "slide_id")):
        key = clean_value(row.get(col))
        if not key:
            continue
        found = label_index.get((dataset, key_type, key))
        if found:
            labels.update({k: v for k, v in found.items() if v is not None})
            break
    if labels["cms"] is None and clean_value(row.get("cms_label")).upper() in CMS_TO_ID:
        labels["cms"] = CMS_TO_ID[clean_value(row.get("cms_label")).upper()]
    return labels


def select_manifest_rows(
    manifest_csv: str | Path,
    splits: Iterable[str],
    clinical_dir: str | Path,
    require_h5: bool = True,
    require_graph: bool = False,
    deduplicate_patients: bool = True,
) -> List[dict]:
    split_set = set(splits)
    label_index = load_multitask_label_index(clinical_dir)
    selected: List[dict] = []
    seen: set[tuple[str, str]] = set()
    with Path(manifest_csv).open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("split") not in split_set:
                continue
            if require_h5 and not Path(row.get("h5_path", "")).exists():
                continue
            if require_graph and not Path(row.get("cache_path", "")).exists():
                continue
            patient_key = (row.get("dataset", ""), row.get("patient_id") or row.get("sample_id") or row.get("slide_id", ""))
            if deduplicate_patients and patient_key[1]:
                if patient_key in seen:
                    continue
                seen.add(patient_key)
            out = dict(row)
            out["_labels"] = labels_for_manifest_row(out, label_index)
            selected.append(out)
    return selected


def split_rows(rows: Sequence[dict]) -> Dict[str, List[dict]]:
    out = {"train": [], "val": [], "EXTERNAL_COHORT_A_external": []}
    for row in rows:
        split = row.get("split", "")
        if split in out:
            out[split].append(row)
    return out


def task_counts(rows: Sequence[dict]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for task in CLASSIFICATION_TASKS:
        vals = [int(r["_labels"][task]) for r in rows if r["_labels"].get(task) is not None]
        out[task] = {str(k): int(v) for k, v in Counter(vals).items()}
    for task in SURVIVAL_TASKS:
        events = [
            int(r["_labels"][f"{task}_event"])
            for r in rows
            if r["_labels"].get(f"{task}_time") is not None and r["_labels"].get(f"{task}_event") is not None
        ]
        out[task] = {f"event={k}": int(v) for k, v in Counter(events).items()}
    return out


def dataset_summary(rows: Sequence[dict]) -> Dict[str, object]:
    patients = {(r.get("dataset", ""), r.get("patient_id") or r.get("sample_id") or r.get("slide_id", "")) for r in rows}
    return {"n": len(rows), "n_patients": len(patients), "task_counts": task_counts(rows)}


@dataclass
class Batch:
    items: list
    labels: Dict[str, torch.Tensor]
    metas: list[dict]


class MultiTaskSlideDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[dict],
        mode: str = "bag",
        max_patches: int = 0,
        seed: int = 42,
        feature_cache_dir: str | Path | None = None,
    ):
        if mode not in {"bag", "graph"}:
            raise ValueError(f"mode must be bag or graph, got {mode}")
        self.rows = list(rows)
        self.mode = mode
        self.max_patches = int(max_patches)
        self.seed = int(seed)
        self.feature_cache_dir = Path(feature_cache_dir) if feature_cache_dir else None

    def __len__(self) -> int:
        return len(self.rows)

    def label_values(self, task: str) -> List[int]:
        vals = []
        for row in self.rows:
            value = row["_labels"].get(task)
            if value is not None:
                vals.append(int(value))
        return vals

    def _read_bag(self, row: Mapping[str, str], idx: int) -> torch.Tensor:
        cache_path = None
        if clean_value(row.get("feature_cache_path")):
            cache_path = Path(str(row["feature_cache_path"]))
        elif self.feature_cache_dir is not None:
            cache_path = feature_cache_path_for_row(self.feature_cache_dir, row)
        if cache_path is not None and cache_path.exists():
            obj = torch.load(cache_path, map_location="cpu", weights_only=False)
            feats = obj.get("features") if isinstance(obj, Mapping) else obj
            if not torch.is_tensor(feats):
                feats = torch.as_tensor(feats)
            feats = feats.float() if feats.dtype != torch.float32 else feats
            if self.max_patches > 0 and feats.shape[0] > self.max_patches:
                rng = np.random.default_rng(self.seed + idx)
                keep = np.sort(rng.choice(feats.shape[0], size=self.max_patches, replace=False))
                feats = feats[torch.from_numpy(keep).long()]
            return feats.contiguous()

        with h5py.File(row["h5_path"], "r") as f:
            ds = f["features"]
            n_patches = int(ds.shape[0])
            if self.max_patches > 0 and n_patches > self.max_patches:
                rng = np.random.default_rng(self.seed + idx)
                keep = np.sort(rng.choice(n_patches, size=self.max_patches, replace=False))
                feats = ds[keep]
            else:
                feats = ds[:]
        return torch.from_numpy(np.asarray(feats, dtype=np.float32))

    def _read_graph(self, row: Mapping[str, str]):
        data = torch.load(row["cache_path"], weights_only=False)
        if (not hasattr(data, "x")) or data.x is None:
            if hasattr(data, "patch_feats"):
                data.x = data.patch_feats
            elif hasattr(data, "features"):
                data.x = data.features
            else:
                raise RuntimeError("Graph cache has no x/patch_feats/features")
        data.x = data.x.float()
        if not hasattr(data, "batch") or data.batch is None:
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
        if not hasattr(data, "pos") and hasattr(data, "coords"):
            data.pos = data.coords.float()
        return data

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        item = self._read_graph(row) if self.mode == "graph" else self._read_bag(row, idx)
        meta = {
            "dataset": row.get("dataset", ""),
            "split": row.get("split", ""),
            "slide_id": row.get("slide_id", ""),
            "sample_id": row.get("sample_id", ""),
            "patient_id": row.get("patient_id", ""),
        }
        if torch.is_tensor(item):
            meta["n_patches"] = int(item.shape[0])
        else:
            meta["n_nodes"] = int(item.x.shape[0])
        return item, row["_labels"], meta


def cls_label_tensor(labels: Sequence[dict], task: str) -> tuple[torch.Tensor, torch.Tensor]:
    vals, masks = [], []
    for lab in labels:
        value = lab.get(task)
        has = value is not None
        vals.append(int(value) if has else -1)
        masks.append(bool(has))
    return torch.tensor(vals, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)


def surv_label_tensor(labels: Sequence[dict], task: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    times, events, masks = [], [], []
    for lab in labels:
        time = lab.get(f"{task}_time")
        event = lab.get(f"{task}_event")
        has = time is not None and event is not None
        times.append(float(time) if has else -1.0)
        events.append(int(event) if has else -1)
        masks.append(bool(has))
    return torch.tensor(times, dtype=torch.float32), torch.tensor(events, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)


def collate_multitask(batch) -> Batch:
    items, labels, metas = zip(*batch)
    out_labels: Dict[str, torch.Tensor] = {}
    for task in CLASSIFICATION_TASKS:
        y, mask = cls_label_tensor(labels, task)
        out_labels[task] = y
        out_labels[f"{task}_mask"] = mask
    for task in SURVIVAL_TASKS:
        t, e, mask = surv_label_tensor(labels, task)
        out_labels[f"{task}_time"] = t
        out_labels[f"{task}_event"] = e
        out_labels[f"{task}_mask"] = mask
    return Batch(list(items), out_labels, list(metas))


def make_loader(ds: MultiTaskSlideDataset, batch_size: int, workers: int, weighted: bool, shuffle: bool) -> DataLoader:
    if weighted:
        labels = [int(row["_labels"].get("cms", 0)) for row in ds.rows]
        counts = Counter(labels)
        sample_weights = [1.0 / float(counts[y]) for y in labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=workers, collate_fn=collate_multitask, pin_memory=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers, collate_fn=collate_multitask, pin_memory=False)


class MultiTaskHeads(nn.Module):
    def __init__(self, embed_dim: int, projection_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = int(embed_dim)
        if projection_dim and int(projection_dim) > 0:
            self.shared = nn.Sequential(nn.Linear(self.embed_dim, int(projection_dim)), nn.LayerNorm(int(projection_dim)), nn.ReLU(inplace=True), nn.Dropout(dropout))
            head_dim = int(projection_dim)
        else:
            self.shared = nn.Sequential(nn.LayerNorm(self.embed_dim), nn.Dropout(dropout))
            head_dim = self.embed_dim
        self.cms = nn.Linear(head_dim, 4)
        self.msi = nn.Linear(head_dim, 2)
        self.pn = nn.Linear(head_dim, 2)
        self.os = nn.Linear(head_dim, 1)
        self.dfs = nn.Linear(head_dim, 1)
        self.cin = nn.Linear(head_dim, 2)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.shared(z)
        return {"cms": self.cms(h), "msi": self.msi(h), "pn": self.pn(h), "os": self.os(h), "dfs": self.dfs(h), "cin": self.cin(h)}


def cox_ph_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    risk = risk.float().view(-1)
    time = time.float().view(-1)
    event = event.view(-1).long()
    if risk.numel() == 0 or not (event == 1).any():
        return risk.sum() * 0.0
    order = torch.argsort(-time)
    risk = risk[order]
    event = event[order]
    log_risk = torch.logcumsumexp(risk, dim=0)
    uncensored = event == 1
    return -(risk[uncensored] - log_risk[uncensored]).mean()


def zero_like_preds(preds: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return next(iter(preds.values())).sum() * 0.0


def compute_losses(
    preds: Mapping[str, torch.Tensor],
    labels: Mapping[str, torch.Tensor],
    task_weights: Optional[Mapping[str, float]] = None,
    class_weights: Optional[Mapping[str, torch.Tensor]] = None,
    tasks: Optional[Iterable[str]] = None,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, int]]:
    task_weights = dict(DEFAULT_TASK_WEIGHTS if task_weights is None else task_weights)
    class_weights = {} if class_weights is None else dict(class_weights)
    enabled_tasks = set(CLASSIFICATION_TASKS + SURVIVAL_TASKS if tasks is None else tuple(tasks))
    total = zero_like_preds(preds)
    loss_items: Dict[str, float] = {}
    n_items: Dict[str, int] = {}
    for task in CLASSIFICATION_TASKS:
        if task not in enabled_tasks:
            continue
        mask = labels[f"{task}_mask"].bool()
        n_items[task] = int(mask.sum().detach().cpu())
        if not mask.any():
            loss_items[task] = 0.0
            continue
        weight = class_weights.get(task)
        if weight is not None:
            weight = weight.to(preds[task].device)
        loss = F.cross_entropy(preds[task].float()[mask], labels[task][mask].long(), weight=weight)
        total = total + float(task_weights.get(task, 1.0)) * loss
        loss_items[task] = float(loss.detach().cpu())
    for task in SURVIVAL_TASKS:
        if task not in enabled_tasks:
            continue
        mask = labels[f"{task}_mask"].bool()
        n_items[task] = int(mask.sum().detach().cpu())
        event = labels[f"{task}_event"][mask]
        if not mask.any() or not (event == 1).any():
            loss_items[task] = 0.0
            continue
        loss = cox_ph_loss(preds[task][mask], labels[f"{task}_time"][mask], event)
        total = total + float(task_weights.get(task, 1.0)) * loss
        loss_items[task] = float(loss.detach().cpu())
    loss_items["total"] = float(total.detach().cpu())
    return total, loss_items, n_items


def move_labels(labels: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in labels.items()}


def class_weights_for_task(ds: MultiTaskSlideDataset, task: str, n_classes: int, device: torch.device) -> torch.Tensor:
    labels = ds.label_values(task)
    if not labels:
        return torch.ones(n_classes, dtype=torch.float32, device=device)
    counts = Counter(labels)
    total = float(sum(counts.values()))
    vals = []
    for cls in range(n_classes):
        frac = float(counts.get(cls, 0)) / total
        vals.append(1.0 / math.sqrt(max(frac, 1e-6)) if counts.get(cls, 0) > 0 else 1.0)
    w = torch.tensor(vals, dtype=torch.float32, device=device)
    return w / w.mean()


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def multiclass_metrics(y_true: np.ndarray, logits: np.ndarray, n_classes: int) -> Dict[str, object]:
    if y_true.size == 0:
        return {"n": 0}
    prob = softmax_np(logits)
    pred = prob.argmax(axis=1)
    out: Dict[str, object] = {
        "n": int(y_true.size),
        "acc": float(accuracy_score(y_true, pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=list(range(n_classes))).astype(int).tolist(),
    }
    try:
        out["auc_ovr_macro"] = float(roc_auc_score(y_true, prob, multi_class="ovr", average="macro"))
    except Exception:
        out["auc_ovr_macro"] = float("nan")
    return out


def binary_metrics(y_true: np.ndarray, logits: np.ndarray) -> Dict[str, object]:
    if y_true.size == 0:
        return {"n": 0}
    prob = softmax_np(logits)[:, 1]
    pred = (prob >= 0.5).astype(np.int64)
    cm = confusion_matrix(y_true, pred, labels=[0, 1]).astype(int)
    tn, fp, fn, tp = cm.ravel()
    out: Dict[str, object] = {
        "n": int(y_true.size),
        "acc": float(accuracy_score(y_true, pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "confusion_matrix": cm.tolist(),
    }
    try:
        out["auroc"] = float(roc_auc_score(y_true, prob))
    except Exception:
        out["auroc"] = float("nan")
    return out


def concordance_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=np.int64)
    risk = np.asarray(risk, dtype=np.float64)
    concordant = 0.0
    comparable = 0.0
    n = len(time)
    for i in range(n):
        for j in range(i + 1, n):
            if time[i] == time[j]:
                continue
            if event[i] == 1 and time[i] < time[j]:
                comparable += 1.0
                if risk[i] > risk[j]:
                    concordant += 1.0
                elif risk[i] == risk[j]:
                    concordant += 0.5
            elif event[j] == 1 and time[j] < time[i]:
                comparable += 1.0
                if risk[j] > risk[i]:
                    concordant += 1.0
                elif risk[i] == risk[j]:
                    concordant += 0.5
    return float(concordant / comparable) if comparable > 0 else float("nan")


def survival_metrics(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> Dict[str, object]:
    if time.size == 0:
        return {"n": 0}
    return {"n": int(time.size), "events": int(event.sum()), "c_index": concordance_index(time, event, risk)}


@torch.no_grad()
def evaluate_multitask(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    tasks: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    model.eval()
    enabled_tasks = set(CLASSIFICATION_TASKS + SURVIVAL_TASKS if tasks is None else tuple(tasks))
    cls_logits: Dict[str, list[np.ndarray]] = defaultdict(list)
    cls_labels: Dict[str, list[np.ndarray]] = defaultdict(list)
    surv_time: Dict[str, list[np.ndarray]] = defaultdict(list)
    surv_event: Dict[str, list[np.ndarray]] = defaultdict(list)
    surv_risk: Dict[str, list[np.ndarray]] = defaultdict(list)
    for batch in loader:
        labels = move_labels(batch.labels, device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            preds = model(batch.items, device)
        for task in CLASSIFICATION_TASKS:
            if task not in enabled_tasks:
                continue
            mask = labels[f"{task}_mask"].bool()
            if mask.any():
                cls_logits[task].append(preds[task][mask].detach().float().cpu().numpy())
                cls_labels[task].append(labels[task][mask].detach().cpu().numpy())
        for task in SURVIVAL_TASKS:
            if task not in enabled_tasks:
                continue
            mask = labels[f"{task}_mask"].bool()
            if mask.any():
                surv_time[task].append(labels[f"{task}_time"][mask].detach().cpu().numpy())
                surv_event[task].append(labels[f"{task}_event"][mask].detach().cpu().numpy())
                surv_risk[task].append(preds[task][mask].view(-1).detach().float().cpu().numpy())
    out: Dict[str, object] = {}
    for task in CLASSIFICATION_TASKS:
        if task not in enabled_tasks:
            continue
        if cls_labels[task]:
            y = np.concatenate(cls_labels[task]).astype(np.int64)
            z = np.concatenate(cls_logits[task], axis=0)
        else:
            y = np.asarray([], dtype=np.int64)
            z = np.empty((0, 4 if task == "cms" else 2), dtype=np.float32)
        out[task] = multiclass_metrics(y, z, 4) if task == "cms" else binary_metrics(y, z)
    for task in SURVIVAL_TASKS:
        if task not in enabled_tasks:
            continue
        if surv_time[task]:
            t = np.concatenate(surv_time[task])
            e = np.concatenate(surv_event[task]).astype(np.int64)
            r = np.concatenate(surv_risk[task])
        else:
            t = np.asarray([], dtype=np.float64)
            e = np.asarray([], dtype=np.int64)
            r = np.asarray([], dtype=np.float64)
        out[task] = survival_metrics(t, e, r)
    return out
