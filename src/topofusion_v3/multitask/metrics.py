from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Mapping

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

from .data_utils import CLASSIFICATION_TASKS, SURVIVAL_TASKS
from .losses import labels_from_batch


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def multiclass_metrics(y_true: np.ndarray, logits: np.ndarray, n_classes: int) -> Dict[str, object]:
    if y_true.size == 0:
        return {"n": 0}
    prob = _softmax(logits)
    pred = prob.argmax(axis=1)
    out: Dict[str, object] = {
        "n": int(y_true.size),
        "acc": float(accuracy_score(y_true, pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "per_class_f1": [float(x) for x in f1_score(y_true, pred, labels=list(range(n_classes)), average=None, zero_division=0)],
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
    prob = _softmax(logits)[:, 1]
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
    """Harrell-like C-index with higher risk meaning shorter survival."""

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
    return {
        "n": int(time.size),
        "events": int(event.sum()),
        "c_index": concordance_index(time, event, risk),
    }


@torch.no_grad()
def evaluate_multitask(model, loader, device: torch.device, tasks: Iterable[str] | None = None) -> Dict[str, object]:
    model.eval()
    tasks = tuple(tasks or ("cms", "msi", "pn", "os", "dfs", "cin"))
    cls_logits = defaultdict(list)
    cls_labels = defaultdict(list)
    surv_time = defaultdict(list)
    surv_event = defaultdict(list)
    surv_risk = defaultdict(list)

    for data in loader:
        data = data.to(device)
        preds = model(data)
        labels = labels_from_batch(data)
        for task in CLASSIFICATION_TASKS:
            if task not in tasks:
                continue
            mask = labels[f"{task}_mask"].bool()
            if mask.any():
                cls_logits[task].append(preds[task][mask].detach().cpu().numpy())
                cls_labels[task].append(labels[task][mask].detach().cpu().numpy())
        for task in SURVIVAL_TASKS:
            if task not in tasks:
                continue
            mask = labels[f"{task}_mask"].bool()
            if mask.any():
                surv_time[task].append(labels[f"{task}_time"][mask].detach().cpu().numpy())
                surv_event[task].append(labels[f"{task}_event"][mask].detach().cpu().numpy())
                surv_risk[task].append(preds[task][mask].view(-1).detach().cpu().numpy())

    out: Dict[str, object] = {}
    for task in CLASSIFICATION_TASKS:
        if task not in tasks:
            continue
        if cls_labels[task]:
            y = np.concatenate(cls_labels[task]).astype(np.int64)
            z = np.concatenate(cls_logits[task], axis=0)
        else:
            y = np.asarray([], dtype=np.int64)
            z = np.empty((0, 4 if task == "cms" else 2), dtype=np.float32)
        out[task] = multiclass_metrics(y, z, 4) if task == "cms" else binary_metrics(y, z)
    for task in SURVIVAL_TASKS:
        if task not in tasks:
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
