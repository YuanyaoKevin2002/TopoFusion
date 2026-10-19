from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def classification_metrics(y_true: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float32)
    pred = logits.argmax(axis=1)
    out = {
        "acc": float(accuracy_score(y_true, pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
    }
    try:
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        prob = exp / exp.sum(axis=1, keepdims=True)
        out["auc_ovr_macro"] = float(roc_auc_score(y_true, prob, multi_class="ovr", average="macro"))
    except Exception:
        out["auc_ovr_macro"] = float("nan")
    return out

