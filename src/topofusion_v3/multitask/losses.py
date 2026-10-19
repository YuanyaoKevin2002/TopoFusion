from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F


CLASSIFICATION_TASKS = ("cms", "msi", "pn", "cin")
SURVIVAL_TASKS = ("os", "dfs")
DEFAULT_TASK_WEIGHTS = {
    "cms": 0.8,
    "msi": 0.8,
    "pn": 1.5,
    "os": 1.5,
    "dfs": 1.5,
    "cin": 0.8,
}


def cox_ph_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Negative partial log likelihood for Cox proportional hazards.

    risk is higher for shorter survival. time is assumed to be in a common unit
    across samples within the task.
    """

    risk = risk.view(-1)
    time = time.view(-1)
    event = event.view(-1).long()
    if risk.numel() == 0 or not (event == 1).any():
        return risk.sum() * 0.0
    order = torch.argsort(-time)
    risk = risk[order]
    event = event[order]
    log_risk = torch.logcumsumexp(risk, dim=0)
    uncensored = event == 1
    return -(risk[uncensored] - log_risk[uncensored]).mean()


def labels_from_batch(data) -> Dict[str, torch.Tensor]:
    labels: Dict[str, torch.Tensor] = {}
    for task in CLASSIFICATION_TASKS:
        labels[task] = getattr(data, task).view(-1)
        labels[f"{task}_mask"] = getattr(data, f"{task}_mask").view(-1).bool()
    for task in SURVIVAL_TASKS:
        labels[f"{task}_time"] = getattr(data, f"{task}_time").view(-1).float()
        labels[f"{task}_event"] = getattr(data, f"{task}_event").view(-1).long()
        labels[f"{task}_mask"] = getattr(data, f"{task}_mask").view(-1).bool()
    return labels


def _zero_like_preds(preds: Mapping[str, torch.Tensor]) -> torch.Tensor:
    first = next(iter(preds.values()))
    return first.sum() * 0.0


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=1)
    prob = log_prob.exp()
    return -(prob * log_prob).sum(dim=1).mean()


def multiclass_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    alpha: Optional[float] = None,
) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=1)
    prob_t = log_prob.gather(1, target.view(-1, 1)).exp().view(-1)
    ce = F.nll_loss(log_prob, target, weight=weight, reduction="none")
    loss = (1.0 - prob_t).pow(float(gamma)) * ce
    if alpha is not None:
        loss = float(alpha) * loss
    return loss.mean()


def compute_losses(
    preds: Mapping[str, torch.Tensor],
    labels: Mapping[str, torch.Tensor],
    task_weights: Optional[Mapping[str, float]] = None,
    class_weights: Optional[Mapping[str, torch.Tensor]] = None,
    tasks: Optional[Iterable[str]] = None,
    cms_entropy_lambda: float = 0.0,
    survival_risk_scale: float = 1.0,
    cms_loss_type: str = "ce",
    cms_focal_gamma: float = 2.0,
    cms_focal_alpha: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, int]]:
    task_weights = dict(DEFAULT_TASK_WEIGHTS if task_weights is None else task_weights)
    class_weights = {} if class_weights is None else dict(class_weights)
    enabled_tasks = set(CLASSIFICATION_TASKS + SURVIVAL_TASKS if tasks is None else tuple(tasks))
    total = _zero_like_preds(preds)
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
        logits = preds[task][mask]
        target = labels[task][mask].long()
        if task == "cms" and str(cms_loss_type).lower() == "focal":
            loss = multiclass_focal_loss(logits, target, weight=weight, gamma=float(cms_focal_gamma), alpha=cms_focal_alpha)
            loss_items["cms_loss_type"] = 1.0
            if cms_focal_alpha is not None:
                loss_items["cms_focal_alpha"] = float(cms_focal_alpha)
        else:
            loss = F.cross_entropy(logits, target, weight=weight)
        if task == "cms" and float(cms_entropy_lambda) != 0.0:
            entropy = _entropy_from_logits(logits)
            loss = loss - float(cms_entropy_lambda) * entropy
            loss_items["cms_entropy"] = float(entropy.detach().cpu())
        total = total + float(task_weights.get(task, 1.0)) * loss
        loss_items[task] = float(loss.detach().cpu())

    for task in SURVIVAL_TASKS:
        if task not in enabled_tasks:
            continue
        mask = labels[f"{task}_mask"].bool()
        event = labels[f"{task}_event"][mask]
        n_items[task] = int(mask.sum().detach().cpu())
        if not mask.any() or not (event == 1).any():
            loss_items[task] = 0.0
            continue
        risk = preds[task][mask] / max(float(survival_risk_scale), 1e-6)
        loss = cox_ph_loss(risk, labels[f"{task}_time"][mask], event)
        total = total + float(task_weights.get(task, 1.0)) * loss
        loss_items[task] = float(loss.detach().cpu())

    loss_items["total"] = float(total.detach().cpu())
    return total, loss_items, n_items
