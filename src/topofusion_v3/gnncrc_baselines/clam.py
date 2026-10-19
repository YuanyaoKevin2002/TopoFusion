"""CLAM-SB / CLAM-MB — line-for-line port of the official implementation.

Source upstream:
    https://github.com/mahmoodlab/CLAM, file models/model_clam.py
    Lu et al., 2021, Nature Biomedical Engineering.
    "Data-efficient and weakly supervised computational pathology on
     whole-slide images" (CLAM).

This file ports the official `Attn_Net_Gated`, `CLAM_SB`, `CLAM_MB`
classes verbatim, including:
  * Gated tanh-sigmoid 3-layer attention network
  * Per-class instance classifiers + top-k positive / bottom-k negative
    instance sampling (inst_eval / inst_eval_out)
  * Bag-level classifier (SB: shared Linear; MB: per-class Linear)
  * Subtyping switch (out-of-class instance pseudo-labels)

The class signatures match upstream so reviewers can diff against the
public repo. The only changes are:
  * `default size_arg='small'` → matches public default; embed_dim is taken
    from `input_dim` to fit our UNI v2 1536-d features
  * `forward()` signature unchanged

These classes do NOT fit the SlideEncoder/task_head split, by design — the
official model bundles bag-classifier + auxiliary instance loss in one
module. They are trained via gnn_crc_eval.engine.clam_trainer.CLAMTrainer,
not the generic SlideTrainer.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Attention network with sigmoid gating (3 fc layers).
# Verbatim port of Attn_Net_Gated in upstream models/model_clam.py.
# ---------------------------------------------------------------------------
class Attn_Net_Gated(nn.Module):
    def __init__(self, L: int = 1024, D: int = 256, dropout: bool = False,
                 n_classes: int = 1):
        super().__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x: torch.Tensor):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)   # N x n_classes
        return A, x


# ---------------------------------------------------------------------------
# CLAM-SB (single-branch).
# Verbatim port of CLAM_SB in upstream models/model_clam.py.
# ---------------------------------------------------------------------------
class CLAM_SB(nn.Module):
    """
    Args (match upstream):
        gate:             use sigmoid-gated attention (always True in defaults)
        size_arg:         'small' -> [embed_dim, 512, 256] (D-fc layer sizes)
        dropout:          float dropout probability
        k_sample:         number of positive/negative patches sampled for the
                          instance-level loss (default 8 in the paper)
        n_classes:        number of slide-level classes
        instance_loss_fn: loss applied on instance pseudo-labels
                          (default CrossEntropyLoss())
        subtyping:        if True, also sample out-of-class top-k as
                          negative-pseudo-label instances (paper §3.4)
        embed_dim:        input feature dim (UNI v2 = 1536)
    """

    def __init__(
        self,
        gate: bool = True,
        size_arg: str = "small",
        dropout: float = 0.0,
        k_sample: int = 8,
        n_classes: int = 2,
        instance_loss_fn=None,
        subtyping: bool = False,
        embed_dim: int = 1024,
    ):
        super().__init__()
        if instance_loss_fn is None:
            instance_loss_fn = nn.CrossEntropyLoss()
        self.size_dict = {
            "small": [embed_dim, 512, 256],
            "big":   [embed_dim, 512, 384],
        }
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2],
                                           dropout=bool(dropout), n_classes=1)
        else:
            # Non-gated branch retained from upstream code path for parity;
            # never selected in practice because gate=True is the default.
            attention_net = nn.Sequential(nn.Linear(size[1], size[2]), nn.Tanh(),
                                          nn.Linear(size[2], 1))
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        self.instance_classifiers = nn.ModuleList(
            [nn.Linear(size[1], 2) for _ in range(n_classes)]
        )
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    def inst_eval(self, A, h, classifier):
        device = h.device
        if A.dim() == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if A.dim() == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False,
                return_features=False, attention_only=False):
        A, h = self.attention_net(h)            # N x K
        A = torch.transpose(A, 1, 0)            # K x N
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)                 # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss
            if self.subtyping:
                total_inst_loss = total_inst_loss / len(self.instance_classifiers)

        M = torch.mm(A, h)
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {}
        if instance_eval:
            results_dict["instance_loss"] = total_inst_loss
            results_dict["inst_labels"] = np.array(all_targets)
            results_dict["inst_preds"] = np.array(all_preds)
        if return_features:
            results_dict["features"] = M
        return logits, Y_prob, Y_hat, A_raw, results_dict


# ---------------------------------------------------------------------------
# CLAM-MB (multi-branch).
# Verbatim port of CLAM_MB in upstream models/model_clam.py.
# ---------------------------------------------------------------------------
class CLAM_MB(CLAM_SB):
    def __init__(
        self,
        gate: bool = True,
        size_arg: str = "small",
        dropout: float = 0.0,
        k_sample: int = 8,
        n_classes: int = 2,
        instance_loss_fn=None,
        subtyping: bool = False,
        embed_dim: int = 1024,
    ):
        # NB: explicitly call nn.Module.__init__ (not super().__init__) — upstream
        # bypasses CLAM_SB.__init__ to build MB-specific classifier list.
        nn.Module.__init__(self)
        if instance_loss_fn is None:
            instance_loss_fn = nn.CrossEntropyLoss()
        self.size_dict = {
            "small": [embed_dim, 512, 256],
            "big":   [embed_dim, 512, 384],
        }
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2],
                                           dropout=bool(dropout),
                                           n_classes=n_classes)
        else:
            attention_net = nn.Sequential(nn.Linear(size[1], size[2]), nn.Tanh(),
                                          nn.Linear(size[2], n_classes))
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for _ in range(n_classes)]
        self.classifiers = nn.ModuleList(bag_classifiers)
        self.instance_classifiers = nn.ModuleList(
            [nn.Linear(size[1], 2) for _ in range(n_classes)]
        )
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

    def forward(self, h, label=None, instance_eval=False,
                return_features=False, attention_only=False):
        A, h = self.attention_net(h)
        A = torch.transpose(A, 1, 0)
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss
            if self.subtyping:
                total_inst_loss = total_inst_loss / len(self.instance_classifiers)

        M = torch.mm(A, h)
        logits = torch.empty(1, self.n_classes, device=M.device, dtype=M.dtype)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {}
        if instance_eval:
            results_dict["instance_loss"] = total_inst_loss
            results_dict["inst_labels"] = np.array(all_targets)
            results_dict["inst_preds"] = np.array(all_preds)
        if return_features:
            results_dict["features"] = M
        return logits, Y_prob, Y_hat, A_raw, results_dict
