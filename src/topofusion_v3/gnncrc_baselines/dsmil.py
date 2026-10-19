"""DSMIL: Dual-Stream Multiple Instance Learning.

Reference: Li, Li & Eliceiri 2021 (CVPR) — Dual-Stream MIL Network for WSI.
Architecture:
    stream 1 — instance classifier emits per-class logits for every patch
               (`[N, num_classes]`). The critical instance is `argmax` over
               the per-instance max-class score; its full per-class logit
               row is fed to `L_max`.
    stream 2 — cross-attention between the critical instance and all
               instances → bag representation → bag classifier emits
               slide-level logits (`[1, num_classes]`).

D5 PI directive (2026-06-04, "DSMIL: restore L_max, do not relabel"):
DSMIL no longer flows through the shared `ClassificationHead`. Its
`forward` returns BOTH the per-critical-instance logits AND the bag
logits as a dict; `DSMILTrainer` consumes both and computes the
composite L = 0.5·L_bag + 0.5·L_max using D1's weighted CE.

Upstream reference (binli123/dsmil-wsi, file `dsmil.py`):
  * `IClassifier.fc: Linear(feature_size, output_class)` — per-instance
    output is `[N, output_class]`, NOT `[N, 1]`. The classes vs
    instance-score interpretation: `c = fc(feats)` gives `[N, C]` and the
    critical instance per class is chosen by `argmax(c, dim=0)` (paper §3.2
    eq. 2, code lines ~30-40).
  * `BClassifier` then computes attention between the critical instance
    embedding and all instances and emits a bag prediction `[C]`.
  * `MILNet.forward` returns `(classes, prediction_bag, A, B)`:
    classes = per-instance logits [N, C]; prediction_bag = bag logits [C].
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from topofusion_v3.gnncrc_baselines.base import SlideEncoder
from topofusion_v3.gnncrc_baselines.registry import register_model


class _IClassifier(nn.Module):
    """Per-instance classifier: feature → `[N, num_classes]` logits.

    The critical instance (per-class) is picked by `argmax(logits, dim=0)`;
    the critical instance's full per-class row is what L_max consumes.
    """

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)  # [N, num_classes]


class _BClassifier(nn.Module):
    """Bag aggregator via cross-attention to the critical instance, then
    a per-class slide-logit head.

    Inputs:
      h: [N, D]               — feature-projected instances
      crit_idx_per_class: [C] — critical-instance index per class
    Returns:
      bag_logits: [num_classes]
    """

    def __init__(self, feature_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.q = nn.Linear(feature_dim, feature_dim)
        self.v = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.Dropout(dropout))
        self.scale = feature_dim ** -0.5
        # Per-class bag head: concat(critical_per_class, bag_per_class) → 1 logit per class.
        # Mirrors upstream BClassifier where the per-class bag prediction is
        # `Linear(2*D, 1)` applied per class then stacked.
        self.bag_head = nn.Conv1d(num_classes, num_classes, kernel_size=feature_dim * 2,
                                  groups=num_classes)

    def forward(self, h: torch.Tensor, crit_idx_per_class: torch.Tensor) -> torch.Tensor:
        # h: [N, D]; crit_idx_per_class: [C]
        # For each class c, take the critical instance, compute attention
        # over all instances using that critical's query, then aggregate.
        C = crit_idx_per_class.shape[0]
        critical = h[crit_idx_per_class]               # [C, D]
        Q = self.q(critical)                           # [C, D]
        V = self.v(h)                                  # [N, D]
        # Attention scores: [C, N] = Q @ V^T (scaled).
        attn = F.softmax((Q @ V.t()) * self.scale, dim=1)   # [C, N]
        bag_per_class = attn @ V                       # [C, D]
        # Per-class bag head: concat(critical, bag_per_class) → 1 logit per class.
        concat = torch.cat([critical, bag_per_class], dim=1)   # [C, 2D]
        concat = concat.unsqueeze(0)                            # [1, C, 2D]
        logits = self.bag_head(concat).view(-1)                 # [C]
        return logits


@register_model("dsmil")
class DSMIL(SlideEncoder):
    """Dual-stream MIL.

    Args:
        input_dim:   patch feature dim (1536 for UNI v2)
        hidden_dim:  per-stream feature dim
        num_classes: slide-level class count (matches `cfg.task.num_classes`)
        dropout:     dropout in feature proj and stream-2 value projection
    """

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        # DSMIL no longer flows into the shared ClassificationHead (D5).
        # `embedding_dim` is reported as `hidden_dim` for any introspection
        # that asks (e.g. CLAM-style state-dict probes), but it is not used
        # downstream because DSMILTrainer dispatches the slide-level logits
        # directly.
        self._embedding_dim = hidden_dim
        self.num_classes = num_classes

        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.i_classifier = _IClassifier(hidden_dim, num_classes)
        self.b_classifier = _BClassifier(hidden_dim, num_classes, dropout=dropout)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def forward(self, representation: torch.Tensor) -> dict:
        """representation: [N, input_dim] bag of patch features (single slide).

        Returns:
          dict with
            * `instance_logits`: [num_classes]  — critical instance per
              class is selected by `argmax`; for the L_max term the trainer
              uses the row corresponding to the per-class argmax. We expose
              the FULL `[N, C]` per-instance logit tensor so the trainer
              can replicate upstream's exact selection logic.
            * `bag_logits`: [num_classes]
            * `instance_logits_all`: [N, num_classes] — the raw per-instance
              per-class logits (for inspection / future extensions).
        """
        h = self.feature_proj(representation)              # [N, hidden]
        instance_logits_all = self.i_classifier(h)         # [N, num_classes]
        # Critical instance per class via per-class argmax over instances.
        crit_idx_per_class = instance_logits_all.argmax(dim=0)  # [num_classes]
        bag_logits = self.b_classifier(h, crit_idx_per_class)   # [num_classes]
        # For L_max upstream uses `max(instance_logits, dim=0)` which gives
        # the per-class max-pooled instance prediction; we return that as
        # `instance_logits`.
        instance_logits, _ = instance_logits_all.max(dim=0)     # [num_classes]
        return {
            "instance_logits": instance_logits,
            "bag_logits": bag_logits,
            "instance_logits_all": instance_logits_all,
        }
