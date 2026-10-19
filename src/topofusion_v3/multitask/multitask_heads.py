from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from topofusion_v3.model import CMSGraphModel


class MultiTaskHeads(nn.Module):
    """Independent prediction heads attached to a shared slide embedding."""

    def __init__(self, embed_dim: int, projection_dim: int = 0):
        super().__init__()
        self.embed_dim = int(embed_dim)
        if projection_dim and int(projection_dim) > 0:
            self.shared_projection = nn.Sequential(
                nn.Linear(self.embed_dim, int(projection_dim)),
                nn.LayerNorm(int(projection_dim)),
                nn.ReLU(inplace=True),
            )
            head_dim = int(projection_dim)
        else:
            self.shared_projection = nn.Identity()
            head_dim = self.embed_dim

        self.cms = nn.Linear(head_dim, 4)
        self.msi = nn.Linear(head_dim, 2)
        self.pn = nn.Linear(head_dim, 2)
        self.os = nn.Linear(head_dim, 1)
        self.dfs = nn.Linear(head_dim, 1)
        self.cin = nn.Linear(head_dim, 2)

    def forward(self, slide_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.shared_projection(slide_embedding)
        return {
            "cms": self.cms(z),
            "msi": self.msi(z),
            "pn": self.pn(z),
            "os": self.os(z),
            "dfs": self.dfs(z),
            "cin": self.cin(z),
        }


class DecoupledMultiTaskHeads(nn.Module):
    """Separate classification and survival heads for decoupled embeddings."""

    def __init__(self, cls_dim: int, surv_dim: int):
        super().__init__()
        self.cls_dim = int(cls_dim)
        self.surv_dim = int(surv_dim)
        self.cms = nn.Linear(self.cls_dim, 4)
        self.msi = nn.Linear(self.cls_dim, 2)
        self.pn = nn.Linear(self.cls_dim, 2)
        self.cin = nn.Linear(self.cls_dim, 2)
        self.os = nn.Linear(self.surv_dim, 1)
        self.dfs = nn.Linear(self.surv_dim, 1)

    def forward(self, embeddings: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        z_cls = embeddings["cls"]
        z_surv = embeddings["surv"]
        return {
            "cms": self.cms(z_cls),
            "msi": self.msi(z_cls),
            "pn": self.pn(z_cls),
            "cin": self.cin(z_cls),
            "os": self.os(z_surv),
            "dfs": self.dfs(z_surv),
        }


class MultiTaskCMSGraphModel(CMSGraphModel):
    """TopoFusion v3 backbone plus multi-task heads.

    The inherited modules are exactly the CMSGraphModel backbone. The only new
    computation is the task head block after the slide-level embedding.
    """

    def __init__(self, cfg: Dict[str, object], projection_dim: int = 0):
        super().__init__(cfg)
        self.task_head_mode = str(cfg.get("task_head_mode", "shared"))
        if self.task_head_mode.startswith("decoupled"):
            cls_dim = int(cfg.get("cls_embed_dim", 128 + self.gnn_out_dim))
            surv_dim = int(cfg.get("surv_embed_dim", self.gnn_out_dim * 3))
            self.slide_embed_dims = {"cls": cls_dim, "surv": surv_dim}
            self.slide_embed_dim = cls_dim + surv_dim
            self.heads = DecoupledMultiTaskHeads(cls_dim, surv_dim)
        else:
            embed_dim = int(self.classifier.in_features)
            self.slide_embed_dims = {"shared": embed_dim}
            self.slide_embed_dim = embed_dim
            self.heads = MultiTaskHeads(embed_dim, projection_dim=projection_dim)

    def forward_embedding(self, data):
        if self.task_head_mode.startswith("decoupled"):
            return self.compute_decoupled_embeddings(data)
        return self.compute_slide_embedding(data)

    def forward(self, data) -> Dict[str, torch.Tensor]:
        return self.heads(self.forward_embedding(data))
