from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from topofusion_v3.gnncrc_baselines.meanpool import MeanPool
from topofusion_v3.gnncrc_baselines.transmil import TransMIL
from topofusion_v3.gnncrc_baselines.clam import CLAM_SB, CLAM_MB
from topofusion_v3.gnncrc_baselines.dsmil import DSMIL
from topofusion_v3.gnncrc_baselines.patch_gcn import PatchGCN
from topofusion_v3.gnncrc_baselines.heat_pyg import HEATPyG
from topofusion_v3.gnncrc_baselines.hact_pyg import HACTPyG
from topofusion_v3.baselines.multitask_utils import MultiTaskHeads


class _BaseEmbedding(nn.Module):
    embedding_dim: int

    def forward_one(self, item) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError


class MeanPoolEmbedding(_BaseEmbedding):
    def __init__(self, input_dim: int = 1536, output_dim: int = 512):
        super().__init__()
        self.encoder = MeanPool(input_dim=input_dim, output_dim=output_dim)
        self.embedding_dim = int(self.encoder.embedding_dim)

    def forward_one(self, item: torch.Tensor) -> torch.Tensor:
        return self.encoder(item)




class ABMILEmbedding(_BaseEmbedding):
    def __init__(self, input_dim: int = 1536, embed_dim: int = 512, attn_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.embedding_dim = int(embed_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.attn_v = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Tanh())
        self.attn_u = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Sigmoid())
        self.attn_w = nn.Linear(attn_dim, 1)

    def forward_one(self, item: torch.Tensor) -> torch.Tensor:
        h = self.encoder(item)
        a = self.attn_w(self.attn_v(h) * self.attn_u(h)).squeeze(-1)
        attn = torch.softmax(a, dim=0)
        return torch.sum(h * attn.unsqueeze(-1), dim=0)


class TransMILEmbedding(_BaseEmbedding):
    def __init__(self, input_dim: int = 1536, hidden_dim: int = 512, heads: int = 8, output_dim: int = 512):
        super().__init__()
        self.encoder = TransMIL(input_dim=input_dim, hidden_dim=hidden_dim, heads=heads, output_dim=output_dim)
        self.embedding_dim = int(self.encoder.embedding_dim)

    def forward_one(self, item: torch.Tensor) -> torch.Tensor:
        return self.encoder(item)


class CLAMSBEmbedding(_BaseEmbedding):
    def __init__(self, input_dim: int = 1536, dropout: float = 0.25, size_arg: str = "small", n_classes: int = 4):
        super().__init__()
        self.encoder = CLAM_SB(gate=True, size_arg=size_arg, dropout=dropout, n_classes=n_classes, subtyping=False, embed_dim=input_dim)
        self.embedding_dim = self.encoder.size_dict[size_arg][1]

    def forward_one(self, item: torch.Tensor) -> torch.Tensor:
        _, _, _, _, results = self.encoder(item, instance_eval=False, return_features=True)
        return results["features"].squeeze(0)


class CLAMMBEmbedding(_BaseEmbedding):
    def __init__(self, input_dim: int = 1536, dropout: float = 0.25, size_arg: str = "small", n_classes: int = 4, pool: str = "flatten"):
        super().__init__()
        if pool not in {"flatten", "mean"}:
            raise ValueError("CLAM-MB pool must be flatten or mean")
        self.pool = pool
        self.n_classes = int(n_classes)
        self.encoder = CLAM_MB(gate=True, size_arg=size_arg, dropout=dropout, n_classes=n_classes, subtyping=True, embed_dim=input_dim)
        base_dim = self.encoder.size_dict[size_arg][1]
        self.embedding_dim = base_dim * self.n_classes if pool == "flatten" else base_dim

    def forward_one(self, item: torch.Tensor) -> torch.Tensor:
        _, _, _, _, results = self.encoder(item, instance_eval=False, return_features=True)
        features = results["features"]  # [K, D]
        if self.pool == "mean":
            return features.mean(dim=0)
        return features.reshape(-1)


class DSMILEmbedding(_BaseEmbedding):
    def __init__(self, input_dim: int = 1536, hidden_dim: int = 512, num_classes: int = 4, dropout: float = 0.25, pool: str = "flatten"):
        super().__init__()
        if pool not in {"flatten", "mean"}:
            raise ValueError("DSMIL pool must be flatten or mean")
        self.pool = pool
        self.num_classes = int(num_classes)
        self.encoder = DSMIL(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout)
        self.embedding_dim = hidden_dim * self.num_classes if pool == "flatten" else hidden_dim

    def forward_one(self, item: torch.Tensor) -> torch.Tensor:
        h = self.encoder.feature_proj(item)
        instance_logits_all = self.encoder.i_classifier(h)
        crit_idx_per_class = instance_logits_all.argmax(dim=0)
        critical = h[crit_idx_per_class]
        q = self.encoder.b_classifier.q(critical)
        v = self.encoder.b_classifier.v(h)
        attn = F.softmax((q @ v.t()) * self.encoder.b_classifier.scale, dim=1)
        bag_per_class = attn @ v  # [C, D]
        if self.pool == "mean":
            return bag_per_class.mean(dim=0)
        return bag_per_class.reshape(-1)


class PatchGCNEmbedding(_BaseEmbedding):
    def __init__(self, input_dim: int = 1536, num_layers: int = 4, hidden_dim: int = 128, dropout: float = 0.25, output_dim: int = 512):
        super().__init__()
        self.encoder = PatchGCN(input_dim=input_dim, num_layers=num_layers, hidden_dim=hidden_dim, edge_agg="spatial", dropout=dropout, output_dim=output_dim)
        self.embedding_dim = int(self.encoder.embedding_dim)

    def forward_one(self, item) -> torch.Tensor:
        return self.encoder(item)


class HEATPyGEmbedding(_BaseEmbedding):
    def __init__(self, input_dim: int = 1536, hidden_dim: int = 256, n_layers: int = 2, n_heads: int = 8, dropout: float = 0.25, output_dim: int = 512):
        super().__init__()
        self.encoder = HEATPyG(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_layers=n_layers, n_heads=n_heads, dropout=dropout)
        self.embedding_dim = int(self.encoder.embedding_dim)

    def forward_one(self, item) -> torch.Tensor:
        return self.encoder(item)




class HACTPyGEmbedding(_BaseEmbedding):
    def __init__(self, input_dim: int = 1536, hidden_dim: int = 256, n_layers: int = 2, n_heads: int = 8, dropout: float = 0.25, output_dim: int = 512):
        super().__init__()
        self.encoder = HACTPyG(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_layers=n_layers, n_heads=n_heads, dropout=dropout)
        self.embedding_dim = int(self.encoder.embedding_dim)

    def forward_one(self, item) -> torch.Tensor:
        return self.encoder(item)


class MultiTaskBaselineModel(nn.Module):
    def __init__(self, encoder: _BaseEmbedding, projection_dim: int = 0, head_dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.heads = MultiTaskHeads(encoder.embedding_dim, projection_dim=projection_dim, dropout=head_dropout)
        self.embedding_dim = encoder.embedding_dim

    def forward_embedding(self, items: Sequence, device: torch.device) -> torch.Tensor:
        embeds = []
        for item in items:
            if torch.is_tensor(item):
                obj = item.to(device, non_blocking=True)
            else:
                obj = item.to(device)
            embeds.append(self.encoder.forward_one(obj))
        return torch.stack(embeds, dim=0)

    def forward(self, items: Sequence, device: torch.device) -> dict[str, torch.Tensor]:
        return self.heads(self.forward_embedding(items, device))


def make_multitask_baseline(
    model_name: str,
    input_dim: int = 1536,
    embed_dim: int = 512,
    hidden_dim: int = 512,
    attn_heads: int = 8,
    dropout: float = 0.25,
    head_dropout: float = 0.1,
    projection_dim: int = 0,
    clam_mb_pool: str = "flatten",
    dsmil_pool: str = "flatten",
    patch_gcn_hidden_dim: int = 128,
    patch_gcn_layers: int = 4,
    heat_hidden_dim: int = 256,
    heat_layers: int = 2,
    hact_hidden_dim: int = 256,
    hact_layers: int = 2,
) -> MultiTaskBaselineModel:
    name = model_name.lower().replace("-", "_")
    if name == "meanpool":
        encoder = MeanPoolEmbedding(input_dim=input_dim, output_dim=embed_dim)
    elif name == "abmil":
        encoder = ABMILEmbedding(input_dim=input_dim, embed_dim=embed_dim, dropout=dropout)
    elif name == "transmil":
        encoder = TransMILEmbedding(input_dim=input_dim, hidden_dim=hidden_dim, heads=attn_heads, output_dim=embed_dim)
    elif name == "clam_sb":
        encoder = CLAMSBEmbedding(input_dim=input_dim, dropout=dropout, n_classes=4)
    elif name == "clam_mb":
        encoder = CLAMMBEmbedding(input_dim=input_dim, dropout=dropout, n_classes=4, pool=clam_mb_pool)
    elif name == "dsmil":
        encoder = DSMILEmbedding(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=4, dropout=dropout, pool=dsmil_pool)
    elif name == "patch_gcn":
        encoder = PatchGCNEmbedding(input_dim=input_dim, num_layers=patch_gcn_layers, hidden_dim=patch_gcn_hidden_dim, dropout=dropout, output_dim=embed_dim)
    elif name == "heat_pyg":
        encoder = HEATPyGEmbedding(input_dim=input_dim, hidden_dim=heat_hidden_dim, n_layers=heat_layers, n_heads=attn_heads, dropout=dropout, output_dim=embed_dim)
    elif name == "hact_pyg":
        encoder = HACTPyGEmbedding(input_dim=input_dim, hidden_dim=hact_hidden_dim, n_layers=hact_layers, n_heads=attn_heads, dropout=dropout, output_dim=embed_dim)
    else:
        raise ValueError(f"Unknown multitask baseline: {model_name}")
    return MultiTaskBaselineModel(encoder, projection_dim=projection_dim, head_dropout=head_dropout)


def model_uses_graph(model_name: str) -> bool:
    return model_name.lower().replace("-", "_") in {"patch_gcn", "heat_pyg", "hact_pyg"}
