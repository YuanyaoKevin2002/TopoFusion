"""PyG-compatible HACT-style hierarchical graph encoder.

The original HACT-Net uses a cell graph, a tissue graph, and an assignment
matrix from cells to tissue regions. The BIBM formal runs use cached PyG patch
graphs rather than HACT's BRACS graph files. This adapter preserves the same
hierarchical idea on the available representation:

  - patch/tissue branch: UNI patch feature + tissue probability identity
  - cell branch: CellViT per-patch counts + topology features
  - assignment: every patch owns its aggregated cell descriptor, then a learned
    gate injects the cell descriptor into the tissue node
  - tissue graph branch: message passing over the existing KNN patch graph
  - readout: mean, tissue-attention, and cell-attention pooled slide embedding

This keeps the comparison fair: it uses exactly the same graph cache as Patch-GCN,
HEATPyG, and TopoFusion baselines, without introducing a separate BRACS/HACT
preprocessing pipeline.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.utils import softmax as pyg_softmax
except Exception:  # pragma: no cover
    pyg_softmax = None


def _group_softmax(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    if src.numel() == 0:
        return src
    if pyg_softmax is not None:
        return pyg_softmax(src, index, num_nodes=dim_size)
    out = torch.empty_like(src)
    for node in torch.unique(index):
        mask = index == node
        out[mask] = torch.softmax(src[mask], dim=0)
    return out


def _pad_or_trim(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.shape[1] == dim:
        return x
    if x.shape[1] > dim:
        return x[:, :dim]
    pad = x.new_zeros(x.shape[0], dim - x.shape[1])
    return torch.cat([x, pad], dim=1)


class HACTGraphLayer(nn.Module):
    def __init__(self, hidden_dim: int = 256, n_heads: int = 8, dropout: float = 0.2):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads")
        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(n_heads)
        self.head_dim = self.hidden_dim // self.n_heads
        self.scale = math.sqrt(float(self.head_dim))
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.edge_bias = nn.Linear(1, n_heads)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_signal: torch.Tensor | None = None) -> torch.Tensor:
        if edge_index.numel() == 0:
            y = self.norm1(x)
            return self.norm2(y + self.dropout(self.ffn(y)))
        src = edge_index[0].long()
        dst = edge_index[1].long()
        q = self.q(x).view(-1, self.n_heads, self.head_dim)
        k = self.k(x).view(-1, self.n_heads, self.head_dim)
        v = self.v(x).view(-1, self.n_heads, self.head_dim)
        score = (q[dst] * k[src]).sum(dim=-1) / self.scale
        if edge_signal is not None and edge_signal.numel() == src.numel():
            score = score + self.edge_bias(edge_signal.float().view(-1, 1))
        alpha = torch.stack([_group_softmax(score[:, h], dst, x.size(0)) for h in range(self.n_heads)], dim=1)
        msg = (alpha.unsqueeze(-1) * v[src]).reshape(-1, self.hidden_dim)
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msg)
        x = self.norm1(x + self.dropout(self.out(agg)))
        return self.norm2(x + self.dropout(self.ffn(x)))


class HACTPyG(nn.Module):
    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 256,
        output_dim: int = 512,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.embedding_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.patch_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.tissue_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.cell_encoder = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.assign_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 4, hidden_dim),
            nn.Sigmoid(),
        )
        self.fuse_norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList(HACTGraphLayer(hidden_dim, n_heads=n_heads, dropout=dropout) for _ in range(n_layers))
        self.tissue_pool = nn.Sequential(nn.Linear(hidden_dim + 4, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1))
        self.cell_pool = nn.Sequential(nn.Linear(hidden_dim + 5, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    @staticmethod
    def _features(data) -> torch.Tensor:
        if hasattr(data, "patch_feats") and data.patch_feats is not None:
            return data.patch_feats.float()
        if hasattr(data, "x") and data.x is not None:
            return data.x.float()
        if hasattr(data, "features") and data.features is not None:
            return data.features.float()
        raise RuntimeError("HACTPyG needs data.patch_feats, data.x, or data.features")

    @staticmethod
    def _tissue_probs(data, n: int, device: torch.device) -> torch.Tensor:
        if hasattr(data, "tissue_probs_4") and data.tissue_probs_4 is not None:
            return _pad_or_trim(data.tissue_probs_4.float().to(device), 4)
        if hasattr(data, "tissue_probs") and data.tissue_probs is not None:
            return _pad_or_trim(data.tissue_probs.float().to(device), 4)
        return torch.zeros(n, 4, device=device)

    @staticmethod
    def _cell_input(data, n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(data, "cell_counts") and data.cell_counts is not None:
            counts = _pad_or_trim(data.cell_counts.float().to(device), 5)
        else:
            counts = torch.zeros(n, 5, device=device)
        if hasattr(data, "cell_topo") and data.cell_topo is not None:
            topo = _pad_or_trim(data.cell_topo.float().to(device), 7)
        else:
            topo = torch.zeros(n, 7, device=device)
        return torch.cat([counts, topo], dim=1), counts

    @staticmethod
    def _edge_signal(data, x_raw: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge_attr = None
        if hasattr(data, "edge_attr_raw") and data.edge_attr_raw is not None and data.edge_attr_raw.numel() > 0:
            edge_attr = data.edge_attr_raw
        elif hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.numel() > 0:
            edge_attr = data.edge_attr
        if edge_attr is not None:
            ea = edge_attr.float().to(x_raw.device)
            if ea.ndim == 2 and ea.shape[1] >= 3:
                return (1.0 - ea[:, 2].clamp(0, 1)).contiguous()
        src, dst = edge_index[0], edge_index[1]
        return (F.normalize(x_raw[src], dim=1) * F.normalize(x_raw[dst], dim=1)).sum(dim=1).clamp(-1, 1)

    @staticmethod
    def _weighted_pool(x: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        if score.numel() == 0:
            return x.mean(dim=0, keepdim=True)
        w = torch.softmax(score.view(-1), dim=0)
        return (x * w.unsqueeze(-1)).sum(dim=0, keepdim=True)

    def forward(self, data) -> torch.Tensor:
        x_raw = self._features(data)
        device = x_raw.device
        n = x_raw.shape[0]
        tissue = self._tissue_probs(data, n, device)
        cell_in, cell_counts = self._cell_input(data, n, device)
        patch_h = self.patch_encoder(x_raw)
        tissue_h = self.tissue_encoder(tissue)
        cell_h = self.cell_encoder(cell_in)
        gate = self.assign_gate(torch.cat([patch_h + tissue_h, cell_h, tissue], dim=1))
        h = self.fuse_norm(patch_h + tissue_h + gate * cell_h)
        edge_index = data.edge_index.long().to(device)
        edge_signal = self._edge_signal(data, x_raw, edge_index)
        for layer in self.layers:
            h = layer(h, edge_index, edge_signal=edge_signal)
        mean_pool = h.mean(dim=0, keepdim=True)
        tissue_score = self.tissue_pool(torch.cat([h, tissue], dim=1)).squeeze(-1)
        cell_score = self.cell_pool(torch.cat([h, cell_counts], dim=1)).squeeze(-1)
        tissue_attn = self._weighted_pool(h, tissue_score)
        cell_attn = self._weighted_pool(h, cell_score)
        return self.head(torch.cat([mean_pool, tissue_attn, cell_attn], dim=1)).squeeze(0)

