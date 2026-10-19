"""PyG-compatible HEAT-style heterogeneous graph encoder.

This adapter ports the usable HEAT idea from gnn-crc/WSI-HGNN into the BIBM
pipeline without DGL. The original gnn-crc HEAT implementation expects DGL
heterographs built from CellViT cells.json. The BIBM environment currently has
PyG graph caches but no DGL, so this module uses the existing PyG graph fields:

  - patch features: data.patch_feats or data.x
  - patch cell type counts: data.cell_counts, used to assign dominant node type
  - edge_index and optional edge_attr

It preserves the HEAT abstraction that nodes are patch-level and typed by the
dominant cell class, then applies relation-specific attention by source/target
node type plus edge sign.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.utils import softmax as pyg_softmax
except Exception:  # pragma: no cover - fallback for minimal env checks
    pyg_softmax = None


def _scatter_softmax(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Softmax over edges grouped by destination node."""
    if src.numel() == 0:
        return src
    if pyg_softmax is not None:
        return pyg_softmax(src, index, num_nodes=dim_size)
    out = torch.empty_like(src)
    for node in torch.unique(index):
        mask = index == node
        out[mask] = torch.softmax(src[mask], dim=0)
    return out


class HEATPyGLayer(nn.Module):
    def __init__(self, hidden_dim: int = 256, n_heads: int = 8, n_node_types: int = 6, dropout: float = 0.2):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads")
        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(n_heads)
        self.d_k = self.hidden_dim // self.n_heads
        self.n_node_types = int(n_node_types)
        self.scale = math.sqrt(float(self.d_k))
        self.k_linears = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(n_node_types))
        self.q_linears = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(n_node_types))
        self.v_linears = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(n_node_types))
        self.out_linears = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(n_node_types))
        self.edge_gate = nn.Linear(1, 1)
        self.skip = nn.Parameter(torch.ones(n_node_types))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, node_type: torch.Tensor, edge_sim: torch.Tensor | None = None) -> torch.Tensor:
        if edge_index.numel() == 0:
            return x
        src, dst = edge_index[0].long(), edge_index[1].long()
        node_type = node_type.clamp(0, self.n_node_types - 1).long()
        if edge_sim is None:
            edge_sim = x.new_ones(src.numel())
        edge_sim = edge_sim.float().view(-1, 1)
        out = torch.zeros_like(x)
        counts = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)

        for st in range(self.n_node_types):
            st_mask = node_type[src] == st
            if not st_mask.any():
                continue
            src_st = src[st_mask]
            dst_st = dst[st_mask]
            q_all = self.q_linears[st](x).view(-1, self.n_heads, self.d_k)
            k = self.k_linears[st](x[src_st]).view(-1, self.n_heads, self.d_k)
            v = self.v_linears[st](x[src_st]).view(-1, self.n_heads, self.d_k)
            q = q_all[dst_st]
            gate = torch.tanh(self.edge_gate(edge_sim[st_mask])).view(-1, 1)
            score = ((q * k).sum(dim=-1) / self.scale) * gate
            alpha = torch.stack([_scatter_softmax(score[:, h], dst_st, x.size(0)) for h in range(self.n_heads)], dim=1)
            msg = (alpha.unsqueeze(-1) * v).reshape(-1, self.hidden_dim)
            out.index_add_(0, dst_st, msg)
            unique_dst = torch.unique(dst_st)
            counts.index_add_(0, unique_dst, torch.ones(unique_dst.numel(), 1, device=x.device, dtype=x.dtype))

        out = out / counts.clamp_min(1.0)
        final = x.clone()
        for dt in range(self.n_node_types):
            mask = node_type == dt
            if not mask.any():
                continue
            trans = self.dropout(self.out_linears[dt](out[mask]))
            alpha = torch.sigmoid(self.skip[dt])
            final[mask] = alpha * trans + (1.0 - alpha) * x[mask]
        return final


class HEATPyG(nn.Module):
    """Slide-level HEAT-style encoder for BIBM PyG graph caches."""

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 256,
        output_dim: int = 512,
        n_layers: int = 2,
        n_heads: int = 8,
        n_node_types: int = 6,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_node_types = int(n_node_types)
        self.embedding_dim = int(output_dim)
        self.input_proj = nn.ModuleList(
            nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True))
            for _ in range(self.n_node_types)
        )
        self.layers = nn.ModuleList(
            HEATPyGLayer(hidden_dim=hidden_dim, n_heads=n_heads, n_node_types=n_node_types, dropout=dropout)
            for _ in range(n_layers)
        )
        self.type_proj = nn.ModuleList(nn.Linear(hidden_dim, 256) for _ in range(self.n_node_types))
        self.type_attn = nn.ModuleList(nn.Linear(256, 1, bias=False) for _ in range(self.n_node_types))
        self.head = nn.Sequential(
            nn.LayerNorm(256 * self.n_node_types),
            nn.Linear(256 * self.n_node_types, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
        )

    @staticmethod
    def _features(data) -> torch.Tensor:
        if hasattr(data, "patch_feats"):
            return data.patch_feats.float()
        if hasattr(data, "x"):
            return data.x.float()
        raise RuntimeError("HEATPyG needs data.patch_feats or data.x")

    def _node_type(self, data, n: int, device: torch.device) -> torch.Tensor:
        if hasattr(data, "cell_counts") and data.cell_counts is not None:
            counts = data.cell_counts.float().to(device)
            if counts.numel() and counts.shape[1] >= 5:
                # CellViT counts are 5 non-background classes:
                # Neoplastic, Inflammatory, Connective, Dead, Epithelial.
                # Shift by +1 to match HEAT's 0=Background, 1..5=cell types.
                has_cell = counts.sum(dim=1) > 0
                out = torch.zeros(counts.shape[0], dtype=torch.long, device=device)
                out[has_cell] = counts[has_cell].argmax(dim=1).long() + 1
                return out.clamp(0, self.n_node_types - 1)
        return torch.zeros(n, dtype=torch.long, device=device)

    @staticmethod
    def _edge_sim(data, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge_attr = None
        if hasattr(data, "edge_attr_raw") and data.edge_attr_raw is not None and data.edge_attr_raw.numel() > 0:
            edge_attr = data.edge_attr_raw
        elif hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.numel() > 0:
            edge_attr = data.edge_attr
        if edge_attr is not None:
            ea = edge_attr.float().to(x.device)
            if ea.ndim == 2 and ea.shape[1] >= 3:
                # Use distance-normalized sign proxy: closer edges get larger positive weight.
                return (1.0 - ea[:, 2].clamp(0, 1)).contiguous()
        src, dst = edge_index[0], edge_index[1]
        xs = F.normalize(x[src].float(), dim=1)
        xd = F.normalize(x[dst].float(), dim=1)
        return (xs * xd).sum(dim=1).clamp(-1, 1)

    def forward(self, data) -> torch.Tensor:
        x_raw = self._features(data)
        device = x_raw.device
        node_type = self._node_type(data, x_raw.shape[0], device)
        x = torch.empty(x_raw.shape[0], self.input_proj[0][0].out_features, device=device, dtype=x_raw.dtype)
        for t in range(self.n_node_types):
            mask = node_type == t
            if mask.any():
                x[mask] = self.input_proj[t](x_raw[mask])
        edge_index = data.edge_index.long().to(device)
        edge_sim = self._edge_sim(data, x_raw, edge_index)
        for layer in self.layers:
            x = layer(x, edge_index, node_type, edge_sim=edge_sim)

        global_context = torch.zeros(1, 256, device=device, dtype=x.dtype)
        per_type: Dict[int, torch.Tensor] = {}
        for t in range(self.n_node_types):
            mask = node_type == t
            if mask.any():
                h_t = self.type_proj[t](x[mask]).mean(dim=0, keepdim=True)
            else:
                h_t = torch.zeros(1, 256, device=device, dtype=x.dtype)
            per_type[t] = h_t
            global_context = global_context + h_t

        pooled = []
        for t in range(self.n_node_types):
            h_t = per_type[t]
            weight = torch.sigmoid(self.type_attn[t](h_t + global_context))
            pooled.append(weight * h_t)
        return self.head(torch.cat(pooled, dim=1)).squeeze(0)
