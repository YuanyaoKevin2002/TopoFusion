from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.utils import softmax, subgraph, to_undirected

from topofusion_v3.constants import LYM4, STR4, TUM4


class PatchEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 1536,
        hidden_dim: int = 512,
        out_dim: int = 128,
        dropout1: float = 0.2,
        dropout2: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout1)),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout2)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SinusoidalCoordEncoding(nn.Module):
    def __init__(self, dim: int = 32, base_scale: float = 1000.0):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError("2D sinusoidal coord encoding dim must be divisible by 4")
        self.dim = int(dim)
        self.base_scale = float(base_scale)
        n_freq = dim // 4
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(10000.0), n_freq))
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        c = coords / self.base_scale
        x = c[:, 0:1] / self.freqs.view(1, -1)
        y = c[:, 1:2] / self.freqs.view(1, -1)
        return torch.cat([torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)], dim=1)


class CMSGraphModel(nn.Module):
    def __init__(self, cfg: Dict[str, object]):
        super().__init__()
        patch_in = int(cfg.get("patch_in_dim", 1536))
        patch_hidden = int(cfg.get("patch_hidden_dim", 512))
        fused_dim = int(cfg.get("fused_dim", 128))
        cell_in = int(cfg.get("cell_in_dim", 12))
        pos_dim = int(cfg.get("pos_dim", 32))
        edge_raw_dim = int(cfg.get("edge_raw_dim", 6))
        edge_dim = int(cfg.get("edge_dim", 16))
        heads = int(cfg.get("gat_heads", 4))
        gat_hidden = int(cfg.get("gat_hidden_dim", 128))

        self.patch_encoder = PatchEncoder(
            patch_in,
            patch_hidden,
            fused_dim,
            dropout1=float(cfg.get("patch_dropout1", 0.2)),
            dropout2=float(cfg.get("patch_dropout2", 0.1)),
        )
        self.visual_attn = nn.Linear(fused_dim, 1)
        self.cell_encoder = nn.Sequential(
            nn.Linear(cell_in, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(inplace=True),
        )
        self.gate_patch = nn.Linear(fused_dim, fused_dim)
        self.gate_cell = nn.Linear(fused_dim, fused_dim)
        self.gate_interact = nn.Linear(fused_dim, fused_dim)
        self.gate_norm = nn.LayerNorm(fused_dim)
        self.pos_encoder = SinusoidalCoordEncoding(pos_dim)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_raw_dim, edge_dim),
            nn.ReLU(inplace=True),
            nn.Linear(edge_dim, edge_dim),
            nn.LayerNorm(edge_dim),
        )
        node_dim = fused_dim + pos_dim + 4
        self.conv1 = GATv2Conv(node_dim, gat_hidden, heads=heads, concat=True, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(gat_hidden * heads, gat_hidden, heads=heads, concat=True, edge_dim=edge_dim)
        self.gnn_out_dim = gat_hidden * heads
        self.norm = nn.LayerNorm(self.gnn_out_dim)
        self.slide_pool_include_raw = bool(cfg.get("slide_pool_include_raw", False))
        self.slide_pool_include_interface = bool(cfg.get("slide_pool_include_interface", True))
        self.slide_pool_include_boundary = bool(cfg.get("slide_pool_include_boundary", False))
        self.slide_pool_debug = bool(cfg.get("slide_pool_debug", False))
        self.zero_cell_descriptors = bool(cfg.get("zero_cell_descriptors", False))
        self.zero_edge_tissue_interactions = bool(cfg.get("zero_edge_tissue_interactions", False))
        self.slide_embed_dim = self.gnn_out_dim
        if self.slide_pool_include_raw:
            self.slide_embed_dim += fused_dim
        if self.slide_pool_include_interface:
            self.slide_embed_dim += self.gnn_out_dim * 2
        if self.slide_pool_include_boundary:
            self.slide_embed_dim += self.gnn_out_dim
        self.classifier = nn.Linear(self.slide_embed_dim, 4)
        self._last_embedding_debug: Dict[str, object] = {}
        self.cell_only_prob = float(cfg.get("modality_cell_only_prob", 0.4))
        self.patch_only_prob = float(cfg.get("modality_patch_only_prob", 0.1))
        self.interface_temperature = float(cfg.get("interface_temperature", 1.0))
        self.dropout_in = float(cfg.get("gnn_dropout_in", 0.2))
        self.dropout_mid = float(cfg.get("gnn_dropout_mid", 0.1))
        self.dropedge = float(cfg.get("dropedge", 0.0))
        self.node_dropout = float(cfg.get("node_dropout", 0.0))
        self.patch_feature_noise_std = float(cfg.get("patch_feature_noise_std", 0.0))
        self.decoupled_raw_pool_mode = str(cfg.get("decoupled_raw_pool_mode", "attention")).lower()
        self.decoupled_surv_include_raw = bool(cfg.get("decoupled_surv_include_raw", False))

    def encode_modalities(
        self,
        patch_feats: torch.Tensor,
        cell_counts: torch.Tensor,
        cell_topo: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.training and self.patch_feature_noise_std > 0.0:
            patch_feats = patch_feats + torch.randn_like(patch_feats) * self.patch_feature_noise_std
        patch_z = self.patch_encoder(patch_feats)
        cell_z = self.cell_encoder(torch.cat([cell_counts, cell_topo], dim=1))
        if self.training:
            r = torch.rand((), device=patch_feats.device).item()
            if r < self.cell_only_prob:
                return self.gate_norm(cell_z), patch_z, cell_z
            if r < self.cell_only_prob + self.patch_only_prob:
                return self.gate_norm(patch_z), patch_z, cell_z
        interact = self.gate_interact(patch_z * cell_z)
        gate = torch.sigmoid(self.gate_patch(patch_z) + self.gate_cell(cell_z) + interact)
        fused_z = self.gate_norm(gate * patch_z + (1.0 - gate) * cell_z)
        return fused_z, patch_z, cell_z

    def fuse_modalities(self, patch_feats: torch.Tensor, cell_counts: torch.Tensor, cell_topo: torch.Tensor) -> torch.Tensor:
        return self.encode_modalities(patch_feats, cell_counts, cell_topo)[0]

    @staticmethod
    def _neighbor_mean(tissue: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = tissue.size(0)
        out = torch.zeros_like(tissue)
        deg = torch.zeros((n, 1), dtype=tissue.dtype, device=tissue.device)
        if edge_index.numel() == 0:
            return tissue
        src, dst = edge_index[0], edge_index[1]
        out.index_add_(0, src, tissue[dst])
        deg.index_add_(0, src, torch.ones((dst.numel(), 1), dtype=tissue.dtype, device=tissue.device))
        return out / deg.clamp_min(1.0)

    def _interface_pool(
        self,
        x: torch.Tensor,
        tissue: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        neigh = self._neighbor_mean(tissue, edge_index)
        if mode == "tl":
            w = tissue[:, TUM4] * neigh[:, LYM4] + tissue[:, LYM4] * neigh[:, TUM4]
        elif mode == "ts":
            w = tissue[:, TUM4] * neigh[:, STR4] + tissue[:, STR4] * neigh[:, TUM4]
        else:
            raise ValueError(mode)
        alpha = softmax(w / max(self.interface_temperature, 1e-6), batch)
        pooled = torch.zeros((int(batch.max().item()) + 1, x.size(1)), dtype=x.dtype, device=x.device)
        pooled.index_add_(0, batch, x * alpha.unsqueeze(1))
        return pooled

    def _visual_attention_pool(self, patch_z: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_logits = self.visual_attn(patch_z).squeeze(-1)
        attn = softmax(attn_logits, batch)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        pooled = torch.zeros((num_graphs, patch_z.size(1)), dtype=patch_z.dtype, device=patch_z.device)
        pooled.index_add_(0, batch, patch_z * attn.unsqueeze(1))
        return pooled, attn

    @staticmethod
    def _weighted_sum_pool(x: torch.Tensor, weight: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        pooled = torch.zeros((num_graphs, x.size(1)), dtype=x.dtype, device=x.device)
        denom = torch.zeros((num_graphs, 1), dtype=x.dtype, device=x.device)
        w = weight.clamp_min(0.0).unsqueeze(1)
        pooled.index_add_(0, batch, x * w)
        denom.index_add_(0, batch, w)
        return pooled / denom.clamp_min(1e-8)

    @staticmethod
    def _tumor_boundary_score(tissue: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = int(tissue.size(0))
        boundary_score = torch.zeros(n, dtype=tissue.dtype, device=tissue.device)
        if n == 0 or edge_index.numel() == 0:
            return boundary_score
        dominant = tissue.argmax(dim=1)
        is_tumor = dominant == TUM4
        undirected_edge_index = to_undirected(edge_index, num_nodes=n)
        if undirected_edge_index.numel() == 0:
            return boundary_score
        src, dst = undirected_edge_index[0], undirected_edge_index[1]
        same_class = (dominant[src] == dominant[dst]).to(tissue.dtype)
        deg = torch.zeros(n, dtype=tissue.dtype, device=tissue.device)
        same_sum = torch.zeros(n, dtype=tissue.dtype, device=tissue.device)
        deg.index_add_(0, src, torch.ones_like(same_class))
        same_sum.index_add_(0, src, same_class)
        neighbor_ratio = same_sum / deg.clamp_min(1e-8)
        boundary_score[is_tumor] = 1.0 - neighbor_ratio[is_tumor]
        return boundary_score

    @staticmethod
    def _boundary_debug(boundary_score: torch.Tensor, tissue: torch.Tensor) -> Dict[str, object]:
        dominant = tissue.argmax(dim=1) if tissue.numel() else torch.empty(0, dtype=torch.long, device=tissue.device)
        is_tumor = dominant == TUM4
        positive = boundary_score[boundary_score > 0]
        non_tumor_nonzero = ((~is_tumor) & (boundary_score > 0)).sum().item() if boundary_score.numel() else 0
        out: Dict[str, object] = {
            "enabled": 1,
            "n_nodes": int(boundary_score.numel()),
            "tumor_node_fraction": float(is_tumor.float().mean().item()) if is_tumor.numel() else 0.0,
            "nonzero_fraction": float((boundary_score > 0).float().mean().item()) if boundary_score.numel() else 0.0,
            "non_tumor_nonzero_count": int(non_tumor_nonzero),
            "min": float(boundary_score.min().item()) if boundary_score.numel() else 0.0,
            "max": float(boundary_score.max().item()) if boundary_score.numel() else 0.0,
            "mean": float(boundary_score.mean().item()) if boundary_score.numel() else 0.0,
            "positive_count": int(positive.numel()),
        }
        if positive.numel():
            out.update({
                "positive_min": float(positive.min().item()),
                "positive_max": float(positive.max().item()),
                "positive_mean": float(positive.mean().item()),
            })
        return out

    def _maybe_drop_edges(self, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.dropedge <= 0.0 or edge_index.numel() == 0:
            return edge_index, edge_attr
        keep = torch.rand(edge_index.size(1), device=edge_index.device) >= self.dropedge
        if not keep.any():
            keep[torch.randint(0, edge_index.size(1), (1,), device=edge_index.device)] = True
        return edge_index[:, keep], edge_attr[keep]

    @staticmethod
    def _batch_vector(data, n_nodes: int, device: torch.device) -> torch.Tensor:
        if hasattr(data, "batch") and data.batch is not None:
            return data.batch.to(device)
        return torch.zeros(n_nodes, dtype=torch.long, device=device)

    def _node_dropout_keep(self, batch: torch.Tensor) -> torch.Tensor:
        n = int(batch.numel())
        if not self.training or self.node_dropout <= 0.0 or n <= 1:
            return torch.ones(n, dtype=torch.bool, device=batch.device)
        keep = torch.rand(n, device=batch.device) >= self.node_dropout
        for graph_id in torch.unique(batch, sorted=True):
            idx = torch.nonzero(batch == graph_id, as_tuple=False).view(-1)
            if idx.numel() and not keep[idx].any():
                keep[idx[torch.randint(0, idx.numel(), (1,), device=batch.device)]] = True
        return keep

    def _graph_inputs(self, data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        patch_feats = data.patch_feats
        coords = data.coords
        tissue = data.tissue_probs4
        cell_counts = data.cell_counts
        cell_topo = data.cell_topo
        edge_index = data.edge_index
        edge_attr_raw = data.edge_attr_raw
        if self.zero_cell_descriptors:
            cell_counts = torch.zeros_like(cell_counts)
            cell_topo = torch.zeros_like(cell_topo)
        if self.zero_edge_tissue_interactions and edge_attr_raw.numel() and edge_attr_raw.size(1) >= 6:
            edge_attr_raw = edge_attr_raw.clone()
            edge_attr_raw[:, 3:6] = 0.0
        batch = self._batch_vector(data, patch_feats.size(0), patch_feats.device)
        keep = self._node_dropout_keep(batch)
        if keep.all():
            return patch_feats, coords, tissue, cell_counts, cell_topo, edge_index, edge_attr_raw, batch
        edge_index, edge_attr_raw = subgraph(
            keep,
            edge_index,
            edge_attr=edge_attr_raw,
            relabel_nodes=True,
            num_nodes=int(keep.numel()),
        )
        return patch_feats[keep], coords[keep], tissue[keep], cell_counts[keep], cell_topo[keep], edge_index, edge_attr_raw, batch[keep]

    def compute_slide_embedding(self, data) -> torch.Tensor:
        patch_feats, coords, tissue, cell_counts, cell_topo, edge_index, edge_attr_raw, batch = self._graph_inputs(data)
        fz, patch_z, _cell_z = self.encode_modalities(patch_feats, cell_counts, cell_topo)
        pos = self.pos_encoder(coords)
        x = torch.cat([fz, pos, tissue], dim=1)
        edge_index_for_pool = edge_index
        boundary_score = self._tumor_boundary_score(tissue, edge_index_for_pool) if self.slide_pool_include_boundary else None
        edge_attr = self.edge_encoder(edge_attr_raw)
        edge_index, edge_attr = self._maybe_drop_edges(edge_index, edge_attr)
        x = F.dropout(x, p=self.dropout_in, training=self.training)
        x1 = self.conv1(x, edge_index, edge_attr)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=self.dropout_mid, training=self.training)
        x2 = self.conv2(x1, edge_index, edge_attr)
        x2 = F.elu(x2)
        x_out = self.norm(x1 + x2)
        h_parts = [global_mean_pool(x_out, batch)]
        if self.slide_pool_include_raw:
            h_parts.append(global_mean_pool(patch_z, batch))
        if self.slide_pool_include_interface:
            h_parts.extend([
                self._interface_pool(x_out, tissue, edge_index, batch, "tl"),
                self._interface_pool(x_out, tissue, edge_index, batch, "ts"),
            ])
        boundary_debug: Dict[str, object] = {"enabled": 0}
        if self.slide_pool_include_boundary:
            assert boundary_score is not None
            h_parts.append(self._weighted_sum_pool(x_out, boundary_score, batch))
            boundary_debug = self._boundary_debug(boundary_score.detach(), tissue.detach())
        h_slide = torch.cat(h_parts, dim=1)
        self._last_embedding_debug = {
            "slide_embedding_shape": list(h_slide.shape),
            "pool_include_raw": int(self.slide_pool_include_raw),
            "pool_include_interface": int(self.slide_pool_include_interface),
            "pool_include_boundary": int(self.slide_pool_include_boundary),
            "zero_cell_descriptors": int(self.zero_cell_descriptors),
            "zero_edge_tissue_interactions": int(self.zero_edge_tissue_interactions),
            "gnn_out_dim": int(x_out.size(1)),
            "raw_pool_dim": int(patch_z.size(1)),
            "boundary_score": boundary_debug,
        }
        return h_slide

    def compute_decoupled_embeddings(self, data) -> Dict[str, torch.Tensor]:
        patch_feats, coords, tissue, cell_counts, cell_topo, edge_index, edge_attr_raw, batch = self._graph_inputs(data)
        fz, patch_z, _cell_z = self.encode_modalities(patch_feats, cell_counts, cell_topo)
        pos = self.pos_encoder(coords)
        x = torch.cat([fz, pos, tissue], dim=1)
        edge_attr = self.edge_encoder(edge_attr_raw)
        edge_index, edge_attr = self._maybe_drop_edges(edge_index, edge_attr)
        x = F.dropout(x, p=self.dropout_in, training=self.training)
        x1 = self.conv1(x, edge_index, edge_attr)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=self.dropout_mid, training=self.training)
        x2 = self.conv2(x1, edge_index, edge_attr)
        x2 = F.elu(x2)
        x_out = self.norm(x1 + x2)

        raw_attn_debug: Dict[str, object]
        if self.decoupled_raw_pool_mode in {"attention", "attn", "visual_attention"}:
            raw_pool, raw_attn = self._visual_attention_pool(patch_z, batch)
            raw_attn_debug = {
                "enabled": 1,
                "min": float(raw_attn.min().detach().cpu()) if raw_attn.numel() else 0.0,
                "max": float(raw_attn.max().detach().cpu()) if raw_attn.numel() else 0.0,
                "mean": float(raw_attn.mean().detach().cpu()) if raw_attn.numel() else 0.0,
            }
        elif self.decoupled_raw_pool_mode in {"mean", "avg", "global_mean"}:
            raw_pool = global_mean_pool(patch_z, batch)
            raw_attn_debug = {"enabled": 0}
        else:
            raise ValueError(f"Unsupported decoupled_raw_pool_mode: {self.decoupled_raw_pool_mode}")
        h_gnn_mean = global_mean_pool(x_out, batch)
        h_tl = self._interface_pool(x_out, tissue, edge_index, batch, "tl")
        h_ts = self._interface_pool(x_out, tissue, edge_index, batch, "ts")
        h_cls = torch.cat([raw_pool, h_gnn_mean], dim=1)
        if self.decoupled_surv_include_raw:
            h_surv = torch.cat([raw_pool, h_gnn_mean, h_tl, h_ts], dim=1)
        else:
            h_surv = torch.cat([h_gnn_mean, h_tl, h_ts], dim=1)
        self._last_embedding_debug = {
            "embedding_mode": "decoupled",
            "cls_embedding_shape": list(h_cls.shape),
            "surv_embedding_shape": list(h_surv.shape),
            "slide_embedding_shape": {"cls": list(h_cls.shape), "surv": list(h_surv.shape)},
            "gnn_out_dim": int(x_out.size(1)),
            "raw_pool_dim": int(patch_z.size(1)),
            "raw_pool_mode": self.decoupled_raw_pool_mode,
            "surv_include_raw": int(self.decoupled_surv_include_raw),
            "visual_attn": raw_attn_debug,
        }
        return {"cls": h_cls, "surv": h_surv}

    def forward(self, data):
        return self.classifier(self.compute_slide_embedding(data))
