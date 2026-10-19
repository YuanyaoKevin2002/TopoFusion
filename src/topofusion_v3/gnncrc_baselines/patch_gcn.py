"""Patch-GCN — line-for-line port of the official implementation.

Source upstream:
    https://github.com/mahmoodlab/Patch-GCN
    models/model_graph_mil.py   -> PatchGCN_Surv (the published architecture)
    models/model_utils.py        -> Attn_Net_Gated

    Chen et al., MICCAI 2021. "Whole slide images are 2D point clouds:
    context-aware survival prediction using patch-based graph convolutional
    networks."

Architecture (single-slide forward):

    fc(input_dim → hidden_dim)               # 1024 → 128 upstream; here 1536 → 128
    DeepGCNLayer × num_layers                # GENConv(softmax, learn_t=True), LayerNorm, ReLU, residual
        first layer: keep conv-only output, then concat residual after each layer
    Concat[x, x1, x2, x3, x4]                # hidden_dim * (num_layers) features per node
    path_phi → path_attention_head           # Attn_Net_Gated (gated tanh-sigmoid + softmax)
    Linear(hidden*4, output_dim)             # slide-level embedding

Differences from upstream — minimal and explicit:
  * `input_dim` is parameterised (upstream hard-codes 1024; we use UNI v2 1536)
  * `output_dim` is parameterised so the encoder fits our SlideEncoder contract
    and is followed by our generic task_head (ClassificationHead / SurvivalHead).
    The upstream PatchGCN_Surv bundles its own `Linear(hidden*4, n_classes)`
    classifier and the NLLSurvLoss discrete-time hazards path; we keep the
    encoder identical and let the task_head produce the final logits or Cox
    risk score. This is the same composition pattern used by ABMIL/TransMIL/
    DSMIL in this repo, so all baselines plug into the same trainer and the
    final-head choice is held constant across model comparisons.
  * `DataParallel.relocate()` from upstream is dropped — we run one slide per
    GPU and Lightning handles device placement.

Forward expects the output of `GraphKNNBuilder.build(...)`: a PyG `Data`
object with attributes `x [N, D]`, `edge_index [2, E]`, optional `edge_attr`,
and `batch [N]`. The `edge_agg` switch from upstream (`'spatial'` vs
`'latent'`) is preserved; the builder must populate the matching slot.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, ReLU
from torch_geometric.nn import GENConv, DeepGCNLayer

from topofusion_v3.gnncrc_baselines.base import SlideEncoder
from topofusion_v3.gnncrc_baselines.registry import register_model


class _AttnNetGated(nn.Module):
    """Verbatim port of Attn_Net_Gated from upstream models/model_utils.py."""

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

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A, x


@register_model("patch_gcn")
class PatchGCN(SlideEncoder):
    """Patch-GCN encoder (Chen et al., MICCAI 2021).

    Args (match upstream PatchGCN_Surv defaults where applicable):
        input_dim   : feature dim of the input bag (UNI v2 = 1536; upstream = 1024)
        num_layers  : number of DeepGCNLayer blocks (upstream default 4)
        hidden_dim  : per-layer hidden dim (upstream default 128)
        edge_agg    : 'spatial' uses Data.edge_index; 'latent' uses Data.edge_latent
        dropout     : applied inside Attn_Net_Gated
        output_dim  : slide-level embedding dim returned to the task head
    """

    def __init__(
        self,
        input_dim: int = 1536,
        num_layers: int = 4,
        hidden_dim: int = 128,
        edge_agg: str = "spatial",
        dropout: float = 0.25,
        output_dim: int = 512,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("Patch-GCN needs num_layers >= 2 (upstream default 4).")
        self._embedding_dim = output_dim
        self.edge_agg = edge_agg
        self.num_layers = num_layers - 1            # mirrors upstream off-by-one

        # Upstream: nn.Linear(1024, 128); we generalise input_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        self.layers = nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            conv = GENConv(hidden_dim, hidden_dim, aggr="softmax",
                           t=1.0, learn_t=True, num_layers=2, norm="layer")
            norm = LayerNorm(hidden_dim, elementwise_affine=True)
            act  = ReLU(inplace=True)
            self.layers.append(
                DeepGCNLayer(conv, norm, act, block="res", dropout=0.1, ckpt_grad=(i % 3 == 0))
            )

        # `* (num_layers)` because the upstream concat keeps the initial fc
        # output plus one tensor per DeepGCNLayer (after the conv-only first
        # layer):  x, x1, x2, x3  →  4 * hidden_dim for num_layers=4.
        concat_dim = hidden_dim * num_layers

        self.path_phi = nn.Sequential(nn.Linear(concat_dim, concat_dim), nn.ReLU(), nn.Dropout(0.25))
        self.path_attention_head = _AttnNetGated(L=concat_dim, D=concat_dim,
                                                 dropout=bool(dropout), n_classes=1)
        self.path_rho = nn.Sequential(nn.Linear(concat_dim, concat_dim), nn.ReLU())
        self.output_proj = nn.Linear(concat_dim, output_dim)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def forward(self, representation) -> torch.Tensor:
        # `representation` is a PyG Data object from GraphKNNBuilder.
        data = representation
        if self.edge_agg == "spatial":
            edge_index = data.edge_index
        elif self.edge_agg == "latent":
            edge_index = getattr(data, "edge_latent", None)
            if edge_index is None:
                raise RuntimeError("edge_agg='latent' but Data has no edge_latent.")
        else:
            raise ValueError(f"Unknown edge_agg: {self.edge_agg}")

        # No edge attributes used during message passing (matches upstream).
        x = self.fc(data.x)
        x_ = x

        # First DeepGCNLayer is conv-only (matches upstream L163).
        x = self.layers[0].conv(x_, edge_index)
        x_ = torch.cat([x_, x], dim=1)
        for layer in self.layers[1:]:
            x = layer(x, edge_index)
            x_ = torch.cat([x_, x], dim=1)

        h = self.path_phi(x_)
        A_path, h = self.path_attention_head(h)
        A_path = torch.transpose(A_path, 1, 0)             # [1, N]
        h = torch.mm(F.softmax(A_path, dim=1), h)          # [1, concat_dim]
        h = self.path_rho(h).squeeze(0)                     # [concat_dim]
        return self.output_proj(h)                          # [output_dim]
