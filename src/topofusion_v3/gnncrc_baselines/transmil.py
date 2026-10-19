"""TransMIL: Transformer-based Correlated Multiple Instance Learning.

Reference: Shao et al. 2021 (NeurIPS) — TransMIL: Transformer-based MIL.
Architecture:
    project → [CLS] prepend → square-pad to N=ceil(sqrt) → 2× Nyström-attention
    block (with PPEG between them) → CLS readout

Nyström attention is O(N) instead of O(N^2), so this scales to bags of
thousands of patches. We use the canonical 6-landmark setting from the paper.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from topofusion_v3.gnncrc_baselines.base import SlideEncoder
from topofusion_v3.gnncrc_baselines.registry import register_model


def _moore_penrose_iter_pinv(x: torch.Tensor, iters: int = 6) -> torch.Tensor:
    """Iterative Moore-Penrose pseudoinverse (Razavi et al. 2014).
    Used by Nyström self-attention to approximate softmax(QK^T) on landmarks."""
    abs_x = x.abs()
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    z = x.transpose(-1, -2) / (torch.max(col) * torch.max(row))
    I = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
    I = I.expand_as(torch.matmul(z, x))
    for _ in range(iters):
        xz = torch.matmul(x, z)
        z = 0.25 * torch.matmul(z, 13 * I - torch.matmul(xz, 15 * I - torch.matmul(xz, 7 * I - xz)))
    return z


class NystromAttention(nn.Module):
    """Nyström approximation of multi-head self-attention. O(N·m + m^3)."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64,
                 num_landmarks: int = 256, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.num_landmarks = num_landmarks
        inner_dim = heads * dim_head

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        b, n, _ = x.shape
        h, m = self.heads, self.num_landmarks

        # Pad sequence so N is divisible by m
        remainder = n % m
        if remainder != 0:
            pad = m - remainder
            x = F.pad(x, (0, 0, pad, 0))
            n = x.shape[1]

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (t.reshape(b, n, h, -1).transpose(1, 2) for t in qkv)  # [B, H, N, Dh]
        q = q * self.scale

        # Landmarks via segmented mean
        seg = n // m
        q_l = q.reshape(b, h, m, seg, -1).mean(dim=-2)  # [B, H, m, Dh]
        k_l = k.reshape(b, h, m, seg, -1).mean(dim=-2)

        sim1 = torch.einsum("bhid,bhjd->bhij", q,   k_l).softmax(dim=-1)
        sim2 = torch.einsum("bhid,bhjd->bhij", q_l, k_l).softmax(dim=-1)
        sim3 = torch.einsum("bhid,bhjd->bhij", q_l, k  ).softmax(dim=-1)

        out = torch.einsum("bhij,bhjk->bhik", sim1,
              torch.einsum("bhij,bhjk->bhik", _moore_penrose_iter_pinv(sim2),
              torch.einsum("bhij,bhjk->bhik", sim3, v)))

        if remainder != 0:
            out = out[:, :, -(n - (m - remainder)):]

        out = out.transpose(1, 2).reshape(b, -1, h * (k.shape[-1]))
        return self.to_out(out)


class PPEG(nn.Module):
    """Pyramid Position Encoding Generator: 3 depthwise convs at scales 7/5/3."""

    def __init__(self, dim: int = 512):
        super().__init__()
        self.proj  = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # x: [B, 1+H*W, D]; first token is CLS
        b, _, d = x.shape
        cls_token, feat = x[:, :1], x[:, 1:]
        cnn_feat = feat.transpose(1, 2).reshape(b, d, h, w)
        x_pe = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x_pe = x_pe.flatten(2).transpose(1, 2)
        return torch.cat([cls_token, x_pe], dim=1)


class TransLayer(nn.Module):
    def __init__(self, dim: int = 512, heads: int = 8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = NystromAttention(dim=dim, heads=heads, dim_head=dim // heads,
                                     num_landmarks=dim // 2, dropout=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.norm(x))


@register_model("transmil")
class TransMIL(SlideEncoder):
    """Transformer MIL with Nyström attention + PPEG.

    Args:
        input_dim:    feature dim of input bag (default 1536 for UNI v2)
        hidden_dim:   internal transformer dim
        heads:        attention heads
        output_dim:   slide embedding dim (matches task head's expected dim)
    """

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 512,
        heads: int = 8,
        output_dim: int = 512,
    ):
        super().__init__()
        self._embedding_dim = output_dim

        self.feature_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.layer1 = TransLayer(dim=hidden_dim, heads=heads)
        self.pos_layer = PPEG(dim=hidden_dim)
        self.layer2 = TransLayer(dim=hidden_dim, heads=heads)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        """representation: [N, input_dim] bag of patch features (single slide)."""
        # Single-slide → add batch dim for transformer ops
        if representation.dim() == 2:
            x = representation.unsqueeze(0)  # [1, N, D]
            squeeze = True
        else:
            x = representation
            squeeze = False

        b = x.shape[0]
        h = self.feature_proj(x)  # [B, N, hidden]

        # Pad so N becomes a perfect square (PPEG needs H × W grid)
        n = h.shape[1]
        side = int(math.ceil(math.sqrt(n)))
        pad = side * side - n
        if pad > 0:
            h = F.pad(h, (0, 0, 0, pad))  # right-pad sequence dim

        cls = self.cls_token.expand(b, -1, -1)
        h = torch.cat([cls, h], dim=1)  # [B, 1+side^2, hidden]

        h = self.layer1(h)
        h = self.pos_layer(h, side, side)
        h = self.layer2(h)
        h = self.norm(h)

        cls_out = h[:, 0]  # [B, hidden]
        out = self.output_proj(cls_out)  # [B, output_dim]
        return out.squeeze(0) if squeeze else out
