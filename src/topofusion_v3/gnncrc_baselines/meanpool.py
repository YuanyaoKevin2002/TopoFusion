import torch
import torch.nn as nn
from topofusion_v3.gnncrc_baselines.base import SlideEncoder
from topofusion_v3.gnncrc_baselines.registry import register_model


@register_model("meanpool")
class MeanPool(SlideEncoder):
    """Mean-pool MIL baseline. Accepts bag representation [N, D].

    Simplest possible lower bound. If nothing beats this, the task
    has no learnable signal beyond patch-average statistics.
    """

    def __init__(self, input_dim: int = 1536, output_dim: int = 512):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self._embedding_dim = output_dim

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        # representation: [N, D] bag of patch features
        pooled = representation.mean(dim=0)     # [D]
        return self.proj(pooled)                # [output_dim]
