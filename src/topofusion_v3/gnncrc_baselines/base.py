from __future__ import annotations
from abc import ABC, abstractmethod
import torch.nn as nn


class SlideEncoder(nn.Module, ABC):
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        raise NotImplementedError
