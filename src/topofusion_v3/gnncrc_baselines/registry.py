from __future__ import annotations
from typing import Callable

model_registry = {}


def register_model(name: str) -> Callable:
    def decorator(cls):
        model_registry[name] = cls
        return cls
    return decorator
