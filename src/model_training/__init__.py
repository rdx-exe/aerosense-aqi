"""Model training sub-package — dataset split, model registry, fitting."""

from __future__ import annotations

from typing import Any

__all__ = ["create_models", "train_pipeline"]


def __getattr__(name: str) -> Any:
    """Lazily expose training helpers without importing the module eagerly.

    Eager imports here trigger a ``runpy`` warning when the training module is
    invoked with ``python -m src.model_training.train``.
    """
    if name == "create_models":
        from .train import create_models

        return create_models
    if name == "train_pipeline":
        from .train import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
