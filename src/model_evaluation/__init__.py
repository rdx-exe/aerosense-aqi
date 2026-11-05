"""Model evaluation sub-package — metrics, residual analysis, reporting."""

from __future__ import annotations

from typing import Any

__all__ = ["evaluate_models"]


def __getattr__(name: str) -> Any:
    """Lazily expose the final evaluation entry point.

    Eagerly importing :mod:`evaluate_final` here causes a ``runpy`` warning
    when the module is executed with
    ``python -m src.model_evaluation.evaluate_final``.
    """
    if name == "evaluate_models":
        from .evaluate_final import evaluate_final_model

        return evaluate_final_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
