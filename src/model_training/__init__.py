"""Model training sub-package — dataset split, model registry, fitting."""

from .train import create_models, main as train_pipeline

__all__ = ["create_models", "train_pipeline"]