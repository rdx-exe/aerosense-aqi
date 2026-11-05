"""Utilities for reproducible ML experiment tracking and logging.

Provides:
- Reproducible experiment runs with unique IDs
- Structured logging of hyperparameters, metrics, and model info
- Comparison tables with persistence baseline
- Model persistence using Joblib
- Experiment configuration serialization
"""

from __future__ import annotations

import logging
import json
import logging
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..utils.config import get_path


class ExperimentLogger:
    """Structured logging for ML experiments with persistence baseline comparison."""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.experiment_dir = get_path("experiments", experiment_name)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Setup file logger
        log_file = self.experiment_dir / "experiment.log"
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        # Store experiment info
        self.experiment_info = {
            "name": experiment_name,
            "timestamp": pd.Timestamp.now().isoformat(),
            "data_source": "kolkata_city_day.csv",
            "forecast_horizon": "next_day"
        }

    def log_experiment_config(self, config: Dict[str, Any]) -> None:
        """Log experiment configuration."""
        self.logger.info(f"Experiment Configuration: {json.dumps(config, indent=2, default=str)}")
        self.experiment_info["config"] = config

    def log_model_metrics(
        self,
        model_name: str,
        model: Any,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        feature_importance: Dict[str, float] | None = None,
        hyperparameters: Dict[str, Any] | None = None,
    ) -> None:
        """Log model performance metrics."""
        metrics = {
            "model_name": model_name,
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_r2": train_metrics["r2"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "hyperparameters": hyperparameters or {},
            "feature_importance": feature_importance or {}
        }

        self.logger.info(f"Model Metrics - {model_name}: {json.dumps(metrics, indent=2)}")
        self.experiment_info.setdefault("models", {})[model_name] = metrics

    def log_comparison_table(self, baseline_metrics: Dict[str, float], model_metrics: List[Dict[str, Any]]) -> None:
        """Create and log model comparison table."""
        comparison = {
            "baseline": baseline_metrics,
            "models": model_metrics
        }

        df = pd.DataFrame(model_metrics)
        df.insert(0, "Model", [m["name"] for m in model_metrics])
        self.logger.info(f"Model Comparison:\n{df.to_string(index=False)}")

        self.experiment_info["comparison_table"] = comparison
        comparison_path = self.experiment_dir / "comparison_table.csv"
        df.to_csv(comparison_path, index=False)

    def save_experiment_info(self) -> None:
        """Save experiment information to JSON file."""
        info_path = self.experiment_dir / "experiment_info.json"
        with open(info_path, "w") as f:
            json.dump(self.experiment_info, f, indent=2)


def compare_to_baseline(
    baseline_pred: np.ndarray,
    baseline_true: np.ndarray,
    model_preds: Dict[str, np.ndarray],
    model_true: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """Calculate metrics for baseline and all models."""
    metrics = {}

    # Baseline metrics
    baseline_metrics = {
        "mae": mean_absolute_error(baseline_true, baseline_pred),
        "rmse": np.sqrt(mean_squared_error(baseline_true, baseline_pred)),
        "r2": r2_score(baseline_true, baseline_pred)
    }
    metrics["persistence_baseline"] = baseline_metrics

    # Model metrics
    for model_name, pred in model_preds.items():
        model_metrics = {
            "mae": mean_absolute_error(model_true, pred),
            "rmse": np.sqrt(mean_squared_error(model_true, pred)),
            "r2": r2_score(model_true, pred)
        }
        metrics[model_name] = model_metrics

    return metrics


def save_model_with_metadata(model: Any, model_name: str, metadata: Dict[str, Any]) -> str:
    """Save model using Joblib with metadata."""
    model_dir = get_path("models", model_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.pkl"
    from joblib import dump

    dump({
        "model": model,
        "metadata": metadata
    }, model_path)

    return str(model_path)