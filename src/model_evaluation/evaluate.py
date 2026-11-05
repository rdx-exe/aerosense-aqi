"""Legacy evaluation entry point.

The project's authoritative evaluation is
:func:`src.model_evaluation.evaluate_final.evaluate_final_model`, which loads
the validation-selected model and evaluates it once on the frozen test split.

This module is retained as a thin compatibility wrapper for older callers
(e.g. ``src.scripts.run_pipeline``).  It does **not** retrain models and it
does not compute metrics from any data other than the saved test predictions.
"""

from __future__ import annotations

import pandas as pd

from ..utils.config import get_path
from .evaluate_final import evaluate_final_model


def evaluate_models(
    features: pd.DataFrame | None = None,
    *,
    config: dict | None = None,
    features_path: str | None = None,
) -> dict:
    """Run the final evaluation and return a legacy-compatible result dict.

    ``features``, ``config`` and ``features_path`` are accepted for backwards
    compatibility but are intentionally unused: evaluation must use the same
    frozen test split and the same validation-selected model every time.
    """
    result = evaluate_final_model()

    predictions_path = get_path("results", "metrics", "test_predictions.csv")
    predictions_df = pd.read_csv(predictions_path, parse_dates=["date"])

    model_name = result["best_model"]
    return {
        "metrics": {
            model_name: result["test_metrics"],
            "persistence_baseline": result["baseline_metrics"],
        },
        "best": model_name,
        "y_test": predictions_df["actual_aqi"],
        "predictions": {
            model_name: predictions_df["predicted_aqi"].to_numpy(),
            "persistence_baseline": predictions_df["baseline_aqi"].to_numpy(),
        },
    }


def main() -> None:
    """CLI entry point: run the final evaluation."""
    evaluate_models()


if __name__ == "__main__":
    main()
