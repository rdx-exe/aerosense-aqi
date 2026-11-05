"""Model evaluation pipeline.

Computes test-set metrics (MAE, RMSE, R²), generates residual plots and
feature-importance figures, and writes a markdown evaluation report.

All figures are saved to the configured ``visualizations_dir``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.config import get_path, load_config
from ..utils.io import read_csv
from ..model_training.train import train_models


def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE and R²."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _plot_residuals(y_true: pd.Series, y_pred: np.ndarray, out_path: Path) -> None:
    """Save a residual plot (pred vs residual) to ``out_path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.6, s=28, edgecolors="none")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted AQI")
    ax.set_ylabel("Residual (actual − predicted)")
    ax.set_title("Residual Plot — AQI Prediction")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_feature_importance(model, feature_names: list, out_path: Path) -> None:
    """Save a feature-importance bar chart for tree-based models."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not hasattr(model, "feature_importances_"):
        return

    importances = np.asarray(model.feature_importances_)
    order = np.argsort(importances)[::-1][:15]
    names = [feature_names[i] for i in order]
    vals = importances[order]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names[::-1], vals[::-1], color="#4C72B0")
    ax.set_xlabel("Feature importance")
    ax.set_title("Top 15 Feature Importances")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def evaluate_models(
    features: pd.DataFrame | None = None,
    *,
    config: dict | None = None,
    features_path: str | None = None,
) -> dict:
    """Train models, evaluate on the test set, and write artifacts.

    Returns
    -------
    dict
        ``{"metrics": {name: dict}, "best": str, "y_test": Series,
        "predictions": {name: ndarray}}``
    """
    if config is None:
        config = load_config()

    vis_dir = get_path(config.get("visualizations_dir", "visualizations"))
    vis_dir.mkdir(parents=True, exist_ok=True)

    trained = train_models(features, config=config, features_path=features_path)
    best = trained["best"]
    X_test = trained["X_test"]
    y_test = trained["y_test"]
    feature_names = trained["feature_names"]

    metrics: dict[str, dict] = {}
    predictions: dict[str, np.ndarray] = {}

    for name, model in trained["models"].items():
        pred = model.predict(X_test)
        m = _metrics(y_test, pred)
        metrics[name] = m
        predictions[name] = pred
        print(
            f"[eval ] {name:<18} test_mae={m['mae']:7.2f}  "
            f"test_rmse={m['rmse']:7.2f}  test_r2={m['r2']:6.3f}"
        )

    # Residual plot for the best model
    _plot_residuals(y_test, predictions[best], vis_dir / "residuals.png")
    _plot_feature_importance(
        trained["models"][best], feature_names, vis_dir / "feature_importance.png"
    )

    # Markdown report
    _write_report(metrics, best, vis_dir)

    return {
        "metrics": metrics,
        "best": best,
        "y_test": y_test,
        "predictions": predictions,
    }


def _write_report(metrics: dict, best: str, vis_dir: Path) -> None:
    """Write a markdown evaluation report."""
    lines = [
        "# AQI Model Evaluation Report",
        "",
        f"**Best model:** `{best}`",
        "",
        "| Model | MAE | RMSE | R² |",
        "|-------|-----|------|----|",
    ]
    for name, m in metrics.items():
        lines.append(
            f"| {name} | {m['mae']:.2f} | {m['rmse']:.2f} | {m['r2']:.3f} |"
        )
    lines += ["", "## Artifacts", "", "- `residuals.png`", "- `feature_importance.png`"]
    report_path = vis_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[eval ] Report written → {report_path}")


def main() -> None:
    """CLI entry point: evaluate models on the test set."""
    evaluate_models()


if __name__ == "__main__":
    main()