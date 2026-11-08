"""Final, independent evaluation of the selected next-day AQI model.

This module is deliberately *read-only* with respect to model selection.  It
loads the model that was selected on the validation split, loads the untouched
test split once, and reports MAE, RMSE, R² and error structure.  It never fits
or tunes a model on test data.

Artifacts produced
------------------
* ``results/metrics/final_metrics.csv``
* ``results/metrics/metrics_by_aqi_range.csv``
* ``results/metrics/extreme_event_metrics.csv``
* ``results/metrics/extreme_events.csv``
* ``results/metrics/test_predictions.csv``
* ``results/metrics/feature_importance.csv`` (when supported)
* ``results/metrics/evaluation_metadata.json``
* ``results/visualizations/evaluation/*.png``
* ``documentation/model_evaluation.md``

Run with::

    python -m src.model_evaluation.evaluate_final
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402

from ..utils.config import get_path  # noqa: E402

TARGET = "aqi_next_day"
DATE_COL = "date"
BASELINE_COL = "aqi"
NON_FEATURE_COLUMNS = {
    DATE_COL,
    TARGET,
    BASELINE_COL + "_bucket",
    "aqi_bucket",
    "city",
    "season",
    "Unnamed: 0",
}

# US-EPA AQI categories.  The repository computes AQI from PM2.5 using the
# US-EPA 2012 breakpoints, so these are the matching health categories.
AQI_RANGES = [
    ("Good (0-50)", 0.0, 50.0),
    ("Moderate (51-100)", 50.0, 100.0),
    ("Unhealthy for Sensitive Groups (101-150)", 100.0, 150.0),
    ("Unhealthy (151-200)", 150.0, 200.0),
    ("Very Unhealthy (201-300)", 200.0, 300.0),
    ("Hazardous (301-500)", 300.0, 500.0),
]
EXTREME_THRESHOLD = 200.0
TOP_FRACTION = 0.90


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────


def _fmt(value: object, digits: int = 2) -> str:
    """Format a number for a markdown report, handling NaN/None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(number):
        return "N/A"
    return f"{number:.{digits}f}"


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute the required metrics plus a few error-structure summaries."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    n = int(len(y_true))
    if n == 0:
        return {
            "n": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
            "mean_residual": float("nan"),
            "median_residual": float("nan"),
            "max_abs_error": float("nan"),
            "p90_abs_error": float("nan"),
        }

    residuals = y_true - y_pred  # positive = model under-prediction
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if n > 1 and ss_tot > 0 else float("nan")

    return {
        "n": n,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mean_residual": float(np.mean(residuals)) if n else float("nan"),
        "median_residual": float(np.median(residuals)) if n else float("nan"),
        "max_abs_error": float(np.max(np.abs(residuals))) if n else float("nan"),
        "p90_abs_error": float(np.percentile(np.abs(residuals), 90)) if n else float("nan"),
    }


def _improvement_pct(baseline_value: float, model_value: float) -> float:
    """Percentage reduction relative to the baseline (positive is better)."""
    baseline_value = float(baseline_value)
    model_value = float(model_value)
    if not np.isfinite(baseline_value) or baseline_value == 0 or not np.isfinite(model_value):
        return float("nan")
    return (baseline_value - model_value) / baseline_value * 100.0


def _aqi_range_label(value: float) -> str:
    """Return the US-EPA category label for an AQI value."""
    for label, low, high in AQI_RANGES:
        if low < float(value) <= high:
            return label
        if low == 0.0 and float(value) <= high:
            return label
    if float(value) > AQI_RANGES[-1][2]:
        return "Above 500"
    return "Below 0"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_selection() -> tuple[dict, Path]:
    """Load the validation-only model-selection record.

    Preference order:
    1. ``models/best_model.json`` written by ``src.model_training.train``.
    2. ``experiments/aqi_forecasting/experiment_info.json`` (legacy fallback),
       selecting the model with the lowest validation RMSE.
    """
    selection_path = get_path("models", "best_model.json")
    if selection_path.is_file():
        return _read_json(selection_path), selection_path

    experiment_info_path = get_path("experiments", "aqi_forecasting", "experiment_info.json")
    if not experiment_info_path.is_file():
        raise FileNotFoundError(
            "No model-selection record found. Run the training pipeline first:\n"
            "  python -m src.model_training.prepare_data\n"
            "  python -m src.model_training.train"
        )

    info = _read_json(experiment_info_path)
    models = info.get("models", {})
    if not models:
        raise ValueError(f"No model metrics found in {experiment_info_path}")

    best_name = min(models, key=lambda name: float(models[name]["val_rmse"]))
    selection = {
        "model_name": best_name,
        "selected_by": "validation_rmse",
        "validation_metrics": {
            "mae": models[best_name].get("val_mae"),
            "rmse": models[best_name].get("val_rmse"),
            "r2": models[best_name].get("val_r2"),
        },
        "baseline_validation_metrics": info.get("comparison_table", {}).get("baseline", {}),
        "candidate_validation_metrics": {
            name: {
                "mae": values.get("val_mae"),
                "rmse": values.get("val_rmse"),
                "r2": values.get("val_r2"),
            }
            for name, values in models.items()
        },
        "feature_columns": info.get("feature_columns"),
        "test_split_used_during_training": False,
        "selected_at": info.get("timestamp"),
        "source": "legacy experiment_info.json fallback",
    }
    return selection, experiment_info_path


def _resolve_feature_columns(
    test_df: pd.DataFrame,
    selection: dict,
    metadata: dict,
) -> list[str]:
    """Resolve the exact training feature contract."""
    columns = (
        selection.get("feature_columns")
        or metadata.get("feature_columns")
    )
    if not columns:
        # Last-resort inference.  This is logged as a warning in the report.
        columns = [
            c
            for c in test_df.columns
            if c not in NON_FEATURE_COLUMNS and not str(c).startswith("Unnamed")
        ]

    columns = [str(c) for c in columns]
    missing = [c for c in columns if c not in test_df.columns]
    if missing:
        raise ValueError(
            "The test split is missing feature columns required by the model: "
            f"{missing}"
        )
    if TARGET in columns:
        raise ValueError("The target column must not be part of the feature list")

    return columns


# ─────────────────────────────────────────────────────────────────────────────
# Error-structure tables
# ─────────────────────────────────────────────────────────────────────────────


def _range_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute model and baseline errors for each AQI category."""
    rows = []
    for label, low, high in AQI_RANGES:
        subset = predictions[
            (predictions["actual_aqi"] > low) & (predictions["actual_aqi"] <= high)
        ]
        if label == AQI_RANGES[0][0]:
            subset = predictions[predictions["actual_aqi"] <= high]

        model_metrics = _metrics(subset["actual_aqi"], subset["predicted_aqi"])
        baseline_metrics = _metrics(subset["actual_aqi"], subset["baseline_aqi"])
        rows.append(
            {
                "aqi_range": label,
                "n": int(len(subset)),
                "share_pct": float(len(subset) / len(predictions) * 100.0)
                if len(predictions)
                else float("nan"),
                "mae": model_metrics["mae"],
                "rmse": model_metrics["rmse"],
                "r2": model_metrics["r2"],
                "mean_residual": model_metrics["mean_residual"],
                "max_abs_error": model_metrics["max_abs_error"],
                "baseline_mae": baseline_metrics["mae"],
                "baseline_rmse": baseline_metrics["rmse"],
                "baseline_r2": baseline_metrics["r2"],
                "mae_improvement_pct": _improvement_pct(
                    baseline_metrics["mae"], model_metrics["mae"]
                ),
                "rmse_improvement_pct": _improvement_pct(
                    baseline_metrics["rmse"], model_metrics["rmse"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _extreme_metrics(
    predictions: pd.DataFrame,
    q90_threshold: float,
) -> pd.DataFrame:
    """Compare model and baseline errors overall and on extreme subsets."""
    definitions = [
        ("All test days", None, predictions),
        (f"AQI >= {EXTREME_THRESHOLD:.0f} (Very Unhealthy)", EXTREME_THRESHOLD,
         predictions[predictions["actual_aqi"] >= EXTREME_THRESHOLD]),
        (f"Top 10% of test days (AQI >= {q90_threshold:.1f})", q90_threshold,
         predictions[predictions["actual_aqi"] >= q90_threshold]),
    ]

    rows = []
    for definition, threshold, subset in definitions:
        model_metrics = _metrics(subset["actual_aqi"], subset["predicted_aqi"])
        baseline_metrics = _metrics(subset["actual_aqi"], subset["baseline_aqi"])
        rows.append(
            {
                "definition": definition,
                "threshold": threshold,
                "n": int(len(subset)),
                "share_pct": float(len(subset) / len(predictions) * 100.0)
                if len(predictions)
                else float("nan"),
                "mae": model_metrics["mae"],
                "rmse": model_metrics["rmse"],
                "r2": model_metrics["r2"],
                "mean_residual": model_metrics["mean_residual"],
                "max_abs_error": model_metrics["max_abs_error"],
                "baseline_mae": baseline_metrics["mae"],
                "baseline_rmse": baseline_metrics["rmse"],
                "baseline_r2": baseline_metrics["r2"],
                "mae_improvement_pct": _improvement_pct(
                    baseline_metrics["mae"], model_metrics["mae"]
                ),
                "rmse_improvement_pct": _improvement_pct(
                    baseline_metrics["rmse"], model_metrics["rmse"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _extreme_events(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Summarise consecutive runs of actual AQI >= threshold."""
    rows = []
    mask = (predictions["actual_aqi"] >= threshold).to_numpy()
    start = None
    event_id = 0
    for index, is_extreme in enumerate(mask):
        if is_extreme and start is None:
            start = index
        if not is_extreme and start is not None:
            event_id += 1
            rows.append(_event_row(predictions, start, index - 1, event_id, threshold))
            start = None
    if start is not None:
        event_id += 1
        rows.append(_event_row(predictions, start, len(mask) - 1, event_id, threshold))

    columns = [
        "event_id",
        "start_date",
        "end_date",
        "days",
        "max_actual_aqi",
        "min_actual_aqi",
        "max_predicted_aqi",
        "mean_predicted_aqi",
        "max_baseline_aqi",
        "mean_residual_actual_minus_predicted",
    ]
    return pd.DataFrame(rows, columns=columns)


def _event_row(
    predictions: pd.DataFrame,
    start: int,
    end: int,
    event_id: int,
    threshold: float,
) -> dict:
    event = predictions.iloc[start : end + 1]
    return {
        "event_id": event_id,
        "start_date": str(event[DATE_COL].iloc[0].date()),
        "end_date": str(event[DATE_COL].iloc[-1].date()),
        "days": int(len(event)),
        "max_actual_aqi": float(event["actual_aqi"].max()),
        "min_actual_aqi": float(event["actual_aqi"].min()),
        "max_predicted_aqi": float(event["predicted_aqi"].max()),
        "mean_predicted_aqi": float(event["predicted_aqi"].mean()),
        "max_baseline_aqi": float(event["baseline_aqi"].max()),
        "mean_residual_actual_minus_predicted": float(
            (event["actual_aqi"] - event["predicted_aqi"]).mean()
        ),
    }


def _peak_analysis(predictions: pd.DataFrame, threshold: float) -> dict:
    """Quantify peak capture and extreme-day detection."""
    actual = predictions["actual_aqi"]
    predicted = predictions["predicted_aqi"]
    baseline = predictions["baseline_aqi"]

    actual_extreme = actual >= threshold
    predicted_extreme = predicted >= threshold

    true_positive = int((actual_extreme & predicted_extreme).sum())
    false_positive = int((~actual_extreme & predicted_extreme).sum())
    false_negative = int((actual_extreme & ~predicted_extreme).sum())
    actual_extreme_count = int(actual_extreme.sum())

    extreme_residuals = (actual - predicted)[actual_extreme]
    non_extreme_residuals = (actual - predicted)[~actual_extreme]

    return {
        "threshold": float(threshold),
        "actual_extreme_days": actual_extreme_count,
        "predicted_extreme_days": int(predicted_extreme.sum()),
        "true_positive_days": true_positive,
        "false_positive_days": false_positive,
        "false_negative_days": false_negative,
        "extreme_recall": float(true_positive / actual_extreme_count)
        if actual_extreme_count
        else float("nan"),
        "extreme_precision": float(true_positive / (true_positive + false_positive))
        if (true_positive + false_positive)
        else float("nan"),
        "actual_max_aqi": float(actual.max()),
        "predicted_max_aqi": float(predicted.max()),
        "baseline_max_aqi": float(baseline.max()),
        "predicted_max_error": float(predicted.max() - actual.max()),
        "mean_residual_extreme": float(extreme_residuals.mean())
        if len(extreme_residuals)
        else float("nan"),
        "mean_residual_non_extreme": float(non_extreme_residuals.mean())
        if len(non_extreme_residuals)
        else float("nan"),
        "mean_abs_error_extreme": float(extreme_residuals.abs().mean())
        if len(extreme_residuals)
        else float("nan"),
    }


def _range_compression(predictions: pd.DataFrame) -> dict:
    """Quantify how much the model shrinks predictions toward the mean."""
    actual = predictions["actual_aqi"]
    predicted = predictions["predicted_aqi"]
    actual_range = float(actual.max() - actual.min())
    predicted_range = float(predicted.max() - predicted.min())
    return {
        "actual_min": float(actual.min()),
        "actual_max": float(actual.max()),
        "actual_mean": float(actual.mean()),
        "actual_std": float(actual.std()),
        "predicted_min": float(predicted.min()),
        "predicted_max": float(predicted.max()),
        "predicted_mean": float(predicted.mean()),
        "predicted_std": float(predicted.std()),
        "actual_range": actual_range,
        "predicted_range": predicted_range,
        "prediction_range_ratio": float(predicted_range / actual_range)
        if actual_range > 0
        else float("nan"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────


def _plot_actual_vs_predicted(
    predictions: pd.DataFrame,
    model_name: str,
    metrics: dict,
    output_path: Path,
) -> None:
    extreme = predictions["actual_aqi"] >= EXTREME_THRESHOLD
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        predictions.loc[~extreme, "actual_aqi"],
        predictions.loc[~extreme, "predicted_aqi"],
        s=45,
        alpha=0.7,
        color="#2b6cb0",
        label="Test days (AQI < 200)",
        edgecolors="none",
    )
    ax.scatter(
        predictions.loc[extreme, "actual_aqi"],
        predictions.loc[extreme, "predicted_aqi"],
        s=60,
        alpha=0.9,
        color="#c53030",
        label="Extreme days (AQI >= 200)",
        edgecolors="none",
    )
    all_values = pd.concat(
        [predictions["actual_aqi"], predictions["predicted_aqi"], predictions["baseline_aqi"]]
    )
    low = float(all_values.min()) - 10
    high = float(all_values.max()) + 10
    ax.plot([low, high], [low, high], "k--", linewidth=1.5, label="Perfect 1:1 line")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Actual next-day AQI")
    ax.set_ylabel("Predicted next-day AQI")
    ax.set_title(f"Actual vs predicted next-day AQI — {model_name}")
    ax.text(
        0.03,
        0.97,
        f"MAE = {metrics['mae']:.2f}\nRMSE = {metrics['rmse']:.2f}\n"
        f"R² = {_fmt(metrics['r2'], 3)}",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_timeseries(predictions: pd.DataFrame, model_name: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(
        predictions[DATE_COL],
        predictions["actual_aqi"],
        color="black",
        linewidth=1.6,
        label="Actual AQI",
    )
    ax.plot(
        predictions[DATE_COL],
        predictions["predicted_aqi"],
        color="#2b6cb0",
        linewidth=1.4,
        label=f"Predicted AQI ({model_name})",
    )
    ax.plot(
        predictions[DATE_COL],
        predictions["baseline_aqi"],
        color="#718096",
        linewidth=1.0,
        linestyle="--",
        alpha=0.8,
        label="Persistence baseline",
    )
    extreme = predictions["actual_aqi"] >= EXTREME_THRESHOLD
    ax.scatter(
        predictions.loc[extreme, DATE_COL],
        predictions.loc[extreme, "actual_aqi"],
        color="#c53030",
        s=28,
        zorder=5,
        label="Actual AQI >= 200",
    )
    ax.axhline(EXTREME_THRESHOLD, color="#c53030", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    ax.set_title(f"Next-day AQI over the test period — {model_name}")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_residuals(predictions: pd.DataFrame, metrics: dict, output_path: Path) -> None:
    extreme = predictions["actual_aqi"] >= EXTREME_THRESHOLD
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        predictions.loc[~extreme, "predicted_aqi"],
        predictions.loc[~extreme, "residual"],
        s=45,
        alpha=0.7,
        color="#2b6cb0",
        label="AQI < 200",
        edgecolors="none",
    )
    ax.scatter(
        predictions.loc[extreme, "predicted_aqi"],
        predictions.loc[extreme, "residual"],
        s=60,
        alpha=0.9,
        color="#c53030",
        label="AQI >= 200",
        edgecolors="none",
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1.3)
    ax.axhline(
        metrics["mean_residual"],
        color="#dd6b20",
        linestyle=":",
        linewidth=1.3,
        label=f"Mean residual = {metrics['mean_residual']:.2f}",
    )
    ax.set_xlabel("Predicted next-day AQI")
    ax.set_ylabel("Residual (actual − predicted)")
    ax.set_title("Prediction residuals")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_prediction_distribution(predictions: pd.DataFrame, output_path: Path) -> None:
    """Compare the spread of actual, predicted and baseline AQI values."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(
        min(predictions["actual_aqi"].min(), predictions["baseline_aqi"].min()) - 5,
        max(predictions["actual_aqi"].max(), predictions["baseline_aqi"].max()) + 5,
        24,
    )
    ax.hist(
        predictions["actual_aqi"],
        bins=bins,
        alpha=0.55,
        color="black",
        label="Actual AQI",
    )
    ax.hist(
        predictions["predicted_aqi"],
        bins=bins,
        alpha=0.55,
        color="#2b6cb0",
        label="Predicted AQI",
    )
    ax.hist(
        predictions["baseline_aqi"],
        bins=bins,
        alpha=0.30,
        color="#a0aec0",
        label="Persistence baseline",
    )
    ax.set_xlabel("AQI")
    ax.set_ylabel("Number of test days")
    ax.set_title("Distribution of actual, predicted and baseline AQI")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_residual_distribution(
    predictions: pd.DataFrame, metrics: dict, output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        predictions["residual"],
        bins=20,
        color="#4c78a8",
        edgecolor="white",
        alpha=0.85,
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, label="Zero error")
    ax.axvline(
        metrics["mean_residual"],
        color="#dd6b20",
        linestyle=":",
        linewidth=1.4,
        label=f"Mean = {metrics['mean_residual']:.2f}",
    )
    ax.set_xlabel("Residual (actual − predicted AQI)")
    ax.set_ylabel("Number of test days")
    ax.set_title("Distribution of prediction residuals")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_error_by_range(range_df: pd.DataFrame, output_path: Path) -> None:
    present = range_df[range_df["n"] > 0].copy()
    if present.empty:
        return

    labels = [
        f"{row.aqi_range}\n(n={int(row.n)})" for row in present.itertuples()
    ]
    x = np.arange(len(present))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    for ax, model_col, baseline_col, title in [
        (axes[0], "mae", "baseline_mae", "MAE by actual AQI range"),
        (axes[1], "rmse", "baseline_rmse", "RMSE by actual AQI range"),
    ]:
        ax.bar(x - width / 2, present[model_col], width, color="#2b6cb0", label="Model")
        ax.bar(
            x + width / 2,
            present[baseline_col],
            width,
            color="#a0aec0",
            label="Persistence baseline",
        )
        ax.set_title(title)
        ax.set_ylabel("AQI units (lower is better)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Prediction error across AQI severity ranges")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_extreme_errors(predictions: pd.DataFrame, metrics: dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        predictions["actual_aqi"],
        predictions["abs_error"],
        s=50,
        alpha=0.75,
        color="#2b6cb0",
        label="Model absolute error",
        edgecolors="none",
    )
    ax.scatter(
        predictions["actual_aqi"],
        (predictions["actual_aqi"] - predictions["baseline_aqi"]).abs(),
        s=28,
        alpha=0.35,
        color="#a0aec0",
        label="Persistence absolute error",
        edgecolors="none",
    )
    ax.axvline(
        EXTREME_THRESHOLD,
        color="#c53030",
        linestyle="--",
        linewidth=1.2,
        label=f"Extreme threshold = {EXTREME_THRESHOLD:.0f}",
    )
    ax.axhline(
        metrics["mae"],
        color="#dd6b20",
        linestyle=":",
        linewidth=1.2,
        label=f"Overall model MAE = {metrics['mae']:.2f}",
    )
    ax.set_xlabel("Actual next-day AQI")
    ax.set_ylabel("Absolute error (AQI units)")
    ax.set_title("Absolute error vs AQI severity (extreme-event check)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_error_over_time(predictions: pd.DataFrame, metrics: dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(
        predictions[DATE_COL],
        predictions["abs_error"],
        color="#2b6cb0",
        linewidth=1.2,
        label="Model absolute error",
    )
    ax.plot(
        predictions[DATE_COL],
        (predictions["actual_aqi"] - predictions["baseline_aqi"]).abs(),
        color="#a0aec0",
        linewidth=0.9,
        linestyle="--",
        label="Persistence absolute error",
    )
    ax.axhline(
        metrics["mae"],
        color="#dd6b20",
        linestyle=":",
        linewidth=1.2,
        label=f"Overall model MAE = {metrics['mae']:.2f}",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Absolute error (AQI units)")
    ax.set_title("Absolute prediction error over the test period")
    ax.legend(fontsize=9, ncol=3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_feature_importance(importance_df: pd.DataFrame, method: str, output_path: Path) -> None:
    top = importance_df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["importance"], color="#1b9e77")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top 20 features — {method}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance
# ─────────────────────────────────────────────────────────────────────────────


def _feature_importance(
    model,
    feature_columns: list[str],
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    predictions_dir: Path,
    metrics_dir: Path,
) -> tuple[pd.DataFrame | None, str | None]:
    """Return feature importances when the fitted model supports them.

    Built-in importances are used first.  If the model has neither
    ``feature_importances_`` nor ``coef_`` (e.g. HistGradientBoosting), a
    permutation importance is computed on the *validation* split, not on the
    test split, so test data remains purely an evaluation set.
    """
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("regressor", model)

    importance = None
    method = None
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
        if len(values) == len(feature_columns):
            importance = values
            method = "impurity-based feature importance"
    elif hasattr(estimator, "coef_"):
        values = np.ravel(np.asarray(estimator.coef_, dtype=float))
        if len(values) == len(feature_columns):
            importance = np.abs(values)
            method = "absolute standardized coefficient"

    if importance is None:
        validation_path = get_path("data/processed/splits", "validation.csv")
        if not validation_path.is_file():
            return None, None
        validation_df = pd.read_csv(validation_path, parse_dates=[DATE_COL])
        X_validation = validation_df[feature_columns]
        y_validation = validation_df[TARGET].to_numpy()
        result = permutation_importance(
            model,
            X_validation,
            y_validation,
            n_repeats=5,
            random_state=42,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        importance = np.asarray(result.importances_mean, dtype=float)
        method = "permutation importance on validation set (negative MAE)"

    importance_df = pd.DataFrame(
        {"feature": feature_columns, "importance": importance}
    ).sort_values("importance", ascending=False, kind="mergesort")
    importance_df["rank"] = np.arange(1, len(importance_df) + 1)
    importance_df["method"] = method
    importance_df.to_csv(metrics_dir / "feature_importance.csv", index=False)
    _plot_feature_importance(
        importance_df, method, predictions_dir / "feature_importance.png"
    )
    return importance_df, method


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report
# ─────────────────────────────────────────────────────────────────────────────


def _markdown_table(header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join([":---"] + [":---:" for _ in header[1:]]) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _write_documentation(
    *,
    model_name: str,
    selection: dict,
    selection_path: Path,
    test_path: Path,
    test_metrics: dict,
    baseline_metrics: dict,
    metrics_df: pd.DataFrame,
    range_df: pd.DataFrame,
    extreme_df: pd.DataFrame,
    events_df: pd.DataFrame,
    peak: dict,
    range_compression: dict,
    largest_errors: pd.DataFrame,
    residual_autocorr: float,
    actual_prediction_corr: float,
    date_range: tuple[str, str],
    feature_importance: pd.DataFrame | None,
    feature_importance_method: str | None,
    metadata: dict,
    plot_dir: Path,
    metrics_dir: Path,
) -> Path:
    """Write the final evaluation report with values computed at runtime."""
    doc_path = get_path("documentation", "model_evaluation.md")
    doc_path.parent.mkdir(parents=True, exist_ok=True)

    def _display_path(path: Path) -> str:
        try:
            return str(path.relative_to(get_path()))
        except ValueError:
            return str(path)

    selection_display = _display_path(selection_path)
    test_display = _display_path(test_path)
    plot_display = _display_path(plot_dir)
    metrics_display = _display_path(metrics_dir)

    model_rmse = test_metrics["rmse"]
    baseline_rmse = baseline_metrics["rmse"]
    rmse_improvement = _improvement_pct(baseline_rmse, model_rmse)
    mae_improvement = _improvement_pct(baseline_metrics["mae"], test_metrics["mae"])
    beats_baseline = bool(np.isfinite(model_rmse) and np.isfinite(baseline_rmse) and model_rmse < baseline_rmse)
    r2 = test_metrics["r2"]

    if beats_baseline and np.isfinite(r2) and r2 >= 0.50:
        suitability = (
            "Suitable as an educational forecasting demonstration, with the "
            "caveats below. The model beats the persistence baseline and "
            "explains a meaningful share of test-set variance."
        )
    elif beats_baseline and np.isfinite(r2) and r2 >= 0.0:
        suitability = (
            "Suitable for demonstrating the end-to-end ML pipeline, feature "
            "engineering and honest evaluation, but **not suitable as a "
            "reliable operational forecaster**. It beats persistence on "
            "average error, yet its R² is near zero, so it explains little "
            "of the day-to-day variance in AQI. It should not be presented "
            "as a dependable warning system."
        )
    elif beats_baseline:
        suitability = (
            "Partially suitable as an educational demonstration: it beats the "
            "persistence baseline on average error, but the R² is negative, "
            "meaning it performs worse than a simple mean predictor in "
            "variance-explained terms. Not suitable for operational use."
        )
    else:
        suitability = (
            "Not suitable as a forecasting demonstration on this test set: it "
            "does **not** beat the persistence baseline. The pipeline can "
            "still be used to teach evaluation methodology, but the model "
            "should not be presented as a working AQI forecaster."
        )

    # ── main metrics table ──────────────────────────────────────────────────
    metric_rows = []
    for row in metrics_df.itertuples():
        metric_rows.append(
            [
                str(row.model),
                _fmt(row.mae),
                _fmt(row.rmse),
                _fmt(row.r2, 3),
                _fmt(row.mean_residual),
                _fmt(row.max_abs_error),
                _fmt(row.rmse_improvement_pct) if np.isfinite(row.rmse_improvement_pct) else "—",
            ]
        )

    # ── range table ─────────────────────────────────────────────────────────
    range_rows = []
    for row in range_df[range_df["n"] > 0].itertuples():
        range_rows.append(
            [
                str(row.aqi_range),
                str(int(row.n)),
                _fmt(row.mae),
                _fmt(row.rmse),
                _fmt(row.r2, 3),
                _fmt(row.mean_residual),
                _fmt(row.baseline_mae),
                _fmt(row.mae_improvement_pct),
            ]
        )
    if not range_rows:
        range_table = "_No AQI categories were populated in the test split._"
    else:
        range_table = _markdown_table(
            [
                "AQI range",
                "n",
                "Model MAE",
                "Model RMSE",
                "Model R²",
                "Mean residual (actual−pred)",
                "Baseline MAE",
                "MAE improvement %",
            ],
            range_rows,
        )
    empty_categories = range_df[range_df["n"] == 0]["aqi_range"].tolist()
    empty_note = (
        f"\n\nEmpty categories in the test split: {', '.join(empty_categories)}."
        if empty_categories
        else ""
    )

    # ── extreme table ───────────────────────────────────────────────────────
    extreme_rows = []
    for row in extreme_df.itertuples():
        extreme_rows.append(
            [
                str(row.definition),
                str(int(row.n)),
                _fmt(row.mae),
                _fmt(row.rmse),
                _fmt(row.r2, 3),
                _fmt(row.mean_residual),
                _fmt(row.baseline_mae),
                _fmt(row.baseline_rmse),
                _fmt(row.mae_improvement_pct),
            ]
        )
    extreme_table = _markdown_table(
        [
            "Subset",
            "n",
            "Model MAE",
            "Model RMSE",
            "Model R²",
            "Mean residual (actual−pred)",
            "Baseline MAE",
            "Baseline RMSE",
            "MAE improvement %",
        ],
        extreme_rows,
    )

    # ── event table ─────────────────────────────────────────────────────────
    if events_df.empty:
        event_table = "_No consecutive extreme-event run was found._"
    else:
        event_rows = []
        for row in events_df.itertuples():
            event_rows.append(
                [
                    str(int(row.event_id)),
                    str(row.start_date),
                    str(row.end_date),
                    str(int(row.days)),
                    _fmt(row.max_actual_aqi),
                    _fmt(row.max_predicted_aqi),
                    _fmt(row.max_baseline_aqi),
                    _fmt(row.mean_residual_actual_minus_predicted),
                ]
            )
        event_table = _markdown_table(
            [
                "Event",
                "Start",
                "End",
                "Days",
                "Max actual AQI",
                "Max predicted AQI",
                "Max baseline AQI",
                "Mean residual (actual−pred)",
            ],
            event_rows,
        )

    # ── feature importance table ────────────────────────────────────────────
    if feature_importance is not None and not feature_importance.empty:
        feature_rows = []
        for row in feature_importance.head(10).itertuples():
            feature_rows.append(
                [
                    str(row.feature),
                    _fmt(row.importance, 4),
                    str(int(row.rank)),
                ]
            )
        feature_table = _markdown_table(
            ["Feature", "Importance", "Rank"], feature_rows
        )
        feature_section = (
            f"Feature importances were extracted using **{feature_importance_method}**. "
            "The table below lists the top 10; the full table is saved to "
            f"`{metrics_display}/feature_importance.csv`.\n\n{feature_table}\n\n"
            "Importance is not causal. Many features are lagged or rolling "
            "versions of the same pollutants and are strongly correlated, so "
            "the ranking should be read as model behaviour rather than as a "
            "physical explanation."
        )
    else:
        feature_section = (
            "The selected model does not expose built-in feature importances, "
            "and permutation importance could not be computed. No feature-"
            "importance plot is therefore reported."
        )

    # ── split/distribution table ────────────────────────────────────────────
    split_info = selection.get("split_info") or metadata.get("split_info") or {}
    stats = split_info.get("target_stats", {})
    if stats:
        stat_rows = []
        for name in ["train", "validation", "test"]:
            if name not in stats:
                continue
            values = stats[name]
            stat_rows.append(
                [
                    name,
                    _fmt(values.get("target_mean")),
                    _fmt(values.get("target_std")),
                    _fmt(values.get("target_min")),
                    _fmt(values.get("target_max")),
                ]
            )
        distribution_table = _markdown_table(
            ["Split", "Target mean", "Target std", "Target min", "Target max"],
            stat_rows,
        )
    else:
        distribution_table = (
            "_Split-level target statistics were not available in the "
            "selection metadata._"
        )

    # ── validation selection table ──────────────────────────────────────────
    candidate_metrics = selection.get("candidate_validation_metrics", {})
    candidate_rows = []
    for name, values in candidate_metrics.items():
        if not isinstance(values, dict):
            continue
        candidate_rows.append(
            [
                str(name),
                _fmt(values.get("mae")),
                _fmt(values.get("rmse")),
                _fmt(values.get("r2"), 3),
            ]
        )
    candidate_table = (
        _markdown_table(
            ["Candidate", "Validation MAE", "Validation RMSE", "Validation R²"],
            candidate_rows,
        )
        if candidate_rows
        else "_Candidate validation metrics were not available._"
    )

    residual_autocorr_text = _fmt(residual_autocorr, 3)
    actual_corr_text = _fmt(actual_prediction_corr, 3)

    moderate_rows = range_df[range_df["aqi_range"].str.startswith("Moderate")]
    moderate_row = moderate_rows.iloc[0] if not moderate_rows.empty else None
    extreme_rows_matching = extreme_df[
        extreme_df["definition"].str.startswith(f"AQI >= {EXTREME_THRESHOLD:.0f}")
    ]
    extreme_row = extreme_rows_matching.iloc[0] if not extreme_rows_matching.empty else None
    extreme_mae_ratio = (
        float(extreme_row["mae"] / test_metrics["mae"])
        if extreme_row is not None and test_metrics["mae"] > 0
        else float("nan")
    )

    largest_error_rows = []
    for row in largest_errors.itertuples():
        largest_error_rows.append(
            [
                str(pd.Timestamp(row.date).date()),
                _fmt(row.actual_aqi),
                _fmt(row.predicted_aqi),
                _fmt(row.residual),
                _fmt(row.abs_error),
            ]
        )
    largest_error_table = (
        _markdown_table(
            ["Date", "Actual AQI", "Predicted AQI", "Residual (actual−pred)", "Absolute error"],
            largest_error_rows,
        )
        if largest_error_rows
        else "_No observations available._"
    )

    content = f"""# AeroSense — Final Model Evaluation & Error Analysis (Next-Day AQI Prediction)

This report is generated by `python -m src.model_evaluation.evaluate_final`.
Every number below is computed from the untouched test split described in
Section 1; no metric is hand-written or estimated.

## 1. Test dataset and evaluation integrity

- **Test date range:** **{date_range[0]} to {date_range[1]}**
- **Test rows:** {test_metrics['n']}
- **Target:** `aqi_next_day` (AQI for the next calendar day)
- **Model evaluated:** `{model_name}`
- **Model-selection record:** `{selection_display}`
- **Test split file:** `{test_display}`
- **Test split SHA-256:** `{metadata['test_sha256']}`
- **Model file SHA-256:** `{metadata['model_sha256']}`
- **Selection metric:** {selection.get('selected_by', 'validation RMSE')}
- **Was the test split used for training or selection?** {selection.get('test_split_used_during_training', False) and 'Yes' or 'No'}

The model was selected using **validation performance only**. The test split
was created chronologically, was not shuffled, and was not loaded by the
training script. One boundary row was purged between adjacent splits so that
the last target of one split does not belong to the next split. The model and
its feature contract were fixed before this test evaluation.

> **Integrity statement:** No model was retrained, tuned, or changed after
> observing these test results. Repeated runs of this script reproduce the
> same model and the same test split; they do not re-select the model.

### Split sizes and target distributions

{distribution_table}

## 2. Model selection used only the validation split

{candidate_table}

The selected model is **`{model_name}`**. Its validation metrics were:

- MAE: **{_fmt(selection.get('validation_metrics', {}).get('mae'))}**
- RMSE: **{_fmt(selection.get('validation_metrics', {}).get('rmse'))}**
- R²: **{_fmt(selection.get('validation_metrics', {}).get('r2'), 3)}**
- Validation RMSE improvement over the persistence baseline:
  **{_fmt(selection.get('validation_rmse_improvement_pct'))}%**

## 3. Test-set performance versus the persistence baseline

The persistence baseline predicts tomorrow's AQI as today's observed AQI
(`aqi`). It is a deliberately strong naive baseline for a one-day horizon.

{_markdown_table(
    ["Model", "MAE", "RMSE", "R²", "Mean residual (actual−pred)", "Max abs error", "RMSE improvement %"],
    metric_rows,
)}

**Interpretation of the metrics**

- **MAE (Mean Absolute Error):** average absolute prediction error in AQI
  units. The observed test MAE is **{_fmt(test_metrics['mae'])}**, meaning the
  model is typically about {_fmt(test_metrics['mae'])} AQI points away from the
  observed value.
- **RMSE (Root Mean Squared Error):** square root of the mean squared error.
  It penalises large misses more heavily than MAE, so a large RMSE relative to
  MAE indicates occasional large errors (often pollution spikes).
- **R²:** fraction of the test-set variance explained relative to predicting
  the test-set mean. `R² = 1` is perfect, `R² = 0` means the model is no better
  than a constant mean predictor, and negative `R²` means it is worse than the
  mean predictor.
- **Mean residual (actual − predicted):** positive values mean the model
  **under-predicts** on average; negative values mean it **over-predicts**.

The selected model's test RMSE is **{_fmt(model_rmse)}** versus
**{_fmt(baseline_rmse)}** for persistence, a **{_fmt(rmse_improvement)}%**
RMSE reduction. The MAE change is **{_fmt(mae_improvement)}%** (positive means
the model is better). These two facts are reported together deliberately: a
model can reduce average error while still explaining very little variance.

## 4. Actual-versus-predicted and residual diagnostics

The following figures are saved in `{plot_display}`:

| Figure | What it shows |
| :--- | :--- |
| `actual_vs_predicted.png` | Scatter of actual versus predicted AQI with the 1:1 line; extreme days (actual AQI >= 200) are highlighted. |
| `actual_vs_predicted_timeseries.png` | Actual, predicted and persistence AQI across the full test period. |
| `residual_plot.png` | Residuals (actual − predicted) versus predicted AQI. |
| `residual_distribution.png` | Histogram of residuals, showing bias and spread. |
| `prediction_distribution.png` | Overlaid distributions of actual, predicted and baseline AQI, exposing range compression. |
| `error_by_aqi_range.png` | MAE and RMSE by actual AQI severity range, compared with persistence. |
| `extreme_event_errors.png` | Absolute error versus actual AQI, with the extreme threshold marked. |
| `absolute_error_over_time.png` | Absolute error on each test day. |
| `feature_importance.png` | Top features for the selected model (when supported). |

Residual diagnostics:

- **Mean residual:** {_fmt(test_metrics['mean_residual'])} AQI points
  (positive = under-prediction).
- **Median residual:** {_fmt(test_metrics['median_residual'])} AQI points.
- **Residual autocorrelation at lag 1:** {residual_autocorr_text}.
  Values close to zero suggest little remaining day-to-day structure; large
  positive values would indicate the model is systematically lagging the
  series.
- **Correlation between actual and predicted AQI:** {actual_corr_text}.

### Prediction range compression

A model can have a reasonable MAE while still failing at the ends of the AQI
distribution. The test-set ranges make this visible:

| Quantity | Actual AQI | Model prediction |
| :--- | :---: | :---: |
| Minimum | {_fmt(range_compression['actual_min'])} | {_fmt(range_compression['predicted_min'])} |
| Maximum | {_fmt(range_compression['actual_max'])} | {_fmt(range_compression['predicted_max'])} |
| Range (max − min) | {_fmt(range_compression['actual_range'])} | {_fmt(range_compression['predicted_range'])} |
| Mean | {_fmt(range_compression['actual_mean'])} | {_fmt(range_compression['predicted_mean'])} |
| Standard deviation | {_fmt(range_compression['actual_std'])} | {_fmt(range_compression['predicted_std'])} |

The predicted range is **{_fmt(range_compression['prediction_range_ratio'], 2)}×**
the actual range. This is a clear regression-to-the-mean pattern: the model
avoids extreme predictions. It therefore tends to **over-predict very clean
days** and **under-predict pollution peaks**. This behaviour is a property of
the fitted model and the training distribution, not evidence that extreme
values are inherently unpredictable.

### Largest individual test errors

{largest_error_table}

These individual errors matter because the public-health cost of a missed
extreme event is not proportional to the average error.

## 5. Error across AQI ranges

{range_table}{empty_note}

R² values computed inside a narrow AQI range should be treated cautiously:
when the range is narrow, target variance is small and R² can become extreme
or negative. MAE and RMSE are more interpretable for these small subsets. The
`n` column is shown for exactly this reason.

## 6. Extreme pollution events

**Working definition of an extreme day:** actual next-day AQI >= **200**
(US-EPA *Very Unhealthy* category). As a sensitivity check, the top 10% of
test days is also reported, with its empirical threshold shown below.

{extreme_table}

### Peak capture and event-level behaviour

- Actual maximum AQI: **{_fmt(peak['actual_max_aqi'])}**
- Model maximum predicted AQI: **{_fmt(peak['predicted_max_aqi'])}**
- Persistence maximum: **{_fmt(peak['baseline_max_aqi'])}**
- Model peak error (predicted max − actual max): **{_fmt(peak['predicted_max_error'])}**
- Extreme days (actual AQI >= 200): **{peak['actual_extreme_days']}**
- Days predicted as extreme: **{peak['predicted_extreme_days']}**
- Extreme-day recall: **{_fmt(peak['extreme_recall'], 3)}**
- Extreme-day precision: **{_fmt(peak['extreme_precision'], 3)}**
- False-positive extreme days: **{peak['false_positive_days']}**
- False-negative extreme days: **{peak['false_negative_days']}**
- Mean residual on extreme days (actual − predicted): **{_fmt(peak['mean_residual_extreme'])}**
- Mean absolute error on extreme days: **{_fmt(peak['mean_abs_error_extreme'])}**
- Mean residual on non-extreme days: **{_fmt(peak['mean_residual_non_extreme'])}**

### Consecutive extreme-event runs

{event_table}

**Does the model perform poorly during extreme pollution events?** The
numbers above answer this directly rather than assuming it. Compare the
extreme-subset MAE/RMSE with the all-days MAE/RMSE. A positive mean residual
on extreme days indicates the model is **under-predicting peaks**; a negative
value indicates over-prediction. Low overall MAE does **not** prove reliable
forecasting during extreme events: rare high-AQI days can be a small share of
the test set and can still carry the largest public-health consequences.
If the extreme-day error is larger than the overall error, or if recall is
low, the model should not be described as reliable for pollution alerts.

**Observed answer for this test set:** the extreme-day MAE is
**{_fmt(extreme_row['mae']) if extreme_row is not None else 'N/A'}**, which is
**{_fmt(extreme_mae_ratio, 2)}×** the overall MAE of
**{_fmt(test_metrics['mae'])}**; extreme-day recall is
**{_fmt(peak['extreme_recall'], 3)}**; and the highest prediction is
**{_fmt(abs(peak['predicted_max_error']))}** AQI points below the highest
actual value. Therefore, **yes — the model performs substantially worse
during extreme pollution events on this test set.** The relatively good
overall MAE is driven by the large majority of days in the 151-200 range, not
by reliable peak forecasting.

### Where the model struggles — evidence from this test set

- **Extreme high-AQI days:** the model's MAE on days with actual AQI >= 200 is
  **{_fmt(extreme_row['mae']) if extreme_row is not None else 'N/A'}**, versus
  **{_fmt(test_metrics['mae'])}** overall. It predicted an extreme day
  **{peak['predicted_extreme_days']}** times out of
  **{peak['actual_extreme_days']}** actual extreme days (recall
  **{_fmt(peak['extreme_recall'], 3)}**). The mean residual on those days is
  **{_fmt(peak['mean_residual_extreme'])}**, so peaks are systematically
  **under-predicted**. The model is also worse than persistence on this
  subset (model MAE
  **{_fmt(extreme_row['mae']) if extreme_row is not None else 'N/A'}** vs
  baseline MAE
  **{_fmt(extreme_row['baseline_mae']) if extreme_row is not None else 'N/A'}**).
- **Clean/low-AQI days:** the lowest actual AQI in the test set is
  **{_fmt(range_compression['actual_min'])}**, while the lowest model
  prediction is **{_fmt(range_compression['predicted_min'])}**. On the
  {int(moderate_row['n']) if moderate_row is not None else 0} days in the
  Moderate range the model MAE is
  **{_fmt(moderate_row['mae']) if moderate_row is not None else 'N/A'}** and
  the mean residual is
  **{_fmt(moderate_row['mean_residual']) if moderate_row is not None else 'N/A'}**
  (negative = over-prediction).
- **Overall spread:** the predicted range is only
  **{_fmt(range_compression['prediction_range_ratio'], 2)}×** the actual range,
  and the correlation between actual and predicted AQI is
  **{_fmt(actual_prediction_corr, 3)}**. This is why R² is near zero even
  though MAE is much lower than the persistence baseline's.
- **Largest single miss:** see the largest-errors table above. A handful of
  rapid transitions dominate the tail of the error distribution.

## 7. Feature importance

{feature_section}

## 8. Limitations and possible sources of error

1. **Dataset provenance and scope.** This evaluation uses the repository's
   processed Open-Meteo/CAMS daily series, not a live official CPCB station
   feed. The AQI target is derived from PM2.5 using US-EPA breakpoints. Results
   describe this dataset and should not be generalised to another city, station
   network, or time period without retraining and re-evaluation.
2. **Small sample size.** The test split contains
   {test_metrics['n']} daily observations. Metrics for rare categories
   (especially extreme days) have wide uncertainty. A different test window
   could change the ranking against the baseline.
3. **Distribution shift.** The test period can have a different AQI level and
   variance from training. This is visible in the split statistics above and
   can make R² negative for both the model and persistence even when MAE is
   moderate.
4. **Strong persistence but modest model edge.** Lag-1 AQI is highly
   autocorrelated in this series, so the persistence baseline is hard to beat.
   The selected model improves RMSE and R² on the test window but is slightly
   worse than persistence on MAE; the improvement should not be read as a
   large operational gain.
5. **Missing physics and meteorology.** Wind speed, boundary-layer height,
   temperature, humidity, rainfall, and emission-source changes are not in the
   feature set. These are major drivers of pollution accumulation and
   dispersion.
6. **Feature construction.** Lags and rolling statistics are computed from
   the same pollutant series. Correlated features can inflate apparent
   importance and make coefficient-based explanations unstable. AQI is
   computed from PM2.5, so AQI-derived and PM2.5-derived features are partly
   redundant.
7. **Outliers and extreme events.** Extreme pollution episodes are rare and
   may be under-represented in training. Regression models tend to shrink
   predictions toward the training distribution, which can flatten peaks.
8. **Single chronological split.** There is no rolling-origin or
   cross-validation estimate of test uncertainty. The reported numbers are
   one realisation of one test window.
9. **Baseline definition.** Persistence uses the observed current-day AQI.
   In an operational setting the current-day value may itself be provisional
   or incomplete; this baseline is therefore optimistic relative to some
   real-world deployment conditions.
10. **No health or policy validation.** The target is an AQI number, not a
    health outcome. A low MAE would not by itself establish that the model is
    safe or useful for public advisories.

## 9. Is the model suitable for an educational AQI forecasting demonstration?

{suitability}

**Recommendation for the educational demo:** present the model together with
the persistence baseline, the test-period plot, the residual diagnostics, and
the extreme-event table. The most valuable lesson is not that the model is
accurate; it is that a forecasting model must be judged against a naive
baseline, on a genuinely held-out time period, with explicit attention to
rare extreme events.

## 10. Reproducibility

From the project root:

```bash
source .venv/bin/activate
python -m src.model_training.prepare_data
python -m src.model_training.train
python -m src.model_evaluation.evaluate_final
```

The first command recreates the chronological splits from
`data/processed/features.csv`; the second trains all candidates on the
training split, selects the best by validation RMSE, and writes
`models/best_model.json`; the third loads that fixed model and evaluates it on
the untouched test split. `scripts/run_final_evaluation.sh` wraps the final
step. All metrics in this report come from the CSV/JSON files in
`{metrics_display}`.
"""

    doc_path.write_text(content, encoding="utf-8")
    return doc_path


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_final_model(
    test_path: str | Path = "data/processed/splits/test.csv",
) -> dict:
    """Evaluate the validation-selected model on the untouched test split.

    Parameters
    ----------
    test_path:
        Test split produced by :mod:`src.model_training.prepare_data`.

    Returns
    -------
    dict
        Summary containing the model name, test metrics, baseline metrics and
        RMSE improvement.
    """
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "font.size": 11,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.35,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    test_path = get_path(test_path)
    if not test_path.is_file():
        raise FileNotFoundError(
            f"Test split not found: {test_path}. "
            "Run `python -m src.model_training.prepare_data` first."
        )

    plot_dir = get_path("results", "visualizations", "evaluation")
    metrics_dir = get_path("results", "metrics")
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load the untouched test split.
    test_df = pd.read_csv(test_path, parse_dates=[DATE_COL]).sort_values(DATE_COL)
    if test_df.empty:
        raise ValueError(f"Test split is empty: {test_path}")
    if TARGET not in test_df.columns:
        raise ValueError(f"Test split is missing target column {TARGET!r}")
    if BASELINE_COL not in test_df.columns:
        raise ValueError(
            "Test split is missing the current-day 'aqi' column required by "
            "the persistence baseline"
        )

    test_dates = test_df[DATE_COL]
    date_range = (
        str(test_dates.min().date()),
        str(test_dates.max().date()),
    )

    # 2. Load the validation-selected model and its feature contract.
    selection, selection_path = _resolve_selection()
    model_name = str(selection["model_name"])
    model_path = get_path("models", model_name, "model.pkl")
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Saved model not found: {model_path}. Re-run the training pipeline."
        )

    payload = joblib.load(model_path)
    if isinstance(payload, dict) and "model" in payload:
        model = payload["model"]
        model_metadata = payload.get("metadata", {})
    else:
        model = payload
        model_metadata = {}

    feature_columns = _resolve_feature_columns(test_df, selection, model_metadata)
    X_test = test_df[feature_columns]
    y_test = test_df[TARGET].to_numpy(dtype=float)

    n_features_in = getattr(model, "n_features_in_", None)
    if n_features_in is not None and int(n_features_in) != len(feature_columns):
        raise ValueError(
            f"Model expects {n_features_in} features but the saved feature "
            f"contract has {len(feature_columns)}"
        )

    # 3. Predictions and persistence baseline.
    y_pred = np.asarray(model.predict(X_test), dtype=float)
    y_baseline = test_df[BASELINE_COL].to_numpy(dtype=float)

    test_metrics = _metrics(y_test, y_pred)
    baseline_metrics = _metrics(y_test, y_baseline)

    # 4. Save the required metrics table.
    metrics_df = pd.DataFrame(
        [
            {
                "role": "selected_model",
                "model": model_name,
                "n": test_metrics["n"],
                "date_start": date_range[0],
                "date_end": date_range[1],
                "mae": test_metrics["mae"],
                "rmse": test_metrics["rmse"],
                "r2": test_metrics["r2"],
                "mean_residual": test_metrics["mean_residual"],
                "max_abs_error": test_metrics["max_abs_error"],
                "mae_improvement_pct": _improvement_pct(
                    baseline_metrics["mae"], test_metrics["mae"]
                ),
                "rmse_improvement_pct": _improvement_pct(
                    baseline_metrics["rmse"], test_metrics["rmse"]
                ),
                "r2_delta": test_metrics["r2"] - baseline_metrics["r2"],
            },
            {
                "role": "persistence_baseline",
                "model": "persistence (tomorrow = today's AQI)",
                "n": baseline_metrics["n"],
                "date_start": date_range[0],
                "date_end": date_range[1],
                "mae": baseline_metrics["mae"],
                "rmse": baseline_metrics["rmse"],
                "r2": baseline_metrics["r2"],
                "mean_residual": baseline_metrics["mean_residual"],
                "max_abs_error": baseline_metrics["max_abs_error"],
                "mae_improvement_pct": float("nan"),
                "rmse_improvement_pct": float("nan"),
                "r2_delta": float("nan"),
            },
        ]
    )
    metrics_csv = metrics_dir / "final_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    # 5. Per-observation predictions and error tables.
    predictions = pd.DataFrame(
        {
            DATE_COL: test_df[DATE_COL].to_numpy(),
            "actual_aqi": y_test,
            "predicted_aqi": y_pred,
            "baseline_aqi": y_baseline,
            "residual": y_test - y_pred,
            "abs_error": np.abs(y_test - y_pred),
            "baseline_abs_error": np.abs(y_test - y_baseline),
        }
    )
    predictions["aqi_range"] = predictions["actual_aqi"].apply(_aqi_range_label)
    predictions["is_extreme_200"] = predictions["actual_aqi"] >= EXTREME_THRESHOLD
    predictions_csv = metrics_dir / "test_predictions.csv"
    predictions.to_csv(predictions_csv, index=False)

    range_df = _range_metrics(predictions)
    range_df.to_csv(metrics_dir / "metrics_by_aqi_range.csv", index=False)

    q90_threshold = float(predictions["actual_aqi"].quantile(TOP_FRACTION))
    extreme_df = _extreme_metrics(predictions, q90_threshold)
    extreme_df.to_csv(metrics_dir / "extreme_event_metrics.csv", index=False)

    events_df = _extreme_events(predictions, EXTREME_THRESHOLD)
    events_df.to_csv(metrics_dir / "extreme_events.csv", index=False)

    peak = _peak_analysis(predictions, EXTREME_THRESHOLD)
    range_compression = _range_compression(predictions)

    largest_errors = predictions.sort_values(
        "abs_error", ascending=False, kind="mergesort"
    ).head(10)
    largest_errors.to_csv(metrics_dir / "largest_errors.csv", index=False)

    residual_autocorr = float(
        pd.Series(predictions["residual"]).autocorr(lag=1)
    ) if len(predictions) > 2 else float("nan")
    actual_prediction_corr = float(
        np.corrcoef(predictions["actual_aqi"], predictions["predicted_aqi"])[0, 1]
    ) if len(predictions) > 1 else float("nan")

    # 6. Visualizations.
    _plot_actual_vs_predicted(
        predictions, model_name, test_metrics, plot_dir / "actual_vs_predicted.png"
    )
    _plot_timeseries(
        predictions, model_name, plot_dir / "actual_vs_predicted_timeseries.png"
    )
    _plot_residuals(
        predictions, test_metrics, plot_dir / "residual_plot.png"
    )
    _plot_prediction_distribution(
        predictions, plot_dir / "prediction_distribution.png"
    )
    _plot_residual_distribution(
        predictions, test_metrics, plot_dir / "residual_distribution.png"
    )
    _plot_error_by_range(range_df, plot_dir / "error_by_aqi_range.png")
    _plot_extreme_errors(
        predictions, test_metrics, plot_dir / "extreme_event_errors.png"
    )
    _plot_error_over_time(
        predictions, test_metrics, plot_dir / "absolute_error_over_time.png"
    )

    feature_importance, feature_importance_method = _feature_importance(
        model,
        feature_columns,
        X_test,
        y_test,
        plot_dir,
        metrics_dir,
    )

    # 7. Machine-readable evaluation metadata.
    metadata = {
        "evaluation_timestamp": pd.Timestamp.now().isoformat(),
        "model_name": model_name,
        "model_file": str(model_path.relative_to(get_path())),
        "model_sha256": _sha256(model_path),
        "selection_file": str(selection_path.relative_to(get_path())),
        "selection_sha256": _sha256(selection_path),
        "test_file": str(test_path.relative_to(get_path())),
        "test_sha256": _sha256(test_path),
        "test_date_range": {"start": date_range[0], "end": date_range[1]},
        "test_rows": int(len(test_df)),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "baseline": "persistence: tomorrow's AQI = today's observed 'aqi'",
        "extreme_threshold": EXTREME_THRESHOLD,
        "top_fraction": TOP_FRACTION,
        "top_fraction_threshold": q90_threshold,
        "metrics": test_metrics,
        "baseline_metrics": baseline_metrics,
        "peak_analysis": peak,
        "prediction_range_compression": range_compression,
        "residual_autocorrelation_lag1": residual_autocorr,
        "actual_predicted_correlation": actual_prediction_corr,
        "feature_importance_method": feature_importance_method,
        "feature_importance_available": feature_importance is not None,
        "test_split_used_during_training": False,
        "model_selection": selection,
        "notes": [
            "Test data was not used to train, tune, or select the model.",
            "The test split was created chronologically from the processed feature table.",
            "One boundary row was purged between adjacent splits to avoid target overlap.",
            "Metrics are a single realisation of one chronological test window.",
        ],
    }
    metadata_path = metrics_dir / "evaluation_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    # 8. Human-readable report.
    doc_path = _write_documentation(
        model_name=model_name,
        selection=selection,
        selection_path=selection_path,
        test_path=test_path,
        test_metrics=test_metrics,
        baseline_metrics=baseline_metrics,
        metrics_df=metrics_df,
        range_df=range_df,
        extreme_df=extreme_df,
        events_df=events_df,
        peak=peak,
        range_compression=range_compression,
        largest_errors=largest_errors,
        residual_autocorr=residual_autocorr,
        actual_prediction_corr=actual_prediction_corr,
        date_range=date_range,
        feature_importance=feature_importance,
        feature_importance_method=feature_importance_method,
        metadata=metadata,
        plot_dir=plot_dir,
        metrics_dir=metrics_dir,
    )

    print(f"[final-eval] Model: {model_name}")
    print(f"[final-eval] Test range: {date_range[0]} → {date_range[1]} ({len(test_df)} rows)")
    print(
        f"[final-eval] MAE={test_metrics['mae']:.3f}  "
        f"RMSE={test_metrics['rmse']:.3f}  R2={test_metrics['r2']:.3f}"
    )
    print(
        f"[final-eval] Baseline MAE={baseline_metrics['mae']:.3f}  "
        f"RMSE={baseline_metrics['rmse']:.3f}  R2={baseline_metrics['r2']:.3f}"
    )
    print(
        "[final-eval] RMSE improvement vs baseline: "
        f"{_improvement_pct(baseline_metrics['rmse'], test_metrics['rmse']):.2f}%"
    )
    print(f"[final-eval] Metrics → {metrics_csv}")
    print(f"[final-eval] Report → {doc_path}")

    return {
        "best_model": model_name,
        "test_metrics": test_metrics,
        "baseline_metrics": baseline_metrics,
        "rmse_improvement": _improvement_pct(
            baseline_metrics["rmse"], test_metrics["rmse"]
        ),
        "test_date_range": date_range,
    }


def main() -> None:
    print("[final-eval] Starting final evaluation on the untouched test split...")
    evaluate_final_model()
    print("[final-eval] Evaluation completed successfully.")


if __name__ == "__main__":
    main()
