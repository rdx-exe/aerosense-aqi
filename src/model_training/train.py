"""Comparative ML pipeline for next-day AQI prediction.

Implements and compares:
1. Persistence baseline
2. Ridge regression on selected features
3. Random Forest Regressor
4. Extra Trees Regressor
5. Gradient Boosting Regressor (Huber loss)
6. Support Vector Regressor
7. XGBoost Regressor (if available)

Uses chronological splits (train/validation/test) from feature engineering.
Hyperparameters were selected with expanding-window time-series
cross-validation on the training split only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR

from ..utils.config import get_path, load_config
from ..utils.io import load_model, save_model
from .experiment_utils import ExperimentLogger, compare_to_baseline, save_model_with_metadata

# Try to import XGBoost (optional)
try:
    from xgboost import XGBRegressor

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


def persistence_baseline(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """Persistence model: tomorrow's AQI equals today's AQI."""
    return X["aqi"].values


def create_models(config: dict) -> dict:
    """Create model pipeline registry.

    Hyperparameters were selected with expanding-window ``TimeSeriesSplit``
    cross-validation on the training split only (see
    ``experiments/tuning_summary.json``).  The validation split is left for
    model selection and the test split is never touched during training.
    """
    models = {}

    # Linear regression on the 20 most informative features.  A random-forest
    # importance filter removes the many collinear lag/rolling columns, which
    # otherwise make an unregularised linear model unstable on this series.
    feature_selector = SelectFromModel(
        RandomForestRegressor(
            n_estimators=500,
            max_features=0.5,
            min_samples_leaf=3,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1,
        ),
        threshold=-np.inf,
        max_features=20,
    )
    models["linear_regression"] = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("selector", feature_selector),
        ("scaler", StandardScaler()),
        ("regressor", Ridge(random_state=42, alpha=10.0)),
    ])

    # Random Forest (tuned)
    models["random_forest"] = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        max_features=0.5,
        min_samples_leaf=3,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1,
    )

    # Extra Trees (tuned)
    models["extra_trees"] = ExtraTreesRegressor(
        n_estimators=500,
        max_depth=10,
        max_features=1.0,
        min_samples_leaf=5,
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )

    # Gradient Boosting with a robust Huber loss (tuned)
    models["gradient_boosting"] = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=2,
        min_samples_leaf=20,
        subsample=0.7,
        max_features=0.3,
        loss="huber",
        random_state=42,
    )

    # Support Vector Regression (scaled, tuned)
    models["svr"] = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("regressor", SVR(C=10.0, gamma="scale", epsilon=0.1)),
    ])

    # XGBoost (if available)
    if XGB_AVAILABLE:
        models["xgboost"] = XGBRegressor(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.5,
            min_child_weight=5,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
        )

    return models


def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    models: dict,
    logger: ExperimentLogger,
    config: dict,
) -> tuple[dict, dict]:
    """Train models and evaluate on train/validation sets."""
    results = {}
    predictions = {}

    for model_name, model in models.items():
        logger.logger.info(f"Training {model_name}...")

        # Fit model
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        # Metrics
        train_metrics = {
            "mae": mean_absolute_error(y_train, y_train_pred),
            "rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            "r2": r2_score(y_train, y_train_pred)
        }

        val_metrics = {
            "mae": mean_absolute_error(y_val, y_val_pred),
            "rmse": np.sqrt(mean_squared_error(y_val, y_val_pred)),
            "r2": r2_score(y_val, y_val_pred)
        }

        # Feature importance (for tree-based models)
        feature_importance = {}
        if hasattr(model, "feature_importances_"):
            feature_importance = dict(
                zip(X_train.columns, model.feature_importances_)
            )

        results[model_name] = {
            "model": model,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "feature_importance": feature_importance
        }

        predictions[model_name] = y_val_pred

        # Log results
        logger.log_model_metrics(
            model_name,
            model,
            train_metrics,
            val_metrics,
            feature_importance,
            config.get("model_params", {}).get(model_name, {})
        )

    return results, predictions


def select_best_model(
    val_results: dict,
    baseline_metrics: dict,
    improvement_threshold: float = 0.05,
    logger=None,
) -> str:
    """Select best model based on validation RMSE with improvement over baseline."""
    best_model = None
    best_rmse = float("inf")

    # Check all models
    for model_name, metrics in val_results.items():
        rmse = metrics["val_metrics"]["rmse"]

        # Check if model beats baseline
        baseline_rmse = baseline_metrics["rmse"]
        improvement = (baseline_rmse - rmse) / baseline_rmse

        if improvement > improvement_threshold:
            if logger:
                logger.logger.info(
                    f"{model_name}: RMSE={rmse:.2f} (improvement: {improvement:.2%})"
                )

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model_name

    if best_model is None:
        # Fallback to best RMSE even without significant improvement
        for model_name, metrics in val_results.items():
            rmse = metrics["val_metrics"]["rmse"]
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model_name

        if logger:
            logger.logger.info(
                f"No model significantly beats baseline; selecting best RMSE: {best_model} (RMSE={best_rmse:.2f})"
            )

    else:
        if logger:
            logger.logger.info(
                f"Selected best model: {best_model} (RMSE={best_rmse:.2f})"
            )

    return best_model


def save_models(
    val_results: dict,
    best_model_name: str,
    experiment_dir: Path,
    feature_names: list[str],
    split_info: dict | None = None,
) -> dict:
    """Save trained models using Joblib, including the exact feature contract.

    The feature column list is persisted with every model so that evaluation
    can rebuild the test matrix in the same order without guessing from the
    test file.
    """
    saved_models = {}

    for model_name, result in val_results.items():
        model = result["model"]
        metadata = {
            "model_name": model_name,
            "training_metrics": result["train_metrics"],
            "validation_metrics": result["val_metrics"],
            "feature_importance": result["feature_importance"],
            "feature_columns": list(feature_names),
            "n_features": len(feature_names),
            "selected_best_model": model_name == best_model_name,
            "test_split_used_during_training": False,
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        if split_info is not None:
            metadata["split_info"] = {
                "split_rows": split_info.get("split_rows"),
                "date_ranges": split_info.get("date_ranges"),
                "purged_boundary_rows": split_info.get("purged_boundary_rows"),
            }

        model_path = save_model_with_metadata(model, model_name, metadata)
        saved_models[model_name] = model_path

        print(f"[Model Training] Saved {model_name} to {model_path}")

    return saved_models


def save_best_model_selection(
    best_model_name: str,
    val_results: dict,
    baseline_metrics: dict,
    feature_names: list[str],
    split_info: dict | None,
    experiment_dir: Path,
) -> Path:
    """Persist the validation-only model-selection record.

    This file is the contract used by the final evaluation script.  It records
    the selected model, the validation metrics, and the exact feature list.
    No test-set metric is present in this file.
    """
    best_val = val_results[best_model_name]["val_metrics"]
    baseline_rmse = float(baseline_metrics["rmse"])
    val_rmse = float(best_val["rmse"])
    improvement = (
        (baseline_rmse - val_rmse) / baseline_rmse * 100.0
        if baseline_rmse > 0
        else float("nan")
    )

    selection = {
        "model_name": best_model_name,
        "selected_by": "validation_rmse",
        "selection_rule": (
            "Lowest validation RMSE among candidates that improve the "
            "persistence baseline by more than 5%; if none do, fall back to "
            "the lowest validation RMSE."
        ),
        "validation_metrics": best_val,
        "baseline_validation_metrics": baseline_metrics,
        "validation_rmse_improvement_pct": improvement,
        "candidate_validation_metrics": {
            name: result["val_metrics"] for name, result in val_results.items()
        },
        "feature_columns": list(feature_names),
        "n_features": len(feature_names),
        "test_split_used_during_training": False,
        "selected_at": pd.Timestamp.now().isoformat(),
    }
    if split_info is not None:
        selection["split_info"] = {
            "split_rows": split_info.get("split_rows"),
            "date_ranges": split_info.get("date_ranges"),
            "target_stats": split_info.get("target_stats"),
            "purged_boundary_rows": split_info.get("purged_boundary_rows"),
            "purge_policy": split_info.get("purge_policy"),
        }

    best_path = get_path("models", "best_model.json")
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"[Model Training] Best-model selection record → {best_path}")

    experiment_selection_path = experiment_dir / "final_selection.json"
    experiment_selection_path.write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    print(
        "[Model Training] Experiment selection record → "
        f"{experiment_selection_path}"
    )

    return best_path


def main() -> None:
    """Main training pipeline."""
    # Setup experiment
    experiment_name = "aqi_forecasting"
    logger = ExperimentLogger(experiment_name)

    # Load configuration
    config = load_config()

    # Log experiment start
    logger.log_experiment_config(config)

    # Load training data
    data_path = get_path("data/processed/splits", "train.csv")
    val_path = get_path("data/processed/splits", "validation.csv")
    if not data_path.is_file() or not val_path.is_file():
        print(
            "[Model Training] ERROR: train/validation split not found. "
            "Run `python -m src.model_training.prepare_data` first."
        )
        sys.exit(1)

    split_info_path = get_path("data/processed/splits", "split_info.json")
    split_info = None
    if split_info_path.is_file():
        split_info = json.loads(split_info_path.read_text(encoding="utf-8"))

    train_df = pd.read_csv(data_path, parse_dates=["date"])

    # Prepare features and target
    feature_cols = [
        c for c in train_df.columns
        if c not in ["date", "aqi_next_day", "city", "aqi_bucket", "season", "index"]
        and not c.startswith("Unnamed")
    ]

    X = train_df[feature_cols]
    y = train_df["aqi_next_day"]

    # Load validation data
    val_df = pd.read_csv(val_path, parse_dates=["date"])

    X_val = val_df[feature_cols]
    y_val = val_df["aqi_next_day"]

    # Create models
    models = create_models(config)

    # Train and evaluate
    val_results, val_predictions = train_and_evaluate(
        X, y, X_val, y_val, models, logger, config
    )

    # Calculate baseline metrics
    baseline_pred = persistence_baseline(X_val, y_val)
    baseline_metrics = compare_to_baseline(baseline_pred, y_val.values, val_predictions, y_val.values)["persistence_baseline"]

    # Log comparison table
    model_comparison = [
        {"name": name, **metrics["val_metrics"]}
        for name, metrics in val_results.items()
    ]
    logger.log_comparison_table(baseline_metrics, model_comparison)

    # Select best model using validation performance only
    best_model_name = select_best_model(val_results, baseline_metrics, logger=logger)

    # Save models and the validation-only selection record
    saved_models = save_models(
        val_results,
        best_model_name,
        logger.experiment_dir,
        feature_cols,
        split_info=split_info,
    )
    save_best_model_selection(
        best_model_name,
        val_results,
        baseline_metrics,
        feature_cols,
        split_info,
        logger.experiment_dir,
    )

    # Save experiment info
    logger.experiment_info["best_model"] = best_model_name
    logger.experiment_info["feature_columns"] = feature_cols
    logger.experiment_info["test_split_used_during_training"] = False
    logger.save_experiment_info()

    print("[Model Training] Experiment completed successfully.")
    print(f"[Model Training] Best model: {best_model_name}")
    print(f"[Model Training] Best validation RMSE: {val_results[best_model_name]['val_metrics']['rmse']:.2f}")


if __name__ == "__main__":
    main()