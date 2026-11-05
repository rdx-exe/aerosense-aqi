"""Comparative ML pipeline for next-day AQI prediction.

Implements and compares:
1. Persistence baseline
2. Linear Regression
3. Random Forest Regressor
4. HistGradientBoostingRegressor
5. XGBoost Regressor (if available)

Uses chronological splits (train/validation/test) from feature engineering.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

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
    """Create model pipeline registry."""
    models = {}

    # Linear Regression with scaling and imputation
    models["linear_regression"] = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("regressor", Ridge(random_state=42, alpha=1.0))
    ])

    # Random Forest
    models["random_forest"] = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    # Hist Gradient Boosting
    models["gradient_boosting"] = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=8,
        random_state=42
    )

    # XGBoost (if available)
    if XGB_AVAILABLE:
        models["xgboost"] = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
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
) -> dict:
    """Save trained models using Joblib."""
    saved_models = {}

    for model_name, result in val_results.items():
        model = result["model"]
        metadata = {
            "model_name": model_name,
            "training_metrics": result["train_metrics"],
            "validation_metrics": result["val_metrics"],
            "feature_importance": result["feature_importance"],
            "timestamp": pd.Timestamp.now().isoformat()
        }

        model_path = save_model_with_metadata(model, model_name, metadata)
        saved_models[model_name] = model_path

        print(f"[Model Training] Saved {model_name} to {model_path}")

    return saved_models


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
    if not data_path.is_file():
        print("[Model Training] ERROR: Training data not found. Run feature engineering first.")
        sys.exit(1)

    train_df = pd.read_csv(data_path, parse_dates=["date"])

    # Prepare features and target
    feature_cols = [
        c for c in train_df.columns
        if c not in ["date", "aqi_next_day", "city", "aqi_bucket", "season"]
        and not c.startswith("Unnamed")
    ]

    X = train_df[feature_cols]
    y = train_df["aqi_next_day"]

    # Load validation data
    val_path = get_path("data/processed/splits", "validation.csv")
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

    # Select best model
    best_model_name = select_best_model(val_results, baseline_metrics, logger=logger)

    # Save models
    saved_models = save_models(val_results, best_model_name, logger.experiment_dir)

    # Save experiment info
    logger.save_experiment_info()

    print("[Model Training] Experiment completed successfully.")
    print(f"[Model Training] Best model: {best_model_name}")
    print(f"[Model Training] Best validation RMSE: {val_results[best_model_name]['val_metrics']['rmse']:.2f}")


if __name__ == "__main__":
    main()