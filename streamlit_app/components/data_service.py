"""Cached data access and leakage-safe forecasting for the dashboard.

Every loader fails with an actionable :class:`DashboardDataError` instead of
returning fabricated values.  The model is loaded from disk; the dashboard
never trains a model at start-up.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.feature_engineering.features import build_inference_features, compute_aqi
from src.utils.config import get_path

from .theme import POLLUTANT_META, aqi_category

TARGET = "aqi_next_day"
DATE_COL = "date"
POLLUTANT_COLUMNS = tuple(POLLUTANT_META)


class DashboardDataError(RuntimeError):
    """Raised when a required dashboard artifact is missing or invalid."""


class InputValidationError(ValueError):
    """Raised when user-supplied pollutant inputs are invalid."""


@dataclass(frozen=True)
class ModelBundle:
    """Everything the dashboard needs to call the trained model safely."""

    model: Any
    metadata: dict[str, Any]
    selection: dict[str, Any]
    feature_columns: tuple[str, ...]
    model_name: str
    model_path: Path
    training_cutoff: pd.Timestamp | None


# ── Generic file helpers ─────────────────────────────────────────────────────


def _require_file(path: Path, hint: str) -> Path:
    if not path.is_file():
        raise DashboardDataError(f"Required file not found: {path}. {hint}")
    return path


def _read_json(path: Path, hint: str) -> dict[str, Any]:
    _require_file(path, hint)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DashboardDataError(f"Could not parse JSON file {path}: {exc}") from exc


def _read_csv(path: Path, hint: str, **kwargs: Any) -> pd.DataFrame:
    _require_file(path, hint)
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as exc:  # pragma: no cover - pandas error surface
        raise DashboardDataError(f"Could not read CSV file {path}: {exc}") from exc


# ── Historical data ──────────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def load_clean_data() -> pd.DataFrame:
    """Load the cleaned daily pollutant series (no AQI column).

    AQI is intentionally *not* added here: the prediction path must derive it
    from the (possibly user-edited) PM2.5 value using the training formula.
    """
    path = get_path("data", "processed", "clean.csv")
    df = _read_csv(
        path,
        "Run the data pipeline first: `python -m src.preprocessing.clean`.",
    )
    if DATE_COL not in df.columns:
        raise DashboardDataError(f"{path} must contain a '{DATE_COL}' column.")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL)
    df = df.drop_duplicates(subset=[DATE_COL], keep="last").set_index(DATE_COL)
    df.index.name = DATE_COL

    missing = [col for col in POLLUTANT_COLUMNS if col not in df.columns]
    if missing:
        raise DashboardDataError(
            f"{path} is missing required pollutant columns: {missing}."
        )

    df = df.loc[:, list(POLLUTANT_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    if df.empty:
        raise DashboardDataError("The cleaned dataset contains no usable rows.")

    return df


@st.cache_data(show_spinner=False)
def load_historical_data() -> pd.DataFrame:
    """Clean series plus AQI, category, and calendar helper columns."""
    df = load_clean_data().copy()
    df = compute_aqi(df)
    df["aqi_category"] = df["aqi"].apply(lambda value: aqi_category(value)["label"])
    df["aqi_color"] = df["aqi"].apply(lambda value: aqi_category(value)["color"])
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["month_name"] = df.index.strftime("%b")
    df["year_month"] = df.index.to_period("M").astype(str)
    return df


# ── Model and evaluation artifacts ───────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def load_model_bundle() -> ModelBundle:
    """Load the validation-selected model and its exact feature contract."""
    selection_path = get_path("models", "best_model.json")
    selection = _read_json(
        selection_path,
        "Run `./scripts/run_model_training.sh` to create the selection record.",
    )

    model_name = str(selection.get("model_name", "")).strip()
    if not model_name:
        raise DashboardDataError(f"{selection_path} does not record a model name.")

    model_path = get_path("models", model_name, "model.pkl")
    _require_file(
        model_path,
        "Train and save the selected model with `./scripts/run_model_training.sh`.",
    )

    try:
        payload = joblib.load(model_path)
    except Exception as exc:  # pragma: no cover - joblib error surface
        raise DashboardDataError(f"Could not load model {model_path}: {exc}") from exc

    if isinstance(payload, dict) and "model" in payload:
        model = payload["model"]
        metadata = dict(payload.get("metadata", {}))
    else:
        model = payload
        metadata = {}

    feature_columns = (
        selection.get("feature_columns")
        or metadata.get("feature_columns")
        or []
    )
    feature_columns = [str(col) for col in feature_columns]
    if not feature_columns:
        raise DashboardDataError(
            "The saved model has no recorded feature contract "
            "(models/best_model.json or model metadata)."
        )
    if len(set(feature_columns)) != len(feature_columns):
        raise DashboardDataError("The saved feature contract contains duplicates.")
    if TARGET in feature_columns:
        raise DashboardDataError("The target column must not be a model feature.")

    n_features_in = getattr(model, "n_features_in_", None)
    if n_features_in is not None and int(n_features_in) != len(feature_columns):
        raise DashboardDataError(
            "The saved model expects "
            f"{n_features_in} features but the feature contract lists "
            f"{len(feature_columns)}."
        )

    training_cutoff: pd.Timestamp | None = None
    split_info = selection.get("split_info") or metadata.get("split_info") or {}
    train_range = (split_info.get("date_ranges") or {}).get("train")
    if train_range and len(train_range) == 2:
        training_cutoff = pd.Timestamp(train_range[1])

    return ModelBundle(
        model=model,
        metadata=metadata,
        selection=selection,
        feature_columns=tuple(feature_columns),
        model_name=model_name,
        model_path=model_path,
        training_cutoff=training_cutoff,
    )


@st.cache_data(show_spinner=False)
def load_final_metrics() -> pd.DataFrame:
    """Model and persistence metrics from the final evaluation."""
    path = get_path("results", "metrics", "final_metrics.csv")
    df = _read_csv(
        path,
        "Run `./scripts/run_final_evaluation.sh` to generate final metrics.",
    )
    required = {"role", "model", "mae", "rmse", "r2"}
    missing = required.difference(df.columns)
    if missing:
        raise DashboardDataError(
            f"{path} is missing required columns: {sorted(missing)}."
        )
    return df


def metric_row(role: str) -> pd.Series:
    """Return one row of ``final_metrics.csv`` by role."""
    metrics = load_final_metrics()
    matches = metrics[metrics["role"] == role]
    if matches.empty:
        raise DashboardDataError(f"No metrics row with role '{role}'.")
    return matches.iloc[0]


@st.cache_data(show_spinner=False)
def load_test_predictions() -> pd.DataFrame:
    """Per-day test predictions produced by the final evaluation."""
    path = get_path("results", "metrics", "test_predictions.csv")
    df = _read_csv(
        path,
        "Run `./scripts/run_final_evaluation.sh` to generate test predictions.",
        parse_dates=[DATE_COL],
    )
    required = {"date", "actual_aqi", "predicted_aqi", "baseline_aqi", "residual"}
    missing = required.difference(df.columns)
    if missing:
        raise DashboardDataError(
            f"{path} is missing required columns: {sorted(missing)}."
        )
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    # ``date`` in the evaluation artifact is the feature/origin date; the
    # actual value belongs to the following day.
    df["origin_date"] = df[DATE_COL]
    df["target_date"] = df[DATE_COL] + pd.Timedelta(days=1)
    return df


@st.cache_data(show_spinner=False)
def load_feature_importance() -> pd.DataFrame:
    """Feature importance table, or an empty frame when unavailable."""
    path = get_path("results", "metrics", "feature_importance.csv")
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if {"feature", "importance"}.difference(df.columns):
        return pd.DataFrame()
    return df.sort_values("importance", ascending=False, kind="mergesort")


@st.cache_data(show_spinner=False)
def load_split_info() -> dict[str, Any]:
    """Chronological split metadata."""
    path = get_path("data", "processed", "splits", "split_info.json")
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


@st.cache_data(show_spinner=False)
def load_model_comparison() -> pd.DataFrame:
    """Validation metrics for every candidate plus the persistence baseline."""
    selection = load_model_bundle().selection
    candidates = selection.get("candidate_validation_metrics", {}) or {}
    baseline = selection.get("baseline_validation_metrics", {}) or {}

    rows: list[dict[str, Any]] = []
    if baseline:
        rows.append(
            {
                "model": "Persistence baseline",
                "mae": baseline.get("mae"),
                "rmse": baseline.get("rmse"),
                "r2": baseline.get("r2"),
                "selected": False,
                "kind": "baseline",
            }
        )
    for name, values in candidates.items():
        if not isinstance(values, dict):
            continue
        rows.append(
            {
                "model": name,
                "mae": values.get("mae"),
                "rmse": values.get("rmse"),
                "r2": values.get("r2"),
                "selected": name == selection.get("model_name"),
                "kind": "candidate",
            }
        )

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values("rmse", na_position="last").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_metrics_by_range() -> pd.DataFrame:
    """Error metrics grouped by actual AQI range."""
    path = get_path("results", "metrics", "metrics_by_aqi_range.csv")
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_extreme_metrics() -> pd.DataFrame:
    """Error metrics for extreme / top-decile subsets."""
    path = get_path("results", "metrics", "extreme_event_metrics.csv")
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_largest_errors() -> pd.DataFrame:
    """Largest individual test errors."""
    path = get_path("results", "metrics", "largest_errors.csv")
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    return df.sort_values("abs_error", ascending=False).reset_index(drop=True)


# ── Input validation and forecasting ─────────────────────────────────────────


def validate_pollutant_values(values: dict[str, Any]) -> dict[str, float]:
    """Validate and normalise user-supplied pollutant values.

    Raises
    ------
    InputValidationError
        If a value is missing, non-numeric, non-finite, negative, or outside
        the physically meaningful range used by the dashboard.
    """
    errors: list[str] = []
    cleaned: dict[str, float] = {}

    for column, meta in POLLUTANT_META.items():
        raw = values.get(column)
        if raw is None:
            errors.append(f"{meta['label']} is required.")
            continue
        try:
            number = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{meta['label']} must be a number.")
            continue
        if not math.isfinite(number):
            errors.append(f"{meta['label']} must be a finite number.")
            continue
        if number < meta["min"] or number > meta["max"]:
            errors.append(
                f"{meta['label']} must be between "
                f"{meta['min']:.1f} and {meta['max']:.1f} {meta['unit']}."
            )
            continue
        cleaned[column] = number

    if errors:
        raise InputValidationError(" ".join(errors))
    return cleaned


def out_of_training_range(values: dict[str, float]) -> list[str]:
    """Return human-readable warnings for inputs outside the training range."""
    bundle = load_model_bundle()
    clean = load_clean_data()
    cutoff = bundle.training_cutoff or clean.index.max()
    train = clean.loc[:cutoff]
    if train.empty:
        return []

    warnings: list[str] = []
    for column, value in values.items():
        if column not in train.columns:
            continue
        low = float(train[column].min())
        high = float(train[column].max())
        if value < low or value > high:
            meta = POLLUTANT_META[column]
            warnings.append(
                f"{meta['label']} = {value:,.1f} {meta['unit']} is outside the "
                f"training range ({low:,.1f}–{high:,.1f} {meta['unit']}). "
                "Treat this scenario as an extrapolation."
            )
    return warnings


def origin_dates() -> list[pd.Timestamp]:
    """Available forecast origins, newest first."""
    return list(load_clean_data().index.sort_values(ascending=False))


def predict_next_day(
    origin_date: str | pd.Timestamp,
    pollutant_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a next-day forecast using only data up to ``origin_date``.

    The history is truncated *before* feature construction, so no future
    observation can influence the features or the prediction.  The next-day
    actual value (when it exists because the origin is a historical date) is
    returned only as an evaluation reference and is never passed to the model.
    """
    clean = load_clean_data()
    origin_ts = pd.Timestamp(origin_date).normalize()
    if origin_ts not in clean.index:
        raise InputValidationError(
            f"{origin_ts.date()} is not an available observation date."
        )

    history = clean.loc[:origin_ts]
    if len(history) < 8:
        raise InputValidationError(
            "At least 8 daily observations are required before the origin "
            "date to build the 7-day lag features."
        )

    scenario_values: dict[str, float]
    if pollutant_values is None:
        scenario_values = {
            column: float(history[column].iloc[-1])
            for column in POLLUTANT_COLUMNS
        }
    else:
        scenario_values = validate_pollutant_values(pollutant_values)

    scenario = history.copy()
    for column, value in scenario_values.items():
        scenario.loc[origin_ts, column] = value

    try:
        features = build_inference_features(scenario, origin_date=origin_ts)
    except ValueError as exc:
        raise InputValidationError(str(exc)) from exc

    bundle = load_model_bundle()
    missing = [column for column in bundle.feature_columns if column not in features.columns]
    if missing:
        raise DashboardDataError(
            "The feature pipeline did not produce columns required by the "
            f"model: {missing[:5]}{'…' if len(missing) > 5 else ''}."
        )

    # Exact training order.  Never rely on the incoming column order.
    X = features.loc[:, list(bundle.feature_columns)]
    if X.shape != (1, len(bundle.feature_columns)):
        raise DashboardDataError(
            "Unexpected inference matrix shape: "
            f"{X.shape} (expected 1 × {len(bundle.feature_columns)})."
        )
    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise DashboardDataError(
            f"Inference features contain missing values: {bad[:5]}"
            f"{'…' if len(bad) > 5 else ''}."
        )

    try:
        raw_prediction = bundle.model.predict(X)
    except Exception as exc:  # pragma: no cover - sklearn error surface
        raise DashboardDataError(f"The model could not generate a prediction: {exc}") from exc

    predicted = float(np.asarray(raw_prediction).reshape(-1)[0])
    if not math.isfinite(predicted):
        raise DashboardDataError("The model returned a non-finite prediction.")

    derived_aqi = float(features["aqi"].iloc[0])
    if not math.isfinite(derived_aqi):
        raise InputValidationError(
            "The supplied PM2.5 value cannot be converted to AQI. "
            "PM2.5 must be within the US-EPA breakpoint range (0–500 µg/m³)."
        )

    target_date = origin_ts + pd.Timedelta(days=1)
    actual_next: float | None = None
    if target_date in clean.index:
        # Reference only.  This value is computed after the prediction and is
        # never part of the feature matrix.
        next_row = clean.loc[[target_date]]
        actual_next = float(compute_aqi(next_row)["aqi"].iloc[0])

    category = aqi_category(predicted)

    return {
        "model_name": bundle.model_name,
        "origin_date": origin_ts,
        "target_date": target_date,
        "predicted_aqi": predicted,
        "category": category,
        "derived_aqi": derived_aqi,
        "delta_vs_origin": predicted - derived_aqi,
        "scenario_values": scenario_values,
        "actual_next_aqi": actual_next,
        "warnings": out_of_training_range(scenario_values),
        "feature_count": len(bundle.feature_columns),
        "training_cutoff": bundle.training_cutoff,
        "feature_row": features.iloc[0],
    }
