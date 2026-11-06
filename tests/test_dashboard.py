"""Tests for the Streamlit dashboard: data contract, leakage safety and UI.

These tests exercise the real saved artifacts.  They are skipped only when the
model/evaluation files are absent, so a fresh clone with the full pipeline
output can run the whole suite.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.feature_engineering.features import (
    build_features,
    build_inference_features,
    compute_aqi,
)
from streamlit_app.components import charts, data_service
from streamlit_app.components.theme import (
    AQI_BANDS,
    aqi_category,
    data_table,
    inject_css,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = PROJECT_ROOT / "streamlit_app" / "app.py"

pytestmark = pytest.mark.skipif(
    not (PROJECT_ROOT / "models" / "best_model.json").is_file(),
    reason="Dashboard tests need trained model artifacts.",
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def clean() -> pd.DataFrame:
    return data_service.load_clean_data()


@pytest.fixture(scope="module")
def historical() -> pd.DataFrame:
    return data_service.load_historical_data()


@pytest.fixture(scope="module")
def bundle() -> data_service.ModelBundle:
    return data_service.load_model_bundle()


@pytest.fixture(scope="module")
def test_predictions() -> pd.DataFrame:
    return data_service.load_test_predictions()


# ── Historical data ─────────────────────────────────────────────────────────


def test_clean_dataset_contract(clean: pd.DataFrame) -> None:
    assert not clean.empty
    assert isinstance(clean.index, pd.DatetimeIndex)
    assert clean.index.is_monotonic_increasing
    assert clean.index.is_unique
    assert list(clean.columns) == list(data_service.POLLUTANT_COLUMNS)
    assert not clean.isna().any().any()
    assert (clean >= 0).all().all()


def test_historical_aqi_is_derived_from_pm25(
    clean: pd.DataFrame, historical: pd.DataFrame
) -> None:
    derived = compute_aqi(clean)["aqi"]
    pd.testing.assert_series_equal(
        historical["aqi"], derived, check_names=False, rtol=0, atol=1e-9
    )
    assert set(historical["aqi_category"]) <= {band["label"] for band in AQI_BANDS}
    assert historical["aqi"].between(0, 500).all()


def test_historical_calendar_columns(historical: pd.DataFrame) -> None:
    assert historical["year"].equals(historical.index.year.to_series(index=historical.index))
    assert historical["month"].equals(historical.index.month.to_series(index=historical.index))
    assert historical["month_name"].str.len().eq(3).all()


# ── Model contract ──────────────────────────────────────────────────────────


def test_model_bundle_contract(bundle: data_service.ModelBundle) -> None:
    assert bundle.model_path.is_file()
    assert len(bundle.feature_columns) == 119
    assert len(set(bundle.feature_columns)) == len(bundle.feature_columns)
    assert data_service.TARGET not in bundle.feature_columns
    assert bundle.model.n_features_in_ == len(bundle.feature_columns)
    assert bundle.training_cutoff is not None
    assert bundle.training_cutoff <= data_service.load_clean_data().index.max()


def test_metrics_and_comparison_artifacts() -> None:
    metrics = data_service.load_final_metrics()
    assert {"role", "model", "mae", "rmse", "r2"} <= set(metrics.columns)
    roles = set(metrics["role"])
    assert {"selected_model", "persistence_baseline"} <= roles

    comparison = data_service.load_model_comparison()
    assert not comparison.empty
    assert comparison["selected"].sum() == 1
    assert comparison.loc[comparison["selected"], "model"].iloc[0] == data_service.load_model_bundle().model_name


def test_test_predictions_artifact(test_predictions: pd.DataFrame) -> None:
    assert not test_predictions.empty
    assert test_predictions["date"].is_monotonic_increasing
    assert not test_predictions[["actual_aqi", "predicted_aqi"]].isna().any().any()
    assert np.allclose(
        test_predictions["actual_aqi"] - test_predictions["predicted_aqi"],
        test_predictions["residual"],
    )


# ── Category logic ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "label"),
    [
        (0, "Good"),
        (50, "Good"),
        (51, "Moderate"),
        (100, "Moderate"),
        (101, "Unhealthy for Sensitive Groups"),
        (150, "Unhealthy for Sensitive Groups"),
        (151, "Unhealthy"),
        (200, "Unhealthy"),
        (201, "Very Unhealthy"),
        (300, "Very Unhealthy"),
        (301, "Hazardous"),
        (500, "Hazardous"),
        (501, "Beyond Hazardous"),
    ],
)
def test_aqi_category_boundaries(value: float, label: str) -> None:
    assert aqi_category(value)["label"] == label


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), "abc"])
def test_aqi_category_missing_values_are_total(value: object) -> None:
    assert aqi_category(value)["label"] == "Unknown"


def test_aqi_category_rejects_negative() -> None:
    assert aqi_category(-1)["label"] == "Invalid"


# ── Feature pipeline and leakage safety ────────────────────────────────────


def test_inference_features_match_training_features(
    clean: pd.DataFrame, bundle: data_service.ModelBundle
) -> None:
    training = build_features(clean)
    for origin in ["2022-06-01", "2023-09-13", "2023-12-30"]:
        row = build_inference_features(clean, origin_date=origin)
        assert list(row.columns) == [c for c in training.columns if c != data_service.TARGET]
        common = [c for c in row.columns if c in training.columns]
        difference = (row.iloc[0][common] - training.loc[pd.Timestamp(origin), common]).abs()
        assert float(difference.max()) < 1e-8
        # Every model feature is present, in any incoming order.
        assert set(bundle.feature_columns) <= set(row.columns)
        assert data_service.TARGET not in row.columns


def test_inference_ignores_future_observations(clean: pd.DataFrame) -> None:
    origin = pd.Timestamp("2023-06-15")
    original = build_inference_features(clean, origin_date=origin)

    tampered = clean.copy()
    future = tampered.index > origin
    assert future.any()
    tampered.loc[future, :] = tampered.loc[future, :] * 100.0

    altered = build_inference_features(tampered, origin_date=origin)
    pd.testing.assert_frame_equal(original, altered, check_freq=False)


def test_inference_requires_enough_history(clean: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="No observations|Not enough complete history"):
        build_inference_features(clean, origin_date="2021-12-31")
    with pytest.raises(ValueError, match="No observation exists"):
        build_inference_features(clean, origin_date="2023-07-15 12:00:00")


def test_inference_recomputes_aqi_from_edited_pm25(clean: pd.DataFrame) -> None:
    origin = pd.Timestamp("2023-12-30")
    edited = clean.copy()
    edited.loc[origin, "pm25"] = 250.0
    row = build_inference_features(edited, origin_date=origin)
    assert math.isclose(float(row["aqi"].iloc[0]), float(compute_aqi(edited.loc[[origin]])["aqi"].iloc[0]))
    assert float(row["pm25_lag1"].iloc[0]) != 250.0  # lag uses the true previous day


# ── Prediction ──────────────────────────────────────────────────────────────


def test_prediction_uses_saved_model_and_feature_order(
    clean: pd.DataFrame, bundle: data_service.ModelBundle
) -> None:
    origin = clean.index.max()
    result = data_service.predict_next_day(origin)
    features = build_inference_features(clean, origin_date=origin)
    expected = float(bundle.model.predict(features.loc[:, list(bundle.feature_columns)])[0])

    assert math.isclose(result["predicted_aqi"], expected, rel_tol=0, abs_tol=1e-9)
    assert result["origin_date"] == origin
    assert result["target_date"] == origin + pd.Timedelta(days=1)
    assert result["feature_count"] == len(bundle.feature_columns)
    assert math.isfinite(result["predicted_aqi"])
    assert result["actual_next_aqi"] is None  # no observation exists after the last day


def test_prediction_is_deterministic(clean: pd.DataFrame) -> None:
    origin = pd.Timestamp("2023-09-13")
    first = data_service.predict_next_day(origin)
    second = data_service.predict_next_day(origin)
    # RandomForest averages trees with a parallel reduction; the sum order can
    # change the last floating-point bit, so compare numerically rather than
    # requiring bit-identical floats.
    assert math.isclose(
        first["predicted_aqi"], second["predicted_aqi"], rel_tol=0, abs_tol=1e-9
    )
    assert first["scenario_values"] == second["scenario_values"]


def test_prediction_matches_saved_test_evaluation(test_predictions: pd.DataFrame) -> None:
    sample = test_predictions.iloc[[0, len(test_predictions) // 2, -1]]
    for row in sample.itertuples():
        result = data_service.predict_next_day(row.date)
        assert math.isclose(result["predicted_aqi"], float(row.predicted_aqi), abs_tol=1e-6), row.date


def test_prediction_actual_next_is_reference_only(clean: pd.DataFrame) -> None:
    origin = pd.Timestamp("2023-11-20")
    result = data_service.predict_next_day(origin)
    expected_actual = float(compute_aqi(clean.loc[[origin + pd.Timedelta(days=1)]])["aqi"].iloc[0])
    assert result["actual_next_aqi"] is not None
    # The reference value is not a model feature.
    assert "aqi_next_day" not in result["feature_row"].index
    assert math.isclose(result["actual_next_aqi"], expected_actual)


def test_prediction_what_if_changes_forecast(clean: pd.DataFrame) -> None:
    origin = pd.Timestamp("2023-12-30")
    observed = {c: float(clean.loc[origin, c]) for c in data_service.POLLUTANT_COLUMNS}
    baseline = data_service.predict_next_day(origin)
    elevated = dict(observed, pm25=300.0)
    scenario = data_service.predict_next_day(origin, elevated)
    assert scenario["predicted_aqi"] != baseline["predicted_aqi"]
    assert scenario["scenario_values"]["pm25"] == 300.0
    assert any("outside the training range" in warning for warning in scenario["warnings"])


@pytest.mark.parametrize("origin", ["2024-01-15", "1900-01-01"])
def test_prediction_rejects_unknown_origin(origin: str) -> None:
    with pytest.raises(data_service.InputValidationError):
        data_service.predict_next_day(origin)


# ── Input validation ────────────────────────────────────────────────────────


def test_validate_pollutant_values_accepts_valid_input() -> None:
    values = {"pm25": 80, "pm10": 120, "no2": 30, "so2": 10, "o3": 40, "co": 0.8}
    assert data_service.validate_pollutant_values(values) == {
        key: float(value) for key, value in values.items()
    }


@pytest.mark.parametrize(
    "override",
    [
        {"pm25": -1},
        {"pm25": 501},
        {"pm10": 2000},
        {"no2": float("nan")},
        {"so2": float("inf")},
        {"o3": "not-a-number"},
    ],
)
def test_validate_pollutant_values_rejects_bad_values(override: dict) -> None:
    values = {"pm25": 80, "pm10": 120, "no2": 30, "so2": 10, "o3": 40, "co": 0.8}
    with pytest.raises(data_service.InputValidationError):
        data_service.validate_pollutant_values({**values, **override})


def test_validate_pollutant_values_requires_every_pollutant() -> None:
    with pytest.raises(data_service.InputValidationError, match="required"):
        data_service.validate_pollutant_values({"pm25": 80})


# ── Charts ──────────────────────────────────────────────────────────────────


def _assert_figure(figure: go.Figure) -> None:
    assert isinstance(figure, go.Figure)
    assert figure.layout.height and figure.layout.height > 0


def test_charts_render_with_real_data(historical: pd.DataFrame, bundle: data_service.ModelBundle) -> None:
    predictions = data_service.load_test_predictions()
    comparison = data_service.load_model_comparison()
    importance = data_service.load_feature_importance()
    by_range = data_service.load_metrics_by_range()
    origin = historical.index.max()
    target = origin + pd.Timedelta(days=1)

    figures = [
        charts.aqi_trend(historical),
        charts.monthly_aqi(historical),
        charts.monthly_pollutant_grid(historical),
        charts.category_donut(historical),
        charts.pollutant_correlation(historical),
        charts.pollutant_scatter(historical, "pm25"),
        charts.forecast_history(historical, origin, target, 160.0),
        charts.aqi_gauge(160.0),
        charts.actual_vs_predicted(predictions),
        charts.residual_chart(predictions),
        charts.model_comparison_chart(comparison),
        charts.feature_importance_chart(importance),
        charts.error_by_range_chart(by_range),
        charts.monthly_heatmap(historical),
        charts.high_pollution_timeline(historical),
    ]
    for figure in figures:
        _assert_figure(figure)
        assert len(figure.data) > 0


def test_charts_handle_empty_data_gracefully() -> None:
    empty = pd.DataFrame()
    figures = [
        charts.aqi_trend(empty),
        charts.monthly_aqi(empty),
        charts.monthly_pollutant_grid(empty),
        charts.category_donut(empty),
        charts.pollutant_correlation(empty),
        charts.pollutant_scatter(empty, "pm25"),
        charts.forecast_history(empty, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), 100),
        charts.actual_vs_predicted(empty),
        charts.residual_chart(empty),
        charts.model_comparison_chart(empty),
        charts.feature_importance_chart(empty),
        charts.error_by_range_chart(empty),
        charts.monthly_heatmap(empty),
        charts.high_pollution_timeline(empty),
    ]
    for figure in figures:
        _assert_figure(figure)
        assert len(figure.data) == 0


def test_forecast_chart_separates_forecast_from_history(historical: pd.DataFrame) -> None:
    origin = pd.Timestamp("2023-12-30")
    figure = charts.forecast_history(historical, origin, origin + pd.Timedelta(days=1), 175.0)
    names = [trace.name for trace in figure.data]
    assert "Historical observation" in names
    assert "Model forecast" in names
    # The forecast trace is labelled as model-generated in hover text.
    forecast = next(trace for trace in figure.data if trace.name == "Model forecast")
    assert "not observed" in str(forecast.hovertemplate)


def test_dashboard_uses_only_inter_and_jetbrains_mono() -> None:
    css = inject_css()
    assert "@import" in css
    assert "family=Inter" in css
    assert "JetBrains+Mono" in css
    assert "font-family: 'Inter'" in css
    assert "font-family: 'JetBrains Mono'" in css
    for forbidden in (
        "Segoe UI",
        "Roboto",
        "Arial",
        "Helvetica",
        "system-ui",
        "Source Sans",
    ):
        assert forbidden not in css
    # Plotly charts must use the same sans-serif typeface.
    assert charts.FONT_FAMILY.startswith("Inter")
    assert "Segoe" not in charts.FONT_FAMILY
    # BaseWeb select menus render in a portal outside .stApp.
    assert '[data-baseweb="popover"] *' in css
    assert '[data-baseweb="menu"] *' in css


def test_tables_use_theme_fonts_and_escape_text() -> None:
    frame = pd.DataFrame(
        {
            "Category": ["Good <script>"],
            "Days": [3],
            "Mean AQI": [42.5],
            "Selected": [np.True_],
        }
    )
    markup = data_table(frame)
    assert 'class="aqi-table"' in markup
    assert "Good &lt;script&gt;" in markup
    assert ">3<" in markup
    assert ">42.5<" in markup
    assert ">Yes<" in markup
    assert 'class="num"' in markup
    # No canvas dataframe is used, so cell glyphs stay in the theme fonts.
    assert "stDataFrame" not in markup


# ── Streamlit app ───────────────────────────────────────────────────────────


def test_app_renders_every_section_without_errors() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP_PATH), default_timeout=120)
    app.run()
    assert not app.exception, [str(e.value) for e in app.exception]
    assert not app.error, [str(e.value) for e in app.error]

    sections = [
        "Overview",
        "Historical air quality",
        "AQI prediction",
        "Model performance",
        "Environmental insights",
        "Project information",
    ]
    assert app.sidebar.radio[0].options == sections

    for section in sections:
        app.sidebar.radio[0].set_value(section).run()
        assert not app.exception, (section, [str(e.value) for e in app.exception])
        assert not app.error, (section, [str(e.value) for e in app.error])


def test_app_prediction_controls() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP_PATH), default_timeout=120)
    app.run()
    app.sidebar.radio[0].set_value("AQI prediction").run()
    assert not app.exception

    # The forecast card identifies target, origin and model provenance.
    markdown = "\n".join(block.value for block in app.markdown)
    assert "Model-generated next-day forecast" in markdown
    assert "Forecast target" in markdown
    assert "Forecast origin" in markdown

    app.selectbox[0].set_value(pd.Timestamp("2023-12-30")).run()
    assert not app.exception
    assert not app.error

    app.toggle[0].set_value(True).run()
    assert not app.exception
    assert len(app.number_input) == len(data_service.POLLUTANT_COLUMNS)

    app.number_input[0].set_value(300.0).run()
    assert not app.exception
    assert not app.error
    assert any("outside the training range" in warning.value for warning in app.warning)


def test_app_historical_controls() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP_PATH), default_timeout=120)
    app.run()
    app.sidebar.radio[0].set_value("Historical air quality").run()
    assert not app.exception

    window = next(radio for radio in app.radio if radio.label == "Time window")
    window.set_value("Last 90 days").run()
    assert not app.exception
    assert not app.error

    app.selectbox[0].set_value("o3").run()
    assert not app.exception
    assert not app.error
