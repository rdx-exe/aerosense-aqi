"""Model performance section: honest test-set metrics and diagnostics."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .charts import (
    PLOTLY_CONFIG,
    actual_vs_predicted,
    error_by_range_chart,
    feature_importance_chart,
    model_comparison_chart,
    residual_chart,
)
from .data_service import (
    load_extreme_metrics,
    load_feature_importance,
    load_final_metrics,
    load_largest_errors,
    load_metrics_by_range,
    load_model_bundle,
    load_model_comparison,
    load_test_predictions,
)
from .theme import (
    PALETTE,
    callout,
    data_table,
    format_number,
    metric_card,
    page_header,
)


def _format_date(value: object) -> str:
    try:
        return pd.Timestamp(value).strftime("%d %b %Y")
    except (TypeError, ValueError):
        return "N/A"


def render() -> None:
    metrics = load_final_metrics()
    predictions = load_test_predictions()
    comparison = load_model_comparison()
    importance = load_feature_importance()
    range_metrics = load_metrics_by_range()
    extreme_metrics = load_extreme_metrics()
    largest_errors = load_largest_errors()
    bundle = load_model_bundle()

    selected = metrics[metrics["role"] == "selected_model"]
    baseline = metrics[metrics["role"] == "persistence_baseline"]
    if selected.empty or baseline.empty:
        st.error("final_metrics.csv must contain selected_model and persistence_baseline rows.")
        return
    model_row = selected.iloc[0]
    baseline_row = baseline.iloc[0]

    st.markdown(
        page_header(
            "Model performance",
            "Independent test-set evaluation of the validation-selected model. "
            "The test split was created chronologically and was not used for "
            "training or model selection.",
            eyebrow="Section 04",
        ),
        unsafe_allow_html=True,
    )

    st.caption(
        f"Test period: {_format_date(model_row.get('date_start'))} → "
        f"{_format_date(model_row.get('date_end'))} · "
        f"{int(model_row.get('n', len(predictions)))} daily observations · "
        f"model: {bundle.model_name.replace('_', ' ').title()}"
    )

    cards = st.columns(4, gap="medium")
    with cards[0]:
        st.markdown(
            metric_card(
                "MAE",
                format_number(model_row.get("mae"), 2),
                subtitle="Average absolute error (AQI units)",
                accent=PALETTE["primary"],
            ),
            unsafe_allow_html=True,
        )
    with cards[1]:
        st.markdown(
            metric_card(
                "RMSE",
                format_number(model_row.get("rmse"), 2),
                subtitle="Penalises large misses more heavily",
                accent=PALETTE["primary_light"],
            ),
            unsafe_allow_html=True,
        )
    with cards[2]:
        st.markdown(
            metric_card(
                "R²",
                format_number(model_row.get("r2"), 3),
                subtitle="Variance explained; 0 ≈ mean predictor",
                accent=PALETTE["accent"],
            ),
            unsafe_allow_html=True,
        )
    with cards[3]:
        st.markdown(
            metric_card(
                "RMSE improvement",
                format_number(model_row.get("rmse_improvement_pct"), 1, "%"),
                subtitle=(
                    "vs persistence RMSE "
                    f"{format_number(baseline_row.get('rmse'), 2)}"
                ),
                accent=PALETTE["forecast"],
            ),
            unsafe_allow_html=True,
        )

    st.markdown(
        callout(
            "The model beats persistence on average error, but its R² is close to "
            "zero: it explains very little day-to-day variance and systematically "
            "under-predicts extreme pollution days. The metrics below are reported "
            "together so that the average-error improvement cannot be mistaken for "
            "reliable peak forecasting.",
            kind="warn",
        ),
        unsafe_allow_html=True,
    )

    predictions_tab, comparison_tab, importance_tab, errors_tab = st.tabs(
        ["Predictions & residuals", "Model comparison", "Feature importance", "Error analysis"]
    )

    with predictions_tab:
        left, right = st.columns(2, gap="large")
        with left:
            st.plotly_chart(
                actual_vs_predicted(predictions),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )
        with right:
            st.plotly_chart(
                residual_chart(predictions),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )
        st.caption(
            "Rows are indexed by the forecast origin: each predicted value is "
            "compared with the observation from the following day. Residuals are "
            "actual minus predicted; positive residuals mean the model under-predicted."
        )

    with comparison_tab:
        if comparison.empty:
            st.info("No model-comparison data recorded in models/best_model.json.")
        else:
            st.plotly_chart(
                model_comparison_chart(comparison),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )
            display = comparison.copy()
            display["model"] = (
                display["model"].str.replace("_", " ").str.title()
            )
            display = display.rename(
                columns={
                    "model": "Model",
                    "mae": "Validation MAE",
                    "rmse": "Validation RMSE",
                    "r2": "Validation R²",
                    "selected": "Selected",
                }
            )
            columns = ["Model", "Validation MAE", "Validation RMSE", "Validation R²", "Selected"]
            display = display[[column for column in columns if column in display.columns]]
            for column in ["Validation MAE", "Validation RMSE", "Validation R²"]:
                if column in display.columns:
                    display[column] = display[column].astype(float).round(3)
            st.markdown(data_table(display), unsafe_allow_html=True)
            st.caption(
                "Comparison metrics are from the validation split, which is the "
                "only split used for model selection. The test split is reserved "
                "for the final evaluation shown above."
            )

    with importance_tab:
        if importance.empty:
            st.info(
                "Feature importance is not available. The final evaluation script "
                "writes results/metrics/feature_importance.csv when the model "
                "supports it."
            )
        else:
            left, right = st.columns([1.25, 1.0], gap="large")
            with left:
                st.plotly_chart(
                    feature_importance_chart(importance, top_n=15),
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )
            with right:
                display = importance.head(15).copy()
                display = display.rename(
                    columns={
                        "feature": "Feature",
                        "importance": "Importance",
                        "rank": "Rank",
                    }
                )
                display = display[[column for column in ["Rank", "Feature", "Importance"] if column in display.columns]]
                st.markdown(data_table(display), unsafe_allow_html=True)
                method = str(importance.get("method", pd.Series(["unknown"])).iloc[0])
                st.caption(
                    f"Method: {method}. Importance is model behaviour, not a causal "
                    "effect; lagged and rolling features are strongly correlated."
                )

    with errors_tab:
        if not range_metrics.empty:
            st.plotly_chart(
                error_by_range_chart(range_metrics),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )
            display = range_metrics[range_metrics["n"] > 0].copy()
            display = display[
                [
                    "aqi_range",
                    "n",
                    "mae",
                    "rmse",
                    "r2",
                    "baseline_mae",
                    "mae_improvement_pct",
                ]
            ].rename(
                columns={
                    "aqi_range": "AQI range",
                    "n": "Days",
                    "mae": "Model MAE",
                    "rmse": "Model RMSE",
                    "r2": "Model R²",
                    "baseline_mae": "Baseline MAE",
                    "mae_improvement_pct": "MAE improvement (%)",
                }
            )
            for column in ["Model MAE", "Model RMSE", "Model R²", "Baseline MAE", "MAE improvement (%)"]:
                display[column] = display[column].astype(float).round(2)
            st.markdown(data_table(display), unsafe_allow_html=True)

        if not extreme_metrics.empty:
            st.markdown("#### Extreme-event subsets")
            display = extreme_metrics[
                ["definition", "n", "mae", "rmse", "baseline_mae", "mae_improvement_pct"]
            ].rename(
                columns={
                    "definition": "Subset",
                    "n": "Days",
                    "mae": "Model MAE",
                    "rmse": "Model RMSE",
                    "baseline_mae": "Baseline MAE",
                    "mae_improvement_pct": "MAE improvement (%)",
                }
            )
            for column in ["Model MAE", "Model RMSE", "Baseline MAE", "MAE improvement (%)"]:
                display[column] = display[column].astype(float).round(2)
            st.markdown(data_table(display), unsafe_allow_html=True)

        if not largest_errors.empty:
            st.markdown("#### Largest individual errors")
            display = largest_errors.head(10).copy()
            display["date"] = pd.to_datetime(display["date"]).dt.strftime("%d %b %Y")
            display = display[
                ["date", "actual_aqi", "predicted_aqi", "residual", "abs_error"]
            ].rename(
                columns={
                    "date": "Date",
                    "actual_aqi": "Actual AQI",
                    "predicted_aqi": "Predicted AQI",
                    "residual": "Residual",
                    "abs_error": "Absolute error",
                }
            )
            for column in ["Actual AQI", "Predicted AQI", "Residual", "Absolute error"]:
                display[column] = display[column].astype(float).round(2)
            st.markdown(data_table(display), unsafe_allow_html=True)
