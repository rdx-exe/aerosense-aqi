"""Overview section: project introduction and latest snapshot."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .charts import PLOTLY_CONFIG, aqi_trend
from .data_service import (
    load_historical_data,
    load_model_bundle,
    metric_row,
)
from .theme import PALETTE, aqi_category, format_number, hero, metric_card


def _format_date(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%d %b %Y")


def render() -> None:
    historical = load_historical_data()
    bundle = load_model_bundle()

    latest = historical.iloc[-1]
    latest_date = historical.index[-1]
    category = aqi_category(latest["aqi"])
    model_metrics = metric_row("selected_model")
    baseline_metrics = metric_row("persistence_baseline")

    st.markdown(
        hero(
            "Kolkata Air Quality Intelligence",
            "A historical air-quality explorer and a leakage-safe next-day AQI "
            "forecast, built on the project's validation-selected machine-learning model.",
            badges=[
                "Next-day AQI",
                "Kolkata, West Bengal",
                "Historical snapshot",
                f"{bundle.model_name.replace('_', ' ').title()} model",
            ],
        ),
        unsafe_allow_html=True,
    )

    columns = st.columns(4, gap="medium")
    with columns[0]:
        st.markdown(
            metric_card(
                "Latest observed AQI",
                format_number(latest["aqi"], 1),
                subtitle=(
                    f"{category['label']} · {_format_date(latest_date)} · "
                    "historical observation, not live"
                ),
                accent=category["color"],
            ),
            unsafe_allow_html=True,
        )
    with columns[1]:
        st.markdown(
            metric_card(
                "Dataset coverage",
                f"{len(historical):,} days",
                subtitle=(
                    f"{_format_date(historical.index.min())} → "
                    f"{_format_date(historical.index.max())}"
                ),
                accent=PALETTE["primary_light"],
            ),
            unsafe_allow_html=True,
        )
    with columns[2]:
        st.markdown(
            metric_card(
                "Selected model",
                bundle.model_name.replace("_", " ").title(),
                subtitle=(
                    f"Validation-selected · {len(bundle.feature_columns)} features"
                ),
                accent=PALETTE["primary"],
            ),
            unsafe_allow_html=True,
        )
    with columns[3]:
        st.markdown(
            metric_card(
                "Test RMSE",
                format_number(model_metrics.get("rmse"), 2),
                subtitle=(
                    "Persistence: "
                    f"{format_number(baseline_metrics.get('rmse'), 2)} · "
                    "lower is better"
                ),
                accent=PALETTE["accent"],
            ),
            unsafe_allow_html=True,
        )

    st.plotly_chart(
        aqi_trend(historical.tail(120), "Last 120 days of observed AQI"),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.markdown("#### What you can do here")
            st.markdown(
                """
                - Explore the historical AQI and pollutant record.
                - Generate a **next-day AQI forecast** from the saved model.
                - Test **what-if** pollutant scenarios with validated inputs.
                - Review honest test-set performance and its limitations.
                """
            )
    with right:
        with st.container(border=True):
            st.markdown("#### Data & provenance")
            st.markdown(
                f"""
                - Snapshot: **{_format_date(historical.index.min())} → {_format_date(historical.index.max())}**.
                - {len(historical):,} daily observations across six pollutants.
                - AQI is **derived from PM2.5** with US-EPA breakpoints.
                - Historical research data, **not a live feed**; model trained
                  through **{_format_date(bundle.training_cutoff) if bundle.training_cutoff is not None else 'N/A'}**.
                """
            )
