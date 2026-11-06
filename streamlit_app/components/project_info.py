"""Project information, provenance, reproduction and limitations."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .data_service import (
    load_final_metrics,
    load_historical_data,
    load_model_bundle,
    load_split_info,
)
from .theme import PALETTE, callout, format_number, metric_card, page_header


def _format_date(value: object) -> str:
    try:
        return pd.Timestamp(value).strftime("%d %b %Y")
    except (TypeError, ValueError):
        return "N/A"


def render() -> None:
    historical = load_historical_data()
    bundle = load_model_bundle()
    metrics = load_final_metrics()
    split_info = load_split_info()

    selected = metrics[metrics["role"] == "selected_model"]
    baseline = metrics[metrics["role"] == "persistence_baseline"]
    model_row = selected.iloc[0] if not selected.empty else pd.Series(dtype=object)
    baseline_row = baseline.iloc[0] if not baseline.empty else pd.Series(dtype=object)

    st.markdown(
        page_header(
            "Project information",
            "Purpose, data provenance, model contract, reproduction steps and the "
            "limitations that bound how this dashboard should be used.",
            eyebrow="Reference",
        ),
        unsafe_allow_html=True,
    )

    cards = st.columns(4, gap="medium")
    with cards[0]:
        st.markdown(
            metric_card(
                "Location",
                "Kolkata",
                subtitle="West Bengal, India",
                accent=PALETTE["primary"],
            ),
            unsafe_allow_html=True,
        )
    with cards[1]:
        st.markdown(
            metric_card(
                "Task",
                "Next-day AQI",
                subtitle="Regression from lagged pollutant features",
                accent=PALETTE["primary_light"],
            ),
            unsafe_allow_html=True,
        )
    with cards[2]:
        st.markdown(
            metric_card(
                "Selected model",
                bundle.model_name.replace("_", " ").title(),
                subtitle=f"{len(bundle.feature_columns)} model features",
                accent=PALETTE["accent"],
            ),
            unsafe_allow_html=True,
        )
    with cards[3]:
        st.markdown(
            metric_card(
                "Data coverage",
                f"{len(historical):,} days",
                subtitle=(
                    f"{_format_date(historical.index.min())} → "
                    f"{_format_date(historical.index.max())}"
                ),
                accent=PALETTE["forecast"],
            ),
            unsafe_allow_html=True,
        )

    left, right = st.columns(2, gap="large")

    with left:
        with st.container(border=True):
            st.markdown("### Purpose")
            st.markdown(
                """
                This dashboard supports an academic project on **AI-based air quality
                index prediction and analysis using machine learning**. It serves two
                purposes:

                1. **Explore** the historical AQI and pollutant record for Kolkata.
                2. **Forecast** the next day's AQI from a saved, validation-selected
                   model, including transparent what-if scenarios.

                It is a research and teaching artifact. It is **not** an official
                air-quality service and must not be used as the sole basis for health
                or policy decisions.
                """
            )

        with st.container(border=True):
            st.markdown("### Pipeline architecture")
            st.markdown(
                """
                ```
                raw pollutant data
                    → cleaning and validation      (src/preprocessing)
                    → chronological splits         (data/processed/splits)
                    → shared feature engineering   (src/feature_engineering)
                    → candidate model training     (src/models)
                    → validation-based selection   (models/best_model.json)
                    → final test evaluation        (results/metrics)
                    → Streamlit dashboard          (streamlit_app)
                ```

                The dashboard **loads** `models/<selected>/model.pkl` with Joblib and
                never retrains a model at start-up. Feature construction is shared with
                training through `build_features` / `build_inference_features`, which
                prevents the dashboard and training pipelines from drifting apart.
                """
            )

        with st.container(border=True):
            st.markdown("### Data provenance")
            st.markdown(
                f"""
                - Source table: `data/processed/clean.csv` (cleaned daily snapshot).
                - Coverage: **{_format_date(historical.index.min())} → {_format_date(historical.index.max())}**,
                  {len(historical):,} daily rows.
                - Pollutants: PM2.5, PM10, NO₂, SO₂, O₃ and CO.
                - AQI is **computed from PM2.5** with US-EPA 2012 breakpoints. It is a
                  derived indicator, not a live CPCB/CPCB composite feed.
                - No live API, web scrape or synthetic reading is used anywhere in the
                  dashboard.
                """
            )

    with right:
        with st.container(border=True):
            st.markdown("### Model and evaluation")
            st.markdown(
                f"""
                - Selected model: **{bundle.model_name.replace('_', ' ').title()}**
                  (`models/{bundle.model_name}/model.pkl`).
                - Selection rule: lowest **validation** RMSE among candidates that beat
                  the persistence baseline by more than 5%.
                - Training data ends **{_format_date(bundle.training_cutoff) if bundle.training_cutoff is not None else 'N/A'}**;
                  the test window was never used for training or selection.
                - Test window: **{_format_date(model_row.get('date_start'))} → {_format_date(model_row.get('date_end'))}**
                  ({int(model_row.get('n', 0)) if not model_row.empty else 'N/A'} days).
                - Test MAE **{format_number(model_row.get('mae'), 2)}**,
                  RMSE **{format_number(model_row.get('rmse'), 2)}**,
                  R² **{format_number(model_row.get('r2'), 3)}**.
                - Persistence baseline MAE **{format_number(baseline_row.get('mae'), 2)}**,
                  RMSE **{format_number(baseline_row.get('rmse'), 2)}**,
                  R² **{format_number(baseline_row.get('r2'), 3)}**.
                - Full report: `documentation/model_evaluation.md`.
                """
            )

        with st.container(border=True):
            st.markdown("### Integrity controls")
            split_note = ""
            if split_info:
                ranges = split_info.get("date_ranges") or {}
                train_range = ranges.get("train")
                test_range = ranges.get("test")
                if train_range and test_range:
                    split_note = (
                        f"\n- Chronological splits: train ends **{train_range[1]}**, "
                        f"test starts **{test_range[0]}**.\n"
                        "- Boundary rows are purged so adjacent targets cannot overlap."
                    )
            st.markdown(
                f"""
                - The model is loaded with Joblib; **no training occurs on dashboard start-up**.
                - Features are passed in the exact order recorded in
                  `models/best_model.json` / model metadata ({len(bundle.feature_columns)} columns).
                - Prediction history is truncated at the forecast origin **before**
                  lag/rolling features are built.
                - User inputs are range-validated and AQI is recomputed from the edited
                  PM2.5 value using the training formula.
                - Forecasts and historical observations are labelled separately in every
                  chart and card.
                {split_note}
                """
            )

        with st.container(border=True):
            st.markdown("### Reproduce and launch")
            st.code(
                "source .venv/bin/activate\n"
                "./scripts/run_forecasting.sh\n"
                "./scripts/run_model_training.sh\n"
                "./scripts/run_final_evaluation.sh\n"
                "streamlit run streamlit_app/app.py",
                language="bash",
            )
            st.caption(
                "The final command must be run from the project root. The dashboard is "
                "served by Streamlit and opens in the browser."
            )

    with st.container(border=True):
        st.markdown("### Limitations and responsible use")
        st.markdown(
            f"""
            - **Near-zero explanatory power.** Test R² is
              **{format_number(model_row.get('r2'), 3)}**, close to a mean predictor,
              despite a lower average error than persistence. Average-error
              improvement must not be read as reliable forecasting.
            - **Extreme peaks are missed.** In the test window, 14 days reached AQI 200
              and the model predicted none of them. Extreme-day MAE was worse than the
              persistence baseline.
            - **Predictions are compressed.** The predicted range was roughly 28% of
              the actual range, so the model is biased toward the middle of the
              distribution.
            - **Derived AQI.** Because AQI is computed from PM2.5, pollutant
              correlations with AQI are partly mechanical; the dashboard does not
              present them as causal findings.
            - **Single historical window.** Metrics are one realisation of one
              chronological test period and are sensitive to the season sampled.
            - **Not an alerting system.** No live data, no uncertainty interval, and no
              official status is implied by any forecast shown.
            """
        )

    st.markdown(
        callout(
            "<b>Credits and stack.</b> Built with Python, pandas, scikit-learn, "
            "Streamlit, Plotly and Joblib. Model artifacts, metrics and documentation "
            "are produced by the repository's reproducible scripts. All displayed "
            "figures are read from those artifacts at runtime.",
            kind="info",
        ),
        unsafe_allow_html=True,
    )
