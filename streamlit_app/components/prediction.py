"""Next-day AQI forecast section with validated what-if inputs."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.feature_engineering.features import compute_aqi

from .charts import (
    PLOTLY_CONFIG,
    aqi_gauge,
    forecast_history,
)
from .data_service import (
    DashboardDataError,
    InputValidationError,
    load_clean_data,
    load_historical_data,
    origin_dates,
    predict_next_day,
)
from .theme import (
    PALETTE,
    POLLUTANT_META,
    callout,
    data_table,
    format_number,
    metric_card,
    page_header,
    pill,
)


def _format_date(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%d %b %Y")


def _origin_values(clean: pd.DataFrame, origin: pd.Timestamp) -> dict[str, float]:
    row = clean.loc[origin]
    return {column: float(row[column]) for column in POLLUTANT_META}


def _scenario_table(values: dict[str, float]) -> pd.DataFrame:
    rows = []
    for column, meta in POLLUTANT_META.items():
        rows.append(
            {
                "Pollutant": meta["label"],
                "Value": float(values[column]),
                "Unit": meta["unit"],
            }
        )
    return pd.DataFrame(rows)


def _forecast_card(prediction: dict) -> str:
    category = prediction["category"]
    return f"""
    <div class="aqi-forecast-card">
        <div class="kicker">Model-generated next-day forecast</div>
        <div class="big" style="color:{category['color']};">{prediction['predicted_aqi']:.1f}</div>
        <div>{pill(category['label'], category['color'])}</div>
        <div style="margin-top:0.8rem; color:{PALETTE['muted']}; font-size:0.86rem; line-height:1.55;">
            <b>Forecast target:</b> {_format_date(prediction['target_date'])}<br>
            <b>Forecast origin:</b> {_format_date(prediction['origin_date'])} (last observation used)<br>
            <b>Model:</b> {prediction['model_name'].replace('_', ' ').title()}
        </div>
    </div>
    """


def render() -> None:
    clean = load_clean_data()
    historical = load_historical_data()

    st.markdown(
        page_header(
            "Next-day AQI prediction",
            "Generate a model forecast from a historical forecast origin. The "
            "history is truncated at the origin before feature construction, so "
            "future observations cannot influence the prediction.",
            eyebrow="Section 03",
        ),
        unsafe_allow_html=True,
    )

    st.caption(
        f"Historical snapshot ending {_format_date(historical.index.max())}. "
        "Forecasts are model-generated scenarios for the following day, not live AQI readings."
    )

    left, right = st.columns([1.02, 1.35], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("#### Forecast inputs")
            origins = origin_dates()
            origin = st.selectbox(
                "Forecast origin (last observed day)",
                options=origins,
                format_func=lambda value: value.strftime("%d %b %Y (%a)"),
                index=0,
                help="Only observations on or before this date are used.",
            )
            observed = _origin_values(clean, origin)
            st.caption(
                f"Forecast target: {_format_date(origin + pd.Timedelta(days=1))} · "
                f"available history: {int((clean.index <= origin).sum()):,} days"
            )

            adjust = st.toggle(
                "Adjust the origin-day pollutant values (what-if scenario)",
                value=False,
                help=(
                    "When enabled, the six pollutant values for the origin day "
                    "are replaced with your inputs. Lag and rolling features are "
                    "recomputed from the historical window using the training "
                    "pipeline."
                ),
            )

            values = dict(observed)
            if adjust:
                input_columns = st.columns(2)
                for index, (column, meta) in enumerate(POLLUTANT_META.items()):
                    with input_columns[index % 2]:
                        values[column] = st.number_input(
                            f"{meta['label']} ({meta['unit']})",
                            min_value=float(meta["min"]),
                            max_value=float(meta["max"]),
                            value=float(observed[column]),
                            step=float(meta["step"]),
                            help=meta["description"],
                            key=f"pollutant_{origin.date()}_{column}",
                        )
                st.caption(
                    "PM2.5 is capped at 500 µg/m³ because the AQI conversion is "
                    "defined up to that breakpoint."
                )
            else:
                st.caption("Using the observed pollutant values for this origin date.")

            preview = pd.DataFrame([values])
            preview_aqi = float(compute_aqi(preview)["aqi"].iloc[0])
            st.markdown(
                f"**Derived origin-day AQI:** {format_number(preview_aqi, 1)} "
                "<span style='color:#5F6B7A;font-size:0.82rem;'>"
                "(from PM2.5 using the training formula)</span>",
                unsafe_allow_html=True,
            )

    prediction: dict | None = None
    error: str | None = None
    try:
        prediction = predict_next_day(origin, values if adjust else None)
    except (InputValidationError, DashboardDataError) as exc:
        error = str(exc)

    with right:
        if error is not None:
            st.error("The forecast could not be generated.")
            st.markdown(f"**Reason:** {error}")
            st.info(
                "Check that the selected origin has at least eight days of history "
                "and that all pollutant values are within the allowed ranges."
            )
            return

        if prediction is None:
            st.error("The forecast could not be generated.")
            return

        st.markdown(_forecast_card(prediction), unsafe_allow_html=True)
        st.plotly_chart(
            aqi_gauge(prediction["predicted_aqi"]),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

        metric_columns = st.columns(2, gap="medium")
        with metric_columns[0]:
            st.markdown(
                metric_card(
                    "Origin-day AQI",
                    format_number(prediction["derived_aqi"], 1),
                    subtitle="Derived from PM2.5",
                    accent=PALETTE["primary_light"],
                ),
                unsafe_allow_html=True,
            )
        with metric_columns[1]:
            delta = prediction["delta_vs_origin"]
            st.markdown(
                metric_card(
                    "Forecast change",
                    f"{delta:+.1f}",
                    subtitle="vs origin-day AQI",
                    accent=PALETTE["forecast"],
                ),
                unsafe_allow_html=True,
            )

        st.plotly_chart(
            forecast_history(
                historical,
                origin_date=prediction["origin_date"],
                target_date=prediction["target_date"],
                predicted_aqi=prediction["predicted_aqi"],
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

        for warning in prediction["warnings"]:
            st.warning(warning)

        if prediction["actual_next_aqi"] is not None:
            actual = prediction["actual_next_aqi"]
            error_value = actual - prediction["predicted_aqi"]
            st.markdown(
                callout(
                    "<b>Evaluation reference only.</b> The observed AQI on "
                    f"{_format_date(prediction['target_date'])} was "
                    f"<b>{actual:.1f}</b> (forecast error {error_value:+.1f}). "
                    "This observation was <b>not</b> used to generate the forecast.",
                    kind="info",
                ),
                unsafe_allow_html=True,
            )

        st.markdown(
            callout(
                "Model caveat: predictions are compressed toward the training mean "
                "and can miss pollution peaks. Treat this as an educational estimate, "
                "not an official alert.",
                kind="warn",
            ),
            unsafe_allow_html=True,
        )

        with st.expander("Forecast basis & integrity", expanded=False):
            st.markdown("**Scenario values used**")
            st.markdown(
                data_table(_scenario_table(prediction["scenario_values"])),
                unsafe_allow_html=True,
            )
            st.caption(
                f"{prediction['feature_count']} features passed in the saved training "
                "order. History is truncated at the origin before lag/rolling features "
                "are built, and AQI is recomputed from PM2.5 using the training formula."
            )
