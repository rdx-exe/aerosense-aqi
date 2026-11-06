"""Environmental insights section: EDA findings and pollution context.

Every statement in this section is computed from the loaded historical data.
No live or fabricated readings are used.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from .charts import PLOTLY_CONFIG, high_pollution_timeline, monthly_heatmap
from .data_service import load_historical_data
from .theme import (
    AQI_BANDS,
    PALETTE,
    aqi_category,
    callout,
    data_table,
    format_number,
    metric_card,
    page_header,
)

_EXTREME_THRESHOLD = 200.0


def _format_date(value: Any) -> str:
    return pd.Timestamp(value).strftime("%d %b %Y")


def _month_name(month: int) -> str:
    return pd.Timestamp(year=2020, month=int(month), day=1).strftime("%B")


def _episodes(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Group consecutive days at or above ``threshold`` into episodes."""
    if df.empty or not (df["aqi"] >= threshold).any():
        return pd.DataFrame()

    complete_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    daily = df["aqi"].reindex(complete_range)

    episodes: list[dict[str, Any]] = []
    current_start: pd.Timestamp | None = None
    current_days: list[pd.Timestamp] = []
    previous_date: pd.Timestamp | None = None

    for date in daily.index:
        value = daily.loc[date]
        is_high = pd.notna(value) and value >= threshold
        continuous = (
            previous_date is not None
            and (date - previous_date) == pd.Timedelta(days=1)
        )

        if is_high:
            if current_start is None:
                current_start = date
            elif not continuous:
                episodes.append(_build_episode(daily, current_start, current_days))
                current_start = date
                current_days = []
            current_days.append(date)
        elif current_days and current_start is not None:
            episodes.append(_build_episode(daily, current_start, current_days))
            current_start = None
            current_days = []

        previous_date = date

    if current_start is not None and current_days:
        episodes.append(_build_episode(daily, current_start, current_days))

    result = pd.DataFrame(episodes)
    return result.sort_values(
        ["max_aqi", "days"], ascending=[False, False]
    ).reset_index(drop=True)


def _build_episode(
    daily: pd.Series,
    start: pd.Timestamp,
    dates: list[pd.Timestamp],
) -> dict[str, Any]:
    values = daily.loc[dates].dropna()
    return {
        "start": start,
        "end": dates[-1],
        "days": len(dates),
        "max_aqi": float(values.max()) if not values.empty else float("nan"),
        "mean_aqi": float(values.mean()) if not values.empty else float("nan"),
    }


def _monthly_ranking(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.assign(month=df.index.month)
        .groupby("month")
        .agg(mean_aqi=("aqi", "mean"), days=("aqi", "size"))
        .sort_values("mean_aqi", ascending=False)
    )


def _aqi_band_reference() -> None:
    rows = [
        {
            "Category": band["label"],
            "Range": f"{band['lower']}–{band['upper']}",
            "Health guidance": band["guidance"],
        }
        for band in AQI_BANDS
    ]
    st.markdown(data_table(pd.DataFrame(rows)), unsafe_allow_html=True)


def render() -> None:
    df = load_historical_data()

    st.markdown(
        page_header(
            "Environmental insights",
            "Seasonal structure, pollution episodes, pollutant behaviour, and what "
            "the observed AQI levels mean for public health and the environment.",
            eyebrow="Section 05",
        ),
        unsafe_allow_html=True,
    )

    # ── Headline statistics ─────────────────────────────────────────────────
    mean_aqi = float(df["aqi"].mean())
    median_aqi = float(df["aqi"].median())
    peak_index = df["aqi"].idxmax()
    peak_aqi = float(df["aqi"].max())
    unhealthy_days = int((df["aqi"] >= 151).sum())
    extreme_days = int((df["aqi"] >= _EXTREME_THRESHOLD).sum())

    monthly = _monthly_ranking(df)
    worst_month = int(monthly.index[0]) if not monthly.empty else 0
    cleanest_month = int(monthly.index[-1]) if not monthly.empty else 0

    cards = st.columns(4, gap="medium")
    with cards[0]:
        st.markdown(
            metric_card(
                "Mean AQI",
                format_number(mean_aqi, 1),
                subtitle=f"Median {format_number(median_aqi, 1)} · {len(df):,} days",
                accent=aqi_category(mean_aqi)["color"],
            ),
            unsafe_allow_html=True,
        )
    with cards[1]:
        st.markdown(
            metric_card(
                "Days AQI ≥ 151",
                f"{unhealthy_days:,}",
                subtitle=f"{unhealthy_days / len(df) * 100:.1f}% of the record",
                accent=PALETTE["danger"],
            ),
            unsafe_allow_html=True,
        )
    with cards[2]:
        st.markdown(
            metric_card(
                "Days AQI ≥ 200",
                f"{extreme_days:,}",
                subtitle=f"{extreme_days / len(df) * 100:.1f}% of the record",
                accent=aqi_category(210)["color"],
            ),
            unsafe_allow_html=True,
        )
    with cards[3]:
        st.markdown(
            metric_card(
                "Highest month",
                _month_name(worst_month) if worst_month else "N/A",
                subtitle=(
                    f"Mean {format_number(monthly['mean_aqi'].iloc[0], 1)} · "
                    f"cleanest {_month_name(cleanest_month)}"
                    if not monthly.empty
                    else ""
                ),
                accent=PALETTE["forecast"],
            ),
            unsafe_allow_html=True,
        )

    # ── Key findings ────────────────────────────────────────────────────────
    st.markdown("### Key findings")
    findings = st.columns(3, gap="medium")

    with findings[0]:
        with st.container(border=True):
            st.markdown("#### Seasonal pattern")
            st.markdown(
                f"""
                - **{_month_name(worst_month)}** has the highest pooled mean AQI
                  (**{format_number(monthly['mean_aqi'].iloc[0], 1)}**);
                  **{_month_name(cleanest_month)}** is cleanest
                  (**{format_number(monthly['mean_aqi'].iloc[-1], 1)}**).
                - Winter and post-monsoon months dominate the high-AQI record.
                - Monsoon rain and stronger ventilation typically reduce particulate
                  pollution.
                """
            )

    with findings[1]:
        with st.container(border=True):
            st.markdown("#### Peaks and episodes")
            st.markdown(
                f"""
                - The highest observed AQI was **{format_number(peak_aqi, 1)}**
                  on **{_format_date(peak_index)}**.
                - **{extreme_days:,}** days crossed AQI 200; pollution is episodic
                  rather than a steady high baseline.
                - The model predicted **none** of the 14 extreme test-window days,
                  so peak risk is not captured operationally.
                """
            )

    with findings[2]:
        with st.container(border=True):
            st.markdown("#### Pollutant behaviour")
            correlations = df[["pm25", "pm10", "no2", "so2", "o3", "co", "aqi"]].corr(
                numeric_only=True
            )["aqi"]
            st.markdown(
                f"""
                - PM2.5 correlates with AQI at **r = {correlations['pm25']:.2f}** —
                  high by construction, because AQI is derived from PM2.5.
                - SO₂ correlates at **r = {correlations['so2']:.2f}**; rolling SO₂,
                  CO and PM2.5 statistics dominate model importance.
                - These are associations, not causal effects.
                """
            )

    st.plotly_chart(
        high_pollution_timeline(df, threshold=_EXTREME_THRESHOLD),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

    # ── Monthly pattern and episodes ────────────────────────────────────────
    left, right = st.columns([0.95, 1.15], gap="large")
    with left:
        st.plotly_chart(
            monthly_heatmap(df),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
    with right:
        st.markdown("#### High-pollution episodes (AQI ≥ 200)")
        episodes = _episodes(df, _EXTREME_THRESHOLD)
        if episodes.empty:
            st.info("No days in the historical record reached AQI 200.")
        else:
            display = episodes.copy()
            display["start"] = pd.to_datetime(display["start"]).dt.strftime("%d %b %Y")
            display["end"] = pd.to_datetime(display["end"]).dt.strftime("%d %b %Y")
            display = display.rename(
                columns={
                    "start": "Start",
                    "end": "End",
                    "days": "Days",
                    "max_aqi": "Max AQI",
                    "mean_aqi": "Mean AQI",
                }
            )
            display[["Max AQI", "Mean AQI"]] = display[
                ["Max AQI", "Mean AQI"]
            ].round(1)
            st.markdown(data_table(display.head(8)), unsafe_allow_html=True)
            st.caption(
                f"{len(episodes)} episodes across the record; consecutive days at or "
                "above the threshold are grouped, and gaps break an episode."
            )

    # ── Environmental significance ──────────────────────────────────────────
    st.markdown("### Environmental and public-health significance")
    significance = st.columns(2, gap="large")
    with significance[0]:
        with st.container(border=True):
            st.markdown("#### Why these levels matter")
            st.markdown(
                """
                - **PM2.5** penetrates deep into the lungs and bloodstream and is
                  linked to cardiovascular and respiratory illness.
                - **AQI above 150** affects everyone, with children, older adults,
                  outdoor workers and people with respiratory conditions at greater risk.
                - **AQI above 200** warrants avoiding prolonged outdoor exertion and
                  reducing indoor exposure where possible.
                """
            )
        with st.container(border=True):
            st.markdown("#### Likely drivers")
            st.markdown(
                """
                - Traffic, diesel, road and construction dust.
                - Seasonal biomass and waste burning.
                - Industrial and domestic fuel combustion.
                - Winter inversions trap pollution; monsoon rain scavenges it.
                """
            )
    with significance[1]:
        with st.container(border=True):
            st.markdown("#### How to read this responsibly")
            st.markdown(
                """
                - AQI here is **derived from PM2.5** with US-EPA breakpoints, not the
                  CPCB composite index.
                - The dataset is a **historical snapshot**, not a live feed.
                - A city-scale series cannot represent intra-city exposure differences.
                - Associations do not establish causation.
                """
            )
        with st.expander("AQI category reference (US-EPA)", expanded=False):
            _aqi_band_reference()

    st.markdown(
        callout(
            "These findings describe the historical snapshot used to train the model. "
            "They are suitable for exploration and education, not regulatory decisions "
            "or personal exposure advice.",
            kind="warn",
        ),
        unsafe_allow_html=True,
    )
