"""Historical air-quality exploration section."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .charts import (
    PLOTLY_CONFIG,
    aqi_trend,
    category_donut,
    monthly_aqi,
    monthly_pollutant_grid,
    pollutant_correlation,
    pollutant_scatter,
)
from .data_service import load_historical_data
from .theme import (
    AQI_BAND_LABELS,
    PALETTE,
    POLLUTANT_META,
    aqi_category,
    data_table,
    format_number,
    metric_card,
    page_header,
)

WINDOW_OPTIONS = {
    "Full record": None,
    "Last 365 days": 365,
    "Last 180 days": 180,
    "Last 90 days": 90,
    "Last 30 days": 30,
}


def _format_date(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%d %b %Y")


def _prepare_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("aqi_category")
        .agg(days=("aqi", "size"), mean_aqi=("aqi", "mean"), max_aqi=("aqi", "max"))
        .reindex([label for label in AQI_BAND_LABELS if label in df["aqi_category"].unique()])
        .dropna(subset=["days"])
    )
    if summary.empty:
        return summary
    summary = summary.copy()
    summary["share_pct"] = summary["days"] / len(df) * 100.0
    return summary


def render() -> None:
    df = load_historical_data()

    st.markdown(
        page_header(
            "Historical air quality",
            "Explore the observed AQI record, monthly pollution patterns, "
            "category mix, and pollutant relationships. All values are historical "
            "observations from the repository snapshot.",
            eyebrow="Section 02",
        ),
        unsafe_allow_html=True,
    )

    window = st.radio(
        "Time window",
        options=list(WINDOW_OPTIONS),
        horizontal=True,
        index=0,
        help="Filter every chart and table in this section.",
    )
    days = WINDOW_OPTIONS[window]
    if days is None:
        filtered = df.copy()
    else:
        cutoff = df.index.max() - pd.Timedelta(days=days - 1)
        filtered = df.loc[df.index >= cutoff].copy()

    if filtered.empty:
        st.warning("No observations fall inside the selected window.")
        return

    peak_date = filtered["aqi"].idxmax()
    peak_category = aqi_category(filtered["aqi"].max())
    high_days = int((filtered["aqi"] >= 200).sum())

    metric_columns = st.columns(4, gap="medium")
    with metric_columns[0]:
        st.markdown(
            metric_card(
                "Days in view",
                f"{len(filtered):,}",
                subtitle=f"{_format_date(filtered.index.min())} → {_format_date(filtered.index.max())}",
                accent=PALETTE["primary_light"],
            ),
            unsafe_allow_html=True,
        )
    with metric_columns[1]:
        st.markdown(
            metric_card(
                "Mean AQI",
                format_number(filtered["aqi"].mean(), 1),
                subtitle=f"Median {format_number(filtered['aqi'].median(), 1)}",
                accent=PALETTE["primary"],
            ),
            unsafe_allow_html=True,
        )
    with metric_columns[2]:
        st.markdown(
            metric_card(
                "Peak AQI",
                format_number(filtered["aqi"].max(), 1),
                subtitle=f"{_format_date(peak_date)} · {peak_category['label']}",
                accent=peak_category["color"],
            ),
            unsafe_allow_html=True,
        )
    with metric_columns[3]:
        st.markdown(
            metric_card(
                "High-pollution days",
                f"{high_days:,}",
                subtitle=f"AQI ≥ 200 · {high_days / len(filtered) * 100:.1f}% of days",
                accent=PALETTE["danger"],
            ),
            unsafe_allow_html=True,
        )

    st.plotly_chart(
        aqi_trend(filtered, f"Observed AQI — {window.lower()}"),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )
    st.caption(
        "AQI in this dataset is derived from PM2.5 using US-EPA breakpoints; it is "
        "not a live official CPCB composite index."
    )

    trends_tab, categories_tab, pollutants_tab = st.tabs(
        ["Monthly trends", "AQI categories", "Pollutant analysis"]
    )

    with trends_tab:
        st.plotly_chart(
            monthly_aqi(filtered),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
        st.plotly_chart(
            monthly_pollutant_grid(filtered),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    with categories_tab:
        left, right = st.columns([1.0, 1.1], gap="large")
        with left:
            st.plotly_chart(
                category_donut(filtered),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )
        with right:
            summary = _prepare_category_summary(filtered)
            if summary.empty:
                st.info("No AQI category data available for this window.")
            else:
                display = summary.reset_index().rename(
                    columns={
                        "aqi_category": "Category",
                        "days": "Days",
                        "mean_aqi": "Mean AQI",
                        "max_aqi": "Max AQI",
                        "share_pct": "Share (%)",
                    }
                )
                display[["Mean AQI", "Max AQI", "Share (%)"]] = display[
                    ["Mean AQI", "Max AQI", "Share (%)"]
                ].round(1)
                st.markdown("#### Category summary")
                st.markdown(data_table(display), unsafe_allow_html=True)
                st.caption(
                    "Thresholds follow the US-EPA AQI categories used to derive "
                    "the target in this project."
                )

    with pollutants_tab:
        pollutant = st.selectbox(
            "Pollutant",
            options=list(POLLUTANT_META),
            format_func=lambda key: (
                f"{POLLUTANT_META[key]['label']} ({POLLUTANT_META[key]['unit']})"
            ),
            index=0,
        )
        left, right = st.columns(2, gap="large")
        with left:
            st.plotly_chart(
                pollutant_scatter(filtered, pollutant),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )
        with right:
            st.plotly_chart(
                pollutant_correlation(filtered),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )

        rows = []
        correlations = filtered[list(POLLUTANT_META) + ["aqi"]].corr(numeric_only=True)["aqi"]
        for column, meta in POLLUTANT_META.items():
            series = filtered[column]
            rows.append(
                {
                    "Pollutant": meta["label"],
                    "Unit": meta["unit"],
                    "Mean": series.mean(),
                    "Median": series.median(),
                    "Maximum": series.max(),
                    "Correlation with AQI": correlations.get(column),
                }
            )
        stats = pd.DataFrame(rows)
        numeric_columns = ["Mean", "Median", "Maximum", "Correlation with AQI"]
        stats[numeric_columns] = stats[numeric_columns].round(3)
        st.markdown("#### Pollutant summary")
        st.markdown(data_table(stats), unsafe_allow_html=True)
        st.caption(
            "PM2.5 is the pollutant used to derive AQI in this dataset, so its "
            "correlation with AQI is high by construction rather than an "
            "independent finding."
        )
