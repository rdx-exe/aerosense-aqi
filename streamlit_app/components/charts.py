"""Plotly figure builders for the AQI dashboard.

All figures use the shared environmental theme, responsive sizing and explicit
axis labels/units.  Builders never fetch data themselves; they receive the
already-loaded frames so that the data path stays in one place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .theme import (
    AQI_BAND_COLORS,
    AQI_BANDS,
    PALETTE,
    POLLUTANT_META,
    aqi_category,
)

FONT_FAMILY = "Inter, sans-serif"
PLOTLY_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
    "displaylogo": False,
}


def _rgba(hex_color: str, alpha: float) -> str:
    """Convert ``#RRGGBB`` to an ``rgba()`` string accepted by Plotly."""
    value = hex_color.lstrip("#")
    if len(value) != 6:
        return f"rgba(95, 107, 122, {alpha})"
    red = int(value[0:2], 16)
    green = int(value[2:4], 16)
    blue = int(value[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def _empty_figure(message: str, height: int = 260) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=13, color=PALETTE["muted"], family=FONT_FAMILY),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _apply_layout(fig, "", height=height, showlegend=False)


def _apply_layout(
    fig: go.Figure,
    title: str,
    height: int = 380,
    showlegend: bool = True,
) -> go.Figure:
    fig.update_layout(
        template=None,
        title=dict(
            text=title,
            x=0,
            xanchor="left",
            font=dict(size=16, color="#14281D", family=FONT_FAMILY),
        )
        if title
        else None,
        height=height,
        margin=dict(l=12, r=14, t=54 if title else 18, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(family=FONT_FAMILY, size=12, color=PALETTE["text"]),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor=PALETTE["border"],
            font=dict(size=12, family=FONT_FAMILY, color=PALETTE["text"]),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=showlegend,
        colorway=[
            PALETTE["primary"],
            PALETTE["primary_light"],
            PALETTE["accent"],
            "#7CB342",
            "#8D6E63",
            "#A1887F",
        ],
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor=PALETTE["border"],
        tickcolor=PALETTE["border"],
        zeroline=False,
        automargin=True,
        title_font=dict(size=12, color=PALETTE["muted"]),
        tickfont=dict(size=11, color=PALETTE["muted"]),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=PALETTE["grid"],
        zeroline=False,
        automargin=True,
        linecolor=PALETTE["border"],
        title_font=dict(size=12, color=PALETTE["muted"]),
        tickfont=dict(size=11, color=PALETTE["muted"]),
    )
    return fig


def aqi_trend(df: pd.DataFrame, title: str = "Observed AQI trend") -> go.Figure:
    """Line chart of historical AQI with severity reference lines."""
    if df.empty or "aqi" not in df.columns:
        return _empty_figure("No AQI observations available.")

    marker_colors = (
        df["aqi_color"].tolist() if "aqi_color" in df.columns else PALETTE["primary"]
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["aqi"],
            mode="lines+markers",
            name="Observed AQI",
            line=dict(color=PALETTE["primary"], width=2.1),
            marker=dict(size=4.5, color=marker_colors, line=dict(width=0)),
            fill="tozeroy",
            fillcolor="rgba(95, 107, 122, 0.05)",
            hovertemplate="%{x|%d %b %Y}<br>AQI: %{y:.1f}<extra></extra>",
        )
    )

    for level, color in [
        (100, "#A68A00"),
        (150, "#D9691B"),
        (200, "#C62828"),
        (300, "#6A1B9A"),
    ]:
        if float(df["aqi"].max()) >= level:
            fig.add_hline(
                y=level,
                line=dict(color=color, width=1, dash="dot"),
                opacity=0.45,
            )

    fig.update_yaxes(title_text="AQI (index)")
    fig.update_xaxes(
        title_text="Date",
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor="#FFFFFF",
            activecolor=PALETTE["primary_soft"],
            bordercolor=PALETTE["border"],
            font=dict(size=10, color=PALETTE["muted"]),
        ),
    )
    return _apply_layout(fig, title, height=400)


def monthly_aqi(df: pd.DataFrame) -> go.Figure:
    """Monthly mean AQI bars coloured by severity."""
    if df.empty or "aqi" not in df.columns:
        return _empty_figure("No monthly AQI data available.")

    monthly = (
        df["aqi"]
        .resample("MS")
        .agg(["mean", "max", "count"])
        .dropna(subset=["mean"])
    )
    if monthly.empty:
        return _empty_figure("No monthly AQI data available.")

    colors = [aqi_category(value)["color"] for value in monthly["mean"]]
    fig = go.Figure(
        go.Bar(
            x=monthly.index,
            y=monthly["mean"],
            marker=dict(color=colors, line=dict(width=0)),
            customdata=np.column_stack(
                [monthly["max"].to_numpy(), monthly["count"].to_numpy()]
            ),
            hovertemplate=(
                "%{x|%b %Y}<br>Mean AQI: %{y:.1f}"
                "<br>Max AQI: %{customdata[0]:.1f}"
                "<br>Days: %{customdata[1]:.0f}<extra></extra>"
            ),
            name="Monthly mean AQI",
        )
    )
    overall = float(df["aqi"].mean())
    fig.add_hline(
        y=overall,
        line=dict(color=PALETTE["dark"], width=1.2, dash="dash"),
        annotation_text=f"Overall mean {overall:.1f}",
        annotation_position="top right",
        annotation_font=dict(size=10, color=PALETTE["muted"]),
    )
    fig.update_yaxes(title_text="AQI (monthly mean)")
    fig.update_xaxes(title_text="Month")
    return _apply_layout(fig, "Monthly mean AQI", height=390, showlegend=False)


def monthly_pollutant_grid(df: pd.DataFrame) -> go.Figure:
    """Small-multiple monthly trends for all six pollutants."""
    available = [column for column in POLLUTANT_META if column in df.columns]
    if df.empty or not available:
        return _empty_figure("No pollutant observations available.")

    rows, cols = 2, 3
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[
            f"{POLLUTANT_META[column]['label']} ({POLLUTANT_META[column]['unit']})"
            for column in available
        ],
        vertical_spacing=0.17,
        horizontal_spacing=0.08,
    )

    for index, column in enumerate(available):
        row = index // cols + 1
        col = index % cols + 1
        series = df[column].resample("MS").mean().dropna()
        if series.empty:
            continue
        meta = POLLUTANT_META[column]
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines+markers",
                name=meta["label"],
                line=dict(width=1.9, color=PALETTE["primary"]),
                marker=dict(size=3.5, color=PALETTE["primary_light"]),
                showlegend=False,
                hovertemplate=(
                    "%{x|%b %Y}<br>"
                    + meta["label"]
                    + ": %{y:.2f} "
                    + meta["unit"]
                    + "<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )
        fig.update_yaxes(title_text=meta["unit"], row=row, col=col)
        fig.update_xaxes(row=row, col=col, showgrid=False)

    fig.update_annotations(font=dict(size=11, color=PALETTE["muted"]))
    return _apply_layout(fig, "Monthly pollutant trends", height=560, showlegend=False)


def category_donut(df: pd.DataFrame) -> go.Figure:
    """Share of days in each AQI category."""
    if df.empty or "aqi_category" not in df.columns:
        return _empty_figure("No AQI category data available.")

    labels = [band["label"] for band in AQI_BANDS]
    counts = df["aqi_category"].value_counts().reindex(labels, fill_value=0)
    present = counts[counts > 0]
    if present.empty:
        return _empty_figure("No AQI category data available.")

    colors = [
        AQI_BAND_COLORS.get(label, PALETTE["muted"]) for label in present.index
    ]
    fig = go.Figure(
        go.Pie(
            labels=present.index.tolist(),
            values=present.values.tolist(),
            hole=0.62,
            sort=False,
            direction="clockwise",
            marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
            textinfo="percent",
            textfont=dict(size=12, color="#FFFFFF", family=FONT_FAMILY),
            hovertemplate="%{label}<br>%{value} days (%{percent})<extra></extra>",
        )
    )
    fig.add_annotation(
        text=f"<b>{int(present.sum()):,}</b><br><span style='font-size:11px'>days</span>",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=18, color="#14281D", family=FONT_FAMILY),
    )
    fig.update_layout(
        legend=dict(
            orientation="h",
            y=-0.08,
            x=0.5,
            xanchor="center",
            font=dict(size=10),
        )
    )
    return _apply_layout(fig, "AQI category distribution", height=420)


def pollutant_correlation(df: pd.DataFrame) -> go.Figure:
    """Correlation heatmap for pollutants and AQI."""
    columns = [column for column in POLLUTANT_META if column in df.columns]
    if "aqi" in df.columns:
        columns.append("aqi")
    if len(columns) < 2:
        return _empty_figure("Not enough pollutant columns for a correlation matrix.")

    labels = [POLLUTANT_META[c]["label"] if c in POLLUTANT_META else "AQI" for c in columns]
    corr = df[columns].corr(numeric_only=True)

    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale=[
                [0.0, "#B71C1C"],
                [0.5, "#FFFFFF"],
                [1.0, "#1B5E20"],
            ],
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=11, family=FONT_FAMILY),
            hovertemplate="%{y} × %{x}<br>r = %{z:.2f}<extra></extra>",
            colorbar=dict(title="r", thickness=12, len=0.75),
        )
    )
    fig.update_xaxes(side="bottom")
    return _apply_layout(fig, "Pollutant correlation matrix", height=460, showlegend=False)


def pollutant_scatter(df: pd.DataFrame, pollutant: str) -> go.Figure:
    """Scatter of one pollutant against AQI with a linear trend line."""
    if pollutant not in df.columns or "aqi" not in df.columns or df.empty:
        return _empty_figure("Pollutant data unavailable.")

    meta = POLLUTANT_META[pollutant]
    colors = df["aqi_color"].tolist() if "aqi_color" in df.columns else PALETTE["primary"]
    fig = go.Figure(
        go.Scatter(
            x=df[pollutant],
            y=df["aqi"],
            mode="markers",
            name=meta["label"],
            marker=dict(size=7, color=colors, opacity=0.72, line=dict(width=0)),
            customdata=df.index.strftime("%d %b %Y"),
            hovertemplate=(
                "%{customdata}<br>"
                + meta["label"]
                + ": %{x:.1f} "
                + meta["unit"]
                + "<br>AQI: %{y:.1f}<extra></extra>"
            ),
        )
    )

    paired = df[[pollutant, "aqi"]].dropna()
    if len(paired) >= 2:
        slope, intercept = np.polyfit(paired[pollutant], paired["aqi"], 1)
        x_line = np.linspace(paired[pollutant].min(), paired[pollutant].max(), 50)
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=slope * x_line + intercept,
                mode="lines",
                name="Linear trend",
                line=dict(color=PALETTE["dark"], width=1.6, dash="dash"),
                hoverinfo="skip",
            )
        )
        corr = float(paired[pollutant].corr(paired["aqi"]))
        fig.add_annotation(
            x=0.02,
            y=0.97,
            xref="paper",
            yref="paper",
            showarrow=False,
            text=f"Pearson r = {corr:.3f}",
            font=dict(size=11, color=PALETTE["muted"]),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=PALETTE["border"],
            borderpad=4,
        )

    fig.update_xaxes(title_text=f"{meta['label']} ({meta['unit']})")
    fig.update_yaxes(title_text="AQI (index)")
    return _legend_below(
        _apply_layout(fig, f"AQI vs {meta['label']}", height=440)
    )


def forecast_history(
    df: pd.DataFrame,
    origin_date: pd.Timestamp,
    target_date: pd.Timestamp,
    predicted_aqi: float,
    history_days: int = 90,
) -> go.Figure:
    """Historical AQI plus a clearly separated model forecast point."""
    if df.empty or "aqi" not in df.columns:
        return _empty_figure("No history available for the forecast chart.")

    history = df.loc[:origin_date].tail(history_days)
    if history.empty:
        return _empty_figure("No history available for the forecast chart.")

    color = aqi_category(predicted_aqi)["color"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history["aqi"],
            mode="lines",
            name="Historical observation",
            line=dict(color=PALETTE["primary"], width=2.1),
            fill="tozeroy",
            fillcolor="rgba(95, 107, 122, 0.05)",
            hovertemplate="%{x|%d %b %Y}<br>Observed AQI: %{y:.1f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[origin_date, target_date],
            y=[float(history["aqi"].iloc[-1]), predicted_aqi],
            mode="lines",
            line=dict(color=PALETTE["forecast"], width=1.8, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[target_date],
            y=[predicted_aqi],
            mode="markers+text",
            name="Model forecast",
            marker=dict(
                symbol="star",
                size=18,
                color=color,
                line=dict(color="#FFFFFF", width=1.5),
            ),
            text=[f"Forecast {predicted_aqi:.1f}"],
            textposition="top center",
            textfont=dict(size=11, color=PALETTE["dark"]),
            hovertemplate=(
                "%{x|%d %b %Y}<br>Forecast AQI: %{y:.1f}"
                "<br><i>Model-generated, not observed</i><extra></extra>"
            ),
        )
    )
    fig.add_vline(
        x=origin_date,
        line=dict(color=PALETTE["muted"], width=1, dash="dot"),
        opacity=0.7,
    )
    fig.add_annotation(
        x=origin_date,
        y=1.04,
        xref="x",
        yref="paper",
        text="Forecast origin",
        showarrow=False,
        font=dict(size=10, color=PALETTE["muted"]),
    )
    fig.update_yaxes(title_text="AQI (index)")
    fig.update_xaxes(title_text="Date")
    return _apply_layout(fig, "Recent observations and next-day forecast", height=410)


def aqi_gauge(value: float, title: str = "Forecast AQI") -> go.Figure:
    """Gauge showing the predicted AQI against category bands."""
    category = aqi_category(value)
    steps = [
        dict(range=[band["lower"], band["upper"]], color=_rgba(band["color"], 0.16))
        for band in AQI_BANDS
    ]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(value),
            number=dict(
                font=dict(size=40, color="#14281D", family=FONT_FAMILY),
                suffix="",
            ),
            gauge=dict(
                axis=dict(
                    range=[0, 500],
                    tickwidth=1,
                    tickcolor=PALETTE["border"],
                    tickfont=dict(size=10, color=PALETTE["muted"]),
                ),
                bar=dict(color=category["color"], thickness=0.28),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=steps,
                threshold=dict(
                    line=dict(color=PALETTE["dark"], width=3),
                    thickness=0.8,
                    value=float(value),
                ),
            ),
            domain=dict(x=[0, 1], y=[0, 1]),
        )
    )
    return _apply_layout(fig, title, height=280, showlegend=False)


def _date_labels(df: pd.DataFrame) -> list[str]:
    """Return per-row ``origin → target`` labels for evaluation hover text."""
    if {"origin_date", "target_date"} <= set(df.columns):
        return [
            f"Origin {pd.Timestamp(origin):%d %b} → target {pd.Timestamp(target):%d %b %Y}"
            for origin, target in zip(df["origin_date"], df["target_date"])
        ]
    if "date" in df.columns:
        return [f"Origin {value:%d %b %Y}" for value in pd.to_datetime(df["date"])]
    return ["Origin date unavailable"] * len(df)


def _legend_below(fig: go.Figure, bottom_margin: int = 92) -> go.Figure:
    """Place a horizontal legend below the x-axis title."""
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(b=bottom_margin),
    )
    return fig


def actual_vs_predicted(df: pd.DataFrame) -> go.Figure:
    """Test-set actual versus predicted AQI with the 1:1 line."""
    if df.empty or {"actual_aqi", "predicted_aqi"}.difference(df.columns):
        return _empty_figure("No test predictions available.")

    extreme = df["actual_aqi"] >= 200
    labels = np.asarray(_date_labels(df), dtype=object)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.loc[~extreme, "actual_aqi"],
            y=df.loc[~extreme, "predicted_aqi"],
            mode="markers",
            name="AQI < 200",
            marker=dict(size=8, color=PALETTE["primary_light"], opacity=0.70, line=dict(width=0)),
            customdata=labels[~extreme.to_numpy()],
            hovertemplate=(
                "%{customdata}<br>Actual (next day): %{x:.1f}"
                "<br>Predicted: %{y:.1f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.loc[extreme, "actual_aqi"],
            y=df.loc[extreme, "predicted_aqi"],
            mode="markers",
            name="AQI ≥ 200",
            marker=dict(size=10, color=PALETTE["danger"], opacity=0.85, line=dict(width=0)),
            customdata=labels[extreme.to_numpy()],
            hovertemplate=(
                "%{customdata}<br>Actual (next day): %{x:.1f}"
                "<br>Predicted: %{y:.1f}<extra></extra>"
            ),
        )
    )

    values = pd.concat([df["actual_aqi"], df["predicted_aqi"]])
    low = float(values.min()) - 8
    high = float(values.max()) + 8
    fig.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            name="Perfect 1:1",
            line=dict(color=PALETTE["dark"], width=1.4, dash="dash"),
            hoverinfo="skip",
        )
    )

    mae = float(np.mean(np.abs(df["actual_aqi"] - df["predicted_aqi"])))
    rmse = float(np.sqrt(np.mean((df["actual_aqi"] - df["predicted_aqi"]) ** 2)))
    ss_res = float(np.sum((df["actual_aqi"] - df["predicted_aqi"]) ** 2))
    ss_tot = float(np.sum((df["actual_aqi"] - df["actual_aqi"].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="left",
        text=f"MAE {mae:.2f}<br>RMSE {rmse:.2f}<br>R² {r2:.3f}",
        font=dict(size=11, color=PALETTE["text"]),
        bgcolor="rgba(255,255,255,0.90)",
        bordercolor=PALETTE["border"],
        borderpad=5,
    )

    fig.update_xaxes(title_text="Actual next-day AQI", range=[low, high])
    fig.update_yaxes(
        title_text="Predicted next-day AQI",
        range=[low, high],
        scaleanchor="x",
        scaleratio=1,
    )
    return _legend_below(
        _apply_layout(fig, "Actual vs predicted AQI (test set)", height=560)
    )


def residual_chart(df: pd.DataFrame) -> go.Figure:
    """Residuals against predicted AQI, coloured by actual severity."""
    if df.empty or {"predicted_aqi", "residual"}.difference(df.columns):
        return _empty_figure("No residuals available.")

    colors = (
        df["aqi_range"].map(
            lambda label: aqi_category(
                {
                    "Moderate (51-100)": 75,
                    "Unhealthy for Sensitive Groups (101-150)": 125,
                    "Unhealthy (151-200)": 175,
                    "Very Unhealthy (201-300)": 250,
                }.get(str(label), 175)
            )["color"]
        )
        if "aqi_range" in df.columns
        else PALETTE["primary_light"]
    )
    fig = go.Figure(
        go.Scatter(
            x=df["predicted_aqi"],
            y=df["residual"],
            mode="markers",
            marker=dict(size=8, color=colors, opacity=0.72, line=dict(width=0)),
            customdata=_date_labels(df),
            hovertemplate=(
                "%{customdata}<br>Predicted: %{x:.1f}<br>Residual: %{y:.1f}<extra></extra>"
            ),
            name="Residual",
        )
    )
    fig.add_hline(y=0, line=dict(color=PALETTE["dark"], width=1.3, dash="dash"))
    fig.update_xaxes(title_text="Predicted next-day AQI")
    fig.update_yaxes(title_text="Residual (actual − predicted)")
    return _apply_layout(fig, "Prediction residuals", height=430, showlegend=False)


def model_comparison_chart(df: pd.DataFrame) -> go.Figure:
    """Validation RMSE for each candidate and the baseline."""
    if df.empty or "rmse" not in df.columns:
        return _empty_figure("No model comparison data available.")

    ordered = df.dropna(subset=["rmse"]).sort_values("rmse", ascending=False)
    if ordered.empty:
        return _empty_figure("No model comparison data available.")

    colors = []
    for row in ordered.itertuples():
        if bool(getattr(row, "selected", False)):
            colors.append(PALETTE["primary"])
        elif getattr(row, "kind", "") == "baseline":
            colors.append("#8A9A8A")
        else:
            colors.append("#C8D8C8")

    fig = go.Figure(
        go.Bar(
            x=ordered["rmse"],
            y=ordered["model"],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{value:.2f}" for value in ordered["rmse"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>Validation RMSE: %{x:.2f}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Validation RMSE (lower is better)")
    fig.update_yaxes(title_text="")
    return _apply_layout(fig, "Candidate models — validation RMSE", height=360, showlegend=False)


def feature_importance_chart(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of the most important model features."""
    if df.empty or {"feature", "importance"}.difference(df.columns):
        return _empty_figure("Feature importance is not available for this model.")

    top = df.head(top_n).iloc[::-1]
    height = max(340, 24 * len(top) + 100)
    fig = go.Figure(
        go.Bar(
            x=top["importance"],
            y=top["feature"],
            orientation="h",
            marker=dict(color=PALETTE["primary_light"], line=dict(width=0)),
            hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Importance")
    fig.update_yaxes(title_text="")
    return _apply_layout(fig, f"Top {len(top)} features", height=height, showlegend=False)


def error_by_range_chart(df: pd.DataFrame) -> go.Figure:
    """Model vs baseline MAE by actual AQI range."""
    if df.empty or "n" not in df.columns:
        return _empty_figure("No range-level error data available.")

    present = df[df["n"] > 0].copy()
    if present.empty:
        return _empty_figure("No populated AQI ranges in the test set.")

    short_labels = {
        "Moderate (51-100)": "Moderate",
        "Unhealthy for Sensitive Groups (101-150)": "Sensitive groups",
        "Unhealthy (151-200)": "Unhealthy",
        "Very Unhealthy (201-300)": "Very unhealthy",
    }
    labels = [
        f"{short_labels.get(value, value)} (n={int(count)})"
        for value, count in zip(present["aqi_range"], present["n"])
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=present["mae"],
            name="Selected model",
            marker=dict(color=PALETTE["primary"], line=dict(width=0)),
            hovertemplate="%{x}<br>Model MAE: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=present["baseline_mae"],
            name="Persistence baseline",
            marker=dict(color="#B8C6B8", line=dict(width=0)),
            hovertemplate="%{x}<br>Baseline MAE: %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Actual AQI range")
    fig.update_yaxes(title_text="MAE (AQI units, lower is better)")
    return _apply_layout(fig, "Error by AQI range", height=400)


def monthly_heatmap(df: pd.DataFrame) -> go.Figure:
    """Year × month mean AQI heatmap."""
    if df.empty or "aqi" not in df.columns:
        return _empty_figure("No monthly AQI data available.")

    pivot = df.pivot_table(index="year", columns="month", values="aqi", aggfunc="mean")
    if pivot.empty:
        return _empty_figure("No monthly AQI data available.")

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=[month_names[int(month) - 1] for month in pivot.columns],
            y=[str(year) for year in pivot.index],
            colorscale=[
                [0.0, "#2E7D32"],
                [0.35, "#A6C46A"],
                [0.55, "#E0B84A"],
                [0.75, "#D9691B"],
                [1.0, "#C62828"],
            ],
            zmin=0,
            zmax=300,
            text=np.round(pivot.values, 0),
            texttemplate="%{text}",
            textfont=dict(size=11, family=FONT_FAMILY, color="#14281D"),
            hovertemplate="%{y} %{x}<br>Mean AQI: %{z:.1f}<extra></extra>",
            colorbar=dict(title="AQI", thickness=12, len=0.75),
        )
    )
    fig.update_xaxes(side="bottom")
    fig.update_yaxes(type="category")
    return _apply_layout(fig, "Monthly mean AQI by year", height=320, showlegend=False)


def high_pollution_timeline(df: pd.DataFrame, threshold: float = 200) -> go.Figure:
    """AQI timeline with days above the extreme threshold highlighted."""
    if df.empty or "aqi" not in df.columns:
        return _empty_figure("No pollution history available.")

    high = df[df["aqi"] >= threshold]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["aqi"],
            mode="lines",
            name="Observed AQI",
            line=dict(color=PALETTE["primary"], width=1.8),
            hovertemplate="%{x|%d %b %Y}<br>AQI: %{y:.1f}<extra></extra>",
        )
    )
    if not high.empty:
        fig.add_trace(
            go.Scatter(
                x=high.index,
                y=high["aqi"],
                mode="markers",
                name=f"AQI ≥ {threshold:.0f}",
                marker=dict(
                    size=8,
                    color=PALETTE["danger"],
                    line=dict(color="#FFFFFF", width=1),
                ),
                hovertemplate="%{x|%d %b %Y}<br>High pollution: %{y:.1f}<extra></extra>",
            )
        )
    fig.add_hline(
        y=threshold,
        line=dict(color=PALETTE["danger"], width=1.2, dash="dot"),
        annotation_text=f"{threshold:.0f} threshold",
        annotation_position="top left",
        annotation_font=dict(size=10, color=PALETTE["danger"]),
    )
    fig.update_yaxes(title_text="AQI (index)")
    fig.update_xaxes(title_text="Date")
    return _apply_layout(fig, "High-pollution periods", height=380)
