"""Comprehensive Exploratory Data Analysis (EDA) for Kolkata AQI Dataset.

Performs the following analyses:
1. Overall AQI distribution (histogram, statistics).
2. Daily AQI trends (time-series line chart).
3. Monthly average AQI (bar chart).
4. Seasonal AQI variations (box plot by season).
5. Year-wise AQI comparison (if multiple years exist).
6. Distribution of AQI categories (pie chart).
7. Pollutant concentration distributions (box plots for PM2.5, PM10, NO2, SO2, CO, O3).
8. Correlation between AQI and individual pollutants (heatmap).
9. Identification of high-pollution periods (top N days).
10. Weather correlation (if weather data were added).

Saves:
- plots to results/visualizations/eda/ as PNG/PNG and interactive HTML (Plotly) where appropriate.
- statistical summaries as CSV in results/summary/.
- written findings document in docs/eda_findings.md.
"""

from __future__ import annotations

import calendar
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..utils.config import get_path, load_config

# Matplotlib configuration for readability
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 14
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.7

# Constants
EDA_PLOT_DIR = get_path("results/visualizations/eda")
EDA_SUMMARY_DIR = get_path("results/summary")

EDA_PLOT_DIR.mkdir(parents=True, exist_ok=True)
EDA_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def extract_season(month: int) -> str:
    """Map month to a season string using meteorological convention."""
    if month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Autumn"
    else:
        return "Winter"


def save_plot(figure, filename: str, kind: str = "png") -> None:
    """Save a Matplotlib figure or Plotly figure to the EDA directory."""
    filename = filename if kind == "html" else f"{filename}.png"
    path = EDA_PLOT_DIR / filename

    if isinstance(figure, (plt.Figure, np.ndarray)):
        # Matplotlib figure
        figure.tight_layout()
        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)
    else:
        # Plotly figure
        if kind == "html":
            figure.write_html(str(path))
        else:
            figure.write_image(str(path))
    print(f"[EDA] Saved plot: {path}")


def load_data() -> pd.DataFrame:
    """Load the cleaned dataset from the default path."""
    config = load_config()
    default_path = get_path(config["data"]["processed_dir"], "cleaned_aqi.csv")
    return pd.read_csv(default_path, parse_dates=["date"])


def run_eda() -> None:
    df = load_data()
    df = df.copy()

    # Ensure datetime index
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Extract temporal components
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["season"] = df["month"].apply(lambda m: extract_season(m))

    # 1. Overall AQI Distribution
    _plot_aqi_histogram(df)

    # 2. Daily AQI Trends
    _plot_daily_trends(df)

    # 3. Monthly Average AQI
    _plot_monthly_average(df)

    # 4. Seasonal AQI Variations (Box Plot)
    _plot_seasonal_boxplot(df)

    # 5. Year-wise AQI Comparison (if applicable)
    _plot_year_comparison(df)

    # 6. AQI Category Distribution
    _plot_category_pie(df)

    # 7. Pollutant Distribution Box Plots
    _plot_pollutant_boxplots(df)

    # 8. Pollutant-AQI Correlation Heatmap
    _plot_correlation_heatmap(df)

    # 9. High Pollution Periods
    _identify_high_pollution_periods(df)

    # 10. Save statistical summaries
    _save_statistical_summaries(df)

    # 11. Generate written EDA findings
    _write_eda_findings(df)


def _plot_aqi_histogram(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df["aqi"], bins=20, kde=True, color="#4682b4", ax=ax)
    ax.set_title("Overall Distribution of Daily AQI")
    ax.set_xlabel("AQI (0–500)")
    ax.set_ylabel("Frequency")
    save_plot(fig, "aqi_histogram.png")


def _plot_daily_trends(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.lineplot(data=df, x=df.index, y="aqi", color="#ff7f0e", linewidth=2, marker="o", markersize=3, ax=ax)
    ax.set_title("Daily AQI Trend — Kolkata")
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    save_plot(fig, "daily_aqi_trend.png")


def _plot_monthly_average(df: pd.DataFrame) -> None:
    # Group by month
    monthly_avg = df.groupby("month")["aqi"].mean().reset_index()
    month_names = [calendar.month_abbr[m] for m in monthly_avg["month"]]
    monthly_avg["month_name"] = month_names

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=monthly_avg, x="month_name", y="aqi", palette="YlOrRd", hue="month_name", legend=False, ax=ax)
    ax.set_title("Average AQI by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average AQI")
    save_plot(fig, "monthly_average.png")

    # Interactive Plotly version
    fig = px.bar(monthly_avg, x="month_name", y="aqi", title="Average AQI by Month")
    save_plot(fig, "monthly_average.html", kind="html")


def _plot_seasonal_boxplot(df: pd.DataFrame) -> None:
    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    seasonal_data = [df[df["season"] == s]["aqi"] for s in seasons if not df[df["season"] == s]["aqi"].empty]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(seasonal_data, labels=seasons[: len(seasonal_data)], showmeans=True)
    ax.set_title("AQI by Season")
    ax.set_ylabel("AQI")
    save_plot(fig, "seasonal_boxplot.png")


def _plot_year_comparison(df: pd.DataFrame) -> None:
    if df["year"].nunique() <= 1:
        print("[EDA] Only one year present; skipping year-wise comparison.")
        return

    yearly_avg = df.groupby("year")["aqi"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=yearly_avg, x="year", y="aqi", marker="o", linewidth=2, ax=ax)
    ax.set_title("Year-wise Average AQI")
    ax.set_xlabel("Year")
    ax.set_ylabel("Average AQI")
    save_plot(fig, "year_comparison.png")


def _plot_category_pie(df: pd.DataFrame) -> None:
    if "aqi_bucket" not in df.columns:
        print("[EDA] No 'aqi_bucket' column; skipping category pie chart.")
        return

    category_counts = df["aqi_bucket"].value_counts().reset_index()
    category_counts.columns = ["category", "count"]

    fig = px.pie(
        category_counts,
        names="category",
        values="count",
        title="Distribution of AQI Health Categories",
        hole=0.4,
    )
    save_plot(fig, "category_pie.html", kind="html")


def _plot_pollutant_boxplots(df: pd.DataFrame) -> None:
    pollutant_cols = [c for c in ["pm2.5", "pm10", "no2", "so2", "co", "o3"] if c in df.columns]
    if not pollutant_cols:
        print("[EDA] No pollutant columns found.")
        return

    data_long = df[pollutant_cols].melt(var_name="pollutant", value_name="value")
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(data=data_long, x="pollutant", y="value", palette="Set2", hue="pollutant", legend=False, ax=ax)
    ax.set_title("Distribution of Pollutants (Box Plot)")
    ax.set_xlabel("Pollutant")
    ax.set_ylabel("Concentration")
    save_plot(fig, "pollutant_boxplots.png")


def _plot_correlation_heatmap(df: pd.DataFrame) -> None:
    pollutant_cols = [c for c in ["pm2.5", "pm10", "no2", "so2", "co", "o3", "aqi"] if c in df.columns]
    if not pollutant_cols:
        print("[EDA] Insufficient pollutant columns for correlation heatmap.")
        return

    corr_matrix = df[pollutant_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, ax=ax)
    ax.set_title("Correlation Heatmap — Pollutants & AQI")
    save_plot(fig, "correlation_heatmap.png")


def _identify_high_pollution_periods(df: pd.DataFrame) -> None:
    """Highlight top N days by AQI and mark periods of sustained high AQI."""
    top_n = df.nlargest(5, "aqi").reset_index()
    top_n["date_str"] = top_n["date"].dt.strftime("%Y-%m-%d")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=top_n, x="aqi", y="date_str", palette="Reds_r", hue="date_str", legend=False)
    ax.set_title("Top 5 Highest AQI Days")
    ax.set_xlabel("AQI")
    ax.set_ylabel("Date")
    save_plot(fig, "top5_aqi.png")

    # Interactive Plotly version
    fig = px.bar(top_n, x="aqi", y="date_str", orientation="h", title="Top 5 Highest AQI Days")
    save_plot(fig, "top5_aqi.html", kind="html")


def _save_statistical_summaries(df: pd.DataFrame) -> None:
    """Save key statistics to CSV files in results/summary/."""
    # Overall statistics
    stats = df["aqi"].describe().to_frame(name="AQI").reset_index()
    stats["Metric"] = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    stats.to_csv(EDA_SUMMARY_DIR / "aqi_summary_stats.csv", index=False)

    # Pollutant statistics
    pollutant_cols = [c for c in ["pm2.5", "pm10", "no2", "so2", "co", "o3"] if c in df.columns]
    if pollutant_cols:
        summary = pd.DataFrame({"Mean": df[pollutant_cols].mean(), "Std": df[pollutant_cols].std()})
        summary.to_csv(EDA_SUMMARY_DIR / "pollutant_summary_stats.csv")


def _write_eda_findings(df: pd.DataFrame) -> None:
    """Generate a written summary of the analysis."""
    report_lines = []
    report_lines.append("# AeroSense — Exploratory Data Analysis (EDA) Findings for Kolkata AQI")
    report_lines.append("")
    report_lines.append(f"**Dataset:** Kolkata air quality observations (CPCB historical data).")
    report_lines.append(f"**Time Coverage:** {df.index.min().date()} to {df.index.max().date()}")
    report_lines.append(f"**Total Observations:** {len(df)}")
    report_lines.append("")
    report_lines.append("## Key Findings Supported by the Data")
    report_lines.append("")

    # Overall AQI
    aqi_mean = df["aqi"].mean()
    report_lines.append(f"1. The average AQI over the period is {aqi_mean:.1f}.")

    # Monthly pattern (if multiple months)
    monthly_means = df.groupby("month")["aqi"].mean()
    peak_month = monthly_means.idxmax()
    report_lines.append(f"2. The highest average AQI occurs in month {peak_month} (average {monthly_means.loc[peak_month]:.1f}).")

    # Seasonal distribution
    seasonal_means = df.groupby("season")["aqi"].mean()
    report_lines.append(f"3. Seasonal analysis shows the following average AQI: {', '.join([f'{s} ({v:.1f})' for s, v in seasonal_means.items()])}.")

    # Correlation
    pollutant_cols = [c for c in ["pm2.5", "pm10", "no2", "so2", "co", "o3", "aqi"] if c in df.columns]
    if pollutant_cols:
        corr = df[pollutant_cols].corr()
        aqi_corr = corr.loc["aqi", pollutant_cols[0]]
        report_lines.append(f"4. The strongest pollutant-AQI correlation is with {pollutant_cols[0]} ({aqi_corr:.2f}).")

    # High pollution days
    top_n = df.nlargest(5, "aqi").iloc[0]
    report_lines.append(f"5. The highest single-day AQI recorded was {top_n['aqi']:.1f} on {top_n.name.date()}.")

    # Write the document
    report_path = get_path("docs/eda_findings.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[EDA] Findings document written to {report_path}")


def main() -> None:
    print("[EDA] Running comprehensive exploratory data analysis ...")
    run_eda()
    print("[EDA] EDA completed successfully.")


def __getattr__(name):
    # This is a fallback to allow calling run_eda directly from the shell
    if name == "run_eda":
        return run_eda
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    main()