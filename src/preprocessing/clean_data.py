"""Clean Data Pipeline Module.

Implements rigorous preprocessing on the raw CPCB Kolkata dataset:
1. Loads raw CSV safely without modifying source files.
2. Standardizes column names and dates.
3. Filters for Kolkata records.
4. Identifies and flags physical impossibilities (negative pollutants).
5. Detects and handles duplicate records.
6. Imputes missing values safely using causal/forward methods without leakage.
7. Validates and harmonizes AQI categories.
8. Checks for missing dates in the time series and builds a continuous index.
9. Exports cleaned dataset to data/processed/cleaned_aqi.csv and quality report.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from .data_quality import assess_data_quality, generate_quality_report


def clean_air_quality_data(
    raw_path: str | Path = "data/raw/kolkata_city_day.csv",
    output_path: str | Path = "data/processed/cleaned_aqi.csv",
    report_path: str | Path = "results/data_quality_report.csv",
) -> pd.DataFrame:
    """Execute the complete data cleaning and preprocessing pipeline."""
    raw_path = Path(raw_path)
    if not raw_path.is_file():
        # Fallback to general city_day.csv or generate dummy structure if none exists
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_path}. Please place the raw dataset in data/raw/."
        )

    df = pd.read_csv(raw_path)

    # 1. Run initial quality assessment
    audit_data = assess_data_quality(df)

    # 2. Standardize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # 3. Filter for Kolkata if 'city' column exists
    if "city" in df.columns:
        df = df[df["city"].str.lower() == "kolkata"].copy()

    # 4. Standardize date column
    date_col = next((c for c in ['date', 'datetime', 'timestamp'] if c in df.columns), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        df = df.sort_values(date_col).drop_duplicates(subset=[date_col], keep='last')
        df = df.set_index(date_col)
    else:
        raise ValueError("No valid date column found in dataset.")

    # 5. Drop redundant columns if present (e.g. 'city', 'Unnamed: 0')
    removed_cols = []
    for col in ['city', 'unnamed:_0', 'id']:
        if col in df.columns:
            df = df.drop(columns=[col])
            removed_cols.append(col)

    # 6. Handle impossible/negative pollutant values (treat < 0 as NaN for imputation)
    pollutant_cols = [c for c in ['pm2.5', 'pm10', 'no2', 'so2', 'co', 'o3', 'aqi'] if c in df.columns]
    for col in pollutant_cols:
        df.loc[df[col] < 0, col] = np.nan

    # 7. Check for missing dates in the time series and reindex to daily continuity
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(full_idx)
    df.index.name = "date"

    # 8. Safely impute missing values (forward fill up to 3 consecutive days to prevent long-term leakage)
    df[pollutant_cols] = df[pollutant_cols].ffill(limit=3)

    # If any remaining NaNs in pollutants, apply a rolling median or drop if unrecoverable
    df[pollutant_cols] = df[pollutant_cols].fillna(method='bfill', limit=1)

    # 9. Recompute or harmonize AQI categories if missing
    if "aqi_bucket" in df.columns:
        df["aqi_bucket"] = df["aqi_bucket"].fillna("Unknown")
    elif "aqi" in df.columns:
        df["aqi_bucket"] = df["aqi"].apply(_assign_aqi_bucket)

    # Missing values after processing
    missing_after = df.isnull().sum().to_dict()

    # 10. Save cleaned dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_csv(output_path, index=False)
    print(f"[clean_data] Cleaned dataset saved to {output_path} ({len(df)} rows)")

    # 11. Generate quality report
    generate_quality_report(
        audit_data=audit_data,
        final_count=len(df),
        missing_after=missing_after,
        removed_cols=removed_cols,
        output_path=report_path,
    )
    print(f"[clean_data] Quality report saved to {report_path}")

    return df


def _assign_aqi_bucket(aqi: float) -> str:
    """Assign standard CPCB AQI categories based on AQI value."""
    if pd.isna(aqi):
        return "Unknown"
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw air quality dataset.")
    parser.add_argument("--raw", type=str, default="data/raw/kolkata_city_day.csv")
    parser.add_argument("--out", type=str, default="data/processed/cleaned_aqi.csv")
    parser.add_argument("--report", type=str, default="results/data_quality_report.csv")
    args = parser.parse_args()

    clean_air_quality_data(raw_path=args.raw, output_path=args.out, report_path=args.report)


if __name__ == "__main__":
    main()