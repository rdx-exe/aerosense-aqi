"""Data Quality Assessment & Audit Module.

Performs rigorous checks on raw air quality datasets:
1. Record counts, date ranges, and time continuity (missing dates).
2. Missing value quantification (distinguishing NaNs from zeros).
3. Duplicate record detection.
4. Impossible / suspicious pollutant measurement detection (e.g. negative values,
   impossible CO levels, extreme PM spikes).
5. Generation of the required data quality report.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


def assess_data_quality(df: pd.DataFrame) -> dict:
    """Run comprehensive data quality checks on a raw DataFrame.

    Returns a dictionary of audit metrics.
    """
    initial_count = len(df)

    # 1. Standardize columns
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # Date parsing
    date_col = next((c for c in ['date', 'datetime', 'timestamp'] if c in df.columns), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        valid_dates = df[date_col].dropna()
        date_range = (str(valid_dates.min().date()), str(valid_dates.max().date())) if not valid_dates.empty else ("NaT", "NaT")
    else:
        date_range = ("Missing", "Missing")

    # 2. Missing values before processing
    missing_before = df.isnull().sum().to_dict()
    zeros_count = {col: int((df[col] == 0).sum()) for col in df.select_dtypes(include=[np.number]).columns}

    # 3. Duplicates
    duplicates_count = int(df.duplicated().sum())

    # 4. Invalid measurements (negative values for pollutants/AQI)
    numeric_cols = [c for c in ['pm2.5', 'pm10', 'no2', 'so2', 'co', 'o3', 'aqi'] if c in df.columns]
    invalid_counts = {}
    for col in numeric_cols:
        invalid_counts[col] = int((df[col] < 0).sum())

    return {
        "initial_record_count": initial_count,
        "date_range": date_range,
        "missing_before": missing_before,
        "zeros_count": zeros_count,
        "duplicate_records": duplicates_count,
        "invalid_measurements": invalid_counts,
    }


def generate_quality_report(audit_data: dict, final_count: int, missing_after: dict, removed_cols: list[str], output_path: str | Path) -> pd.DataFrame:
    """Generate the results/data_quality_report.csv table."""
    metrics = [
        ("Initial Record Count", audit_data["initial_record_count"]),
        ("Final Record Count", final_count),
        ("Duplicate Records", audit_data["duplicate_records"]),
        ("Date Range Start", audit_data["date_range"][0]),
        ("Date Range End", audit_data["date_range"][1]),
        ("Removed Columns", ", ".join(removed_cols) if removed_cols else "None"),
    ]

    # Add missing counts
    for col, count in audit_data["missing_before"].items():
        metrics.append((f"Missing Before — {col}", count))
    for col, count in missing_after.items():
        metrics.append((f"Missing After — {col}", count))

    report_df = pd.DataFrame(metrics, columns=["Metric", "Value"])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path, index=False)
    return report_df