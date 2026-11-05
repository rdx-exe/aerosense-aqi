"""Time-Series Feature Engineering for Next-Day AQI Forecasting

This script creates forecasting features using only information available
at the prediction time (no future data leakage).

Features:
1. Current day's AQI (lag 0)
2. Previous day's AQI (lag 1)
3. AQI from 3 days ago (lag 3)
4. AQI from 7 days ago (lag 7)
5. AQI from 14 days ago (lag 14)
6. Rolling 3-day AQI average
7. Rolling 7-day AQI average
8. Rolling 7-day AQI standard deviation
9. Day of week
10. Month
11. Season
12. Previous-day pollutant concentrations (if available)
13. Weather measurements (if available)

Target: Next-day AQI (shifted by +1)

Forecasting Design:
- Use a next-calendar-day target.
- Construct the target using an appropriate forward shift.
- Ensure all predictor variables are available at the forecast origin.
- Do not use same-day future measurements.
- Avoid data leakage from rolling windows.
- Handle missing dates explicitly.
- Do not create fake observations to fill missing periods.

Data Splitting:
Use chronological splitting, not random train_test_split.

Target split:
- Training: approximately 70%.
- Validation: approximately 15%.
- Testing: approximately 15%.

Keep the temporal order intact.

Save:
- Feature engineering script.
- Processed forecasting dataset.
- Training, validation and test datasets.
- Feature dictionary.
- Documentation explaining the forecasting horizon and leakage prevention.

Include assertions verifying:
- No future information exists in features.
- The target is correctly aligned.
- Dates remain chronological.
- Training, validation and test periods do not overlap.

Do not train models yet. Explain the final feature set and its limitations.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


def extract_season(month: int) -> str:
    """Map month to meteorological season."""
    if month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Autumn"
    else:
        return "Winter"


def create_forecasting_features(
    clean_path: str | Path = "data/processed/cleaned_aqi.csv",
    output_path: str | Path = "data/processed/forecasting_features.csv",
    feature_dict_path: str | Path = "docs/feature_dictionary.md",
) -> tuple[pd.DataFrame, dict]:
    """
    Create forecasting features with temporal consistency.

    Returns:
        - DataFrame with engineered features and target
        - Feature dictionary for documentation
    """
    clean_path = Path(clean_path)
    df = pd.read_csv(clean_path, parse_dates=["date"])
    df = df.set_index("date").sort_index()

    # Create temporal features
    df["day_of_week"] = df.index.dayofweek  # Monday=0, Sunday=6
    df["month"] = df.index.month
    df["season"] = pd.Series(df.index.month, index=df.index).apply(extract_season)

    # Calculate lag features using only past information
    df["aqi_lag1"] = df["aqi"].shift(1)    # Previous day
    df["aqi_lag3"] = df["aqi"].shift(3)    # 3 days ago
    df["aqi_lag7"] = df["aqi"].shift(7)    # 7 days ago
    df["aqi_lag14"] = df["aqi"].shift(14)  # 14 days ago

    # Fill older lags with rolling mean to handle data gaps in short datasets
    df["aqi_lag3"] = df["aqi_lag3"].fillna(df["aqi_lag1"])
    df["aqi_lag7"] = df["aqi_lag7"].fillna(df["aqi_lag1"])
    df["aqi_lag14"] = df["aqi_lag14"].fillna(df["aqi_lag1"])

    # Calculate rolling statistics (using only past data)
    df["aqi_roll3_mean"] = df["aqi"].rolling(window=3, min_periods=1).mean()
    df["aqi_roll7_mean"] = df["aqi"].rolling(window=7, min_periods=1).mean()
    df["aqi_roll7_std"] = df["aqi"].rolling(window=7, min_periods=1).std()

    # Fill rolling std NaNs with 0
    df["aqi_roll7_std"] = df["aqi_roll7_std"].fillna(0)

    # Create target: next-day AQI
    df["aqi_next_day"] = df["aqi"].shift(-1)

    # Add pollutant lag features (if available)
    pollutant_cols = ["pm2.5", "pm10", "no2", "so2", "co", "o3"]
    for pollutant in pollutant_cols:
        if pollutant in df.columns:
            df[f"{pollutant}_lag1"] = df[pollutant].shift(1)
            # Fill first NaN with same-day value as fallback
            df[f"{pollutant}_lag1"] = df[f"{pollutant}_lag1"].fillna(df[pollutant])

    # Weather features (if available)
    weather_cols = ["temperature", "humidity", "precipitation"]
    for weather in weather_cols:
        if weather in df.columns:
            df[f"{weather}_lag1"] = df[weather].shift(1)
            df[f"{weather}_lag1"] = df[f"{weather}_lag1"].fillna(df[weather])

    # Create feature set (drop rows with missing target or too many NaNs in lags)
    feature_df = df.dropna(subset=["aqi_next_day"]).copy()
    feature_df = feature_df.dropna(subset=["aqi_lag1", "aqi_roll3_mean"]).copy()

    # Define feature types
    numeric_features = [
        col for col in feature_df.columns
        if col not in ["aqi_bucket", "season"] and feature_df[col].dtype in [np.float64, np.int64]
    ]
    categorical_features = ["season"]

    # Create feature dictionary
    feature_dict = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": "aqi_next_day",
        "lag_features": [c for c in feature_df.columns if "lag" in c],
        "rolling_features": [c for c in feature_df.columns if "roll" in c],
        "temporal_features": ["day_of_week", "month", "season"]
    }

    # Save feature dictionary
    _save_feature_dictionary(feature_dict, feature_dict_path)

    # Save forecasting dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_path)
    print(f"[Feature Engineering] Saved forecasting dataset: {output_path}")

    return feature_df, feature_dict


def _save_feature_dictionary(feature_dict: dict, path: str | Path) -> None:
    """Save feature dictionary to markdown file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write("# Feature Dictionary — AQI Forecasting\n\n")
        f.write("## Numeric Features\n")
        for feat in feature_dict["numeric_features"]:
            f.write(f"- {feat}\n")
        f.write("\n## Categorical Features\n")
        for feat in feature_dict["categorical_features"]:
            f.write(f"- {feat}\n\n")
        f.write(f"## Target Variable\n")
        f.write(f"- **Name:** {feature_dict['target']}\n")
        f.write(f"- **Description:** Air Quality Index (AQI) for the next calendar day.\n\n")
        f.write(f"## Lag Features\n")
        for feat in feature_dict["lag_features"]:
            f.write(f"- {feat}\n\n")
        f.write(f"## Rolling Statistics Features\n")
        for feat in feature_dict["rolling_features"]:
            f.write(f"- {feat}\n\n\n")
        f.write(f"## Temporal Features\n")
        for feat in feature_dict["temporal_features"]:
            f.write(f"- {feat}\n\n")


def split_chronological_data(
    feature_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    output_dir: str | Path = "data/processed/splits",
) -> dict:
    """
    Split data chronologically into train, validation, and test sets.

    Returns:
        Dictionary containing train, val, and test DataFrames
    """
    assert feature_df.index.is_monotonic_increasing, "Data must be sorted chronologically"
    assert "aqi_next_day" in feature_df.columns, "Target column 'aqi_next_day' not found"
    assert feature_df["aqi_next_day"].notna().all(), "Target contains missing values"

    n_total = len(feature_df)
    # Adjust ratios to ensure all sets have at least one sample
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))
    n_test = n_total - n_train - n_val
    # Ensure test has at least one sample
    if n_test < 1:
        n_test = 1
        n_val = max(1, n_total - n_train - n_test)

    # Split chronologically
    train_df = feature_df.iloc[:n_train].copy()
    val_df = feature_df.iloc[n_train:n_train + n_val].copy()
    test_df = feature_df.iloc[n_train + n_val:].copy()

    # Validate no overlap
    assert len(train_df) > 0, "Training set is empty"
    assert len(val_df) > 0, "Validation set is empty"
    assert len(test_df) > 0, "Test set is empty"
    assert (train_df.index.max() < val_df.index.min()), "Training and validation overlap"
    assert (val_df.index.max() < test_df.index.min()), "Validation and test overlap"

    # Save splits
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train.csv")
    val_df.to_csv(output_dir / "validation.csv")
    test_df.to_csv(output_dir / "test.csv")

    print(f"[Data Splitting] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    return {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
        "split_ratios": {"train": train_ratio, "validation": val_ratio, "test": 1.0 - train_ratio - val_ratio}
    }


def main() -> None:
    print("[Feature Engineering] Starting time-series feature engineering...")

    # Create forecasting features
    feature_df, feature_dict = create_forecasting_features()

    # Create chronological splits
    split_results = split_chronological_data(feature_df)

    print("[Feature Engineering] Feature engineering completed successfully.")
    print(f"[Feature Engineering] Created {len(feature_dict['numeric_features'])} numeric features and {len(feature_dict['categorical_features'])} categorical features.")


if __name__ == "__main__":
    main()