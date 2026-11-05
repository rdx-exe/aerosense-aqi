"""Unit tests for the feature engineering module."""

import pandas as pd
import pytest
from src.feature_engineering.forecast_features import extract_season, create_forecasting_features, split_chronological_data


def test_extract_season() -> None:
    assert extract_season(1) == "Winter"
    assert extract_season(3) == "Spring"
    assert extract_season(6) == "Summer"
    assert extract_season(9) == "Autumn"


def test_create_forecasting_features() -> None:
    # Create a small test DataFrame with required columns
    df = pd.DataFrame({
        "date": pd.date_range("2015-01-01", periods=30, freq="D"),
        "aqi": [100 + i * 10 for i in range(30)],  # Simple increasing sequence
        "pm2.5": [50 + i for i in range(30)],
        "pm10": [80 + i * 2 for i in range(30)],
    })

    df.to_csv("data/raw/test_cleaned.csv", index=False)

    feature_df, feature_dict = create_forecasting_features(
        clean_path="data/raw/test_cleaned.csv",
        output_path="data/processed/test_features.csv",
        feature_dict_path="docs/test_feature_dictionary.md"
    )

    # Check that target exists
    assert "aqi_next_day" in feature_df.columns

    # Check that lag features exist
    assert "aqi_lag1" in feature_df.columns
    assert "aqi_lag3" in feature_df.columns
    assert "aqi_lag7" in feature_df.columns

    # Check that rolling features exist
    assert "aqi_roll3_mean" in feature_df.columns
    assert "aqi_roll7_mean" in feature_df.columns

    # Check that temporal features exist
    assert "day_of_week" in feature_df.columns
    assert "month" in feature_df.columns
    assert "season" in feature_df.columns

    # Check that feature dictionary is created
    assert feature_dict["target"] == "aqi_next_day"


def test_split_chronological_data() -> None:
    # Create a larger test DataFrame
    dates = pd.date_range("2015-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "aqi": [100 + i for i in range(100)],
    })
    df.to_csv("data/raw/test_split.csv", index=False)

    # First create features
    feature_df, _ = create_forecasting_features(
        clean_path="data/raw/test_split.csv",
        output_path="data/processed/test_split_features.csv"
    )

    # Split data
    splits = split_chronological_data(feature_df)

    # Check that splits exist and are non-empty
    assert len(splits["train"]) > 0
    assert len(splits["validation"]) > 0
    assert len(splits["test"]) > 0

    # Check that train comes before validation
    assert splits["train"].index.max() < splits["validation"].index.min()

    # Check that validation comes before test
    assert splits["validation"].index.max() < splits["test"].index.min()

    # Check that ratios are approximately correct
    total = len(splits["train"]) + len(splits["validation"]) + len(splits["test"])
    assert total == len(feature_df)