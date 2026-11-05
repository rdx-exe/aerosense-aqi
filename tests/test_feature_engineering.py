"""Unit tests for the legacy feature-engineering module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.feature_engineering.forecast_features import (
    create_forecasting_features,
    extract_season,
    split_chronological_data,
)


def test_extract_season() -> None:
    assert extract_season(1) == "Winter"
    assert extract_season(3) == "Spring"
    assert extract_season(6) == "Summer"
    assert extract_season(9) == "Autumn"


def test_create_forecasting_features(tmp_path: Path) -> None:
    clean_path = tmp_path / "cleaned.csv"
    output_path = tmp_path / "features.csv"
    feature_dict_path = tmp_path / "feature_dictionary.md"

    df = pd.DataFrame(
        {
            "date": pd.date_range("2015-01-01", periods=30, freq="D"),
            "aqi": [100 + i * 10 for i in range(30)],
            "pm2.5": [50 + i for i in range(30)],
            "pm10": [80 + i * 2 for i in range(30)],
        }
    )
    df.to_csv(clean_path, index=False)

    feature_df, feature_dict = create_forecasting_features(
        clean_path=clean_path,
        output_path=output_path,
        feature_dict_path=feature_dict_path,
    )

    assert "aqi_next_day" in feature_df.columns
    assert "aqi_lag1" in feature_df.columns
    assert "aqi_lag3" in feature_df.columns
    assert "aqi_lag7" in feature_df.columns
    assert "aqi_roll3_mean" in feature_df.columns
    assert "aqi_roll7_mean" in feature_df.columns
    assert "day_of_week" in feature_df.columns
    assert "month" in feature_df.columns
    assert "season" in feature_df.columns
    assert feature_dict["target"] == "aqi_next_day"
    assert output_path.is_file()
    assert feature_dict_path.is_file()


def test_split_chronological_data(tmp_path: Path) -> None:
    clean_path = tmp_path / "cleaned.csv"
    output_path = tmp_path / "features.csv"
    splits_dir = tmp_path / "splits"

    dates = pd.date_range("2015-01-01", periods=100, freq="D")
    df = pd.DataFrame({"date": dates, "aqi": [100 + i for i in range(100)]})
    df.to_csv(clean_path, index=False)

    feature_df, _ = create_forecasting_features(
        clean_path=clean_path,
        output_path=output_path,
        feature_dict_path=tmp_path / "feature_dictionary.md",
    )

    splits = split_chronological_data(feature_df, output_dir=splits_dir)

    assert len(splits["train"]) > 0
    assert len(splits["validation"]) > 0
    assert len(splits["test"]) > 0
    assert splits["train"].index.max() < splits["validation"].index.min()
    assert splits["validation"].index.max() < splits["test"].index.min()
    assert (splits_dir / "train.csv").is_file()
    assert (splits_dir / "validation.csv").is_file()
    assert (splits_dir / "test.csv").is_file()

    total = len(splits["train"]) + len(splits["validation"]) + len(splits["test"])
    assert total == len(feature_df)
