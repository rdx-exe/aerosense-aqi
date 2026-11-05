"""Tests for the preprocessing / cleaning module."""

import numpy as np
import pandas as pd

from src.preprocessing.clean import clean_raw_data


def _make_raw(n: int = 60) -> pd.DataFrame:
    """Build a synthetic raw DataFrame with some missing values."""
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "date": dates,
            "pm25": rng.normal(80, 20, n),
            "pm10": rng.normal(120, 25, n),
            "no2": rng.normal(40, 10, n),
            "so2": rng.normal(15, 5, n),
            "o3": rng.normal(30, 8, n),
            "co": rng.normal(0.6, 0.2, n),
        }
    )
    # Inject some missing values
    df.loc[5:7, "pm25"] = np.nan
    df.loc[10, "pm10"] = np.nan
    return df


def test_clean_drops_excessive_missing_rows() -> None:
    """Rows with too many missing values should be dropped."""
    df = _make_raw()
    drop_date = df.loc[20, "date"]
    df.loc[20, ["pm25", "pm10", "no2", "so2", "o3", "co"]] = np.nan
    clean = clean_raw_data(df, config={"preprocessing": {"min_completeness": 0.6}})
    assert drop_date not in clean.index


def test_clean_forward_fills_short_gaps() -> None:
    """Short gaps should be forward-filled rather than dropped."""
    df = _make_raw()
    gap_date = df.loc[30, "date"]
    df.loc[30, "pm25"] = np.nan
    clean = clean_raw_data(
        df,
        config={
            "preprocessing": {
                "min_completeness": 0.6,
                "forward_fill_limit": 2,
                "clip_outliers_iqr": False,
            }
        },
    )
    assert gap_date in clean.index
    assert not pd.isna(clean.loc[gap_date, "pm25"])  # not NaN


def test_clean_clips_outliers() -> None:
    """Extreme outliers should be clipped to the IQR fence."""
    df = _make_raw()
    spike_date = df.loc[40, "date"]
    df.loc[40, "pm25"] = 1000  # extreme spike
    clean = clean_raw_data(
        df,
        config={
            "preprocessing": {
                "min_completeness": 0.6,
                "forward_fill_limit": 0,
                "clip_outliers_iqr": True,
                "iqr_multiplier": 1.5,
            }
        },
    )
    assert spike_date in clean.index
    assert clean.loc[spike_date, "pm25"] < 1000


def test_clean_returns_sorted_datetime_index() -> None:
    """Output should be sorted by date with a datetime index."""
    df = _make_raw()
    clean = clean_raw_data(df, config={"preprocessing": {"min_completeness": 0.6}})
    assert isinstance(clean.index, pd.DatetimeIndex)
    assert clean.index.is_monotonic_increasing