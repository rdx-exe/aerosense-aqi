"""Tests for the feature-engineering module."""

import numpy as np
import pandas as pd

from src.feature_engineering.features import build_features, compute_aqi


def _make_clean(n: int = 60) -> pd.DataFrame:
    """Build a synthetic clean DataFrame with a pm25 column."""
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "pm25": rng.normal(80, 20, n),
            "pm10": rng.normal(120, 25, n),
            "no2": rng.normal(40, 10, n),
        },
        index=dates,
    )
    df.index.name = "date"
    return df


def test_compute_aqi_creates_column() -> None:
    """``compute_aqi`` should add an ``aqi`` column when missing."""
    df = _make_clean()
    assert "aqi" not in df.columns
    out = compute_aqi(df)
    assert "aqi" in out.columns
    assert out["aqi"].notna().all()


def test_compute_aqi_is_idempotent() -> None:
    """Calling ``compute_aqi`` twice should not duplicate the column."""
    df = _make_clean()
    out = compute_aqi(compute_aqi(df))
    assert list(out.columns).count("aqi") == 1


def test_build_features_has_target() -> None:
    """The feature matrix should contain the next-day target column."""
    df = _make_clean()
    features = build_features(df, config={"features": {"target": "aqi_next_day"}})
    assert "aqi_next_day" in features.columns


def test_build_features_has_lags() -> None:
    """Lag features should be present for the configured lags."""
    df = _make_clean()
    features = build_features(
        df,
        config={
            "features": {
                "target": "aqi_next_day",
                "lags": [1, 2, 3],
                "rolling_windows": [],
            }
        },
    )
    for lag in [1, 2, 3]:
        assert f"pm25_lag{lag}" in features.columns


def test_build_features_drops_leading_nans() -> None:
    """Rows with NaNs introduced by lags should be dropped."""
    df = _make_clean()
    features = build_features(
        df,
        config={
            "features": {
                "target": "aqi_next_day",
                "lags": [7],
                "rolling_windows": [],
            }
        },
    )
    assert not features.isna().any().any()
    # At least the first 7 rows should have been dropped by the 7-day lag
    assert len(features) <= len(df) - 7


def test_build_features_target_is_shifted() -> None:
    """``aqi_next_day`` should equal ``aqi`` shifted by -1."""
    df = _make_clean()
    features = build_features(
        df,
        config={
            "features": {
                "target": "aqi_next_day",
                "lags": [],
                "rolling_windows": [],
            }
        },
    )
    aqi = compute_aqi(df)["aqi"]
    expected = aqi.shift(-1).dropna()
    # Align on the common index
    common = features.index.intersection(expected.index)
    np.testing.assert_allclose(
        features.loc[common, "aqi_next_day"].values,
        expected.loc[common].values,
    )