"""Feature engineering for next-day AQI forecasting.

Given a clean daily DataFrame (indexed by date), this module:

1. Computes AQI from PM2.5 using the US-EPA breakpoint formula when an explicit
   ``aqi`` column is absent.
2. Builds lag features (1, 2, 3, 7 days) and rolling statistics
   (mean/std/min/max over 3, 7, 14 days) for every pollutant.
3. Defines the target ``aqi_next_day`` (AQI shifted by -1 day).
4. Drops rows with NaNs introduced by lags/rolling windows.

All feature construction is deterministic and operates on a single DataFrame,
which makes it trivial to unit-test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.config import load_config

# Pollutants used as feature sources
POLLUTANTS = ["pm25", "pm10", "no2", "so2", "o3", "co"]
AQI_COL = "aqi"


# 24-hour PM2.5 (µg/m³) breakpoints — US EPA 2012 standard
_PM25_BREAKPOINTS = [0.0, 12.0, 35.4, 55.4, 150.4, 250.4, 350.4, 500.4]
_AQI_BREAKPOINTS = [0, 50, 100, 150, 200, 300, 400, 500]


def _pm25_to_aqi(pm25: float) -> float:
    """Convert a 24-hour average PM2.5 (µg/m³) to AQI using US-EPA breakpoints.

    Uses linear interpolation between the EPA breakpoint pairs, which is
    mathematically equivalent to the official piecewise formula but handles
    every value continuously (the official table rounds to one decimal, which
    leaves sub-intervals that would otherwise fall between buckets).

    Returns NaN for values outside the defined range or non-numeric input.
    """
    if pm25 is None or (isinstance(pm25, float) and pd.isna(pm25)):
        return np.nan
    try:
        value = float(pm25)
    except (TypeError, ValueError):
        return np.nan
    if value < _PM25_BREAKPOINTS[0] or value > _PM25_BREAKPOINTS[-1]:
        return np.nan
    return float(np.interp(value, _PM25_BREAKPOINTS, _AQI_BREAKPOINTS))


def compute_aqi(df: pd.DataFrame) -> pd.DataFrame:
    """Add an ``aqi`` column computed from PM2.5 when one is not present."""
    if AQI_COL in df.columns:
        return df

    if "pm25" not in df.columns:
        raise ValueError("Cannot compute AQI: 'pm25' column is missing.")

    df = df.copy()
    df[AQI_COL] = df["pm25"].apply(_pm25_to_aqi)
    return df


def build_features(
    clean: pd.DataFrame | None = None,
    *,
    config: dict | None = None,
    clean_path: str | None = None,
) -> pd.DataFrame:
    """Build the feature matrix and target for next-day AQI prediction.

    Parameters
    ----------
    clean:
        Clean DataFrame indexed by date with pollutant columns. If ``None``,
        the file at ``clean_path`` is read.
    config:
        Optional pre-loaded configuration dict.
    clean_path:
        Path to a CSV used when ``clean`` is ``None``.

    Returns
    -------
    pandas.DataFrame
        Feature matrix indexed by date, including the ``aqi_next_day`` target.
    """
    if config is None:
        config = load_config()

    feat_cfg = config.get("features", {})
    lags = list(feat_cfg.get("lags", [1, 2, 3, 7]))
    rolling_windows = list(feat_cfg.get("rolling_windows", [3, 7, 14]))
    target = feat_cfg.get("target", "aqi_next_day")

    if clean is None:
        from ..utils.io import read_csv

        clean = read_csv(clean_path or "data/processed/clean.csv")
        if "date" in clean.columns:
            clean["date"] = pd.to_datetime(clean["date"])
            clean = clean.set_index("date")

    df = compute_aqi(clean.copy())

    # Pollutant columns actually present
    cols = [c for c in POLLUTANTS if c in df.columns]
    if AQI_COL in df.columns and AQI_COL not in cols:
        cols.append(AQI_COL)

    # Lag features
    for lag in lags:
        for col in cols:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # Rolling statistics
    for window in rolling_windows:
        for col in cols:
            roll = df[col].rolling(window=window, min_periods=1)
            df[f"{col}_roll{window}_mean"] = roll.mean()
            df[f"{col}_roll{window}_std"] = roll.std()
            df[f"{col}_roll{window}_min"] = roll.min()
            df[f"{col}_roll{window}_max"] = roll.max()

    # Target: next-day AQI
    df[target] = df[AQI_COL].shift(-1)

    # Drop rows with NaNs introduced by lags/rolling/target
    df = df.dropna()

    return df.sort_index()


def main() -> None:
    """CLI entry point: build features and save to ``data/processed/features.csv``."""
    from ..utils.config import get_path
    from ..utils.io import read_csv, write_csv

    config = load_config()
    clean_path = get_path(config["data"]["processed_dir"], "clean.csv")
    clean = read_csv(clean_path)

    features = build_features(clean, config=config)

    out = get_path(config["data"]["processed_dir"], "features.csv")
    write_csv(features.reset_index(), out)
    print(f"[features] Saved {len(features)} rows → {out}")


if __name__ == "__main__":
    main()