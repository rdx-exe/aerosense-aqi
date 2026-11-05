"""Raw-data cleaning pipeline.

Transforms raw daily air-quality measurements into a clean, model-ready
DataFrame. Steps:

1. Parse dates and sort chronologically.
2. Drop rows with excessive missing values (configured ``min_completeness``).
3. Forward-fill short gaps (configured ``forward_fill_limit``).
4. Clip outliers using the IQR method (optional, configured ``clip_outliers_iqr``).
5. Drop any remaining NaN rows.

The function is pure (no global state) and composable so it can be reused by
the feature-engineering stage and unit-tested in isolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.config import load_config

# Pollutant columns expected in the raw dataset
POLLUTANTS = ["pm25", "pm10", "no2", "so2", "o3", "co"]
DATE_COL = "date"


def clean_raw_data(
    raw: pd.DataFrame | None = None,
    *,
    config: dict | None = None,
    raw_path: str | None = None,
) -> pd.DataFrame:
    """Clean a raw air-quality DataFrame.

    Parameters
    ----------
    raw:
        Raw DataFrame with a ``date`` column and one column per pollutant.
        If ``None``, the file at ``raw_path`` is read.
    config:
        Optional pre-loaded configuration dict. Loaded from ``config.yaml``
        when omitted.
    raw_path:
        Path to a CSV file used when ``raw`` is ``None``.

    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame indexed by date with one row per day.
    """
    if config is None:
        config = load_config()

    pp_cfg = config.get("preprocessing", {})
    min_completeness = float(pp_cfg.get("min_completeness", 0.60))
    ffill_limit = int(pp_cfg.get("forward_fill_limit", 2))
    clip_outliers = bool(pp_cfg.get("clip_outliers_iqr", True))
    iqr_mult = float(pp_cfg.get("iqr_multiplier", 1.5))

    if raw is None:
        from ..utils.io import read_csv

        if raw_path is None:
            raw_path = config["data"].get(
                "fallback_csv", "data/raw/kolkata_aqi_history.csv"
            )
        raw = read_csv(raw_path)

    df = raw.copy()

    # 1. Parse & sort dates
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df = df.dropna(subset=[DATE_COL])
        df = df.sort_values(DATE_COL).drop_duplicates(
            subset=[DATE_COL], keep="last"
        )
        df = df.set_index(DATE_COL)
        df.index.name = "date"

    # Keep only the pollutant columns we care about
    cols = [c for c in POLLUTANTS if c in df.columns]
    df = df[cols].apply(pd.to_numeric, errors="coerce")

    # 2. Drop rows with excessive missing values
    completeness = df.notna().mean(axis=1)
    df = df[completeness >= min_completeness]

    # 3. Forward-fill short gaps (limit=0 or less disables the step)
    if ffill_limit and ffill_limit > 0:
        df = df.ffill(limit=ffill_limit)

    # 4. Clip outliers via IQR
    if clip_outliers:
        for col in cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_mult * iqr
            upper = q3 + iqr_mult * iqr
            df[col] = df[col].clip(lower=lower, upper=upper)

    # 5. Drop any remaining NaN rows
    df = df.dropna()

    return df.sort_index()


def main() -> None:
    """CLI entry point: clean raw data and save to ``data/processed/clean.csv``."""
    from ..utils.config import get_path
    from ..utils.io import read_csv, write_csv

    config = load_config()
    raw_path = config["data"].get(
        "fallback_csv", "data/raw/kolkata_aqi_history.csv"
    )
    raw = read_csv(get_path(raw_path))

    clean = clean_raw_data(raw, config=config)

    out = get_path(config["data"]["processed_dir"], "clean.csv")
    write_csv(clean.reset_index(), out)
    print(f"[clean] Saved {len(clean)} rows → {out}")


if __name__ == "__main__":
    main()