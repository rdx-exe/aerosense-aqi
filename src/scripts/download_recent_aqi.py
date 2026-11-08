"""Download recent daily air-quality data for Kolkata from Open-Meteo.

The public Open-Meteo Air Quality API is free, needs no API key, and serves
historical hourly CAMS/ECMWF fields from 2022 onward.  This module aggregates
those hourly values into daily means and writes a CSV with the same schema as
the rest of the project (``date,pm25,pm10,no2,so2,o3,co``).

Carbon monoxide is converted from µg/m³ (Open-Meteo) to mg/m³ so that it is
consistent with the historical CPCB-based series used by the project.

Usage::

    python -m src.scripts.download_recent_aqi --start 2024-01-01 --end 2026-11-06
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests

from ..utils.config import get_path

API_BASE = "https://air-quality-api.open-meteo.com/v1/air-quality"
HOURLY_PARAMETERS = [
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "carbon_monoxide",
]
# Open-Meteo hourly name -> project column name
COLUMN_MAP = {
    "pm2_5": "pm25",
    "pm10": "pm10",
    "nitrogen_dioxide": "no2",
    "sulphur_dioxide": "so2",
    "ozone": "o3",
    "carbon_monoxide": "co",
}
POLLUTANT_COLUMNS = ["pm25", "pm10", "no2", "so2", "o3", "co"]
KOLKATA_LAT = 22.5726
KOLKATA_LON = 88.3639


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download recent Kolkata air-quality data from Open-Meteo."
    )
    parser.add_argument("--start", type=str, required=True, help="Start date, e.g. 2024-01-01")
    parser.add_argument("--end", type=str, required=True, help="End date, e.g. 2026-11-06")
    parser.add_argument("--lat", type=float, default=KOLKATA_LAT)
    parser.add_argument("--lon", type=float, default=KOLKATA_LON)
    parser.add_argument(
        "--out",
        type=str,
        default="data/raw/kolkata_aqi_history.csv",
        help="Output CSV path (default: data/raw/kolkata_aqi_history.csv)",
    )
    return parser.parse_args(argv)


def fetch_hourly(
    start: str,
    end: str,
    *,
    lat: float = KOLKATA_LAT,
    lon: float = KOLKATA_LON,
) -> pd.DataFrame:
    """Fetch hourly air-quality fields for a location and date range."""
    query = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_PARAMETERS),
        "timezone": "Asia/Kolkata",
    }
    response = requests.get(API_BASE, params=query, timeout=120)
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise RuntimeError("Open-Meteo returned no hourly data.")

    frame = pd.DataFrame(hourly)
    frame["time"] = pd.to_datetime(frame["time"])
    return frame


def to_daily(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly observations into daily means in the project schema."""
    frame = frame.copy()
    frame["time"] = pd.to_datetime(frame["time"])
    daily = (
        frame.set_index("time")
        .resample("D")
        .mean(numeric_only=True)
        .rename(columns=COLUMN_MAP)
    )

    # Drop days without a complete 24-hour record.  AQI targets must be based
    # on full daily averages, not on partial days at the edge of the request.
    hourly_counts = frame.set_index("time").resample("D").count()
    complete = hourly_counts[HOURLY_PARAMETERS].min(axis=1) == 24
    daily = daily.loc[complete]

    daily = daily.reindex(columns=POLLUTANT_COLUMNS)
    # Open-Meteo reports CO in µg/m³; the project's historical series uses mg/m³.
    daily["co"] = daily["co"] / 1000.0
    daily.index.name = "date"
    return daily.dropna().sort_index()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    hourly = fetch_hourly(args.start, args.end, lat=args.lat, lon=args.lon)
    daily = to_daily(hourly)
    if daily.empty:
        print("[download-recent] No complete daily rows returned.")
        return 1

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = get_path(str(out_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    daily.reset_index().to_csv(out_path, index=False)

    print(
        f"[download-recent] Saved {len(daily)} daily rows "
        f"({daily.index.min().date()} → {daily.index.max().date()}) → {out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
