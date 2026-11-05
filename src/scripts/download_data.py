"""Download daily air-quality measurements for Kolkata from OpenAQ.

OpenAQ API v2 is free and requires no API key. This module aggregates
hourly observations into daily means for the requested pollutants and
writes a CSV to ``data/raw/``.

Usage::

    python -m src.scripts.download_data --start 2020-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from ..utils.config import get_path, load_config

API_BASE = "https://api.openaq.org/v2/measurements"
DEFAULT_PARAMS = ["pm25", "pm10", "no2", "so2", "o3", "co"]
PAGE_SIZE = 10000


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Kolkata AQI data from OpenAQ."
    )
    parser.add_argument(
        "--start", type=str, required=True, help="Start date, e.g. 2020-01-01"
    )
    parser.add_argument(
        "--end", type=str, required=True, help="End date, e.g. 2024-12-31"
    )
    parser.add_argument(
        "--location",
        type=str,
        default="IN-Kolkata",
        help="OpenAQ location name (default: IN-Kolkata)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path (default: data/raw/openaq_kolkata_<start>_<end>.csv)",
    )
    return parser.parse_args(argv)


def _fetch_page(
    params: list[str],
    location: str,
    date_from: str,
    date_to: str,
    page: int,
) -> list[dict]:
    """Fetch a single page of measurements from OpenAQ."""
    query = {
        "parameter": params,
        "location": location,
        "date_from": date_from,
        "date_to": date_to,
        "page": page,
        "limit": PAGE_SIZE,
        "order_by": "datetime",
        "format": "json",
    }
    resp = requests.get(API_BASE, params=query, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("results", [])


def download_data(
    start: str,
    end: str,
    *,
    location: str = "IN-Kolkata",
    params: list[str] | None = None,
) -> pd.DataFrame:
    """Download and aggregate OpenAQ measurements into a daily DataFrame.

    Parameters
    ----------
    start:
        Start date (ISO 8601), inclusive.
    end:
        End date (ISO 8601), inclusive.
    location:
        OpenAQ location name.
    params:
        Pollutant parameters to fetch. Defaults to the project standard set.

    Returns
    -------
    pandas.DataFrame
        Daily-mean measurements indexed by date with one column per
        pollutant.
    """
    params = params or DEFAULT_PARAMS
    date_from = f"{start}T00:00:00Z"
    date_to = f"{end}T23:59:59Z"

    records: list[dict] = []
    page = 1
    while True:
        results = _fetch_page(params, location, date_from, date_to, page)
        if not results:
            break
        records.extend(results)
        if len(results) < PAGE_SIZE:
            break
        page += 1

    if not records:
        print("[download] No measurements returned by OpenAQ.")
        return pd.DataFrame(columns=params)

    rows = []
    for rec in records:
        # OpenAQ returns datetimeLocal (e.g. "2020-01-01 01:00:00")
        dt = rec.get("datetimeLocal") or rec.get("datetime")
        param = rec.get("parameter")
        value = rec.get("value")
        if dt is None or param is None or value is None:
            continue
        try:
            ts = pd.to_datetime(dt, errors="coerce", utc=True)
        except Exception:
            continue
        if pd.isna(ts):
            continue
        rows.append({"date": ts, "parameter": param, "value": float(value)})

    if not rows:
        print("[download] No valid records parsed.")
        return pd.DataFrame(columns=params)

    df = pd.DataFrame(rows)
    df["date"] = df["date"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df["date"] = df["date"].dt.normalize()

    # Aggregate to daily means, one column per pollutant
    daily = df.pivot_table(
        index="date", columns="parameter", values="value", aggfunc="mean"
    )
    daily = daily.reindex(columns=params)

    return daily.sort_index()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_config()

    daily = download_data(args.start, args.end, location=args.location)
    if daily.empty:
        print("[download] No data to save.")
        return 1

    out_path = Path(args.out) if args.out else get_path(
        config["data"]["raw_dir"],
        f"openaq_kolkata_{args.start}_{args.end}.csv",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    daily.reset_index().to_csv(out_path, index=False)
    print(f"[download] Saved {len(daily)} daily rows → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())