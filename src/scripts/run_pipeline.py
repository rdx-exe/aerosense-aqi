"""Run the full AQI prediction pipeline end-to-end.

Stages (in order):

1. Download raw data (if missing).
2. Preprocess → ``data/processed/clean.csv``.
3. Feature engineering → ``data/processed/features.csv``.
4. Model training → ``models/*.pkl``.
5. Model evaluation → ``visualizations/*.png``, ``report.md``.

Usage::

    python -m src.scripts.run_pipeline [--start 2020-01-01] [--end 2024-12-31]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..utils.config import get_path, load_config
from ..utils.io import read_csv
from ..preprocessing.clean import clean_raw_data
from ..feature_engineering.features import build_features
from ..model_evaluation.evaluate import evaluate_models


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full AQI pipeline.")
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument(
        "--skip-download", action="store_true", help="Skip the download stage"
    )
    return parser.parse_args(argv)


def run_pipeline(start: str = "2020-01-01", end: str = "2024-12-31") -> int:
    config = load_config()
    raw_dir = get_path(config["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ensure raw data exists
    raw_files = list(raw_dir.glob("*.csv"))
    if not raw_files:
        print("[pipeline] No raw data found — downloading from OpenAQ...")
        from .download_data import download_data

        daily = download_data(start, end)
        if daily.empty:
            print("[pipeline] ERROR: download returned no data.")
            return 1
        raw_path = raw_dir / f"openaq_kolkata_{start}_{end}.csv"
        daily.reset_index().to_csv(raw_path, index=False)
        print(f"[pipeline] Saved raw data → {raw_path}")
    else:
        raw_path = raw_files[0]
        print(f"[pipeline] Using existing raw data → {raw_path}")

    # 2. Preprocess
    raw = read_csv(raw_path)
    clean = clean_raw_data(raw, config=config)
    clean_out = get_path(config["data"]["processed_dir"], "clean.csv")
    clean_out.parent.mkdir(parents=True, exist_ok=True)
    clean.reset_index().to_csv(clean_out, index=False)
    print(f"[pipeline] Preprocessed → {clean_out} ({len(clean)} rows)")

    # 3. Feature engineering
    features = build_features(clean, config=config)
    features_out = get_path(config["data"]["processed_dir"], "features.csv")
    features.reset_index().to_csv(features_out, index=False)
    print(f"[pipeline] Features → {features_out} ({len(features)} rows)")

    # 4 & 5. Train + evaluate
    results = evaluate_models(features, config=config)
    print(
        f"[pipeline] Done. Best model: {results['best']} "
        f"(test RMSE={results['metrics'][results['best']]['rmse']:.2f})"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.skip_download:
        config = load_config()
        raw_dir = get_path(config["data"]["raw_dir"])
        raw_files = list(raw_dir.glob("*.csv"))
        if not raw_files:
            print("[pipeline] ERROR: --skip-download but no raw data found.")
            return 1
    return run_pipeline(start=args.start, end=args.end)


if __name__ == "__main__":
    sys.exit(main())