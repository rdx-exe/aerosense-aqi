"""Prepare chronological train/validation/test splits from engineered features.

This module exists because the repository previously contained a 5-row
smoke-test split (2 training rows, 1 validation row and 1 test row).  Such a
split cannot support a meaningful final evaluation.  The split created here
uses the full processed feature table (``data/processed/features.csv``) and
preserves temporal order.

Design rules
------------
* Chronological split: 70% train, 15% validation, 15% test.
* One boundary row is purged between adjacent splits.  The target of the last
  row in a split is the current-day AQI of the first row in the next split;
  purging that row prevents validation selection from seeing any test-period
  value and prevents training from using a validation-period target.
* No random shuffling and no resampling.
* The test split is written but never read by the training script.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.config import get_path
from ..utils.io import read_csv, write_csv

TARGET = "aqi_next_day"
DATE_COL = "date"


def prepare_splits(
    features_path: str | Path = "data/processed/features.csv",
    output_dir: str | Path = "data/processed/splits",
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> dict:
    """Create deterministic chronological splits and return split metadata.

    Parameters
    ----------
    features_path:
        Processed feature table containing ``date`` and ``aqi_next_day``.
    output_dir:
        Directory where ``train.csv``, ``validation.csv`` and ``test.csv``
        are written.
    train_ratio, validation_ratio:
        Fractions of the usable rows assigned to training and validation.
        The remaining rows form the test set.
    """
    features_path = get_path(features_path)
    output_dir = get_path(output_dir)

    if not features_path.is_file():
        raise FileNotFoundError(
            f"Processed feature table not found: {features_path}. "
            "Run `python -m src.feature_engineering.features` first."
        )

    df = read_csv(features_path, parse_dates=[DATE_COL])
    if TARGET not in df.columns:
        raise ValueError(f"Target column {TARGET!r} is missing from {features_path}")

    df = df.sort_values(DATE_COL).reset_index(drop=True)
    if df[DATE_COL].duplicated().any():
        raise ValueError("Duplicate dates found in the feature table")
    if df[TARGET].isna().any():
        raise ValueError("Target column contains missing values")

    # Descriptive information on gaps.  Gaps are allowed but documented; the
    # feature table is not filled with synthetic dates.
    gaps = df[DATE_COL].diff().dt.days.dropna()
    gap_counts = {int(k): int(v) for k, v in gaps.value_counts().sort_index().items()}

    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_validation = int(n_total * validation_ratio)
    n_test = n_total - n_train - n_validation
    if min(n_train, n_validation, n_test) < 2:
        raise ValueError(
            "Not enough rows to build train/validation/test splits "
            f"(n_total={n_total})"
        )

    # Purge one boundary row on each side of the validation split.
    # Row at n_train - 1 has a target that belongs to the validation period.
    # Row at n_train + n_validation - 1 has a target that belongs to the test period.
    train_df = df.iloc[: n_train - 1].copy()
    validation_df = df.iloc[n_train : n_train + n_validation - 1].copy()
    test_df = df.iloc[n_train + n_validation :].copy()

    assert train_df[DATE_COL].max() < validation_df[DATE_COL].min()
    assert validation_df[DATE_COL].max() < test_df[DATE_COL].min()
    assert len(train_df) > 0 and len(validation_df) > 0 and len(test_df) > 0

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(train_df, output_dir / "train.csv")
    write_csv(validation_df, output_dir / "validation.csv")
    write_csv(test_df, output_dir / "test.csv")

    feature_columns = [
        c for c in df.columns if c not in {DATE_COL, TARGET}
    ]

    def _range(frame: pd.DataFrame) -> list[str]:
        return [
            str(frame[DATE_COL].min().date()),
            str(frame[DATE_COL].max().date()),
        ]

    def _stats(frame: pd.DataFrame) -> dict:
        return {
            "target_mean": float(frame[TARGET].mean()),
            "target_std": float(frame[TARGET].std()),
            "target_min": float(frame[TARGET].min()),
            "target_max": float(frame[TARGET].max()),
            "current_aqi_mean": float(frame["aqi"].mean()),
            "current_aqi_std": float(frame["aqi"].std()),
        }

    split_info = {
        "source_file": str(features_path.relative_to(get_path())),
        "target": TARGET,
        "total_rows": int(n_total),
        "requested_ratios": {
            "train": float(train_ratio),
            "validation": float(validation_ratio),
            "test": float(1.0 - train_ratio - validation_ratio),
        },
        "split_rows": {
            "train": int(len(train_df)),
            "validation": int(len(validation_df)),
            "test": int(len(test_df)),
        },
        "date_ranges": {
            "train": _range(train_df),
            "validation": _range(validation_df),
            "test": _range(test_df),
        },
        "target_stats": {
            "train": _stats(train_df),
            "validation": _stats(validation_df),
            "test": _stats(test_df),
        },
        "purged_boundary_rows": 2,
        "purge_policy": (
            "One row was removed between train/validation and between "
            "validation/test because its target belongs to the next split."
        ),
        "date_gap_counts_days": gap_counts,
        "feature_columns": feature_columns,
        "created_at": pd.Timestamp.now().isoformat(),
    }

    split_info_path = output_dir / "split_info.json"
    split_info_path.write_text(json.dumps(split_info, indent=2), encoding="utf-8")

    print(
        "[prepare-data] "
        f"train={len(train_df)} ({_range(train_df)[0]} → {_range(train_df)[1]}), "
        f"validation={len(validation_df)} "
        f"({_range(validation_df)[0]} → {_range(validation_df)[1]}), "
        f"test={len(test_df)} ({_range(test_df)[0]} → {_range(test_df)[1]})"
    )
    print(f"[prepare-data] Split metadata → {split_info_path}")

    return split_info


def main() -> None:
    """CLI entry point."""
    prepare_splits()


if __name__ == "__main__":
    main()
