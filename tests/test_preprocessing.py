# Preprocessing & Data Quality Unit Tests

import pandas as pd
import numpy as np
from pathlib import Path
from src.preprocessing.clean_data import clean_air_quality_data
from src.preprocessing.data_quality import assess_data_quality


def test_assess_data_quality(tmp_path: Path) -> None:
    """Test data quality assessment logic."""
    df = pd.DataFrame({
        'City': ['Kolkata', 'Kolkata'],
        'Date': ['2023-01-01', '2023-01-02'],
        'PM2.5': [80.0, -5.0],  # -5 is invalid
        'AQI': [120, 130]
    })
    audit = assess_data_quality(df)
    assert audit["initial_record_count"] == 2
    assert audit["invalid_measurements"]["pm2.5"] == 1


def test_clean_air_quality_pipeline(tmp_path: Path) -> None:
    """Test full cleaning pipeline end-to-end."""
    raw_file = tmp_path / "raw.csv"
    df = pd.DataFrame({
        'City': ['Kolkata', 'Kolkata', 'Kolkata'],
        'Date': ['2023-01-01', '2023-01-02', '2023-01-04'], # Gap on Jan 3
        'PM2.5': [80.0, np.nan, 90.0],
        'PM10': [120.0, 125.0, 130.0],
        'NO2': [40.0, 42.0, 45.0],
        'SO2': [15.0, 16.0, 17.0],
        'CO': [0.5, 0.6, 0.7],
        'O3': [30.0, 32.0, 35.0],
        'AQI': [120, 125, 135],
        'AQI_Bucket': ['Moderate', 'Moderate', 'Moderate']
    })
    df.to_csv(raw_file, index=False)

    out_file = tmp_path / "cleaned.csv"
    report_file = tmp_path / "report.csv"

    cleaned_df = clean_air_quality_data(
        raw_path=raw_file,
        output_path=out_file,
        report_path=report_file
    )

    # Verify daily continuity (Jan 1, 2, 3, 4 -> 4 rows)
    assert len(cleaned_df) == 4
    assert 'date' in cleaned_df.reset_index().columns
    assert out_file.is_file()
    assert report_file.is_file()
