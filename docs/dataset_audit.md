# AeroSense — Dataset Audit Report

## Shipped modelling dataset

- **file**: `data/processed/clean.csv`
- **raw source**: `data/raw/kolkata_aqi_history.csv`
- **provenance**: Open-Meteo Air Quality API (CAMS/ECMWF daily means)
- **row_count**: 1,041
- **date_range**: 2024-01-01 → 2026-11-06
- **columns**: `date`, `pm25`, `pm10`, `no2`, `so2`, `o3`, `co`
- **missing_values**: 0
- **duplicate dates**: 0
- **daily coverage**: every day in the range has a complete 24-hour aggregate
- **AQI**: derived from PM2.5 with the US-EPA 2012 breakpoints

## Legacy fixture (not used by the shipped model)

- **file**: `data/raw/kolkata_city_day.csv`
- **row_count**: 5
- **date_range**: 2015-01-01 → 2015-01-05
- **columns**: `City`, `Date`, `PM2.5`, `PM10`, `NO2`, `SO2`, `CO`, `O3`, `AQI`, `AQI_Bucket`
- **missing_values**: PM2.5 = 1, PM10 = 1, all others = 0
- **duplicates**: 0
