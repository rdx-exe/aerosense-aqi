---
name: dataset-source
description: Historical AQI dataset research and collection strategy
metadata:
  type: project
---

The OpenAQ API v2 is decommissioned (410 Gone). Accessing the v3 API requires an API key, which is not ideal for an open-source project.

**Current source (2026 refresh)**:
1. The shipped model uses the keyless Open-Meteo Air Quality API
   (`https://air-quality-api.open-meteo.com/v1/air-quality`), which serves
   hourly CAMS/ECMWF fields from 2022 onward.
2. `src/scripts/download_recent_aqi.py` downloads the 2024-01-01 → 2026-11-06
   window for Kolkata (22.5726, 88.3639), keeps complete 24-hour days, aggregates
   to daily means, converts CO from µg/m³ to mg/m³, and writes
   `data/raw/kolkata_aqi_history.csv`.
3. The 1,041-row daily series feeds `data/processed/clean.csv`; AQI is derived
   from PM2.5 with the US-EPA 2012 breakpoints already implemented in
   `src/feature_engineering/features.py`.

**Why:** Reproducible, keyless access to recent data with a consistent schema.
The values are modelled/reanalysis fields, not direct CPCB station readings, so
the dashboard and reports must label them as such.

**Legacy plan (not used by the shipped model):** the Kaggle/GitHub-accessible
`city_day.csv` (CPCB India, 2015-2020) and the 5-row smoke-test fixture remain in
`data/raw/` for reference only.
[[project-structure]]
[[data-pipeline]]
