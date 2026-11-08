# AeroSense — Dataset Sources and Documentation

## Overview

This document records the data source, provenance, units, and limitations of the
Kolkata air-quality series used by the AeroSense project.

## Primary Dataset Source

- **Provider**: [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api)
  (hourly CAMS/ECMWF fields).
- **Endpoint**: `https://air-quality-api.open-meteo.com/v1/air-quality`
- **Authentication**: none required.
- **Geographic coverage**: Kolkata, West Bengal, India
  (`latitude=22.5726`, `longitude=88.3639`, `timezone=Asia/Kolkata`).
- **Temporal coverage used by the shipped model**: **2024-01-01 → 2026-11-06**
  (1,041 complete daily rows).
- **License / attribution**: attribute Open-Meteo and the Copernicus Atmosphere
  Monitoring Service (CAMS). Check the
  [Open-Meteo licence](https://open-meteo.com/en/license) and
  [CAMS licence](https://www.copernicus.eu/en/access-data/licence) pages for the
  current terms.

## Download and Aggregation

The reproducible downloader is `src/scripts/download_recent_aqi.py`:

```bash
python -m src.scripts.download_recent_aqi --start 2024-01-01 --end 2026-11-06
```

It performs the following steps:

1. Requests hourly `pm2_5`, `pm10`, `nitrogen_dioxide`, `sulphur_dioxide`,
   `ozone`, and `carbon_monoxide` values.
2. Keeps only days with a complete 24-hour record.
3. Computes daily means and renames the columns to the project schema
   (`pm25`, `pm10`, `no2`, `so2`, `o3`, `co`).
4. Converts CO from µg/m³ to mg/m³ to match the project's unit convention.
5. Writes `data/raw/kolkata_aqi_history.csv`.

## Collected Features

| Feature | Description | Unit |
|---|---|---|
| `date` | Calendar day | YYYY-MM-DD |
| `pm25` | Fine particulate matter, ≤ 2.5 µm (daily mean) | µg/m³ |
| `pm10` | Coarse particulate matter, ≤ 10 µm (daily mean) | µg/m³ |
| `no2` | Nitrogen dioxide (daily mean) | µg/m³ |
| `so2` | Sulphur dioxide (daily mean) | µg/m³ |
| `o3` | Ground-level ozone (daily mean) | µg/m³ |
| `co` | Carbon monoxide (daily mean) | mg/m³ |
| `aqi` | AQI derived from PM2.5 (US-EPA 2012 breakpoints) | index (0–500) |
| `aqi_next_day` | AQI on the following calendar day (target) | index (0–500) |

## Limitations and Known Issues

1. **Modelled values, not station observations.** Open-Meteo serves CAMS/ECMWF
   fields interpolated to the requested coordinates. They are not direct CPCB
   monitor readings and should not be treated as an official city average.
2. **Spatial representativeness.** A single coordinate represents the whole city;
   local hotspots and station-level differences are smoothed out.
3. **Temporal resolution.** The project uses daily means. Sub-daily pollution
   peaks and rush-hour dynamics are not represented.
4. **No meteorology as model input.** The current feature set uses pollutant
   lags and rolling statistics only.
5. **PM2.5-only AQI.** AQI is derived from PM2.5, not the CPCB composite index.
6. **Legacy files.** `data/raw/city_day.csv`, `data/raw/kolkata_city_day.csv`,
   and the 2015 smoke-test artifacts are retained for reference but are not used
   by the shipped 2024–2026 model or dashboard.
7. **API dependency.** OpenAQ API v2 is decommissioned. Open-Meteo is keyless but
   remains an external service; the raw CSV is retained in the working copy so
   training does not require network access.
