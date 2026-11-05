# Dataset Sources and Documentation

## Overview

This document records the data sources, provenance, licensing conditions, and audit results for the Kolkata Air Quality Index (AQI) prediction project.

## Primary Dataset Source

- **Publisher**: Central Pollution Control Board (CPCB), India (curated historical daily aggregates).
- **Source Platform**: Publicly accessible GitHub mirrors and Open Government Data (OGD) platforms.
- **Geographic Coverage**: Kolkata, West Bengal, India.
- **Temporal Coverage**: Historical daily records (2015–2020).
- **License**: Open Data Commons Attribution License (ODC-By) / Public Domain.

## Collected Features

| Feature | Description | Unit |
|---|---|---|
| `City` | Name of the city | String |
| `Date` | Timestamp of the recording | YYYY-MM-DD |
| `PM2.5` | Particulate Matter < 2.5 micrometers | µg/m³ |
| `PM10` | Particulate Matter < 10 micrometers | µg/m³ |
| `NO2` | Nitrogen Dioxide | µg/m³ |
| `SO2` | Sulfur Dioxide | µg/m³ |
| `CO` | Carbon Monoxide | mg/m³ |
| `O3` | Ozone | µg/m³ |
| `AQI` | Calculated Air Quality Index | Index (0–500) |
| `AQI_Bucket` | Qualitative category (Good, Moderate, Unhealthy, etc.) | String |

## Limitations and Known Issues

1. **Missing Values**: Air quality monitoring stations occasionally suffer from downtime, leading to missing pollutant readings (handled via forward-fill and interpolation in the preprocessing stage).
2. **Station Aggregation**: Values represent a city-wide daily mean across multiple CPCB monitoring stations in Kolkata (e.g., Rabindra Bharati, Victoria Memorial, Ballygunge), which can introduce spatial smoothing.
3. **API Deprecation**: OpenAQ API v2 was recently deprecated; we rely on structured static CSV dumps to ensure pipeline reproducibility without relying on external API uptime.
