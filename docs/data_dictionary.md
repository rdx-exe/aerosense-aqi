# Data Dictionary — AeroSense Kolkata AQI Prediction Project

This data dictionary defines all features and attributes present in the cleaned dataset (`data/processed/cleaned_aqi.csv`).

| Column Name | Data Type | Measurement Unit | Description |
| :--- | :--- | :--- | :--- |
| `date` | Datetime (YYYY-MM-DD) | — | Calendar date of observation (indexed continuously). |
| `pm2.5` | Float | µg/m³ | Particulate Matter ≤ 2.5 micrometers in aerodynamic diameter (24-hour average). |
| `pm10` | Float | µg/m³ | Particulate Matter ≤ 10 micrometers in aerodynamic diameter (24-hour average). |
| `no2` | Float | µg/m³ | Nitrogen Dioxide concentration (24-hour average). |
| `so2` | Float | µg/m³ | Sulfur Dioxide concentration (24-hour average). |
| `co` | Float | mg/m³ | Carbon Monoxide concentration (8-hour average). |
| `o3` | Float | µg/m³ | Ozone concentration (8-hour maximum/average). |
| `aqi` | Float | Index (0–500) | Sub-index / overall Air Quality Index calculated via CPCB guidelines. |
| `aqi_bucket` | Categorical | — | Qualitative health advisory category (Good, Satisfactory, Moderate, Poor, Very Poor, Severe). |

## Data Integrity Rules
- **Missing Values**: Handled via causal forward-fill (limit = 3 days) to avoid data leakage from future observations.
- **Negative Values**: Flagged as invalid/sensor errors during auditing and replaced with `NaN` prior to imputation.
- **Outliers**: Extreme pollution spikes (e.g. Diwali celebrations or inversion layers) are **preserved** as legitimate meteorological/anthropogenic phenomena.
