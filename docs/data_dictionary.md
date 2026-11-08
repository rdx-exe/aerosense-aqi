# Data Dictionary — AeroSense Kolkata AQI Prediction Project

This data dictionary defines the columns used by the shipped pipeline
(`data/processed/clean.csv` → `data/processed/features.csv`).

| Column Name | Data Type | Measurement Unit | Description |
| :--- | :--- | :--- | :--- |
| `date` | Datetime (YYYY-MM-DD) | — | Calendar date of observation. |
| `pm25` | Float | µg/m³ | Fine particulate matter ≤ 2.5 µm (daily mean). |
| `pm10` | Float | µg/m³ | Coarse particulate matter ≤ 10 µm (daily mean). |
| `no2` | Float | µg/m³ | Nitrogen dioxide (daily mean). |
| `so2` | Float | µg/m³ | Sulphur dioxide (daily mean). |
| `co` | Float | mg/m³ | Carbon monoxide (daily mean). |
| `o3` | Float | µg/m³ | Ground-level ozone (daily mean). |
| `aqi` | Float | Index (0–500) | AQI derived from PM2.5 using US-EPA 2012 breakpoints. |
| `aqi_next_day` | Float | Index (0–500) | AQI on the following calendar day (training target). |

## Feature Columns

`src/feature_engineering/features.py` adds 119 features:

- Lags 1, 2, 3, and 7 days for the six pollutants and `aqi`.
- Rolling mean, standard deviation, minimum, and maximum over 3, 7, and 14 days.
- The target `aqi_next_day = aqi.shift(-1)`.

Rows with incomplete lag/rolling history or a missing next-day target are
dropped, producing 1,033 usable rows from the 1,041-row clean series.

## Data Integrity Rules

- **Missing values**: the raw download keeps only complete 24-hour days, so the
  shipped clean series has no gaps.
- **Outliers**: real pollution spikes are preserved; the optional IQR clipping
  configured in `config.yaml` is applied during preprocessing.
- **Leakage**: all lags and rolling statistics use current and past values only;
  the target is the single forward shift.
- **AQI derivation**: `aqi` is recomputed from PM2.5 in both training and
  inference code paths.
