# AeroSense — Exploratory Data Analysis (EDA) Findings for Kolkata AQI

**Dataset:** Open-Meteo Air Quality API (CAMS/ECMWF daily means), Kolkata.
**Time coverage:** 2024-01-01 → 2026-11-06.
**Total observations:** 1,041 daily rows.

## Key Findings Supported by the Data

1. The average AQI is **118.5** (median 116.6), with a standard deviation of 50.3.
2. The worst month is **December** (mean AQI 184.8); the cleanest is **August**
   (mean AQI 67.3), reflecting the monsoon seasonal cycle.
3. The AQI distribution spans **22.9 → 216.8**:
   - Good: 66 days (6.3%)
   - Moderate: 396 days (38.0%)
   - Unhealthy for Sensitive Groups: 178 days (17.1%)
   - Unhealthy: 365 days (35.1%)
   - Very Unhealthy: 36 days (3.5%)
4. Because AQI is derived from PM2.5, their correlation is **0.944 by
   construction**. PM10 (0.934), SO₂ (0.882), CO (0.858), NO₂ (0.823), and O₃
   (0.698) are also strongly correlated with the daily AQI level.
5. The highest single-day AQI recorded is **216.8**; the lowest is **22.9**.
6. The one-day-ahead target is highly persistent: the correlation between today's
   AQI and tomorrow's AQI is **0.932** in the training split. This is why the
   persistence baseline is strong and why the model's edge is modest.
