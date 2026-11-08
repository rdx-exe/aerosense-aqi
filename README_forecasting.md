# AeroSense — AQI Forecasting Pipeline (Next-Day Prediction)

> **Task**: Predict the **next calendar day's Air Quality Index (AQI)** for Kolkata using historical air quality measurements and engineered time-series features.

## Project Flow

1. **Data Acquisition** → `data/raw/kolkata_aqi_history.csv` downloaded from the keyless Open-Meteo Air Quality API (CAMS/ECMWF daily means, 2024–2026).
2. **Data Cleaning** → `src/preprocessing/clean_data.py` produces `data/processed/cleaned_aqi.csv`.
3. **Exploratory Data Analysis (EDA)** → `src/exploratory/eda.py` generates visualizations, statistical summaries, and `docs/eda_findings.md`.
4. **Time‑Series Feature Engineering** → `src/feature_engineering/features.py` builds the full lag/rolling feature table from the cleaned 2024–2026 dataset; `src/model_training/prepare_data.py` creates chronological train/validation/test splits:
   * `data/processed/features.csv`
   * `data/processed/splits/train.csv`, `validation.csv`, `test.csv`
   * `data/processed/splits/split_info.json`
5. **Model Training & Evaluation** → `src/model_training/train.py` trains candidates and selects by validation RMSE; `src/model_evaluation/evaluate_final.py` runs the independent test evaluation.
6. **Web Dashboard** → **Not started** (deliberately blocked until final evaluation is complete).

## How to Run the Forecasting Pipeline

From the project root:

```bash
# Ensure the virtual environment is active
source .venv/bin/activate

# 1. Run full preprocessing + EDA + Feature Engineering
./scripts/run_pipeline.sh 2020-01-01 2024-12-31 --skip-download

# OR run the dedicated forecasting script
./scripts/run_forecasting.sh
```

> **Note:** The shipped pipeline uses `data/raw/kolkata_aqi_history.csv`, downloaded from the keyless Open-Meteo Air Quality API. Refresh it with `python -m src.scripts.download_recent_aqi --start 2024-01-01 --end 2026-11-06`. The old 5-row `kolkata_city_day.csv` fixture is retained for reference only.

## Feature Set Overview

| Category | Features |
|----------|----------|
| **Core AQI** | `aqi`, `aqi_next_day` (target) |
| **Lagged AQI** | `aqi_lag1`, `aqi_lag3`, `aqi_lag7`, `aqi_lag14` |
| **Rolling Statistics** | `aqi_roll3_mean`, `aqi_roll7_mean`, `aqi_roll7_std` |
| **Pollutant Lags** | `pm2.5_lag1`, `pm10_lag1`, `no2_lag1`, `so2_lag1`, `co_lag1`, `o3_lag1` |
| **Temporal** | `day_of_week`, `month`, `season` |
| **Daily Measurements** | `pm2.5`, `pm10`, `no2`, `so2`, `co`, `o3` |

## Data Splits (Chronological)

The full processed feature table (1,033 rows) is split into **train / validation / test**
with exact temporal order preservation. One boundary row is purged between
adjacent splits so that the last target in one split does not belong to the
next split.

| Split | Rows | Coverage |
|-------|------|----------|
| **Training** | 722 | 2024‑01‑08 to 2025‑12‑29 |
| **Validation** | 153 | 2025‑12‑31 to 2026‑06‑01 |
| **Test** | 156 | 2026‑06‑03 to 2026‑11‑05 |

> **Rationale**: Chronological splitting ensures realistic forecasting evaluation — models are always trained on past data to predict the future. The test split is written once and is never read by the training script.

## Reproducing Training and Final Evaluation

```bash
source .venv/bin/activate

# 1. Rebuild features + chronological splits
./scripts/run_forecasting.sh

# 2. Train candidates; select the best model on validation only
./scripts/run_model_training.sh

# 3. Independent test evaluation (no retraining)
./scripts/run_final_evaluation.sh
```

The final report is `documentation/model_evaluation.md`; the metrics table is
`results/metrics/final_metrics.csv`.

## Limitations & Considerations

- **Dataset Scope**: The current evaluation uses a modelled Open-Meteo/CAMS series, not a live official CPCB feed. Results should not be generalised to another city or period without retraining and re-evaluation.
- **Feature Leakage Prevention**: All lag and rolling features use only past/current information; no future measurements are included.
- **Seasonal Handling**: Seasons are derived from the calendar month (Winter = Dec‑Feb, Spring = Mar‑May, etc.).
- **Imputation**: Missing values are forward‑filled with a limit of 3 days; outliers are preserved as legitimate events.
- **No Weather**: Weather variables are not included yet; they can be added once a reliable source (e.g., Open-Meteo API) is integrated.
- **Weak day-to-day predictability**: lag‑1 AQI autocorrelation is low in this dataset, so next-day AQI is difficult to predict from lagged pollutant features alone.

## Project Structure (Updated)

```
.
├── config.yaml                     # Central configuration
├── pyproject.toml                  # Dependencies
├── requirements.txt                # Pinned dependencies
├── .gitignore                      # Version control ignores
├── README.md                       # Core project overview
├── README_forecasting.md            # This forecasting guide
├── data/
│   ├── raw/                        # Original source files
│   │   └── kolkata_city_day.csv
│   ├── processed/                 # Pipeline outputs
│   │   ├── cleaned_aqi.csv
│   │   ├── forecasting_features.csv
│   │   └── splits/
├── docs/                           # Design and analysis documentation
│   ├── DESIGN.md
│   ├── DATA_SOURCES.md
│   ├── MODEL_CARD.md
│   ├── API.md
│   ├── data_dictionary.md
│   ├── eda_findings.md
│   └── feature_dictionary.md
├── memory/                         # Project memory hooks
├── results/                        # EDA outputs
│   ├── visualizations/eda/
│   └── summary/
├── scripts/                        # Convenience shell wrappers
│   ├── install.sh
│   ├── run_pipeline.sh
│   ├── run_forecasting.sh
│   └── run_streamlit.sh
├── src/
│   ├── utils/                     # Shared helpers
│   ├── preprocessing/             # Data cleaning
│   ├── exploratory/               # EDA visualizations
│   ├── feature_engineering/       # Forecasting features
│   ├── model_training/            # Model fitting
│   └── model_evaluation/          # Metrics and reporting
├── streamlit_app/                 # Web application
├── tests/                         # Python unit tests
└── visualizations/                # Final evaluation plots
```

## Quick Start Checklist

- [x] Project structure created
- [x] Dependencies installed (`requirements.txt`, `pyproject.toml`)
- [x] Raw dataset sourced (Open-Meteo Air Quality API → `kolkata_aqi_history.csv`)
- [x] Data cleaning pipeline (`clean_data.py`) implemented and tested
- [x] Exploratory analysis (`eda.py`) executed
- [x] Time‑series feature engineering (`forecast_features.py`) completed
- [x] Chronological data splits generated
- [x] Documentation written (`data_dictionary.md`, `feature_dictionary.md`, etc.)
- [x] **Model training** (validation-only model selection)
- [x] **Final test-set evaluation** (`documentation/model_evaluation.md`)
- [ ] **Web dashboard** (blocked until final evaluation is reviewed)

## Next Actions (Post‑Prototype)

1. **Scale and validate the dataset**: Add direct CPCB station observations or an independent monitor series to cross-check the Open-Meteo/CAMS fields.
2. **Add weather**: Integrate the Open‑Meteo daily API (no key required) for temperature, humidity, precipitation.
3. **Expand features**: Include more lags (30‑day), rolling statistics for pollutants, and interaction terms.
4. **Strengthen evaluation**: Add rolling-origin validation and a second chronological test window.
5. **Deployment**: Only after the evaluation limitations are accepted, containerise the Streamlit app and CI/CD pipeline.

> The current model is suitable for an educational pipeline demonstration, not for operational AQI alerts. See `documentation/model_evaluation.md` for the evidence.

---

### Note to Contributors

This repository follows a **modular, script‑first** design. Each sub‑package (`preprocessing`, `feature_engineering`, `model_training`, etc.) can be run independently, making development, testing, and debugging straightforward.

All scripts are **executable from the project root** using the provided shell wrappers. The project is fully self‑contained, with minimal dependencies, and ready for both research and production use.

Co-Authored-By: Claude Code <noreply@anthropic.com>