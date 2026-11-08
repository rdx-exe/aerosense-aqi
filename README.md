# AeroSense

**Next-day Air Quality Index (AQI) prediction for Kolkata, India.**

AeroSense is an end-to-end machine-learning project: it downloads recent air-quality
data, builds time-series features, trains and compares models, evaluates them on a
future test period, and serves the result in an interactive Streamlit dashboard.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.1-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41.0-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-1B5E20)](#license)

---

## What it does

Given today's pollutant measurements for Kolkata, AeroSense estimates **tomorrow's
AQI**. It is a research and learning project, not an official air-quality service.

- **Location:** Kolkata, West Bengal, India
- **Task:** next-day AQI regression
- **Data:** daily PM2.5, PM10, NO₂, SO₂, O₃ and CO
- **Dashboard:** six-section Streamlit app
- **Tests:** 77 passing

## Results

The shipped model is a **Ridge regression** trained on the 20 most useful features
selected from 119 lag and rolling-statistic features. It was chosen on a validation
period and evaluated once on a later, untouched test period.

| Metric | AeroSense | Simple baseline* |
| :--- | ---: | ---: |
| Test MAE | **16.70** | 15.96 |
| Test RMSE | **21.08** | 21.97 |
| Test R² | **0.773** | 0.753 |

\*The baseline predicts *tomorrow = today*. It is surprisingly strong because AQI
changes slowly from day to day.

**What this means:** AeroSense explains more of the day-to-day variation than the
baseline (higher R²) and has a lower RMSE, but its average absolute error is
slightly worse. It should not be used for health alerts or policy decisions.

Test window: **3 Jun 2026 → 5 Nov 2026** (156 days).

## Dataset

| Item | Value |
| :--- | :--- |
| Period | **1 Jan 2024 → 6 Nov 2026** |
| Daily rows | **1,041** |
| Source | [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) (CAMS/ECMWF daily means) |
| AQI | Derived from PM2.5 using US-EPA breakpoints (not the CPCB composite index) |

The source values are **modelled/reanalysis data**, not direct CPCB station
measurements. They are useful for demonstrating the pipeline but are not an
official city average.

## How it works

1. **Download** — fetch recent hourly air-quality values for Kolkata and aggregate
   them to daily means.
2. **Prepare** — clean the data, derive AQI from PM2.5, and build lag/rolling
   features.
3. **Split** — use chronological train (722 days), validation (153 days) and test
   (156 days) periods with no shuffling.
4. **Train and select** — compare five models on validation RMSE; freeze the best
   one before touching the test set.
5. **Evaluate and serve** — report test metrics and error analysis, then load the
   saved model into the Streamlit dashboard.

The test split is never used for training, tuning or model selection.

## Quick start

### 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the model (fresh clone only)

Trained model files are not committed to Git. If `models/` has no `.pkl` files,
build the data and train once:

```bash
python -m src.scripts.download_recent_aqi --start 2024-01-01 --end 2026-11-06
python -m src.preprocessing.clean
python -m src.feature_engineering.features
python -m src.model_training.prepare_data
python -m src.model_training.train
```

### 3. Run the dashboard

```bash
streamlit run streamlit_app/app.py
```

Open the URL printed in the terminal, usually <http://localhost:8501>.

### 4. Run the tests

```bash
python -m pytest -p no:warnings
```

Expected: **77 passed**.

## Retrain the model

Refresh the data (optional — skip this if the raw CSV already exists):

```bash
python -m src.scripts.download_recent_aqi --start 2024-01-01 --end 2026-11-06
python -m src.preprocessing.clean
```

Rebuild features, train and evaluate:

```bash
python -m src.feature_engineering.features
python -m src.model_training.prepare_data
python -m src.model_training.train
python -m src.model_evaluation.evaluate_final
```

Or use the helper scripts:

```bash
./scripts/run_forecasting.sh
./scripts/run_model_training.sh
./scripts/run_final_evaluation.sh
```

## Project structure

```text
aerosense-aqi/
├── streamlit_app/          # Dashboard entry point and components
├── src/
│   ├── scripts/            # Data download and pipeline helpers
│   ├── preprocessing/      # Cleaning and data-quality checks
│   ├── feature_engineering/# Lag and rolling features
│   ├── model_training/     # Chronological splits, training, selection
│   └── model_evaluation/   # Final test evaluation and reports
├── data/
│   ├── raw/                # Downloaded daily series
│   └── processed/          # Clean data, features and splits
├── models/                 # Saved models + best_model.json
├── results/                # Metrics, predictions and evaluation plots
├── tests/                  # Test suite
├── docs/                   # Data, model-card and EDA documentation
├── config.yaml             # Project configuration
└── requirements.txt        # Pinned dependencies
```

## Model details

Five candidates were compared: Ridge regression, Random Forest, Extra Trees,
Gradient Boosting and SVR. Hyperparameters were tuned with expanding-window
time-series cross-validation on the training split.

| | Validation | Test |
| :--- | ---: | ---: |
| Selected model | Ridge regression | Ridge regression |
| RMSE | 17.58 | 21.08 |
| R² | 0.860 | 0.773 |

The validation RMSE is 8.42% better than the persistence baseline. The test
improvement is smaller (4.03%) because the later period is harder to predict.

## Limitations

- **Modelled data, not monitor readings.** Open-Meteo/CAMS values are interpolated
  fields, not direct station observations.
- **PM2.5-only AQI.** The index uses US-EPA PM2.5 breakpoints, not the CPCB
  composite index.
- **Strong persistence.** Tomorrow's AQI is close to today's, so the simple
  baseline is hard to beat.
- **Peak under-prediction.** The model still smooths pollution spikes and should
  not be used for extreme-event alerts.
- **One test window.** The 156-day test period covers the 2026 monsoon and
  post-monsoon seasons; results may differ in another period.
- **Not for health or policy decisions.** Always check official AQI sources.

## Documentation

- [Data sources and units](docs/DATA_SOURCES.md)
- [Model card](docs/MODEL_CARD.md)
- [Full evaluation report](documentation/model_evaluation.md)
- [EDA findings](docs/eda_findings.md)
- [Feature dictionary](docs/feature_dictionary.md)

## License

MIT — see [LICENSE](LICENSE).

Data attribution: Open-Meteo and the Copernicus Atmosphere Monitoring Service
(CAMS). See [Open-Meteo's licence](https://open-meteo.com/en/license) for current
terms.
