# AI-Based Air Quality Index Prediction and Analysis Using Machine Learning

> Forecasting next-day Air Quality Index (AQI) for **Kolkata, West Bengal, India**
> using historical air quality data and machine learning.

---

## 📌 Overview

This project builds a complete, modular ML pipeline to predict the **next day's
AQI for Kolkata**. It covers data acquisition, exploratory analysis,
preprocessing, feature engineering, model training, evaluation, and a
Streamlit web application for interactive forecasting.

All scripts are written to be **executable from the project root** and use
[`pathlib`](https://docs.python.org/3/library/pathlib.html) for portable file
paths. No GPU is required — everything runs on CPU with lightweight,
open-source libraries.

---

## 🗂️ Project Architecture

```
.
├── config.yaml                  # Central configuration (paths, hyper-params)
├── pyproject.toml               # Project metadata + dependency declaration
├── requirements.txt             # Pinned dependency list
├── .gitignore                   # Ignores data, models, caches, venvs
├── README.md                    # This file
│
├── data/
│   ├── raw/                     # Raw downloaded datasets (git-ignored)
│   └── processed/               # Cleaned, feature-engineered datasets
│
├── notebooks/                   # Jupyter notebooks for EDA
│
├── src/
│   ├── utils/                   # Shared helpers: config loader, I/O
│   ├── preprocessing/           # Cleaning, missing-value & outlier handling
│   ├── feature_engineering/     # Lag/rolling features, target definition
│   ├── model_training/          # Model selection, hyper-parameter tuning
│   ├── model_evaluation/        # Metrics, residual analysis, reporting
│   └── scripts/                 # End-to-end pipeline entry points
│
├── models/                      # Trained model artifacts (git-ignored)
├── visualizations/              # EDA & evaluation plots (git-ignored)
├── streamlit_app/               # Streamlit web app source
├── docs/                        # Project documentation
├── tests/                       # pytest test suite
└── scripts/                     # Shell wrappers (install, run, etc.)
```

### Directory Purpose

| Directory | Purpose |
|-----------|---------|
| `data/raw/` | Raw air-quality data downloaded from OpenAQ (or a local fallback CSV). Never committed. |
| `data/processed/` | Cleaned, imputed, feature-engineered datasets ready for modeling. |
| `notebooks/` | Exploratory data analysis notebooks (seasonality, correlations, distributions). |
| `src/utils/` | Shared utilities: YAML config loader, CSV/model I/O helpers. |
| `src/preprocessing/` | Missing-value imputation, outlier clipping, scaling. |
| `src/feature_engineering/` | Lag features, rolling statistics, next-day target construction. |
| `src/model_training/` | Train/validation/test split, model fitting, cross-validation. |
| `src/model_evaluation/` | Metric computation, residual plots, feature importance, reporting. |
| `models/` | Serialized trained models (`.pkl`/`.joblib`). |
| `visualizations/` | Matplotlib/Seaborn/Plotly figures produced by EDA and evaluation. |
| `streamlit_app/` | Interactive web app for AQI prediction and model explanation. |
| `docs/` | Design notes, API docs, model cards. |
| `tests/` | Unit tests for preprocessing, features, and I/O helpers. |
| `scripts/` | Convenience shell scripts (`install.sh`, `run_pipeline.sh`). |

---

## 📦 Installation

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows (PowerShell)
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
# or, using the project metadata:
pip install -e ".[dev]"
```

> **Note:** `requirements.txt` pins exact versions for reproducibility.
> `pyproject.toml` declares the same dependencies with looser bounds and adds
> the `dev` extra (`pytest`, `ipython`).

### 3. Acquire data

The pipeline can fetch daily air-quality measurements for Kolkata from the
free [OpenAQ API](https://docs.openaq.org/) (no API key required):

```bash
python -m src.scripts.download_data --start 2020-01-01 --end 2024-12-31
```

Or place a local CSV at `data/raw/kolkata_aqi_history.csv` with columns:
`date,pm25,pm10,no2,so2,o3,co,aqi`.

---

## 🚀 Usage

### Run the full pipeline

```bash
python -m src.scripts.run_pipeline
```

This sequentially runs: download → preprocess → feature engineering →
train → evaluate → save artifacts.

### Run individual stages

```bash
python -m src.preprocessing.clean          # → data/processed/clean.csv
python -m src.feature_engineering.features # → data/processed/features.csv
python -m src.model_training.prepare_data  # → chronological splits
python -m src.model_training.train         # → models/*.pkl + models/best_model.json
python -m src.model_evaluation.evaluate_final  # → results/, documentation/model_evaluation.md
```

### Launch the Streamlit app

```bash
streamlit run streamlit_app/app.py
```

---

## 📊 Metrics

The models are evaluated on a held-out test set using:

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **R²** — Coefficient of determination

RMSE is the primary metric (sensitive to large AQI spikes, which matter most
for public-health decisions).

---

## ⚠️ Disclaimer

This project is for **educational and research purposes**. AQI predictions are
approximations and should not be used as the sole basis for health-related
decisions. Data is sourced from OpenAQ and may contain measurement gaps.

---

## 📄 License

MIT — see `LICENSE` for details.