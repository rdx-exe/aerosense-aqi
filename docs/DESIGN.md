# AeroSense — Design Notes

## Architecture Principles

- **Modular**: each pipeline stage lives in its own sub-package
  (`preprocessing`, `feature_engineering`, `model_training`,
  `model_evaluation`) and exposes a single public function plus a `main()`
  CLI entry point. Stages are composable and independently testable.
- **Pathlib everywhere**: all file paths are resolved relative to the project
  root via `src.utils.config.get_path()`, so scripts run from any directory.
- **Configuration-driven**: `config.yaml` centralises paths, hyper-parameters
  and model registry. Code reads from it at runtime; nothing is hard-coded.
- **No global state**: functions accept data and config as arguments, making
  them trivial to unit-test.

## Data Flow

```
OpenAQ API ──► data/raw/*.csv
      │
      ▼
src.preprocessing.clean.clean_raw_data() ──► data/processed/clean.csv
      │
      ▼
src.feature_engineering.features.build_features() ──► data/processed/features.csv
      │
      ▼
src.model_training.train.train_models() ──► models/*.pkl
      │
      ▼
src.model_evaluation.evaluate.evaluate_models() ──► visualizations/*.png, report.md
```

## Model Registry

| Name | Estimator | Use case |
|------|-----------|----------|
| `random_forest` | `RandomForestRegressor` | Robust baseline, non-linear, feature importance |
| `gradient_boosting` | `GradientBoostingRegressor` | Strong tabular default |
| `ridge` | `Ridge` | Linear baseline, fast to train |
| `svr` | `SVR` | Small-data non-linear baseline |

XGBoost is listed in `config.yaml` but only used if installed.

## Evaluation

Primary metric: **RMSE** (sensitive to AQI spikes, which matter for public
health). Secondary: MAE and R². Models are selected on validation RMSE using a
time-aware split (no shuffling).