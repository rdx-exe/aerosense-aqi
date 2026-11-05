# API Reference

All public functions accept data and an optional pre-loaded config dict, so
they are easy to call from Python or test.

## `src.utils.config`

```python
from src.utils.config import get_project_root, get_path, load_config

root: Path = get_project_root()          # absolute project root
config: dict = load_config("config.yaml")
path: Path = get_path("data", "raw")     # root / data / raw
```

## `src.utils.io`

```python
from src.utils.io import read_csv, write_csv, save_model, load_model

df: pd.DataFrame = read_csv("data/raw/x.csv")
write_csv(df, "data/processed/y.csv")
save_model(model, "random_forest.pkl", models_dir="models")
model = load_model("models/random_forest.pkl")
```

## `src.preprocessing.clean`

```python
from src.preprocessing.clean import clean_raw_data

clean: pd.DataFrame = clean_raw_data(raw_df, config=config)
```

## `src.feature_engineering.features`

```python
from src.feature_engineering.features import build_features, compute_aqi

features: pd.DataFrame = build_features(clean_df, config=config)
```

## `src.model_training.train`

```python
from src.model_training.train import train_models

result = train_models(features, config=config)
# result["best"], result["models"], result["X_test"], result["y_test"]
```

## `src.model_evaluation.evaluate`

```python
from src.model_evaluation.evaluate import evaluate_models

result = evaluate_models(features, config=config)
# result["metrics"], result["best"], result["predictions"]
```

## `src.scripts.download_data`

```python
from src.scripts.download_data import download_data

daily: pd.DataFrame = download_data("2020-01-01", "2024-12-31")
```