"""I/O helpers for datasets and persisted models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .config import get_project_root


def read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV file into a DataFrame.

    Parameters
    ----------
    path:
        Absolute or project-relative path to the CSV file.
    **kwargs:
        Forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame
    """
    path = _resolve(path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path, **kwargs)


def write_csv(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> Path:
    """Write a DataFrame to CSV, creating parent directories as needed.

    Returns the resolved destination path.
    """
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)
    return path


def save_model(model: Any, filename: str, models_dir: str | Path = "models") -> Path:
    """Pickle a fitted model to the models directory.

    Parameters
    ----------
    model:
        Any object supporting ``joblib.dump`` (sklearn estimators, etc.).
    filename:
        Output filename, e.g. ``"random_forest.pkl"``.
    models_dir:
        Project-relative directory where models are stored.

    Returns
    -------
    Path
        Absolute path to the saved model file.
    """
    dest_dir = _resolve(models_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    joblib.dump(model, dest)
    return dest


def load_model(path: str | Path) -> Any:
    """Load a previously pickled model.

    Parameters
    ----------
    path:
        Absolute or project-relative path to a ``.pkl`` / ``.joblib`` file.

    Returns
    -------
    Any
        The deserialised object.
    """
    path = _resolve(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def _resolve(path: str | Path) -> Path:
    """Resolve a path against the project root when it is not absolute."""
    path = Path(path)
    if path.is_absolute():
        return path
    return get_project_root() / path