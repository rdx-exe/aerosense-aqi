"""Central configuration loader.

All paths are resolved relative to the project root using :mod:`pathlib`,
which keeps the code portable across operating systems and makes scripts
executable from any working directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def get_project_root() -> Path:
    """Return the absolute path to the project root directory.

    The project root is the parent directory of the ``src`` package, so this
    works regardless of the current working directory.
    """
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_name: str = "config.yaml") -> dict[str, Any]:
    """Load a YAML configuration file from the project root.

    Parameters
    ----------
    config_name:
        Filename (or absolute path) of the YAML file. Defaults to
        ``config.yaml`` in the project root.

    Returns
    -------
    dict
        Parsed configuration as a nested dictionary.
    """
    root = get_project_root()
    config_path = Path(config_name)
    if not config_path.is_absolute():
        config_path = root / config_path

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Project root: {root}"
        )

    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_path(*parts: str | Path, config: dict[str, Any] | None = None) -> Path:
    """Resolve a path relative to the project root.

    Parameters
    ----------
    *parts:
        Path components joined to the project root.
    config:
        Unused — kept for API symmetry. Reserved for future relative-path
        resolution against configured directories.

    Returns
    -------
    Path
        Absolute path resolved from the project root.
    """
    return get_project_root().joinpath(*[str(p) for p in parts])