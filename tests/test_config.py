"""Tests for the configuration loader and path helpers."""

from pathlib import Path

import pytest

from src.utils.config import get_project_root, get_path, load_config


def test_get_project_root() -> None:
    """Project root should resolve to the directory containing ``src``."""
    root = get_project_root()
    assert isinstance(root, Path)
    assert (root / "src").is_dir()
    assert (root / "config.yaml").is_file()


def test_load_config_returns_dict() -> None:
    """``load_config`` should return a non-empty nested dictionary."""
    config = load_config()
    assert isinstance(config, dict)
    assert "project" in config
    assert "data" in config
    assert "training" in config


def test_get_path_resolves_relative() -> None:
    """``get_path`` should join parts onto the project root."""
    p = get_path("data", "raw")
    assert p == get_project_root() / "data" / "raw"


def test_get_path_absolute_passthrough() -> None:
    """Absolute paths should be returned unchanged."""
    abs_path = Path("/tmp/some/absolute/path")
    assert get_path(abs_path) == abs_path


def test_load_config_missing_file_raises(tmp_path: Path) -> None:
    """A missing config file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "nope.yaml"))