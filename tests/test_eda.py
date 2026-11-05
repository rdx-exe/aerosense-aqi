"""Unit tests for the Exploratory Data Analysis module."""

import pytest
from pathlib import Path
from src.exploratory.eda import extract_season


def test_extract_season() -> None:
    assert extract_season(1) == "Winter"
    assert extract_season(3) == "Spring"
    assert extract_season(6) == "Summer"
    assert extract_season(9) == "Autumn"


def test_eda_module_import() -> None:
    from src.exploratory.eda import run_eda
    assert callable(run_eda)