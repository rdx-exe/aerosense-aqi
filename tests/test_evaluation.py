"""Tests for the final evaluation module."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.model_evaluation.evaluate_final import evaluate_final_model


@pytest.fixture(scope="module")
def final_results() -> dict:
    """Run the final evaluation once for this test module."""
    test_split = Path("data/processed/splits/test.csv")
    selection = Path("models/best_model.json")
    if not test_split.is_file() or not selection.is_file():
        pytest.skip("Final evaluation artifacts are not available")
    return evaluate_final_model()


def test_evaluate_final_model_execution(final_results: dict) -> None:
    assert "best_model" in final_results
    assert "test_metrics" in final_results
    assert "baseline_metrics" in final_results
    assert "rmse_improvement" in final_results

    metrics_path = Path("results/metrics/final_metrics.csv")
    assert metrics_path.is_file()

    metrics = pd.read_csv(metrics_path)
    assert {"selected_model", "persistence_baseline"} == set(metrics["role"])
    assert len(metrics) == 2


def test_final_evaluation_uses_documented_test_range(final_results: dict) -> None:
    test_split = pd.read_csv("data/processed/splits/test.csv", parse_dates=["date"])
    expected_start = str(test_split["date"].min().date())
    expected_end = str(test_split["date"].max().date())

    assert final_results["test_date_range"] == (expected_start, expected_end)

    doc_path = Path("documentation/model_evaluation.md")
    assert doc_path.is_file()
    doc = doc_path.read_text(encoding="utf-8")
    assert expected_start in doc
    assert expected_end in doc


def test_predictions_cover_the_test_split(final_results: dict) -> None:
    test_split = pd.read_csv("data/processed/splits/test.csv")
    predictions = pd.read_csv("results/metrics/test_predictions.csv")
    assert len(predictions) == len(test_split)
    assert predictions["actual_aqi"].notna().all()
    assert predictions["predicted_aqi"].notna().all()


def test_metadata_records_no_test_training(final_results: dict) -> None:
    metadata = json.loads(
        Path("results/metrics/evaluation_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["test_split_used_during_training"] is False
    assert metadata["model_selection"]["test_split_used_during_training"] is False
