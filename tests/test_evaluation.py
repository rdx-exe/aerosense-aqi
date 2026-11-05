"""Unit tests for the final evaluation module."""

import numpy as np
import pandas as pd
from pathlib import Path
from src.model_evaluation.evaluate_final import evaluate_final_model


def test_evaluate_final_model_execution() -> None:
    # Verify that the final evaluation script executes cleanly with existing artifacts
    results = evaluate_final_model()
    assert "best_model" in results
    assert "test_metrics" in results
    assert "rmse_improvement" in results
    assert Path("results/metrics/final_metrics.csv").is_file()
    assert Path("docs/model_evaluation.md").is_file()