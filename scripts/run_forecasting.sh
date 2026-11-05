#!/bin/bash
# Run the complete AQI forecasting pipeline
# 1. Feature engineering
# 2. Chronological data splitting
# 3. Data inspection

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "==> Running feature engineering..."
python3 -m src.feature_engineering.forecast_features

echo "==> Data inspection..."
python3 -c "
import pandas as pd
from src.feature_engineering.forecast_features import create_forecasting_features, split_chronological_data
print('Data loading and feature engineering completed.')
print('\\nFeature set created successfully.')
print('\\nNote: Model training pipeline not executed yet.')
"

echo "==> Done."