#!/bin/bash
# Run the complete machine learning pipeline
# 1. Feature engineering (if needed)
# 2. Model training and validation
# 3. Model comparison and selection

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Ensure dependencies are installed
source .venv/bin/activate

# Step 1: Run feature engineering if splits don't exist
if [ ! -f "data/processed/splits/train.csv" ] || [ ! -f "data/processed/splits/validation.csv" ]; then
    echo "==> Running feature engineering (forecasting features)..."
    python3 -m src.feature_engineering.forecast_features
fi

# Step 2: Run model training
echo "==> Starting model training..."
python3 -m src.model_training.train

# Step 3: Summary
echo "==> Experiment completed."
echo "==> Check experiments/aqi_forecasting/ for detailed logs and saved models."

echo "==> To run the Streamlit dashboard later:"
echo "   ./scripts/run_streamlit.sh"