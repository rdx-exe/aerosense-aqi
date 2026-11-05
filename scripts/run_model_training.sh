#!/bin/bash
# Run model training and validation-only model selection.
#
# 1. Build chronological train/validation/test splits from the full processed
#    feature table (data/processed/features.csv).
# 2. Train all candidate models on the training split.
# 3. Select the best model using validation RMSE only.
# 4. Save models and models/best_model.json for the final evaluation.
#
# The test split is written by step 1 but is never read by the training code.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

echo "==> Preparing chronological splits from data/processed/features.csv..."
python -m src.model_training.prepare_data

echo "==> Starting model training (selection uses validation only)..."
python -m src.model_training.train

echo "==> Experiment completed."
echo "==> Selection record: models/best_model.json"
echo "==> Check experiments/aqi_forecasting/ for logs and the comparison table."
echo "==> Run ./scripts/run_final_evaluation.sh for the independent test evaluation."
