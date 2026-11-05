#!/bin/bash
# Rebuild the forecasting feature table and chronological splits.
#
# This script intentionally uses the full processed dataset
# (data/processed/clean.csv → features.csv) rather than the legacy 5-row
# smoke-test fixture. Model training and evaluation are separate steps:
#   ./scripts/run_model_training.sh
#   ./scripts/run_final_evaluation.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

echo "==> Building features from data/processed/clean.csv..."
python -m src.feature_engineering.features

echo "==> Creating chronological train/validation/test splits..."
python -m src.model_training.prepare_data

echo "==> Done."
echo "==> Next: ./scripts/run_model_training.sh"
echo "==> Then: ./scripts/run_final_evaluation.sh"
