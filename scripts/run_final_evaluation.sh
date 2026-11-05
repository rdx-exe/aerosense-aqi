#!/bin/bash
# Run the final, independent evaluation on the untouched test split.
#
# Prerequisites (run once, in order):
#   python -m src.model_training.prepare_data
#   python -m src.model_training.train
#
# This script only loads the validation-selected model and the frozen test
# split. It never retrains or tunes a model.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

python -m src.model_evaluation.evaluate_final

echo "==> Final evaluation artifacts:"
echo "    results/metrics/final_metrics.csv"
echo "    results/visualizations/evaluation/"
echo "    documentation/model_evaluation.md"
