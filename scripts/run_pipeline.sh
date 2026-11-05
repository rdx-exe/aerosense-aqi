#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_pipeline.sh — run the full AQI prediction pipeline
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

START="${1:-2020-01-01}"
END="${2:-2024-12-31}"

echo "==> Running full pipeline (${START} → ${END})"
python -m src.scripts.run_pipeline --start "$START" --end "$END"