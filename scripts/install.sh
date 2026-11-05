#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# install.sh — create virtual environment and install project dependencies
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "==> Creating virtual environment (.venv)"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip"
pip install --upgrade pip

echo "==> Installing project dependencies"
pip install -r requirements.txt

echo "==> Installing project in editable mode with dev extras"
pip install -e ".[dev]"

echo "==> Done. Activate with: source .venv/bin/activate"