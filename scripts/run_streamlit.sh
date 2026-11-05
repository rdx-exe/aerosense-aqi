#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_streamlit.sh — launch the Streamlit web app
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
exec streamlit run streamlit_app/app.py