#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ ! -f .venv/bin/python ]]; then "$PYTHON_BIN" -m venv .venv; fi
if [[ ! -f .venv/.dependencias_instaladas ]]; then
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
  touch .venv/.dependencias_instaladas
fi
if [[ ! -f artifacts/manifest.json ]]; then .venv/bin/python scripts/build_index.py; fi
.venv/bin/python -m streamlit run app.py
