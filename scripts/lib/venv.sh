#!/usr/bin/env bash

# Shared helpers for Linux/macOS scripts. Source from scripts/*.sh.

if [[ -n "${_PROTEINTDA_VENV_LOADED:-}" ]]; then
  return 0
fi
_PROTEINTDA_VENV_LOADED=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPTS_DIR="$REPO_ROOT/scripts"
VENV="$REPO_ROOT/.venv"
VENV_PY="$VENV/bin/python"

ensure_venv() {
  if [[ -x "$VENV_PY" ]]; then
    return 0
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 is required to create .venv" >&2
    exit 1
  fi
  echo "Creating virtual environment at $VENV"
  python3 -m venv "$VENV"
}

bootstrap_pip() {
  "$VENV_PY" -m pip install --upgrade pip setuptools wheel
}

require_torch() {
  if ! "$VENV_PY" -c "import torch" >/dev/null 2>&1; then
    echo "error: torch is not installed. Run scripts/setup.sh first." >&2
    exit 1
  fi
}
