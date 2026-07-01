#!/usr/bin/env bash

# Shared helpers for Linux/macOS scripts. Source from scripts/*.sh.

if [[ -n "${_PROTEINTDA_VENV_LOADED:-}" ]]; then
  return 0
fi
_PROTEINTDA_VENV_LOADED=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPTS_DIR="$REPO_ROOT/scripts"
VENV="$REPO_ROOT/.venv"
VENV_PY=""

python_works() {
  "$1" -c "import subprocess, pip" >/dev/null 2>&1
}

resolve_python_candidate() {
  if [[ -n "${PYTHON:-}" ]]; then
    printf '%s\n' "$PYTHON"
    return
  fi
  if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    printf '%s\n' "$CONDA_PREFIX/bin/python"
    return
  fi
  if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    printf '%s\n' "$VIRTUAL_ENV/bin/python"
    return
  fi
  if [[ -x "$VENV/bin/python" ]]; then
    printf '%s\n' "$VENV/bin/python"
  fi
}

ensure_env() {
  local candidate
  candidate="$(resolve_python_candidate)"

  if [[ -n "$candidate" ]] && python_works "$candidate"; then
    VENV_PY="$candidate"
    return 0
  fi

  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    echo "error: active conda env python at $CONDA_PREFIX is not usable" >&2
    exit 1
  fi

  if [[ -d "$VENV" ]]; then
    echo "Removing broken virtual environment at $VENV" >&2
    rm -rf "$VENV"
  fi

  if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 is required to create .venv" >&2
    exit 1
  fi

  echo "Creating virtual environment at $VENV"
  python3 -m venv "$VENV"
  VENV_PY="$VENV/bin/python"
}

# Backwards-compatible alias used by setup.sh / build_vpd.sh.
ensure_venv() {
  ensure_env
}

bootstrap_pip() {
  echo "Using $("$VENV_PY" -c 'import sys; print(sys.executable)')"
  "$VENV_PY" -m pip install --upgrade pip setuptools wheel
}

require_torch() {
  if ! "$VENV_PY" -c "import torch" >/dev/null 2>&1; then
    echo "error: torch is not installed. Run scripts/setup.sh first." >&2
    exit 1
  fi
}
