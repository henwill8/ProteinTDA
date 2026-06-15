#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VPD_DIR="$REPO_ROOT/vpd"

resolve_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    printf '%s\n' "$PYTHON"
    return
  fi
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    printf '%s\n' "$REPO_ROOT/.venv/bin/python"
    return
  fi
  if [[ -x "$REPO_ROOT/.venv/Scripts/python.exe" ]]; then
    printf '%s\n' "$REPO_ROOT/.venv/Scripts/python.exe"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi
  echo "error: no python interpreter found (set PYTHON or create .venv)" >&2
  exit 1
}

PYTHON="$(resolve_python)"
cd "$VPD_DIR"
"$PYTHON" setup.py build_ext --inplace
