#!/usr/bin/env bash

using_windows_python() {
  [[ "${1:-$PYTHON}" == *".exe" ]]
}

python_path() {
  local path="$1"
  if using_windows_python "$PYTHON" && command -v wslpath >/dev/null 2>&1; then
    wslpath -w "$path"
  else
    printf '%s\n' "$path"
  fi
}

wsl_path() {
  local path="$1"
  if using_windows_python "$PYTHON" && command -v wslpath >/dev/null 2>&1; then
    wslpath -u "$path"
  else
    printf '%s\n' "$path"
  fi
}

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

ensure_venv() {
  if [[ -x "$REPO_ROOT/.venv/bin/python" || -x "$REPO_ROOT/.venv/Scripts/python.exe" ]]; then
    return
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 is required to create .venv" >&2
    exit 1
  fi
  echo "Creating virtual environment at $REPO_ROOT/.venv"
  python3 -m venv "$REPO_ROOT/.venv"
}
