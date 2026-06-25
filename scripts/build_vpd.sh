#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VPD_DIR="$REPO_ROOT/vpd"

# shellcheck source=_python.sh
source "$SCRIPT_DIR/_python.sh"

PYTHON="$(resolve_python)"

if ! "$PYTHON" -c "import torch" >/dev/null 2>&1; then
  echo "error: torch is not installed in the active environment." >&2
  echo "Install project requirements first, e.g. pip install -r requirements.txt" >&2
  exit 1
fi

"$PYTHON" -m pip install -e "$(python_path "$VPD_DIR")" --no-build-isolation
