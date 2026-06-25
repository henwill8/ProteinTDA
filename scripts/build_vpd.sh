#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/venv.sh
source "$SCRIPT_DIR/lib/venv.sh"

ensure_venv
require_torch

"$VENV_PY" -m pip install -e "$REPO_ROOT/vpd" --no-build-isolation
