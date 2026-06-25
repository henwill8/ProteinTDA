#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=_python.sh
source "$SCRIPT_DIR/_python.sh"

ensure_venv
PYTHON="$(resolve_python)"

"$PYTHON" -m pip install -r "$(python_path "$REPO_ROOT/requirements.txt")" --no-build-isolation

"$REPO_ROOT/scripts/build_vpd.sh"
"$REPO_ROOT/scripts/download_stereo_chemical_props.sh" && echo "Setup complete"
