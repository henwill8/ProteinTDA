#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/venv.sh
source "$SCRIPT_DIR/lib/venv.sh"

ensure_venv
bootstrap_pip

"$VENV_PY" "$SCRIPTS_DIR/install_requirements.py"

"$SCRIPT_DIR/build_vpd.sh"
"$VENV_PY" "$SCRIPTS_DIR/download_stereo_chemical_props.py"

echo "Setup complete"
