#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

"$REPO_ROOT/scripts/build_vpd.sh"
"$REPO_ROOT/scripts/download_stereo_chemical_props.sh" && echo "Setup complete"