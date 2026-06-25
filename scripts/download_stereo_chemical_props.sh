#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=_python.sh
source "$SCRIPT_DIR/_python.sh"

PYTHON="$(resolve_python)"

"$PYTHON" <<'PY'
import ssl
import urllib.request
from pathlib import Path
import site

URL = (
    "https://git.scicore.unibas.ch/schwede/openstructure/-/raw/"
    "7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt"
)
dest = Path(site.getsitepackages()[0]) / "openfold" / "resources" / "stereo_chemical_props.txt"
dest.parent.mkdir(parents=True, exist_ok=True)

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

with urllib.request.urlopen(URL, context=ctx) as resp:
    dest.write_bytes(resp.read())

print(f"Downloaded {dest}")
PY
