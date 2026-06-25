$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_common.ps1"

Ensure-Venv
Require-Torch

& $Script:VenvPython -m pip install -e (Join-Path $Script:RepoRoot "vpd") --no-build-isolation
