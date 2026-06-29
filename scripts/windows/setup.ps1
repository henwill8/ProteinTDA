$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_common.ps1"

Ensure-Venv
Bootstrap-Pip

& $Script:VenvPython (Join-Path $Script:ScriptsDir "install_requirements.py")

& (Join-Path $PSScriptRoot "build_vpd.ps1")
& $Script:VenvPython (Join-Path $Script:ScriptsDir "download_stereo_chemical_props.py")

Write-Host "Setup complete"
