$ErrorActionPreference = "Stop"

$Script:RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Script:ScriptsDir = Join-Path $Script:RepoRoot "scripts"
$Script:VenvDir = Join-Path $Script:RepoRoot ".venv"
$Script:VenvPython = Join-Path $Script:VenvDir "Scripts\python.exe"

function Ensure-Venv {
    if (Test-Path $Script:VenvPython) {
        return
    }
    Write-Host "Creating virtual environment at $($Script:VenvDir)"
    python -m venv $Script:VenvDir
}

function Bootstrap-Pip {
    & $Script:VenvPython -m pip install --upgrade pip setuptools wheel
}

function Require-Torch {
    & $Script:VenvPython -c "import torch" *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "torch is not installed. Run scripts/windows/setup.ps1 first."
    }
}
