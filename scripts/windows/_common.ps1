$ErrorActionPreference = "Stop"

$Script:RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Script:ScriptsDir = Join-Path $Script:RepoRoot "scripts"
$Script:VenvDir = Join-Path $Script:RepoRoot ".venv"
$Script:VenvPython = Join-Path $Script:VenvDir "Scripts\python.exe"

function Resolve-PythonCandidate {
    if ($env:PYTHON) { return $env:PYTHON }
    if ($env:CONDA_PREFIX) {
        $condaPy = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path $condaPy) { return $condaPy }
    }
    if (Test-Path $Script:VenvPython) { return $Script:VenvPython }
    return $null
}

function Test-PythonWorks([string]$PythonPath) {
    & $PythonPath -c "import subprocess, pip" *> $null
    return $LASTEXITCODE -eq 0
}

function Ensure-Env {
    $candidate = Resolve-PythonCandidate

    if ($candidate -and (Test-PythonWorks $candidate)) {
        $Script:VenvPython = $candidate
        return
    }

    if ($env:CONDA_PREFIX) {
        throw "Active conda env python at $($env:CONDA_PREFIX) is not usable."
    }

    if (Test-Path $Script:VenvDir) {
        Write-Host "Removing broken virtual environment at $($Script:VenvDir)"
        Remove-Item -Recurse -Force $Script:VenvDir
    }

    Write-Host "Creating virtual environment at $($Script:VenvDir)"
    python -m venv $Script:VenvDir
    $Script:VenvPython = Join-Path $Script:VenvDir "Scripts\python.exe"
}

function Ensure-Venv {
    Ensure-Env
}

function Bootstrap-Pip {
    $resolved = & $Script:VenvPython -c "import sys; print(sys.executable)"
    Write-Host "Using $resolved"
    & $Script:VenvPython -m pip install --upgrade pip setuptools wheel
}

function Require-Torch {
    & $Script:VenvPython -c "import torch" *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "torch is not installed. Run scripts/windows/setup.ps1 first."
    }
}
