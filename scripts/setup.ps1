param([string]$Python = "python")
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
& $Python -m venv .venv-biqae
if ($LASTEXITCODE -ne 0) { throw "Failed to create virtual environment" }
& ./.venv-biqae/Scripts/python.exe -m pip install -r environment.lock
if ($LASTEXITCODE -ne 0) { throw "Failed to install locked dependencies" }
& ./.venv-biqae/Scripts/python.exe -m pip install --no-deps --no-build-isolation -e .
if ($LASTEXITCODE -ne 0) { throw "Failed to install package" }

