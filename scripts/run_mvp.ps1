$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
& ./.venv-biqae/Scripts/python.exe -m q3_biqae run --config configs/mvp_4asset_16state.json --with-optimization
if ($LASTEXITCODE -ne 0) { throw "MVP run failed" }
