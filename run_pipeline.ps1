param([string]$Config = "config.local.json", [string]$Action = "run")
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
python -u run_pipeline.py --config $Config --action $Action
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
