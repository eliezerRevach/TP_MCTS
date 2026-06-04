# Copy results/ from the GCE VM to your Windows machine.
# Usage: .\gcp\fetch-results.ps1
# Requires: gcloud CLI, config.env in gcp/

$ErrorActionPreference = "Stop"
$GcpDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ConfigPath = Join-Path $GcpDir "config.env"
if (-not (Test-Path $ConfigPath)) {
    Write-Error "Create gcp\config.env from config.env.example (set PROJECT_ID, ZONE, VM_NAME)."
}

Get-Content $ConfigPath | ForEach-Object {
    if ($_ -match '^\s*([^#=]+)=(.*)$') {
        Set-Variable -Name $matches[1].Trim() -Value $matches[2].Trim() -Scope Script
    }
}

$LocalResults = if ($env:TP_MCTS_RESULTS_DIR) { $env:TP_MCTS_RESULTS_DIR } else {
    "C:\Users\eliezer\Documents\hw2_exam\TP_MCTS\results"
}
New-Item -ItemType Directory -Force -Path $LocalResults | Out-Null

$Remote = "${VM_NAME}:/opt/tp_mcts/results/"
# Uses ZONE and PROJECT_ID from config.env via gcloud flags below
Write-Host "Fetching ${Remote} -> ${LocalResults}"
gcloud compute scp --recurse --zone=$ZONE "${Remote}*" "${LocalResults}\" --project=$PROJECT_ID
Write-Host "Done."
