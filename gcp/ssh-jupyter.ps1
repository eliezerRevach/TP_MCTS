# Open SSH + port-forward for Jupyter Lab on the VM (run Jupyter on the VM first).
# Usage: .\gcp\ssh-jupyter.ps1
# Then on the VM: cd /opt/tp_mcts && jupyter lab --ip=127.0.0.1 --port=8888 --no-browser

$ErrorActionPreference = "Stop"
$GcpDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ConfigPath = Join-Path $GcpDir "config.env"
if (-not (Test-Path $ConfigPath)) {
    Write-Error "Create gcp\config.env from config.env.example."
}
Get-Content $ConfigPath | ForEach-Object {
    if ($_ -match '^\s*([^#=]+)=(.*)$') {
        Set-Variable -Name $matches[1].Trim() -Value $matches[2].Trim() -Scope Script
    }
}

Write-Host @"
On the VM (second terminal or after SSH):
  cd /opt/tp_mcts
  ~/.local/bin/jupyter lab --ip=127.0.0.1 --port=8888 --no-browser

On your PC: open the URL printed by Jupyter (127.0.0.1:8888).
In experiments.ipynb Setup: RUN_ENV = "gce"
"@

gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT_ID -- -L 8888:localhost:8888
