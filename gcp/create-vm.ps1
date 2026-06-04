# Create TP-MCTS experiment VM (Windows). Uses gcp/config.env
$ErrorActionPreference = "Stop"
$GcpDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ConfigPath = Join-Path $GcpDir "config.env"
if (-not (Test-Path $ConfigPath)) { throw "Missing gcp\config.env — copy from config.env.example" }

Get-Content $ConfigPath | ForEach-Object {
    if ($_ -match '^\s*([^#=]+)=(.*)$') {
        Set-Variable -Name $matches[1].Trim() -Value $matches[2].Trim() -Scope Script
    }
}

$startupFile = Join-Path $env:TEMP "tp-mcts-startup.sh"
$branch = if ($REPO_BRANCH) { $REPO_BRANCH } else { "master" }
@"
export REPO_URL='$REPO_URL'
export REPO_BRANCH='$branch'
export GCE_REPO_DIR='$GCE_REPO_DIR'

"@ + (Get-Content -Raw (Join-Path $GcpDir "startup.sh")) | Set-Content -Path $startupFile -Encoding utf8

gcloud config set project $PROJECT_ID
if (gcloud compute instances describe $VM_NAME --zone=$ZONE 2>$null) {
    Write-Host "VM $VM_NAME already exists in $ZONE"
    exit 0
}

Write-Host "Creating $VM_NAME ($MACHINE_TYPE) in $ZONE ..."
gcloud compute instances create $VM_NAME `
    --zone=$ZONE `
    --machine-type=$MACHINE_TYPE `
    --image-family=debian-12 `
    --image-project=debian-cloud `
    --boot-disk-size="${DISK_SIZE_GB}GB" `
    --boot-disk-type=pd-balanced `
    --metadata-from-file=startup-script=$startupFile

Write-Host "Done. Check: gcloud compute ssh $VM_NAME --zone=$ZONE --command='test -f /opt/tp_mcts/.tp_mcts_gce && echo ready'"
