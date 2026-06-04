#!/usr/bin/env bash
# Create the experiment VM. Requires gcloud and gcp/config.env.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -f config.env ]]; then
  # shellcheck disable=SC1091
  source config.env
else
  echo "Copy config.env.example to config.env and set PROJECT_ID."
  exit 1
fi

: "${PROJECT_ID:?Set PROJECT_ID in config.env}"
: "${ZONE:?Set ZONE in config.env}"
: "${VM_NAME:?Set VM_NAME in config.env}"
: "${MACHINE_TYPE:?Set MACHINE_TYPE in config.env}"

gcloud config set project "${PROJECT_ID}"

if gcloud compute instances describe "${VM_NAME}" --zone="${ZONE}" &>/dev/null; then
  echo "VM ${VM_NAME} already exists in ${ZONE}. Stop/delete it first or change VM_NAME."
  exit 1
fi

STARTUP=$(mktemp)
trap 'rm -f "${STARTUP}"' EXIT
{
  echo "export REPO_URL='${REPO_URL:-https://github.com/eliezerRevach/tp_mcts.git}'"
  echo "export REPO_BRANCH='${REPO_BRANCH:-}'"
  echo "export GCE_REPO_DIR='${GCE_REPO_DIR:-/opt/tp_mcts}'"
  cat "${SCRIPT_DIR}/startup.sh"
} > "${STARTUP}"

echo "Creating ${VM_NAME} (${MACHINE_TYPE}) in ${ZONE} ..."
gcloud compute instances create "${VM_NAME}" \
  --zone="${ZONE}" \
  --machine-type="${MACHINE_TYPE}" \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size="${DISK_SIZE_GB:-100}GB" \
  --boot-disk-type=pd-balanced \
  --metadata=startup-script="$(cat "${STARTUP}")"

echo ""
echo "Wait 2–5 minutes, then check:"
echo "  gcloud compute ssh ${VM_NAME} --zone=${ZONE} --command='test -f /opt/tp_mcts/.tp_mcts_gce && git -C /opt/tp_mcts rev-parse --short HEAD'"
echo ""
echo "Next: gcp/README.md (Jupyter or Cursor Remote-SSH)."
