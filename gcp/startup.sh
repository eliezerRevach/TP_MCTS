#!/usr/bin/env bash
# GCE metadata startup script (root). Clones repo then runs bootstrap-vm.sh.
set -euo pipefail

LOG=/var/log/tp-mcts-startup.log
exec > >(tee -a "${LOG}") 2>&1
echo "=== tp-mcts startup $(date -Is) ==="

REPO_URL="${REPO_URL:-https://github.com/eliezerRevach/tp_mcts.git}"
REPO_BRANCH="${REPO_BRANCH:-}"
GCE_REPO_DIR="${GCE_REPO_DIR:-/opt/tp_mcts}"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git

if [[ ! -d "${GCE_REPO_DIR}/.git" ]]; then
  rm -rf "${GCE_REPO_DIR}"
  if [[ -n "${REPO_BRANCH}" ]]; then
    git clone --branch "${REPO_BRANCH}" "${REPO_URL}" "${GCE_REPO_DIR}"
  else
    git clone "${REPO_URL}" "${GCE_REPO_DIR}"
  fi
fi

export REPO_URL REPO_BRANCH GCE_REPO_DIR
bash "${GCE_REPO_DIR}/gcp/bootstrap-vm.sh"

# Allow SSH user to write results (debian default login user varies)
if id -u debian &>/dev/null; then
  chown -R debian:debian "${GCE_REPO_DIR}"
fi

echo "=== startup finished $(date -Is) ==="
