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

# Writable by gcloud SSH user (often local username, not always debian)
for U in debian ubuntu "$(getent passwd | awk -F: '$3>=1000 && $3<65534 {print $1; exit}')"; do
  if id -u "${U}" &>/dev/null 2>&1; then
    chown -R "${U}:${U}" "${GCE_REPO_DIR}"
    sudo -u "${U}" git config --global --add safe.directory "${GCE_REPO_DIR}" || true
    break
  fi
done

echo "=== startup finished $(date -Is) ==="
