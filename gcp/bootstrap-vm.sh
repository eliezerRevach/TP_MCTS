#!/usr/bin/env bash
# One-time (or re-runnable) setup on a Debian/Ubuntu GCE VM.
# Installs Python + Jupyter, clones/updates TP-MCTS under GCE_REPO_DIR.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/eliezerRevach/tp_mcts.git}"
REPO_BRANCH="${REPO_BRANCH:-}"
GCE_REPO_DIR="${GCE_REPO_DIR:-/opt/tp_mcts}"
MARKER="${GCE_REPO_DIR}/.tp_mcts_gce"

export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get install -y -qq git python3 python3-pip python3-venv build-essential

sudo mkdir -p "$(dirname "${GCE_REPO_DIR}")"
if [[ ! -d "${GCE_REPO_DIR}/.git" ]]; then
  sudo rm -rf "${GCE_REPO_DIR}"
  if [[ -n "${REPO_BRANCH}" ]]; then
    sudo git clone --branch "${REPO_BRANCH}" "${REPO_URL}" "${GCE_REPO_DIR}"
  else
    sudo git clone "${REPO_URL}" "${GCE_REPO_DIR}"
  fi
else
  sudo git -C "${GCE_REPO_DIR}" fetch --all --prune
  if [[ -n "${REPO_BRANCH}" ]]; then
    sudo git -C "${GCE_REPO_DIR}" checkout "${REPO_BRANCH}"
  fi
  sudo git -C "${GCE_REPO_DIR}" pull --ff-only || true
fi

sudo touch "${MARKER}"

VENV="${GCE_REPO_DIR}/.venv"
if [[ ! -d "${VENV}" ]]; then
  python3 -m venv "${VENV}"
fi
"${VENV}/bin/pip" install -q --upgrade pip
"${VENV}/bin/pip" install -q dill numpy pandas openpyxl networkx jupyterlab ipykernel pytest

# Kernel for Jupyter / VS Code Remote-SSH (install for every login user that exists)
for U in debian "$(logname 2>/dev/null || true)" "$(whoami)"; do
  [[ -z "${U}" || "${U}" == "root" ]] && continue
  if id -u "${U}" &>/dev/null; then
    sudo -u "${U}" "${VENV}/bin/python" -m ipykernel install --user --name tp-mcts --display-name "TP-MCTS (Python 3)" || true
  fi
done

if id -u debian &>/dev/null; then
  sudo chown -R debian:debian "${GCE_REPO_DIR}"
elif [[ -n "${SUDO_USER:-}" ]]; then
  sudo chown -R "${SUDO_USER}:${SUDO_USER}" "${GCE_REPO_DIR}"
fi

mkdir -p "${GCE_REPO_DIR}/results"
echo "Bootstrap done. Repo: ${GCE_REPO_DIR}"
git -C "${GCE_REPO_DIR}" rev-parse --short HEAD
