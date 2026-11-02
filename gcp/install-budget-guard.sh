#!/usr/bin/env bash
# Install gcp/budget_guard.sh as a boot service on the VM:  sudo bash /opt/tp_mcts/gcp/install-budget-guard.sh
# Remove:  sudo systemctl disable --now tp-mcts-budget-guard && sudo rm /etc/systemd/system/tp-mcts-budget-guard.service
set -euo pipefail
REPO="${GCE_REPO_DIR:-/opt/tp_mcts}"

cat > /etc/systemd/system/tp-mcts-budget-guard.service <<EOF
[Unit]
Description=Stop the VM when the monthly GCP budget is reached
After=network-online.target

[Service]
ExecStart=/bin/bash ${REPO}/gcp/budget_guard.sh
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now tp-mcts-budget-guard.service
systemctl --no-pager --lines=0 status tp-mcts-budget-guard.service | head -n 3
