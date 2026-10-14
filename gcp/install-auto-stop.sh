#!/usr/bin/env bash
# Install gcp/auto_stop.sh as a boot service on the VM:  sudo bash /opt/tp_mcts/gcp/install-auto-stop.sh
# Remove:  sudo systemctl disable --now tp-mcts-auto-stop && sudo rm /etc/systemd/system/tp-mcts-auto-stop.service
set -euo pipefail
REPO="${GCE_REPO_DIR:-/opt/tp_mcts}"
IDLE_MINUTES="${IDLE_MINUTES:-15}"

cat > /etc/systemd/system/tp-mcts-auto-stop.service <<EOF
[Unit]
Description=Stop the VM after ${IDLE_MINUTES} idle minutes (no tp_mcts job, no SSH)
After=network-online.target

[Service]
Environment=IDLE_MINUTES=${IDLE_MINUTES}
ExecStart=/bin/bash ${REPO}/gcp/auto_stop.sh
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now tp-mcts-auto-stop.service
systemctl --no-pager --lines=0 status tp-mcts-auto-stop.service | head -n 3
