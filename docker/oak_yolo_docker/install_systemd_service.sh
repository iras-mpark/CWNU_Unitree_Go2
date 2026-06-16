#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_NAME="oak-yolo-docker.service"
CONTAINER_NAME="oak_yolo_target"
DOCKER_BIN="$(command -v docker)"

sudo tee "/etc/systemd/system/${SERVICE_NAME}" >/dev/null <<UNITEOF
[Unit]
Description=OAK YOLO Target Docker Container
Requires=docker.service
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
WorkingDirectory=${SCRIPT_DIR}
ExecStart=${SCRIPT_DIR}/run_container.sh
ExecStop=${DOCKER_BIN} stop ${CONTAINER_NAME}
RemainAfterExit=yes
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
UNITEOF

sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"

echo "[systemd] Installed: ${SERVICE_NAME}"
echo "Start:  sudo systemctl start ${SERVICE_NAME}"
echo "Status: sudo systemctl status ${SERVICE_NAME}"
