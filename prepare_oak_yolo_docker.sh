#!/usr/bin/env bash
set -euo pipefail

# Prepare a Docker build context for the oak_yolo_target ROS 2 package.
# This script DOES NOT run `docker build`.
# Usage from repository root:
#   bash prepare_oak_yolo_docker.sh
# Optional:
#   bash prepare_oak_yolo_docker.sh /path/to/oak_yolo_target /path/to/docker_context

PKG_SRC="${1:-./oak_yolo_target}"
CTX_DIR="${2:-./docker/oak_yolo_target}"
IMAGE_NAME_DEFAULT="oak-yolo-target:humble"
CONTAINER_NAME_DEFAULT="oak_yolo_target"

if [[ ! -f "${PKG_SRC}/package.xml" ]] || ! grep -q "<name>oak_yolo_target</name>" "${PKG_SRC}/package.xml"; then
  echo "[ERROR] oak_yolo_target package not found at: ${PKG_SRC}" >&2
  echo "        Run from the extracted CWNU_Unitree_Go2-main directory or pass the package path." >&2
  exit 1
fi

echo "[INFO] Package source : ${PKG_SRC}"
echo "[INFO] Docker context : ${CTX_DIR}"

rm -rf "${CTX_DIR}"
mkdir -p "${CTX_DIR}/src"

if command -v rsync >/dev/null 2>&1; then
  rsync -a \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude 'build' \
    --exclude 'install' \
    --exclude 'log' \
    "${PKG_SRC}" "${CTX_DIR}/src/"
else
  cp -a "${PKG_SRC}" "${CTX_DIR}/src/"
  find "${CTX_DIR}/src/oak_yolo_target" -type d \( -name __pycache__ -o -name build -o -name install -o -name log -o -name .git \) -prune -exec rm -rf {} + || true
  find "${CTX_DIR}/src/oak_yolo_target" -type f -name '*.pyc' -delete || true
fi

cat > "${CTX_DIR}/Dockerfile" <<'DOCKERFILE_EOF'
# Default is the official ROS 2 Humble Jammy image.
# This is the safest baseline for ROS 2 apt packages such as depthai_ros_driver.
# On Jetson, this will normally run YOLO on CPU unless you replace the base image
# with a Jetson CUDA/PyTorch image and install ROS 2 Humble there.
ARG BASE_IMAGE=ros:humble-ros-base-jammy
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    udev \
    libusb-1.0-0 \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-opencv \
    python3-numpy \
    ros-${ROS_DISTRO}-launch \
    ros-${ROS_DISTRO}-launch-ros \
    ros-${ROS_DISTRO}-rclpy \
    ros-${ROS_DISTRO}-sensor-msgs \
    ros-${ROS_DISTRO}-geometry-msgs \
    ros-${ROS_DISTRO}-std-msgs \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-image-transport \
    ros-${ROS_DISTRO}-compressed-image-transport \
    ros-${ROS_DISTRO}-depthai-ros-driver \
    ros-${ROS_DISTRO}-depthai-ros-msgs \
    ros-${ROS_DISTRO}-rmw-fastrtps-cpp \
    && rm -rf /var/lib/apt/lists/*

# Ultralytics normally pulls torch. On Jetson, GPU-enabled torch must match
# JetPack/CUDA. This default image is intended as a reliable CPU baseline.
# If you already use a Jetson PyTorch base image, you can modify this line to
# `pip install ultralytics --no-deps` after installing the remaining deps.
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install ultralytics

WORKDIR /ros2_ws
COPY src/oak_yolo_target /ros2_ws/src/oak_yolo_target

RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    colcon build --symlink-install --packages-select oak_yolo_target

COPY startup.sh /startup.sh
RUN chmod +x /startup.sh

ENTRYPOINT ["/startup.sh"]
DOCKERFILE_EOF

cat > "${CTX_DIR}/startup.sh" <<'STARTUP_EOF'
#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/${ROS_DISTRO:-humble}/setup.bash
source /ros2_ws/install/setup.bash

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export RCUTILS_LOGGING_BUFFERED_STREAM="${RCUTILS_LOGGING_BUFFERED_STREAM:-1}"
export PYTHONUNBUFFERED=1

LAUNCH_OAK_DRIVER="${LAUNCH_OAK_DRIVER:-true}"
OAK_CAMERA_MODEL="${OAK_CAMERA_MODEL:-OAK-D-PRO-W}"
YOLO_MODEL_PATH="${YOLO_MODEL_PATH:-yolo11n.pt}"
YOLO_DEVICE="${YOLO_DEVICE:-auto}"
YOLO_HALF="${YOLO_HALF:-true}"
PUBLISH_DEBUG_IMAGE="${PUBLISH_DEBUG_IMAGE:-true}"
PUBLISH_DEBUG_RAW_IMAGE="${PUBLISH_DEBUG_RAW_IMAGE:-true}"
DEBUG_IMAGE_RATE_HZ="${DEBUG_IMAGE_RATE_HZ:-5.0}"

CMD=(
  ros2 launch oak_yolo_target oak_yolo_target.launch.py
  launch_oak_driver:=${LAUNCH_OAK_DRIVER}
  oak_camera_model:=${OAK_CAMERA_MODEL}
  model_path:=${YOLO_MODEL_PATH}
  device:=${YOLO_DEVICE}
  half:=${YOLO_HALF}
  publish_debug_image:=${PUBLISH_DEBUG_IMAGE}
  publish_debug_raw_image:=${PUBLISH_DEBUG_RAW_IMAGE}
  debug_image_rate_hz:=${DEBUG_IMAGE_RATE_HZ}
)

# Optional additional launch args. Example:
#   -e EXTRA_ROS_ARGS="conf_threshold:=0.4 imgsz:=416 publish_debug_raw_image:=false"
if [[ -n "${EXTRA_ROS_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARRAY=(${EXTRA_ROS_ARGS})
  CMD+=("${EXTRA_ARRAY[@]}")
fi

echo "[startup.sh] ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "[startup.sh] RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}"
echo "[startup.sh] command: ${CMD[*]}"
exec "${CMD[@]}"
STARTUP_EOF
chmod +x "${CTX_DIR}/startup.sh"

cat > "${CTX_DIR}/run_container.sh" <<'RUNNER_EOF'
#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-oak-yolo-target:humble}"
CONTAINER_NAME="${CONTAINER_NAME:-oak_yolo_target}"
ROS_DOMAIN_ID_VALUE="${ROS_DOMAIN_ID:-0}"

# Stop/remove old container with the same name, if it exists.
sudo docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

sudo docker run -d \
  --name "${CONTAINER_NAME}" \
  --restart unless-stopped \
  --network host \
  --ipc host \
  --privileged \
  -v /dev/bus/usb:/dev/bus/usb \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID_VALUE}" \
  -e RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}" \
  -e LAUNCH_OAK_DRIVER="${LAUNCH_OAK_DRIVER:-true}" \
  -e OAK_CAMERA_MODEL="${OAK_CAMERA_MODEL:-OAK-D-PRO-W}" \
  -e YOLO_MODEL_PATH="${YOLO_MODEL_PATH:-yolo11n.pt}" \
  -e YOLO_DEVICE="${YOLO_DEVICE:-auto}" \
  -e YOLO_HALF="${YOLO_HALF:-true}" \
  -e PUBLISH_DEBUG_IMAGE="${PUBLISH_DEBUG_IMAGE:-true}" \
  -e PUBLISH_DEBUG_RAW_IMAGE="${PUBLISH_DEBUG_RAW_IMAGE:-true}" \
  -e DEBUG_IMAGE_RATE_HZ="${DEBUG_IMAGE_RATE_HZ:-5.0}" \
  -e EXTRA_ROS_ARGS="${EXTRA_ROS_ARGS:-}" \
  "${IMAGE_NAME}"

echo "[INFO] Container started: ${CONTAINER_NAME}"
echo "[INFO] Logs: sudo docker logs -f ${CONTAINER_NAME}"
RUNNER_EOF
chmod +x "${CTX_DIR}/run_container.sh"

cat > "${CTX_DIR}/install_systemd_service.sh" <<'SYSTEMD_EOF'
#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="oak-yolo-docker.service"
CONTAINER_NAME="${CONTAINER_NAME:-oak_yolo_target}"

cat <<UNIT_EOF | sudo tee /etc/systemd/system/${SERVICE_NAME} >/dev/null
[Unit]
Description=Start OAK YOLO Docker container
Requires=docker.service
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/bin/docker start ${CONTAINER_NAME}
ExecStop=/usr/bin/docker stop ${CONTAINER_NAME}
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
UNIT_EOF

sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"
echo "[INFO] Installed and enabled ${SERVICE_NAME}"
echo "[INFO] Test manually with: sudo systemctl start ${SERVICE_NAME}"
SYSTEMD_EOF
chmod +x "${CTX_DIR}/install_systemd_service.sh"

cat > "${CTX_DIR}/README.md" <<README_EOF
# OAK YOLO Docker Context

Generated for \`oak_yolo_target\`.

## Build

\`\`\`bash
cd ${CTX_DIR}
sudo docker build -t ${IMAGE_NAME_DEFAULT} .
\`\`\`

Optional base image override:

\`\`\`bash
sudo docker build \\
  --build-arg BASE_IMAGE=ros:humble-ros-base-jammy \\
  -t ${IMAGE_NAME_DEFAULT} .
\`\`\`

## Run once and keep auto-restart enabled

\`\`\`bash
cd ${CTX_DIR}
bash run_container.sh
\`\`\`

## Install systemd wrapper

Run after the container has been created once by \`run_container.sh\`:

\`\`\`bash
cd ${CTX_DIR}
bash install_systemd_service.sh
\`\`\`

Note: \`--restart unless-stopped\` already makes Docker restart the container on boot. The systemd wrapper is optional.
README_EOF

echo "[DONE] Docker context generated at: ${CTX_DIR}"
echo
echo "Next commands:"
echo "  cd ${CTX_DIR}"
echo "  sudo docker build -t ${IMAGE_NAME_DEFAULT} ."
echo "  bash run_container.sh"
echo "  bash install_systemd_service.sh   # optional, after run_container.sh"
