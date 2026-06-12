#!/usr/bin/env bash
set -euo pipefail

# Install and build the OAK/YOLO target package on the camera/Jetson device.
# Expected use:
#   bash scripts/install_oak_yolo_device.sh
# or, if this file was copied into oak_yolo_target/:
#   bash install_oak_yolo_device.sh

log() { echo -e "\033[1;32m[OAK-YOLO INSTALL]\033[0m $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $*"; }
err() { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

SUDO=""
if [[ "$(id -u)" -ne 0 ]]; then
  SUDO="sudo"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve package/build root. This supports both:
#   repo_root/scripts/install_oak_yolo_device.sh
#   oak_yolo_target/install_oak_yolo_device.sh
if [[ -f "$SCRIPT_DIR/package.xml" ]] && grep -q "<name>oak_yolo_target</name>" "$SCRIPT_DIR/package.xml"; then
  PACKAGE_DIR="$SCRIPT_DIR"
  BUILD_ROOT="$SCRIPT_DIR"
elif [[ -d "$SCRIPT_DIR/../oak_yolo_target" ]]; then
  PACKAGE_DIR="$(cd "$SCRIPT_DIR/../oak_yolo_target" && pwd)"
  BUILD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
  err "Cannot find oak_yolo_target package. Run this from the extracted project root or from inside oak_yolo_target/."
  exit 1
fi

if [[ -n "${ROS_DISTRO:-}" ]]; then
  ROS_DISTRO_NAME="$ROS_DISTRO"
elif [[ -d /opt/ros/humble ]]; then
  ROS_DISTRO_NAME="humble"
else
  err "ROS 2 Humble was not found at /opt/ros/humble. Install/source ROS 2 first, or set ROS_DISTRO manually."
  exit 1
fi

ROS_SETUP="/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
if [[ ! -f "$ROS_SETUP" ]]; then
  err "ROS setup file not found: $ROS_SETUP"
  exit 1
fi

source "$ROS_SETUP"
log "Using ROS_DISTRO=${ROS_DISTRO_NAME}"
log "Package dir: $PACKAGE_DIR"
log "Build root : $BUILD_ROOT"

SKIP_APT="${SKIP_APT:-0}"
SKIP_PIP="${SKIP_PIP:-0}"
SKIP_ROSDEP="${SKIP_ROSDEP:-0}"
BUILD_AFTER_INSTALL="${BUILD_AFTER_INSTALL:-1}"
INSTALL_DEPTHAI_APT="${INSTALL_DEPTHAI_APT:-1}"

apt_install_required() {
  if [[ "$SKIP_APT" == "1" ]]; then return 0; fi
  log "Installing required apt packages: $*"
  $SUDO apt-get install -y "$@"
}

apt_install_optional() {
  if [[ "$SKIP_APT" == "1" ]]; then return 0; fi
  log "Trying optional apt packages: $*"
  if ! $SUDO apt-get install -y "$@"; then
    warn "Some optional packages failed to install: $*"
    warn "If depthai_ros_driver is installed from source in another workspace, source that workspace before launching."
  fi
}

if [[ "$SKIP_APT" != "1" ]]; then
  log "Running apt update"
  $SUDO apt-get update

  apt_install_required \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    python3-opencv \
    python3-numpy \
    ros-${ROS_DISTRO_NAME}-rclpy \
    ros-${ROS_DISTRO_NAME}-sensor-msgs \
    ros-${ROS_DISTRO_NAME}-geometry-msgs \
    ros-${ROS_DISTRO_NAME}-std-msgs \
    ros-${ROS_DISTRO_NAME}-cv-bridge \
    ros-${ROS_DISTRO_NAME}-image-transport \
    ros-${ROS_DISTRO_NAME}-compressed-image-transport

  if [[ "$INSTALL_DEPTHAI_APT" == "1" ]]; then
    apt_install_optional \
      ros-${ROS_DISTRO_NAME}-depthai-ros-driver \
      ros-${ROS_DISTRO_NAME}-depthai-ros-msgs
  fi
fi

if [[ "$SKIP_ROSDEP" != "1" ]]; then
  if command -v rosdep >/dev/null 2>&1; then
    log "Running rosdep update/install"
    $SUDO rosdep init 2>/dev/null || true
    rosdep update || warn "rosdep update failed; continuing because apt packages were already requested."
    rosdep install --from-paths "$PACKAGE_DIR" --ignore-src -r -y || warn "rosdep install had unresolved entries; continuing."
  else
    warn "rosdep command not found; skipping rosdep."
  fi
fi

if [[ "$SKIP_PIP" != "1" ]]; then
  log "Installing Python YOLO dependencies"
  python3 -m pip install --user --upgrade pip setuptools wheel
  # Do not explicitly install torch here. On Jetson, torch must match JetPack/CUDA.
  # If torch is already installed, pip should keep it when it satisfies Ultralytics requirements.
  python3 -m pip install --user --upgrade ultralytics opencv-python numpy
fi

log "Dependency sanity check"
python3 - <<'PY' || true
import sys
print("python:", sys.version)
try:
    import rclpy; print("rclpy: OK")
except Exception as e: print("rclpy: FAIL", e)
try:
    import cv2; print("cv2: OK", cv2.__version__)
except Exception as e: print("cv2: FAIL", e)
try:
    import numpy as np; print("numpy: OK", np.__version__)
except Exception as e: print("numpy: FAIL", e)
try:
    from ultralytics import YOLO; print("ultralytics: OK")
except Exception as e: print("ultralytics: FAIL", e)
try:
    import torch
    print("torch:", torch.__version__, "cuda_available:", torch.cuda.is_available(), "device_count:", torch.cuda.device_count())
except Exception as e: print("torch: FAIL", e)
PY

if [[ "$BUILD_AFTER_INSTALL" == "1" ]]; then
  log "Building oak_yolo_target"
  cd "$BUILD_ROOT"
  colcon build --symlink-install --packages-select oak_yolo_target
fi

log "Done. Source the workspace before launch:"
echo "  source ${ROS_SETUP}"
echo "  source ${BUILD_ROOT}/install/setup.bash"
echo
log "Launch command on the OAK/YOLO device:"
echo "  ros2 launch oak_yolo_target oak_yolo_target.launch.py"
echo
log "CPU fallback launch, useful if torch CUDA is unavailable:"
echo "  ros2 launch oak_yolo_target oak_yolo_target.launch.py device:=cpu half:=false"
