#!/usr/bin/env bash
set -euo pipefail

# Install and build the Go2 LiDAR planner package on the Go2/control device.
# Expected use:
#   bash scripts/install_go2_lidar_device.sh
# or, if this file was copied into go2_lidar_planner/:
#   bash install_go2_lidar_device.sh

log() { echo -e "\033[1;32m[GO2-LIDAR INSTALL]\033[0m $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $*"; }
err() { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

SUDO=""
if [[ "$(id -u)" -ne 0 ]]; then
  SUDO="sudo"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve package/build root. This supports both:
#   repo_root/scripts/install_go2_lidar_device.sh
#   go2_lidar_planner/install_go2_lidar_device.sh
if [[ -f "$SCRIPT_DIR/package.xml" ]] && grep -q "<name>go2_lidar_planner</name>" "$SCRIPT_DIR/package.xml"; then
  PACKAGE_DIR="$SCRIPT_DIR"
  BUILD_ROOT="$SCRIPT_DIR"
elif [[ -d "$SCRIPT_DIR/../go2_lidar_planner" ]]; then
  PACKAGE_DIR="$(cd "$SCRIPT_DIR/../go2_lidar_planner" && pwd)"
  BUILD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
  err "Cannot find go2_lidar_planner package. Run this from the extracted project root or from inside go2_lidar_planner/."
  exit 1
fi

# Go2 images are often ROS 2 Foxy / Python 3.8, but this script also supports Humble.
if [[ -n "${ROS_DISTRO:-}" ]]; then
  ROS_DISTRO_NAME="$ROS_DISTRO"
elif [[ -d /opt/ros/foxy ]]; then
  ROS_DISTRO_NAME="foxy"
elif [[ -d /opt/ros/humble ]]; then
  ROS_DISTRO_NAME="humble"
else
  err "No ROS 2 install found at /opt/ros/foxy or /opt/ros/humble. Install/source ROS 2 first, or set ROS_DISTRO manually."
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
REQUIRE_UNITREE_API="${REQUIRE_UNITREE_API:-0}"

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
  fi
}

if [[ "$SKIP_APT" != "1" ]]; then
  log "Running apt update"
  $SUDO apt-get update

  apt_install_required \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-numpy \
    ros-${ROS_DISTRO_NAME}-rclpy \
    ros-${ROS_DISTRO_NAME}-geometry-msgs \
    ros-${ROS_DISTRO_NAME}-nav-msgs \
    ros-${ROS_DISTRO_NAME}-sensor-msgs \
    ros-${ROS_DISTRO_NAME}-sensor-msgs-py \
    ros-${ROS_DISTRO_NAME}-std-msgs \
    ros-${ROS_DISTRO_NAME}-tf2-ros

  apt_install_optional ros-${ROS_DISTRO_NAME}-tf-transformations
fi

# Source Unitree ROS2 workspace if available. This is required only for real /api/sport/request control.
source_if_exists() {
  local f="$1"
  if [[ -n "$f" && -f "$f" ]]; then
    log "Sourcing Unitree/extra setup: $f"
    # shellcheck disable=SC1090
    source "$f"
    return 0
  fi
  return 1
}

UNITREE_SOURCED=0
if [[ -n "${UNITREE_ROS2_SETUP:-}" ]]; then
  source_if_exists "$UNITREE_ROS2_SETUP" && UNITREE_SOURCED=1 || warn "UNITREE_ROS2_SETUP was set but not found: $UNITREE_ROS2_SETUP"
fi

if [[ "$UNITREE_SOURCED" == "0" ]]; then
  CANDIDATES=(
    "$HOME/unitree_ros2/setup.sh"
    "$HOME/unitree_ros2/install/setup.bash"
    "$HOME/unitree_go2_ros2/install/setup.bash"
    "$HOME/ros2_ws/install/setup.bash"
    "/unitree/module/Others/unitree_ros2/setup.sh"
    "/opt/unitree_ros2/setup.sh"
    "/opt/unitree_ros2/install/setup.bash"
  )
  for f in "${CANDIDATES[@]}"; do
    if source_if_exists "$f"; then
      UNITREE_SOURCED=1
      break
    fi
  done
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
  log "Installing minimal Python dependencies"
  python3 -m pip install --user --upgrade pip setuptools wheel
  python3 -m pip install --user --upgrade numpy
fi

log "Dependency sanity check"
UNITREE_OK=0
python3 - <<'PY' && UNITREE_OK=1 || UNITREE_OK=0
import sys
print("python:", sys.version)
import rclpy; print("rclpy: OK")
import numpy as np; print("numpy: OK", np.__version__)
from geometry_msgs.msg import PointStamped; print("geometry_msgs: OK")
from nav_msgs.msg import Path; print("nav_msgs: OK")
from sensor_msgs.msg import PointCloud2; print("sensor_msgs: OK")
try:
    from sensor_msgs_py import point_cloud2
    print("sensor_msgs_py: OK")
except Exception as e:
    print("sensor_msgs_py: FAIL", e)
    raise
try:
    import tf_transformations
    print("tf_transformations: OK")
except Exception as e:
    print("tf_transformations: FAIL", e)
    raise
try:
    from unitree_api.msg import Request
    print("unitree_api: OK")
except Exception as e:
    print("unitree_api: MISSING", e)
    raise
PY

if [[ "$UNITREE_OK" != "1" ]]; then
  warn "unitree_api is not currently importable in this shell."
  warn "The planner can still run with api_control_enabled:=false, but real Go2 sport control needs unitree_api."
  warn "If you know the Unitree setup file, rerun with: UNITREE_ROS2_SETUP=/path/to/setup.bash bash $0"
  if [[ "$REQUIRE_UNITREE_API" == "1" ]]; then
    err "REQUIRE_UNITREE_API=1 and unitree_api is missing. Aborting."
    exit 1
  fi
fi

if [[ "$BUILD_AFTER_INSTALL" == "1" ]]; then
  log "Building go2_lidar_planner"
  cd "$BUILD_ROOT"
  colcon build --symlink-install --packages-select go2_lidar_planner
fi

log "Done. Source the workspace before launch:"
echo "  source ${ROS_SETUP}"
if [[ "$UNITREE_SOURCED" == "1" ]]; then
  echo "  # source the same Unitree setup file used above if needed"
fi
echo "  source ${BUILD_ROOT}/install/setup.bash"
echo
log "Launch command on the Go2/LiDAR device:"
echo "  ros2 launch go2_lidar_planner go2_lidar_planner.launch.py"
echo
log "Planner-only/debug launch when unitree_api is unavailable:"
echo "  ros2 launch go2_lidar_planner go2_lidar_planner.launch.py api_control_enabled:=false"
