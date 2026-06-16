#!/usr/bin/env bash
set -e

source /opt/ros/${ROS_DISTRO:-humble}/setup.bash

if [ -f /ros2_ws/install/setup.bash ]; then
  source /ros2_ws/install/setup.bash
fi

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

mkdir -p /models
cd /models

echo "[startup] ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}"
echo "[startup] MODEL_PATH=${MODEL_PATH:-yolo11n.pt}"
echo "[startup] YOLO_DEVICE=${YOLO_DEVICE:-auto}"

exec ros2 launch oak_yolo_target oak_yolo_target.launch.py \
  launch_oak_driver:=${LAUNCH_OAK_DRIVER:-true} \
  oak_camera_model:=${OAK_CAMERA_MODEL:-OAK-D-PRO-W} \
  oak_parent_frame:=${OAK_PARENT_FRAME:-base_link} \
  rectify_rgb:=${RECTIFY_RGB:-false} \
  model_path:=${MODEL_PATH:-yolo11n.pt} \
  device:=${YOLO_DEVICE:-auto} \
  half:=${YOLO_HALF:-true} \
  imgsz:=${YOLO_IMGSZ:-640} \
  conf_threshold:=${YOLO_CONF:-0.35} \
  iou_threshold:=${YOLO_IOU:-0.50} \
  use_yolo_tracking:=${USE_YOLO_TRACKING:-false} \
  publish_debug_image:=${PUBLISH_DEBUG_IMAGE:-true} \
  publish_debug_raw_image:=${PUBLISH_DEBUG_RAW_IMAGE:-true} \
  publish_target_camera_info:=${PUBLISH_TARGET_CAMERA_INFO:-true}
