#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

CONTAINER_NAME="${CONTAINER_NAME:-oak_yolo_target}"
IMAGE_NAME="${IMAGE_NAME:-oak-yolo-target:humble}"
MODELS_DIR="${MODELS_DIR:-$(pwd)/models}"

mkdir -p "${MODELS_DIR}"

if sudo docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  sudo docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

sudo docker run -d \
  --name "${CONTAINER_NAME}" \
  --restart unless-stopped \
  --runtime nvidia \
  --network host \
  --ipc host \
  --privileged \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
  -e RMW_IMPLEMENTATION="rmw_fastrtps_cpp" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e LAUNCH_OAK_DRIVER="${LAUNCH_OAK_DRIVER:-true}" \
  -e OAK_CAMERA_MODEL="${OAK_CAMERA_MODEL:-OAK-D-PRO-W}" \
  -e OAK_PARENT_FRAME="${OAK_PARENT_FRAME:-base_link}" \
  -e RECTIFY_RGB="${RECTIFY_RGB:-false}" \
  -e MODEL_PATH="${MODEL_PATH:-yolo11n.pt}" \
  -e YOLO_DEVICE="${YOLO_DEVICE:-auto}" \
  -e YOLO_HALF="${YOLO_HALF:-true}" \
  -e YOLO_IMGSZ="${YOLO_IMGSZ:-640}" \
  -e YOLO_CONF="${YOLO_CONF:-0.35}" \
  -e YOLO_IOU="${YOLO_IOU:-0.50}" \
  -e USE_YOLO_TRACKING="${USE_YOLO_TRACKING:-false}" \
  -e PUBLISH_DEBUG_IMAGE="${PUBLISH_DEBUG_IMAGE:-true}" \
  -e PUBLISH_DEBUG_RAW_IMAGE="${PUBLISH_DEBUG_RAW_IMAGE:-true}" \
  -e PUBLISH_TARGET_CAMERA_INFO="${PUBLISH_TARGET_CAMERA_INFO:-true}" \
  -v /dev/bus/usb:/dev/bus/usb \
  -v "${MODELS_DIR}:/models" \
  "${IMAGE_NAME}"

echo "[run] Started: ${CONTAINER_NAME}"
echo "[run] Logs: sudo docker logs -f ${CONTAINER_NAME}"
