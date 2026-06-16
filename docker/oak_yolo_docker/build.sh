#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

IMAGE_NAME="${IMAGE_NAME:-oak-yolo-target:humble}"
BASE_IMAGE="${BASE_IMAGE:-dustynv/l4t-pytorch:r36.4.0}"

echo "[build] IMAGE_NAME=${IMAGE_NAME}"
echo "[build] BASE_IMAGE=${BASE_IMAGE}"

sudo docker build \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  -t "${IMAGE_NAME}" \
  .
