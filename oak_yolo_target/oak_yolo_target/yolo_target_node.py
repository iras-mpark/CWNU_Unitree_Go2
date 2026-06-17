#!/usr/bin/env python3
"""OAK RGB-D + Ultralytics YOLO person detector to robot-frame 3D target.

This node intentionally uses only standard ROS 2 message types.  It receives an
RGB image, an aligned depth image, and depth camera intrinsics.  Ultralytics YOLO
detects people; the nearest valid person is converted into a PointStamped in the
robot base frame.  Tracking IDs are intentionally not used for target selection.
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import String

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - handled at runtime on robot
    YOLO = None

try:
    import torch
except Exception:  # pragma: no cover - CPU-only or TensorRT-only environments
    torch = None


@dataclass
class PersonCandidate:
    bbox_xyxy: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int]
    depth_m: float
    point_xyz: Tuple[float, float, float]
    center_uv_rgb: Tuple[float, float]
    center_uv_depth: Tuple[float, float]
    # Raw sensor measurement before Kalman filtering.  These are kept for
    # diagnostics after point_xyz/depth_m are replaced by the filtered output.
    sensor_point_xyz: Optional[Tuple[float, float, float]] = None
    sensor_depth_m: Optional[float] = None
    association_distance_m: Optional[float] = None


class PositionKalmanFilter3D:
    """Constant-position 3D Kalman filter with random-walk process noise.

    State is [x, y, z] in base_link.  The target is assumed to move slowly;
    responsiveness is controlled by process noise rather than an explicit
    velocity model.
    """

    def __init__(self) -> None:
        self.x: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.last_predict_time: Optional[float] = None
        self.last_update_time: Optional[float] = None

    @property
    def initialized(self) -> bool:
        return self.x is not None and self.P is not None

    def reset(self) -> None:
        self.x = None
        self.P = None
        self.last_predict_time = None
        self.last_update_time = None

    def initialize(self, measurement: Sequence[float], variance: float, now: float) -> np.ndarray:
        self.x = np.asarray(measurement, dtype=np.float64).reshape(3)
        self.P = np.eye(3, dtype=np.float64) * max(float(variance), 1e-6)
        self.last_predict_time = now
        self.last_update_time = now
        return self.x.copy()

    def predict(self, process_std: float, now: float) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError("Kalman filter is not initialized")
        assert self.x is not None and self.P is not None
        if self.last_predict_time is None:
            dt = 1.0 / 30.0
        else:
            dt = max(1e-3, min(1.0, float(now - self.last_predict_time)))
        q = max(float(process_std), 0.0)
        self.P = self.P + np.eye(3, dtype=np.float64) * (q * q * dt)
        self.last_predict_time = now
        return self.x.copy()

    def update(self, measurement: Sequence[float], measurement_std_xyz: Sequence[float], now: float) -> np.ndarray:
        if not self.initialized:
            init_var = max(float(measurement_std_xyz[0]) ** 2, 1e-6)
            return self.initialize(measurement, init_var, now)
        assert self.x is not None and self.P is not None
        z = np.asarray(measurement, dtype=np.float64).reshape(3)
        std = np.asarray(measurement_std_xyz, dtype=np.float64).reshape(3)
        std = np.maximum(std, 1e-4)
        R = np.diag(std * std)

        # H = I for direct position measurement.
        S = self.P + R
        K = self.P @ np.linalg.inv(S)
        innovation = z - self.x
        self.x = self.x + K @ innovation
        I = np.eye(3, dtype=np.float64)
        # Joseph form keeps covariance symmetric/positive under numerical noise.
        self.P = (I - K) @ self.P @ (I - K).T + K @ R @ K.T
        self.last_update_time = now
        return self.x.copy()

    def covariance_diag(self) -> Tuple[float, float, float]:
        if self.P is None:
            return (math.nan, math.nan, math.nan)
        d = np.diag(self.P)
        return (float(d[0]), float(d[1]), float(d[2]))


class OakYoloTargetNode(Node):
    """Detect people and publish the nearest valid 3D local goal."""

    def __init__(self) -> None:
        super().__init__("oak_yolo_target_node")

        # Input/output topics.
        self.declare_parameter("rgb_topic", "/oak/rgb/image_raw")
        self.declare_parameter("depth_topic", "/oak/stereo/image_raw")
        self.declare_parameter("camera_info_topic", "/oak/stereo/camera_info")
        self.declare_parameter("target_point_topic", "/local_goal_point")
        self.declare_parameter("target_status_topic", "/target/status")
        self.declare_parameter("debug_image_topic", "/target/debug_image/compressed")
        self.declare_parameter("debug_raw_image_topic", "/target/debug_image")
        self.declare_parameter("target_camera_info_topic", "/target/camera_info")
        self.declare_parameter("target_frame", "base_link")

        # Model / inference parameters.  For Jetson, pass a TensorRT .engine path
        # here after exporting the Ultralytics model.
        self.declare_parameter("model_path", "yolo11n.pt")
        self.declare_parameter("fallback_model_path", "")
        self.declare_parameter("tracker", "bytetrack.yaml")
        self.declare_parameter("use_yolo_tracking", False)
        self.declare_parameter("device", "auto")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf_threshold", 0.35)
        self.declare_parameter("iou_threshold", 0.5)
        self.declare_parameter("person_class_id", 0)
        self.declare_parameter("half", True)
        self.declare_parameter("show_yolo_logs", False)

        # Target selection policy.
        # Tracking IDs are deliberately ignored by default because unstable IDs can
        # cause the robot to lose the person.  Each frame selects the nearest valid
        # person in front of the robot.
        self.declare_parameter("auto_acquire", True)
        self.declare_parameter("acquire_stable_frames", 1)
        self.declare_parameter("acquire_max_distance_m", 6.0)
        self.declare_parameter("acquire_center_gate_norm", 1.00)
        self.declare_parameter("lost_timeout_s", 1.0)
        self.declare_parameter("reacquire_distance_gate_m", 1.0)
        self.declare_parameter("nearest_person_use_center_gate", False)

        # Kalman filtering / association.  The filter output is the final
        # /local_goal_point.  When initialized, target association chooses the
        # person measurement closest to the predicted KF state instead of simply
        # the nearest person in depth.
        self.declare_parameter("use_kalman_filter", True)
        self.declare_parameter("kalman_process_noise_std_m", 0.18)
        self.declare_parameter("kalman_measurement_noise_x_m", 0.35)
        self.declare_parameter("kalman_measurement_noise_y_m", 0.18)
        self.declare_parameter("kalman_measurement_noise_z_m", 0.25)
        self.declare_parameter("kalman_initial_variance_m2", 0.25)
        self.declare_parameter("kalman_association_gate_m", 1.20)
        self.declare_parameter("kalman_reset_timeout_s", 1.50)
        self.declare_parameter("kalman_publish_prediction_without_measurement", False)

        # Depth handling.
        self.declare_parameter("depth_min_m", 0.25)
        self.declare_parameter("depth_max_m", 10.0)
        self.declare_parameter("depth_quantile_low", 0.25)
        self.declare_parameter("depth_quantile_high", 0.50)
        self.declare_parameter("depth_roi_scale", 0.60)
        self.declare_parameter("depth_roi_max_pixels", 180)
        self.declare_parameter("depth_sample_stride", 2)
        self.declare_parameter("min_valid_depth_samples", 20)

        # Camera mounting correction after optical-to-base conversion.
        self.declare_parameter("camera_x_offset_m", 0.0)
        self.declare_parameter("camera_y_offset_m", 0.0)
        self.declare_parameter("camera_z_offset_m", 0.0)

        # Debug image publication.
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("publish_debug_raw_image", True)
        self.declare_parameter("publish_target_camera_info", True)
        self.declare_parameter("debug_image_rate_hz", 5.0)
        self.declare_parameter("debug_jpeg_quality", 35)
        self.declare_parameter("debug_resize_width", 640)

        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.target_point_topic = self.get_parameter("target_point_topic").value
        self.target_status_topic = self.get_parameter("target_status_topic").value
        self.debug_image_topic = self.get_parameter("debug_image_topic").value
        self.debug_raw_image_topic = self.get_parameter("debug_raw_image_topic").value
        self.target_camera_info_topic = self.get_parameter("target_camera_info_topic").value
        self.target_frame = self.get_parameter("target_frame").value

        self.bridge = CvBridge()
        self.model = self._load_model()
        self.inference_device = self._resolve_inference_device(str(self.get_parameter("device").value))
        self.inference_half = self._resolve_half_precision(bool(self.get_parameter("half").value))
        self._last_yolo_error: str = ""
        self.latest_depth: Optional[np.ndarray] = None
        self.latest_depth_encoding: str = ""
        self.latest_camera_info: Optional[CameraInfo] = None

        # Kept only for status/debug backward compatibility.  Target selection no
        # longer locks on YOLO track_id; it chooses the nearest valid person every
        # frame.
        self.locked_track_id: Optional[int] = None
        self.locked_last_point: Optional[Tuple[float, float, float]] = None
        self.locked_last_seen_time: Optional[float] = None
        self._candidate_key: Optional[str] = None
        self._candidate_count: int = 0
        self._last_debug_pub_time: float = 0.0
        self._last_warn_time: float = 0.0
        self.kalman_filter = PositionKalmanFilter3D()
        self._last_kalman_debug: Dict[str, Any] = {"enabled": bool(self.get_parameter("use_kalman_filter").value)}

        self.create_subscription(Image, self.depth_topic, self._depth_callback, 5)
        self.create_subscription(CameraInfo, self.camera_info_topic, self._camera_info_callback, 5)
        self.create_subscription(Image, self.rgb_topic, self._rgb_callback, 2)

        self.target_pub = self.create_publisher(PointStamped, self.target_point_topic, 10)
        self.status_pub = self.create_publisher(String, self.target_status_topic, 10)
        self.debug_pub = self.create_publisher(CompressedImage, self.debug_image_topic, 3)
        self.debug_raw_pub = self.create_publisher(Image, self.debug_raw_image_topic, 3)
        self.target_camera_info_pub = self.create_publisher(CameraInfo, self.target_camera_info_topic, 3)

        self.get_logger().info(
            f"OAK YOLO target node ready: rgb={self.rgb_topic}, depth={self.depth_topic}, "
            f"camera_info={self.camera_info_topic}, point={self.target_point_topic}, "
            f"target_camera_info={self.target_camera_info_topic}, "
            f"selection=nearest_person_no_track_id"
        )

    # ------------------------------------------------------------------ setup
    def _load_model(self) -> Any:
        if YOLO is None:
            raise RuntimeError(
                "Could not import ultralytics. Install it on the Jetson environment "
                "or provide a Python environment containing the Ultralytics package."
            )

        model_path = str(self.get_parameter("model_path").value)
        fallback_model_path = str(self.get_parameter("fallback_model_path").value)

        try:
            model = YOLO(model_path)
            self.get_logger().info(f"Loaded YOLO model: {model_path}")
            return model
        except Exception as exc:
            if fallback_model_path and fallback_model_path != model_path:
                self.get_logger().warn(
                    f"Failed to load '{model_path}' ({exc}). Trying fallback '{fallback_model_path}'."
                )
                model = YOLO(fallback_model_path)
                self.get_logger().info(f"Loaded fallback YOLO model: {fallback_model_path}")
                return model
            raise

    def _resolve_inference_device(self, requested: str) -> str:
        """Return a safe Ultralytics device string.

        ROS launch defaults to 'auto'.  On Jetson it is common for a CUDA device
        to be physically visible while the active Python/PyTorch installation is
        not CUDA-enabled.  In that case Ultralytics rejects device='0', so we must
        fall back to CPU instead of failing every frame.
        """
        req = (requested or "auto").strip().lower()
        if req == "none":
            req = "auto"
        if req == "cpu":
            self.get_logger().warn("YOLO inference device forced to CPU.")
            return "cpu"

        cuda_ok = bool(torch is not None and torch.cuda.is_available())
        cuda_count = int(torch.cuda.device_count()) if torch is not None else 0

        if req in ("auto", "cuda", "gpu"):
            if cuda_ok and cuda_count > 0:
                self.get_logger().info(f"YOLO inference device auto-selected: 0 ({cuda_count} CUDA device(s) visible)")
                return "0"
            self.get_logger().warn(
                "CUDA was requested/auto-selected, but torch.cuda.is_available() is False. "
                "Falling back to device='cpu'. YOLO will run slower until the Jetson PyTorch/CUDA "
                "environment is fixed or a TensorRT engine is used."
            )
            return "cpu"

        # Numeric strings such as '0' are valid only if PyTorch can actually use CUDA.
        if req.replace(",", "").replace(" ", "").isdigit():
            if cuda_ok:
                self.get_logger().info(f"YOLO inference device requested: {requested}")
                return requested
            self.get_logger().warn(
                f"YOLO device='{requested}' was requested, but torch.cuda.is_available() is False. "
                "Using CPU fallback to keep detection running."
            )
            return "cpu"

        # Let Ultralytics handle uncommon backend-specific values, but log them.
        self.get_logger().warn(f"Using non-standard YOLO device argument: {requested}")
        return requested

    def _resolve_half_precision(self, requested_half: bool) -> bool:
        if self.inference_device == "cpu" and requested_half:
            self.get_logger().warn("Disabling half=True because YOLO is running on CPU.")
            return False
        return bool(requested_half)

    # ---------------------------------------------------------------- callbacks
    def _depth_callback(self, msg: Image) -> None:
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.latest_depth_encoding = msg.encoding or ""
        except Exception as exc:
            self._warn_throttled(f"Failed to convert depth image: {exc}")

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        self.latest_camera_info = msg

        # Keep /target/camera_info alive even when no debug image is currently being
        # published.  RViz Image display does not need CameraInfo, but Camera display
        # and some monitoring tools expect this topic to exist.  The debug image
        # publisher below republishes a resized/scaled version with the debug image
        # timestamp whenever a debug frame is emitted.
        if bool(self.get_parameter("publish_target_camera_info").value):
            info = self._make_target_camera_info(
                source=msg,
                stamp=msg.header.stamp,
                out_width=int(msg.width),
                out_height=int(msg.height),
                scale_x=1.0,
                scale_y=1.0,
            )
            self.target_camera_info_pub.publish(info)

    def _rgb_callback(self, msg: Image) -> None:
        if self.latest_depth is None or self.latest_camera_info is None:
            self._publish_status(False, reason="waiting_for_depth_or_camera_info")
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self._warn_throttled(f"Failed to convert RGB image: {exc}")
            self._publish_status(False, reason="rgb_conversion_failed")
            return

        candidates = self._run_yolo_and_build_candidates(frame)
        target = self._select_target(candidates, frame.shape[1])

        if target is not None:
            self._publish_target(target, msg.header.stamp)
            self._publish_status(True, target=target)
        else:
            self._publish_status(False, reason="no_valid_nearest_person")

        if bool(self.get_parameter("publish_debug_image").value):
            self._publish_debug_image(frame, candidates, target, msg.header.stamp)

    # ------------------------------------------------------------- perception
    def _run_yolo_and_build_candidates(self, frame: np.ndarray) -> List[PersonCandidate]:
        person_class_id = int(self.get_parameter("person_class_id").value)
        conf = float(self.get_parameter("conf_threshold").value)
        iou = float(self.get_parameter("iou_threshold").value)
        device = self.inference_device
        imgsz = int(self.get_parameter("imgsz").value)
        tracker = str(self.get_parameter("tracker").value)
        half = self.inference_half
        verbose = bool(self.get_parameter("show_yolo_logs").value)

        results = self._track_with_fallback(
            frame=frame,
            person_class_id=person_class_id,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            half=half,
            tracker=tracker,
            verbose=verbose,
        )
        if results is None:
            return []

        if not results:
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        names = getattr(result, "names", {}) or {}
        candidates: List[PersonCandidate] = []
        boxes = result.boxes

        for box in boxes:
            try:
                xyxy = box.xyxy[0].detach().cpu().numpy().astype(float)
                x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
                confidence = float(box.conf[0].detach().cpu().item()) if box.conf is not None else 0.0
                class_id = int(box.cls[0].detach().cpu().item()) if box.cls is not None else person_class_id
                track_id = None
                if getattr(box, "id", None) is not None:
                    track_id = int(box.id[0].detach().cpu().item())
            except Exception:
                continue

            if class_id != person_class_id or confidence < conf:
                continue

            depth_result = self._estimate_depth_and_point((x1, y1, x2, y2), frame.shape)
            if depth_result is None:
                continue
            depth_m, point_xyz, center_uv_depth = depth_result
            center_uv_rgb = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            class_name = str(names.get(class_id, "person"))

            candidates.append(
                PersonCandidate(
                    bbox_xyxy=(x1, y1, x2, y2),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                    track_id=track_id,
                    depth_m=depth_m,
                    point_xyz=point_xyz,
                    center_uv_rgb=center_uv_rgb,
                    center_uv_depth=center_uv_depth,
                    sensor_point_xyz=point_xyz,
                    sensor_depth_m=depth_m,
                )
            )
        return candidates

    def _track_with_fallback(
        self,
        frame: np.ndarray,
        person_class_id: int,
        conf: float,
        iou: float,
        imgsz: int,
        device: str,
        half: bool,
        tracker: str,
        verbose: bool,
    ) -> Optional[Any]:
        """Run YOLO detection, optionally using track mode, with CPU fallback.

        Target selection does not use tracking IDs.  By default this calls
        YOLO.predict() instead of YOLO.track() so ID switching cannot affect the
        selected person.  The use_yolo_tracking parameter is left available only
        for optional visualization/debug experiments.
        """
        use_tracking = bool(self.get_parameter("use_yolo_tracking").value)
        kwargs = dict(
            source=frame,
            classes=[person_class_id],
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            verbose=verbose,
        )
        if use_tracking:
            kwargs["persist"] = True
            kwargs["tracker"] = tracker
        if half:
            kwargs["half"] = True

        try:
            self._last_yolo_error = ""
            return self._run_yolo_backend(kwargs, use_tracking)
        except TypeError:
            # Some exported/TensorRT backends do not accept every keyword. Retry without half.
            kwargs.pop("half", None)
            try:
                self._last_yolo_error = ""
                return self._run_yolo_backend(kwargs, use_tracking)
            except Exception as exc:
                return self._handle_yolo_exception_and_retry(kwargs, exc, use_tracking)
        except Exception as exc:
            return self._handle_yolo_exception_and_retry(kwargs, exc, use_tracking)

    def _run_yolo_backend(self, kwargs: Dict[str, Any], use_tracking: bool) -> Optional[Any]:
        if use_tracking:
            return self.model.track(**kwargs)
        return self.model.predict(**kwargs)

    def _handle_yolo_exception_and_retry(
        self, kwargs: Dict[str, Any], exc: Exception, use_tracking: bool
    ) -> Optional[Any]:
        text = str(exc)
        self._last_yolo_error = text
        cuda_error = "Invalid CUDA" in text or "CUDA" in text or "cuda" in text
        mode = "tracking" if use_tracking else "detection"
        if kwargs.get("device") != "cpu" and cuda_error:
            self._warn_throttled(
                f"YOLO {mode} failed on device={kwargs.get('device')}: {text}. Retrying on CPU."
            )
            self.inference_device = "cpu"
            self.inference_half = False
            kwargs["device"] = "cpu"
            kwargs.pop("half", None)
            try:
                self._last_yolo_error = ""
                return self._run_yolo_backend(kwargs, use_tracking)
            except Exception as cpu_exc:
                self._last_yolo_error = str(cpu_exc)
                self._warn_throttled(f"YOLO CPU fallback also failed: {cpu_exc}")
                return None

        self._warn_throttled(f"YOLO {mode} failed: {exc}")
        return None

    def _estimate_depth_and_point(
        self, bbox_xyxy: Tuple[int, int, int, int], rgb_shape: Tuple[int, int, int]
    ) -> Optional[Tuple[float, Tuple[float, float, float], Tuple[float, float]]]:
        depth_image = self.latest_depth
        camera_info = self.latest_camera_info
        if depth_image is None or camera_info is None:
            return None

        rgb_h, rgb_w = int(rgb_shape[0]), int(rgb_shape[1])
        depth_h, depth_w = int(depth_image.shape[0]), int(depth_image.shape[1])
        sx = depth_w / max(1.0, float(rgb_w))
        sy = depth_h / max(1.0, float(rgb_h))

        x1, y1, x2, y2 = bbox_xyxy
        x1d = int(np.clip(round(x1 * sx), 0, depth_w - 1))
        x2d = int(np.clip(round(x2 * sx), 0, depth_w - 1))
        y1d = int(np.clip(round(y1 * sy), 0, depth_h - 1))
        y2d = int(np.clip(round(y2 * sy), 0, depth_h - 1))
        if x2d <= x1d or y2d <= y1d:
            return None

        # Use a reasonably large center crop inside the box.  This avoids edges
        # while still retaining enough pixels for robust quantile statistics.
        roi_scale = float(self.get_parameter("depth_roi_scale").value)
        max_pixels = int(self.get_parameter("depth_roi_max_pixels").value)
        stride = max(1, int(self.get_parameter("depth_sample_stride").value))
        cx = (x1d + x2d) / 2.0
        cy = (y1d + y2d) / 2.0
        bw = max(2, x2d - x1d)
        bh = max(2, y2d - y1d)
        crop_w = min(max_pixels, max(2, int(round(bw * roi_scale))))
        crop_h = min(max_pixels, max(2, int(round(bh * roi_scale))))
        xa = int(np.clip(round(cx - crop_w / 2.0), 0, depth_w - 1))
        xb = int(np.clip(round(cx + crop_w / 2.0), xa + 1, depth_w))
        ya = int(np.clip(round(cy - crop_h / 2.0), 0, depth_h - 1))
        yb = int(np.clip(round(cy + crop_h / 2.0), ya + 1, depth_h))

        roi = depth_image[ya:yb:stride, xa:xb:stride]
        values = self._depth_values_to_meters(roi).reshape(-1)
        depth_min = float(self.get_parameter("depth_min_m").value)
        depth_max = float(self.get_parameter("depth_max_m").value)
        valid = values[np.isfinite(values)]
        valid = valid[(valid >= depth_min) & (valid <= depth_max)]

        min_samples = int(self.get_parameter("min_valid_depth_samples").value)
        if valid.size < min_samples:
            return None

        q_low = float(self.get_parameter("depth_quantile_low").value)
        q_high = float(self.get_parameter("depth_quantile_high").value)
        q_low = min(max(q_low, 0.0), 1.0)
        q_high = min(max(q_high, q_low), 1.0)
        lo, hi = np.quantile(valid, [q_low, q_high])
        selected = valid[(valid >= lo) & (valid <= hi)]
        if selected.size < 1:
            selected = valid
        depth_m = float(np.mean(selected))

        # Use depth image intrinsics.  If RGB/depth are aligned but have different
        # image sizes, the bbox center has already been scaled into depth pixels.
        u = float(np.clip(cx, 0, depth_w - 1))
        v = float(np.clip(cy, 0, depth_h - 1))
        fx = float(camera_info.k[0])
        fy = float(camera_info.k[4])
        cx_k = float(camera_info.k[2])
        cy_k = float(camera_info.k[5])
        if abs(fx) < 1e-6 or abs(fy) < 1e-6:
            return None

        # Optical frame: +x right, +y down, +z forward.
        # Robot/base frame assumed here: +x forward, +y left, +z up.
        x_right = (u - cx_k) * depth_m / fx
        y_down = (v - cy_k) * depth_m / fy
        base_x = depth_m + float(self.get_parameter("camera_x_offset_m").value)
        base_y = -x_right + float(self.get_parameter("camera_y_offset_m").value)
        base_z = -y_down + float(self.get_parameter("camera_z_offset_m").value)
        return depth_m, (base_x, base_y, base_z), (u, v)

    def _depth_values_to_meters(self, roi: np.ndarray) -> np.ndarray:
        if roi.dtype == np.uint16 or "16U" in self.latest_depth_encoding.upper():
            return roi.astype(np.float32) / 1000.0
        return roi.astype(np.float32)

    # -------------------------------------------------------------- selection
    def _select_target(
        self, candidates: List[PersonCandidate], image_width: int
    ) -> Optional[PersonCandidate]:
        """Select and filter target.

        Without a Kalman filter, this falls back to nearest valid person.  With
        the KF enabled, the first target initializes the filter; after that,
        candidates are associated by distance to the predicted KF estimate.
        """
        now = time.monotonic()
        self.locked_track_id = None

        if not bool(self.get_parameter("auto_acquire").value):
            self._last_kalman_debug = {"enabled": bool(self.get_parameter("use_kalman_filter").value), "reason": "auto_acquire_false"}
            return None

        max_dist = float(self.get_parameter("acquire_max_distance_m").value)
        center_gate = float(self.get_parameter("acquire_center_gate_norm").value)
        use_center_gate = bool(self.get_parameter("nearest_person_use_center_gate").value)

        valid: List[PersonCandidate] = []
        for cand in candidates:
            if cand.point_xyz[0] <= 0.0:
                continue
            if cand.depth_m > max_dist:
                continue
            if use_center_gate:
                x_norm = 2.0 * (cand.center_uv_rgb[0] / max(1.0, float(image_width)) - 0.5)
                if abs(x_norm) > center_gate:
                    continue
            valid.append(cand)

        use_kf = bool(self.get_parameter("use_kalman_filter").value)
        if not use_kf:
            self._last_kalman_debug = {"enabled": False, "reason": "disabled"}
            return self._select_nearest_valid_without_kf(valid, now)

        if not valid:
            return self._handle_no_kalman_measurement(now)

        process_std = float(self.get_parameter("kalman_process_noise_std_m").value)
        meas_std = (
            float(self.get_parameter("kalman_measurement_noise_x_m").value),
            float(self.get_parameter("kalman_measurement_noise_y_m").value),
            float(self.get_parameter("kalman_measurement_noise_z_m").value),
        )
        init_var = float(self.get_parameter("kalman_initial_variance_m2").value)
        association_gate = float(self.get_parameter("kalman_association_gate_m").value)
        reset_timeout = float(self.get_parameter("kalman_reset_timeout_s").value)

        if not self.kalman_filter.initialized:
            chosen = min(valid, key=lambda c: (c.depth_m, -c.confidence))
            filtered = self.kalman_filter.initialize(chosen.point_xyz, init_var, now)
            return self._finalize_kalman_target(
                chosen=chosen,
                filtered=filtered,
                now=now,
                mode="initialize_nearest",
                association_distance=0.0,
                valid_count=len(valid),
            )

        predicted = self.kalman_filter.predict(process_std=process_std, now=now)
        scored: List[Tuple[float, PersonCandidate]] = []
        for cand in valid:
            z = np.asarray(cand.point_xyz, dtype=np.float64)
            d = float(np.linalg.norm(z - predicted))
            scored.append((d, cand))
        scored.sort(key=lambda item: (item[0], item[1].depth_m, -item[1].confidence))
        association_distance, chosen = scored[0]

        last_update = self.kalman_filter.last_update_time
        age_since_update = math.inf if last_update is None else max(0.0, now - last_update)
        gate_enabled = association_gate > 0.0
        if gate_enabled and association_distance > association_gate:
            if age_since_update >= reset_timeout:
                self.kalman_filter.reset()
                filtered = self.kalman_filter.initialize(chosen.point_xyz, init_var, now)
                return self._finalize_kalman_target(
                    chosen=chosen,
                    filtered=filtered,
                    now=now,
                    mode="reinitialize_after_gate_timeout",
                    association_distance=association_distance,
                    valid_count=len(valid),
                )

            self._candidate_key = "kf_no_association"
            self._candidate_count = len(valid)
            self._last_kalman_debug = {
                "enabled": True,
                "mode": "no_association_outside_gate",
                "association_distance_m": association_distance,
                "association_gate_m": association_gate,
                "age_since_update_s": age_since_update,
                "valid_count": len(valid),
                "prediction": self._xyz_dict(tuple(float(v) for v in predicted)),
                "covariance_diag": self._covariance_dict(),
            }
            if bool(self.get_parameter("kalman_publish_prediction_without_measurement").value):
                pseudo = min(valid, key=lambda c: (c.depth_m, -c.confidence))
                pseudo.sensor_point_xyz = pseudo.point_xyz
                pseudo.sensor_depth_m = pseudo.depth_m
                pseudo.point_xyz = tuple(float(v) for v in predicted)
                pseudo.depth_m = float(predicted[0])
                pseudo.association_distance_m = association_distance
                self.locked_last_point = pseudo.point_xyz
                self.locked_last_seen_time = now
                return pseudo
            return None

        filtered = self.kalman_filter.update(chosen.point_xyz, meas_std, now)
        return self._finalize_kalman_target(
            chosen=chosen,
            filtered=filtered,
            now=now,
            mode="update_associated",
            association_distance=association_distance,
            valid_count=len(valid),
        )

    def _select_nearest_valid_without_kf(self, valid: List[PersonCandidate], now: float) -> Optional[PersonCandidate]:
        if not valid:
            self.locked_last_point = None
            self.locked_last_seen_time = None
            self._candidate_key = None
            self._candidate_count = 0
            return None
        chosen = min(valid, key=lambda c: (c.depth_m, -c.confidence))
        self.locked_last_point = chosen.point_xyz
        self.locked_last_seen_time = now
        self._candidate_key = "nearest_person"
        self._candidate_count = len(valid)
        return chosen

    def _handle_no_kalman_measurement(self, now: float) -> Optional[PersonCandidate]:
        reset_timeout = float(self.get_parameter("kalman_reset_timeout_s").value)
        last_update = self.kalman_filter.last_update_time
        age_since_update = math.inf if last_update is None else max(0.0, now - last_update)
        if self.kalman_filter.initialized and age_since_update >= reset_timeout:
            self.kalman_filter.reset()
        self.locked_last_point = None
        self.locked_last_seen_time = None
        self._candidate_key = None
        self._candidate_count = 0
        self._last_kalman_debug = {
            "enabled": True,
            "mode": "no_measurement",
            "age_since_update_s": age_since_update,
            "reset_timeout_s": reset_timeout,
            "initialized": self.kalman_filter.initialized,
            "covariance_diag": self._covariance_dict(),
        }
        return None

    def _finalize_kalman_target(
        self,
        chosen: PersonCandidate,
        filtered: np.ndarray,
        now: float,
        mode: str,
        association_distance: float,
        valid_count: int,
    ) -> PersonCandidate:
        raw_point = chosen.point_xyz
        raw_depth = chosen.depth_m
        filtered_tuple = tuple(float(v) for v in filtered.reshape(3))
        chosen.sensor_point_xyz = raw_point
        chosen.sensor_depth_m = raw_depth
        chosen.point_xyz = filtered_tuple
        chosen.depth_m = float(filtered_tuple[0])
        chosen.association_distance_m = float(association_distance)
        self.locked_last_point = filtered_tuple
        self.locked_last_seen_time = now
        self._candidate_key = "kalman_associated_person"
        self._candidate_count = valid_count
        self._last_kalman_debug = {
            "enabled": True,
            "mode": mode,
            "association_distance_m": float(association_distance),
            "association_gate_m": float(self.get_parameter("kalman_association_gate_m").value),
            "valid_count": valid_count,
            "sensor_point": self._xyz_dict(raw_point),
            "filtered_point": self._xyz_dict(filtered_tuple),
            "covariance_diag": self._covariance_dict(),
        }
        return chosen

    def _xyz_dict(self, xyz: Sequence[float]) -> Dict[str, float]:
        return {"x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2])}

    def _covariance_dict(self) -> Dict[str, float]:
        cx, cy, cz = self.kalman_filter.covariance_diag()
        return {"x": cx, "y": cy, "z": cz}

    def _update_lock(self, cand: PersonCandidate, now: float) -> None:
        # Backward-compatible helper; track_id is intentionally ignored.
        self.locked_track_id = None
        self.locked_last_point = cand.point_xyz
        self.locked_last_seen_time = now

    def _clear_lock(self) -> None:
        if self.locked_track_id is not None or self.locked_last_point is not None:
            self.get_logger().warn("Target lock cleared.")
        self.locked_track_id = None
        self.locked_last_point = None
        self.locked_last_seen_time = None
        self._candidate_key = None
        self._candidate_count = 0
        if bool(self.get_parameter("use_kalman_filter").value):
            self.kalman_filter.reset()
            self._last_kalman_debug = {"enabled": True, "mode": "reset_by_clear_lock"}

    # --------------------------------------------------------------- publish
    def _publish_target(self, target: PersonCandidate, stamp: Any) -> None:
        msg = PointStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.target_frame
        msg.point.x = float(target.point_xyz[0])
        msg.point.y = float(target.point_xyz[1])
        msg.point.z = float(target.point_xyz[2])
        self.target_pub.publish(msg)

    def _publish_status(
        self, tracked: bool, target: Optional[PersonCandidate] = None, reason: str = ""
    ) -> None:
        payload: Dict[str, Any] = {
            "tracked": bool(tracked),
            "locked_track_id": None,
            "selection_policy": "kalman_association_nearest_to_prediction" if bool(self.get_parameter("use_kalman_filter").value) else "nearest_person_no_track_id",
            "kalman_filter": self._last_kalman_debug,
            "reason": reason,
            "inference_device": self.inference_device,
            "yolo_error": self._last_yolo_error,
            "stamp_monotonic": time.monotonic(),
        }
        if target is not None:
            x, y, z = target.point_xyz
            x1, y1, x2, y2 = target.bbox_xyxy
            payload.update(
                {
                    "track_id": None,
                    "class_name": target.class_name,
                    "confidence": target.confidence,
                    "depth_m": target.depth_m,
                    "point": {"x": x, "y": y, "z": z},
                    "sensor_depth_m": target.sensor_depth_m,
                    "sensor_point": self._xyz_dict(target.sensor_point_xyz if target.sensor_point_xyz is not None else target.point_xyz),
                    "association_distance_m": target.association_distance_m,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "target_heading_rad": math.atan2(y, x),
                }
            )
        self.status_pub.publish(String(data=json.dumps(payload)))

    def _make_target_camera_info(
        self,
        source: CameraInfo,
        stamp: Any,
        out_width: int,
        out_height: int,
        scale_x: float,
        scale_y: float,
    ) -> CameraInfo:
        """Return CameraInfo for the debug/target image topic.

        The source intrinsics come from the OAK camera_info topic used for depth
        projection.  When the debug image is resized before publication, the focal
        lengths and principal point must be scaled by the same factors so that RViz
        or image_transport tools that consume /target/camera_info see a consistent
        image size.
        """
        info = copy.deepcopy(source)
        info.header.stamp = stamp
        info.header.frame_id = self.target_frame
        info.width = int(out_width)
        info.height = int(out_height)

        k = list(info.k)
        if len(k) == 9:
            k[0] *= scale_x
            k[2] *= scale_x
            k[4] *= scale_y
            k[5] *= scale_y
            info.k = k

        p = list(info.p)
        if len(p) == 12:
            p[0] *= scale_x
            p[2] *= scale_x
            p[3] *= scale_x
            p[5] *= scale_y
            p[6] *= scale_y
            p[7] *= scale_y
            info.p = p

        return info

    def _publish_debug_image(
        self,
        frame: np.ndarray,
        candidates: List[PersonCandidate],
        target: Optional[PersonCandidate],
        stamp: Any,
    ) -> None:
        now = time.monotonic()
        rate = float(self.get_parameter("debug_image_rate_hz").value)
        if rate > 0.0 and now - self._last_debug_pub_time < 1.0 / rate:
            return
        self._last_debug_pub_time = now

        vis = frame.copy()
        for cand in candidates:
            x1, y1, x2, y2 = cand.bbox_xyxy
            is_target = target is not None and cand is target
            thickness = 3 if is_target else 1
            color = (0, 255, 0) if is_target else (0, 180, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            label = f"{cand.class_name} {cand.confidence:.2f}"
            if is_target and cand.sensor_depth_m is not None and bool(self.get_parameter("use_kalman_filter").value):
                label += f" raw {cand.sensor_depth_m:.2f}m KF {cand.depth_m:.2f}m"
            else:
                label += f" {cand.depth_m:.2f}m"
            cv2.putText(
                vis,
                label,
                (max(0, x1), max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        if target is None:
            cv2.putText(
                vis,
                "NO VALID PERSON TARGET",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            kf_mode = self._last_kalman_debug.get("mode", "off") if isinstance(self._last_kalman_debug, dict) else "off"
            detail = f"candidates={len(candidates)} policy=KF-nearest-pred mode={kf_mode} device={self.inference_device}"
            cv2.putText(
                vis,
                detail,
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            if self._last_yolo_error:
                cv2.putText(
                    vis,
                    "YOLO ERROR - see /target/status or terminal",
                    (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        src_h, src_w = int(vis.shape[0]), int(vis.shape[1])
        resize_w = int(self.get_parameter("debug_resize_width").value)
        if resize_w > 0 and vis.shape[1] > resize_w:
            scale = resize_w / float(vis.shape[1])
            vis = cv2.resize(vis, (resize_w, int(round(vis.shape[0] * scale))))

        out_h, out_w = int(vis.shape[0]), int(vis.shape[1])
        scale_x = float(out_w) / max(1.0, float(src_w))
        scale_y = float(out_h) / max(1.0, float(src_h))

        if bool(self.get_parameter("publish_debug_raw_image").value):
            raw_msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            raw_msg.header.stamp = stamp
            raw_msg.header.frame_id = self.target_frame
            self.debug_raw_pub.publish(raw_msg)

        if bool(self.get_parameter("publish_target_camera_info").value) and self.latest_camera_info is not None:
            info_msg = self._make_target_camera_info(
                source=self.latest_camera_info,
                stamp=stamp,
                out_width=out_w,
                out_height=out_h,
                scale_x=scale_x,
                scale_y=scale_y,
            )
            self.target_camera_info_pub.publish(info_msg)

        quality = int(self.get_parameter("debug_jpeg_quality").value)
        quality = int(np.clip(quality, 5, 95))
        ok, encoded = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return
        msg = CompressedImage()
        msg.header.stamp = stamp
        msg.header.frame_id = self.target_frame
        msg.format = "jpeg"
        msg.data = encoded.tobytes()
        self.debug_pub.publish(msg)

    # ------------------------------------------------------------- utilities
    def _warn_throttled(self, text: str, period_s: float = 2.0) -> None:
        now = time.monotonic()
        if now - self._last_warn_time >= period_s:
            self._last_warn_time = now
            self.get_logger().warn(text)


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = OakYoloTargetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
