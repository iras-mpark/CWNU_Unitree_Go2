#!/usr/bin/env python3
"""Transform Unitree LiDAR point clouds into the local planning frame.

The node keeps the static mounting transform used by the original package, and
can optionally estimate a residual pitch correction from the observed floor
plane.  The pitch correction is intentionally limited and low-pass filtered so
that the published cloud does not jump when the RANSAC estimate is poor.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf_transformations import euler_matrix, quaternion_from_euler


PlaneFit = Tuple[np.ndarray, float, int, float]


class LidarTransformerNode(Node):
    def __init__(self) -> None:
        super().__init__("go2_lidar_transformer")
        self.declare_parameter("input_topic", "/utlidar/cloud")
        self.declare_parameter("output_topic", "/utlidar/transformed_cloud")
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("lidar_frame", "utlidar_lidar")
        self.declare_parameter("lidar_x", 0.0)
        self.declare_parameter("lidar_y", 0.0)
        self.declare_parameter("lidar_z", 0.0)
        self.declare_parameter("lidar_roll", 0.0)
        self.declare_parameter("lidar_pitch", 2.8782025850555556)
        self.declare_parameter("lidar_yaw", 0.0)

        # Dynamic residual pitch correction from the ground plane.  The static
        # mounting pitch is still the nominal transform; this only compensates
        # small residual errors or body attitude changes relative to the floor.
        self.declare_parameter("dynamic_pitch_enabled", True)
        self.declare_parameter("dynamic_pitch_alpha", 0.08)
        self.declare_parameter("dynamic_pitch_max_correction_rad", 0.25)
        self.declare_parameter("dynamic_pitch_hold_last_on_fail", True)
        self.declare_parameter("dynamic_pitch_debug_topic", "~/debug")

        # RANSAC parameters used only for estimating floor pitch in the already
        # transformed base_link-like cloud.
        self.declare_parameter("ground_ransac_iterations", 60)
        self.declare_parameter("ground_ransac_distance_threshold_m", 0.035)
        self.declare_parameter("ground_ransac_max_sample_points", 900)
        self.declare_parameter("ground_min_inliers", 80)
        self.declare_parameter("ground_min_inlier_ratio", 0.12)
        self.declare_parameter("ground_normal_min_z", 0.70)
        self.declare_parameter("ground_candidate_x_min_m", -0.5)
        self.declare_parameter("ground_candidate_x_max_m", 4.0)
        self.declare_parameter("ground_candidate_y_abs_m", 2.0)
        self.declare_parameter("ground_candidate_z_min_m", -0.45)
        self.declare_parameter("ground_candidate_z_max_m", 0.35)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.target_frame = str(self.get_parameter("target_frame").value)
        self.lidar_frame = str(self.get_parameter("lidar_frame").value)
        self.translation = np.array(
            [
                float(self.get_parameter("lidar_x").value),
                float(self.get_parameter("lidar_y").value),
                float(self.get_parameter("lidar_z").value),
            ],
            dtype=np.float64,
        )
        self.roll = float(self.get_parameter("lidar_roll").value)
        self.pitch = float(self.get_parameter("lidar_pitch").value)
        self.yaw = float(self.get_parameter("lidar_yaw").value)
        self.rotation = np.asarray(euler_matrix(self.roll, self.pitch, self.yaw)[:3, :3], dtype=np.float64)
        self.pitch_correction = 0.0
        self._rng = np.random.default_rng()
        self._last_debug_time = self.get_clock().now() - Duration(seconds=10.0)
        self._last_plane_ok = False
        self._last_plane_info = "not_run"

        self.pub = self.create_publisher(PointCloud2, self.output_topic, 10)
        self.debug_pub = self.create_publisher(String, str(self.get_parameter("dynamic_pitch_debug_topic").value), 5)
        self.create_subscription(PointCloud2, self.input_topic, self._cloud_cb, 10)
        self.tf_pub = StaticTransformBroadcaster(self)
        self._broadcast_static_tf()
        self.get_logger().info(
            f"LiDAR transformer: {self.input_topic} -> {self.output_topic}, "
            f"frame={self.target_frame}, static_rpy=({self.roll:.3f},{self.pitch:.3f},{self.yaw:.3f}), "
            f"dynamic_pitch={bool(self.get_parameter('dynamic_pitch_enabled').value)}"
        )

    def _cloud_cb(self, cloud: PointCloud2) -> None:
        points = list(point_cloud2.read_points(cloud, skip_nans=True))
        if not points:
            out = PointCloud2()
            out.header = cloud.header
            out.header.frame_id = self.target_frame
            self.pub.publish(out)
            return

        raw_xyz = np.asarray([[float(p[0]), float(p[1]), float(p[2])] for p in points], dtype=np.float64)
        static_xyz = raw_xyz @ self.rotation.T + self.translation

        if bool(self.get_parameter("dynamic_pitch_enabled").value):
            self._update_dynamic_pitch(static_xyz)

        if abs(self.pitch_correction) > 1e-9:
            correction_rot = np.asarray(euler_matrix(0.0, self.pitch_correction, 0.0)[:3, :3], dtype=np.float64)
            corrected_rotation = correction_rot @ self.rotation
            transformed_xyz = raw_xyz @ corrected_rotation.T + self.translation
        else:
            transformed_xyz = static_xyz

        transformed: List[Tuple[float, ...]] = []
        for xyz_new, p in zip(transformed_xyz, points):
            transformed.append((float(xyz_new[0]), float(xyz_new[1]), float(xyz_new[2]), *p[3:]))

        msg = point_cloud2.create_cloud(cloud.header, cloud.fields, transformed)
        msg.header.frame_id = self.target_frame
        self.pub.publish(msg)
        self._publish_debug(len(points))

    def _update_dynamic_pitch(self, points_xyz: np.ndarray) -> None:
        fit = self._fit_ground_plane(points_xyz)
        if fit is None:
            self._last_plane_ok = False
            self._last_plane_info = "ground_plane_not_found"
            if not bool(self.get_parameter("dynamic_pitch_hold_last_on_fail").value):
                alpha = float(self.get_parameter("dynamic_pitch_alpha").value)
                self.pitch_correction = (1.0 - alpha) * self.pitch_correction
            return

        normal, _d, inliers, ratio = fit
        target = math.atan2(-float(normal[0]), max(1e-6, float(normal[2])))
        max_corr = abs(float(self.get_parameter("dynamic_pitch_max_correction_rad").value))
        target = max(-max_corr, min(max_corr, target))
        alpha = max(0.0, min(1.0, float(self.get_parameter("dynamic_pitch_alpha").value)))
        self.pitch_correction = (1.0 - alpha) * self.pitch_correction + alpha * target
        self._last_plane_ok = True
        self._last_plane_info = (
            f"normal=({normal[0]:.3f},{normal[1]:.3f},{normal[2]:.3f}), "
            f"inliers={inliers}, ratio={ratio:.2f}, target_pitch_corr={target:.4f}rad"
        )

    def _fit_ground_plane(self, points_xyz: np.ndarray) -> Optional[PlaneFit]:
        if points_xyz.shape[0] < 3:
            return None
        finite = np.isfinite(points_xyz).all(axis=1)
        x_min = float(self.get_parameter("ground_candidate_x_min_m").value)
        x_max = float(self.get_parameter("ground_candidate_x_max_m").value)
        y_abs = abs(float(self.get_parameter("ground_candidate_y_abs_m").value))
        z_min = float(self.get_parameter("ground_candidate_z_min_m").value)
        z_max = float(self.get_parameter("ground_candidate_z_max_m").value)
        mask = (
            finite
            & (points_xyz[:, 0] >= x_min)
            & (points_xyz[:, 0] <= x_max)
            & (np.abs(points_xyz[:, 1]) <= y_abs)
            & (points_xyz[:, 2] >= min(z_min, z_max))
            & (points_xyz[:, 2] <= max(z_min, z_max))
        )
        candidates = points_xyz[mask]
        if candidates.shape[0] < max(3, int(self.get_parameter("ground_min_inliers").value)):
            return None

        max_sample = max(3, int(self.get_parameter("ground_ransac_max_sample_points").value))
        if candidates.shape[0] > max_sample:
            idx = self._rng.choice(candidates.shape[0], size=max_sample, replace=False)
            sample = candidates[idx]
        else:
            sample = candidates

        iters = max(1, int(self.get_parameter("ground_ransac_iterations").value))
        dist_thr = max(0.005, float(self.get_parameter("ground_ransac_distance_threshold_m").value))
        min_normal_z = max(0.0, min(1.0, float(self.get_parameter("ground_normal_min_z").value)))
        best_normal: Optional[np.ndarray] = None
        best_d = 0.0
        best_count = 0

        if sample.shape[0] < 3:
            return None
        for _ in range(iters):
            ids = self._rng.choice(sample.shape[0], size=3, replace=False)
            p1, p2, p3 = sample[ids]
            n = np.cross(p2 - p1, p3 - p1)
            norm = float(np.linalg.norm(n))
            if norm < 1e-8:
                continue
            n = n / norm
            if n[2] < 0.0:
                n = -n
            if n[2] < min_normal_z:
                continue
            d = -float(np.dot(n, p1))
            distances = np.abs(candidates @ n + d)
            count = int(np.count_nonzero(distances <= dist_thr))
            if count > best_count:
                best_count = count
                best_normal = n
                best_d = d

        if best_normal is None:
            return None
        min_inliers = int(self.get_parameter("ground_min_inliers").value)
        min_ratio = float(self.get_parameter("ground_min_inlier_ratio").value)
        ratio = best_count / max(1, candidates.shape[0])
        if best_count < min_inliers or ratio < min_ratio:
            return None

        inlier_mask = np.abs(candidates @ best_normal + best_d) <= dist_thr
        inliers = candidates[inlier_mask]
        if inliers.shape[0] >= 3:
            centroid = inliers.mean(axis=0)
            try:
                _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
                refined = vh[-1]
                refined = refined / max(1e-8, float(np.linalg.norm(refined)))
                if refined[2] < 0.0:
                    refined = -refined
                if refined[2] >= min_normal_z:
                    best_normal = refined
                    best_d = -float(np.dot(best_normal, centroid))
            except Exception:
                pass

        return best_normal, float(best_d), int(best_count), float(ratio)

    def _publish_debug(self, point_count: int) -> None:
        now = self.get_clock().now()
        if now - self._last_debug_time < Duration(seconds=1.0):
            return
        self._last_debug_time = now
        self.debug_pub.publish(
            String(
                data=(
                    f"points={point_count}, dynamic_pitch_enabled={bool(self.get_parameter('dynamic_pitch_enabled').value)}, "
                    f"pitch_correction={self.pitch_correction:.5f}rad ({math.degrees(self.pitch_correction):.2f}deg), "
                    f"ground_ok={self._last_plane_ok}, {self._last_plane_info}"
                )
            )
        )

    def _broadcast_static_tf(self) -> None:
        q = quaternion_from_euler(self.roll, self.pitch, self.yaw)
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = self.target_frame
        tf.child_frame_id = self.lidar_frame
        tf.transform.translation.x = float(self.translation[0])
        tf.transform.translation.y = float(self.translation[1])
        tf.transform.translation.z = float(self.translation[2])
        tf.transform.rotation.x = q[0]
        tf.transform.rotation.y = q[1]
        tf.transform.rotation.z = q[2]
        tf.transform.rotation.w = q[3]
        self.tf_pub.sendTransform(tf)


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = LidarTransformerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
