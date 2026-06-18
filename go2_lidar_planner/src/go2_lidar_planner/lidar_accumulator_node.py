#!/usr/bin/env python3
"""Sliding-window accumulator and ground-aware obstacle-grid generator.

The accumulated cloud is still published unchanged for debugging/visualization.
For the 2-D obstacle grid, the node first tries to estimate the floor plane with
RANSAC in the transformed base_link cloud.  When the plane is found, floor
points are removed by plane-relative height.  If no reliable floor is found, the
node falls back to the legacy fixed z-height filtering.
"""

from __future__ import annotations

from collections import deque
import math
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, String

PlaneFit = Tuple[np.ndarray, float, int, float]


class LidarAccumulatorNode(Node):
    def __init__(self) -> None:
        super().__init__("go2_lidar_accumulator")
        self.declare_parameter("input_topic", "/utlidar/transformed_cloud")
        self.declare_parameter("output_topic", "/utlidar/accumulated_cloud")
        self.declare_parameter("occupancy_grid_topic", "/utlidar/accumulated_obstacle_grid")
        self.declare_parameter("history_duration", 0.8)
        self.declare_parameter("max_clouds", 10)
        self.declare_parameter("publish_rate_hz", 15.0)
        self.declare_parameter("grid_resolution_m", 0.12)
        self.declare_parameter("grid_x_min_m", 0.0)
        self.declare_parameter("grid_x_max_m", 6.0)
        self.declare_parameter("grid_y_min_m", -3.5)
        self.declare_parameter("grid_y_max_m", 3.5)
        # Legacy fallback height filters.  +x and -x can be tuned independently.
        self.declare_parameter("obstacle_z_min_m", 0.1)
        self.declare_parameter("obstacle_z_min_negative_x_m", 0.1)
        self.declare_parameter("obstacle_z_max_m", 1.2)
        self.declare_parameter("min_obstacle_range_m", 0.10)
        self.declare_parameter("max_obstacle_range_m", 6.0)
        self.declare_parameter("min_points_per_cell", 2)

        # Ground-plane RANSAC and ground-relative obstacle filtering.
        self.declare_parameter("use_ground_ransac_filter", True)
        self.declare_parameter("ground_ransac_iterations", 70)
        self.declare_parameter("ground_ransac_distance_threshold_m", 0.035)
        self.declare_parameter("ground_ransac_max_sample_points", 1200)
        self.declare_parameter("ground_min_inliers", 100)
        self.declare_parameter("ground_min_inlier_ratio", 0.10)
        self.declare_parameter("ground_normal_min_z", 0.70)
        self.declare_parameter("ground_candidate_x_min_m", -0.5)
        self.declare_parameter("ground_candidate_x_max_m", 4.0)
        self.declare_parameter("ground_candidate_y_abs_m", 2.0)
        self.declare_parameter("ground_candidate_z_min_m", -0.45)
        self.declare_parameter("ground_candidate_z_max_m", 0.35)
        self.declare_parameter("ground_non_center_min_height_m", 0.04)
        self.declare_parameter("ground_center_strip_y_abs_m", 0.30)
        self.declare_parameter("ground_center_strip_min_height_m", 0.10)
        self.declare_parameter("ground_filter_max_height_m", 1.20)

        # Self/leg return exclusion box near the rear side of the robot.
        # This removes low-height points from the robot legs/body that appear
        # behind the base_link origin from the 2-D obstacle map only.
        self.declare_parameter("rear_self_filter_enabled", True)
        self.declare_parameter("rear_self_filter_x_center_m", -0.10)
        self.declare_parameter("rear_self_filter_x_size_m", 0.30)
        self.declare_parameter("rear_self_filter_y_abs_m", 0.20)
        self.declare_parameter("rear_self_filter_height_m", 0.30)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.history_duration = Duration(seconds=max(0.02, float(self.get_parameter("history_duration").value)))
        self.max_clouds = max(1, int(self.get_parameter("max_clouds").value))
        self.publish_rate_hz = max(0.5, float(self.get_parameter("publish_rate_hz").value))
        self.occupancy_grid_topic = str(self.get_parameter("occupancy_grid_topic").value)
        self.resolution = max(0.01, float(self.get_parameter("grid_resolution_m").value))
        self.x_min = float(self.get_parameter("grid_x_min_m").value)
        self.x_max = float(self.get_parameter("grid_x_max_m").value)
        self.y_min = float(self.get_parameter("grid_y_min_m").value)
        self.y_max = float(self.get_parameter("grid_y_max_m").value)
        self.width = int(math.ceil((self.x_max - self.x_min) / self.resolution)) + 1
        self.height = int(math.ceil((self.y_max - self.y_min) / self.resolution)) + 1

        self.history: Deque[Tuple[Time, List[Tuple[float, ...]]]] = deque()
        self.fields = None
        self.frame_id = ""
        self.last_debug = self.get_clock().now() - Duration(seconds=10.0)
        self._rng = np.random.default_rng()
        self._last_ground_ok = False
        self._last_ground_info = "not_run"
        self._last_obstacle_hits = 0
        self._last_self_filtered_points = 0
        self._last_used_fallback = False
        self.create_subscription(PointCloud2, self.input_topic, self._cloud_cb, 10)
        self.pub = self.create_publisher(PointCloud2, self.output_topic, 10)
        self.occupancy_pub = self.create_publisher(OccupancyGrid, self.occupancy_grid_topic, 10)
        self.debug_pub = self.create_publisher(String, "~/debug", 5)
        self.create_timer(1.0 / self.publish_rate_hz, self._timer_cb)
        self.get_logger().info(
            f"LiDAR accumulator: {self.input_topic} -> {self.output_topic}, "
            f"ogm={self.occupancy_grid_topic}, window={self.history_duration.nanoseconds/1e9:.2f}s, "
            f"max_clouds={self.max_clouds}, ground_ransac={bool(self.get_parameter('use_ground_ransac_filter').value)}"
        )

    def _effective_obstacle_z_bounds(self) -> Tuple[float, float, float]:
        z_min_pos = float(self.get_parameter("obstacle_z_min_m").value)
        z_min_neg = float(self.get_parameter("obstacle_z_min_negative_x_m").value)
        z_max = float(self.get_parameter("obstacle_z_max_m").value)
        return (z_min_pos, z_min_neg, z_max)

    def _effective_obstacle_z_bounds_for_x(self, x: float) -> Tuple[float, float]:
        z_min_pos, z_min_neg, z_max = self._effective_obstacle_z_bounds()
        z_min = z_min_pos if x >= 0.0 else z_min_neg
        return (min(z_min, z_max), max(z_min, z_max))

    def _cloud_cb(self, cloud: PointCloud2) -> None:
        now = self.get_clock().now()
        pts = list(point_cloud2.read_points(cloud, skip_nans=True))
        self.frame_id = cloud.header.frame_id or self.frame_id
        if self.fields is None:
            self.fields = list(cloud.fields)
        if pts:
            self.history.append((now, pts))
            while len(self.history) > self.max_clouds:
                self.history.popleft()
        self._prune(now)

    def _prune(self, now: Time) -> None:
        while self.history and now - self.history[0][0] > self.history_duration:
            self.history.popleft()

    def _timer_cb(self) -> None:
        now = self.get_clock().now()
        self._prune(now)
        all_points: List[Tuple[float, ...]] = []
        for _, pts in self.history:
            all_points.extend(pts)
        all_points = [point for point in all_points if float(point[0]) >= self.x_min]

        header = Header()
        header.stamp = now.to_msg()
        header.frame_id = self.frame_id
        if all_points and self.fields:
            msg = point_cloud2.create_cloud(header, self.fields, all_points)
        elif all_points:
            msg = point_cloud2.create_cloud_xyz32(header, all_points)
        else:
            msg = PointCloud2()
            msg.header = header
        self.pub.publish(msg)
        self.occupancy_pub.publish(self._build_occupancy_grid(all_points, header))

        if now - self.last_debug > Duration(seconds=1.0):
            self.last_debug = now
            z_min_pos, z_min_neg, z_max = self._effective_obstacle_z_bounds()
            r_min = float(self.get_parameter("min_obstacle_range_m").value)
            self.debug_pub.publish(
                String(
                    data=(
                        f"clouds={len(self.history)}, points={len(all_points)}, obstacle_hits={self._last_obstacle_hits}, "
                        f"rear_self_filtered={self._last_self_filtered_points}, "
                        f"ground_ransac={bool(self.get_parameter('use_ground_ransac_filter').value)}, "
                        f"ground_ok={self._last_ground_ok}, fallback_height_filter={self._last_used_fallback}, "
                        f"{self._last_ground_info}, legacy_z_min(+x)={z_min_pos:.2f}, "
                        f"legacy_z_min(-x)={z_min_neg:.2f}, z_max={z_max:.2f}, obstacle_xy_min={r_min:.2f}"
                    )
                )
            )

    def _build_occupancy_grid(self, points: List[Tuple[float, ...]], header: Header) -> OccupancyGrid:
        r_min = float(self.get_parameter("min_obstacle_range_m").value)
        r_max = float(self.get_parameter("max_obstacle_range_m").value)
        r_min, r_max = min(r_min, r_max), max(r_min, r_max)
        min_hits = max(1, int(self.get_parameter("min_points_per_cell").value))
        hit_counts = [0] * (self.width * self.height)
        self._last_obstacle_hits = 0
        self._last_self_filtered_points = 0

        xyz = self._points_to_xyz_array(points)
        fit = self._fit_ground_plane(xyz) if bool(self.get_parameter("use_ground_ransac_filter").value) else None
        if fit is not None:
            self._last_ground_ok = True
            self._last_used_fallback = False
            normal, d, inliers, ratio = fit
            self._last_ground_info = (
                f"plane_n=({normal[0]:.3f},{normal[1]:.3f},{normal[2]:.3f}), d={d:.3f}, "
                f"inliers={inliers}, ratio={ratio:.2f}"
            )
        else:
            self._last_ground_ok = False
            self._last_used_fallback = True
            normal, d = None, 0.0
            self._last_ground_info = "ground_plane_not_found"

        for point in points:
            x, y, z = float(point[0]), float(point[1]), float(point[2])
            distance_xy = math.hypot(x, y)
            if distance_xy < r_min or distance_xy > r_max:
                continue

            if self._is_rear_self_filter_point(x, y, z, normal, d):
                self._last_self_filtered_points += 1
                continue

            if normal is not None:
                if not self._is_ground_relative_obstacle(x, y, z, normal, d):
                    continue
            else:
                if not self._is_legacy_height_obstacle(x, y, z):
                    continue

            ix = int(math.floor((x - self.x_min) / self.resolution))
            iy = int(math.floor((y - self.y_min) / self.resolution))
            if 0 <= ix < self.width and 0 <= iy < self.height:
                hit_counts[iy * self.width + ix] += 1
                self._last_obstacle_hits += 1

        grid = OccupancyGrid()
        grid.header = header
        grid.info.resolution = self.resolution
        grid.info.width = self.width
        grid.info.height = self.height
        grid.info.origin.position.x = self.x_min
        grid.info.origin.position.y = self.y_min
        grid.info.origin.orientation.w = 1.0
        grid.data = [100 if count >= min_hits else 0 for count in hit_counts]
        return grid

    def _height_above_ground(self, x: float, y: float, z: float, normal: Optional[np.ndarray], d: float) -> float:
        if normal is None:
            return z
        return float(np.dot(normal, np.array([x, y, z], dtype=np.float64)) + d)

    def _is_rear_self_filter_point(
        self, x: float, y: float, z: float, normal: Optional[np.ndarray], d: float
    ) -> bool:
        if not bool(self.get_parameter("rear_self_filter_enabled").value):
            return False
        x_center = float(self.get_parameter("rear_self_filter_x_center_m").value)
        x_size = abs(float(self.get_parameter("rear_self_filter_x_size_m").value))
        y_abs = abs(float(self.get_parameter("rear_self_filter_y_abs_m").value))
        max_h = float(self.get_parameter("rear_self_filter_height_m").value)
        if x_size <= 0.0 or y_abs <= 0.0 or max_h <= 0.0:
            return False
        x_min = x_center - 0.5 * x_size
        x_max = x_center + 0.5 * x_size
        if not (x_min <= x <= x_max and abs(y) <= y_abs):
            return False
        height = self._height_above_ground(x, y, z, normal, d)
        # Remove low returns from floor level up to the configured height.  A
        # small negative tolerance is intentional because fitted planes and
        # transformed points can have a few centimeters of noise.
        return height <= max_h

    def _is_ground_relative_obstacle(self, x: float, y: float, z: float, normal: np.ndarray, d: float) -> bool:
        # Signed height above the floor plane.  The fitted normal is forced to
        # have positive z, so positive signed distance means above the floor.
        height = self._height_above_ground(x, y, z, normal, d)
        max_h = float(self.get_parameter("ground_filter_max_height_m").value)
        if height > max_h:
            return False
        center_strip_y = abs(float(self.get_parameter("ground_center_strip_y_abs_m").value))
        if x > 0.0 and abs(y) <= center_strip_y:
            min_h = float(self.get_parameter("ground_center_strip_min_height_m").value)
        else:
            min_h = float(self.get_parameter("ground_non_center_min_height_m").value)
        return height >= min_h

    def _is_legacy_height_obstacle(self, x: float, y: float, z: float) -> bool:
        z_min, z_max = self._effective_obstacle_z_bounds_for_x(x)
        return z_min <= z <= z_max

    def _points_to_xyz_array(self, points: List[Tuple[float, ...]]) -> np.ndarray:
        if not points:
            return np.zeros((0, 3), dtype=np.float64)
        arr = np.asarray([[float(p[0]), float(p[1]), float(p[2])] for p in points], dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            return np.zeros((0, 3), dtype=np.float64)
        return arr

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


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = LidarAccumulatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
