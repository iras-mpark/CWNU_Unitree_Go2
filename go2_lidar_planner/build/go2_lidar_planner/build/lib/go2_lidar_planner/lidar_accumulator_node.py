#!/usr/bin/env python3
"""Sliding-window accumulator for sparse Go2 LiDAR point clouds."""

from __future__ import annotations

from collections import deque
import math
from typing import Deque, List, Optional, Sequence, Tuple

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, String


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
        # Height filter for obstacle-grid generation.  The forward side (+x, robot
        # heading direction) keeps the legacy lower bound, while the rear side
        # (-x, behind the LiDAR/body) can be tuned independently to ignore
        # body/leg/floor returns behind the robot.
        self.declare_parameter("obstacle_z_min_m", 0.1)
        self.declare_parameter("obstacle_z_min_negative_x_m", 0.1)
        self.declare_parameter("obstacle_z_max_m", 1.2)
        self.declare_parameter("min_obstacle_range_m", 0.10)
        self.declare_parameter("max_obstacle_range_m", 6.0)
        self.declare_parameter("min_points_per_cell", 2)

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
        self.last_debug = self.get_clock().now()
        self.create_subscription(PointCloud2, self.input_topic, self._cloud_cb, 10)
        self.pub = self.create_publisher(PointCloud2, self.output_topic, 10)
        self.occupancy_pub = self.create_publisher(OccupancyGrid, self.occupancy_grid_topic, 10)
        self.debug_pub = self.create_publisher(String, "~/debug", 5)
        self.create_timer(1.0 / self.publish_rate_hz, self._timer_cb)
        self.get_logger().info(
            f"LiDAR accumulator: {self.input_topic} -> {self.output_topic}, "
            f"ogm={self.occupancy_grid_topic}, window={self.history_duration.nanoseconds/1e9:.2f}s, "
            f"max_clouds={self.max_clouds}"
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
                        f"clouds={len(self.history)}, points={len(all_points)}, "
                        f"obstacle_z_min(+x)={z_min_pos:.2f}, "
                        f"obstacle_z_min(-x)={z_min_neg:.2f}, "
                        f"obstacle_z_max={z_max:.2f}, obstacle_xy_min={r_min:.2f}"
                    )
                )
            )

    def _build_occupancy_grid(self, points: List[Tuple[float, ...]], header: Header) -> OccupancyGrid:
        r_min = float(self.get_parameter("min_obstacle_range_m").value)
        r_max = float(self.get_parameter("max_obstacle_range_m").value)
        r_min, r_max = min(r_min, r_max), max(r_min, r_max)
        min_hits = max(1, int(self.get_parameter("min_points_per_cell").value))
        hit_counts = [0] * (self.width * self.height)

        for point in points:
            x, y, z = float(point[0]), float(point[1]), float(point[2])
            z_min, z_max = self._effective_obstacle_z_bounds_for_x(x)
            if z < z_min or z > z_max:
                continue
            distance_xy = math.hypot(x, y)
            if distance_xy < r_min or distance_xy > r_max:
                continue
            ix = int(math.floor((x - self.x_min) / self.resolution))
            iy = int(math.floor((y - self.y_min) / self.resolution))
            if 0 <= ix < self.width and 0 <= iy < self.height:
                hit_counts[iy * self.width + ix] += 1

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
