#!/usr/bin/env python3
"""Sliding-window accumulator for sparse Go2 LiDAR point clouds."""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Sequence, Tuple

import rclpy
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
        self.declare_parameter("history_duration", 0.8)
        self.declare_parameter("max_clouds", 10)
        self.declare_parameter("publish_rate_hz", 15.0)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.history_duration = Duration(seconds=max(0.02, float(self.get_parameter("history_duration").value)))
        self.max_clouds = max(1, int(self.get_parameter("max_clouds").value))
        self.publish_rate_hz = max(0.5, float(self.get_parameter("publish_rate_hz").value))

        self.history: Deque[Tuple[Time, List[Tuple[float, ...]]]] = deque()
        self.fields = None
        self.frame_id = ""
        self.last_debug = self.get_clock().now()

        self.create_subscription(PointCloud2, self.input_topic, self._cloud_cb, 10)
        self.pub = self.create_publisher(PointCloud2, self.output_topic, 10)
        self.debug_pub = self.create_publisher(String, "~/debug", 5)
        self.create_timer(1.0 / self.publish_rate_hz, self._timer_cb)
        self.get_logger().info(
            f"LiDAR accumulator: {self.input_topic} -> {self.output_topic}, "
            f"window={self.history_duration.nanoseconds/1e9:.2f}s, max_clouds={self.max_clouds}"
        )

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

        if now - self.last_debug > Duration(seconds=1.0):
            self.last_debug = now
            self.debug_pub.publish(String(data=f"clouds={len(self.history)}, points={len(all_points)}"))


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
