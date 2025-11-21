#!/usr/bin/env python3
"""Aggregate LiDAR scans over a short window to densify local observations."""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Sequence, Tuple

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, String
from std_srvs.srv import Trigger


class SlidingWindowLidarAccumulator(Node):
    """Collect LiDAR scans over a configurable time window and republish them."""

    def __init__(self) -> None:
        super().__init__("lidar_accumulator")
        self.declare_parameter("input_topic", "/utlidar/transformed_cloud")
        self.declare_parameter("output_topic", "/utlidar/accumulated_cloud")
        self.declare_parameter("history_duration", 0.8)
        self.declare_parameter("max_clouds", 10)
        self.declare_parameter("publish_rate_hz", 15.0)

        self.input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        self.output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        history_seconds = self.get_parameter("history_duration").get_parameter_value().double_value
        # Keep the duration positive but allow small windows.
        history_seconds = max(0.01, history_seconds)
        self.history_duration = Duration(seconds=history_seconds)
        self.max_clouds = max(1, int(self.get_parameter("max_clouds").value))
        publish_rate = max(1e-2, self.get_parameter("publish_rate_hz").get_parameter_value().double_value)

        self._history: Deque[Tuple[Time, List[Tuple[float, ...]]]] = deque()
        self._history_point_count = 0
        self._last_frame_id: Optional[str] = None
        self._fields: Optional[List] = None
        self._last_input_stamp: Optional[Time] = None
        self._last_publish_stamp: Optional[Time] = None
        self._total_received = 0
        self._total_published = 0
        self._last_debug_time = self.get_clock().now()

        self._subscription = self.create_subscription(
            PointCloud2, self.input_topic, self._cloud_callback, 10
        )
        self._publisher = self.create_publisher(PointCloud2, self.output_topic, 10)
        self._debug_pub = self.create_publisher(String, "~/debug", 5)
        self.create_service(Trigger, "~/dump_status", self._on_dump_status)
        self._timer = self.create_timer(1.0 / publish_rate, self._on_timer)

        self.get_logger().info(
            "LiDAR accumulator ready "
            f"(window={history_seconds:.2f}s, max_clouds={self.max_clouds}, "
            f"{self.input_topic} -> {self.output_topic})"
        )

    # ------------------------------------------------------------------ Helpers
    def _cloud_callback(self, cloud: PointCloud2) -> None:
        """Cache the incoming cloud and immediately prune old entries."""
        raw_points = list(point_cloud2.read_points(cloud, skip_nans=True))
        now = self.get_clock().now()
        if not raw_points:
            self._prune_history(now)
            return

        self._last_frame_id = cloud.header.frame_id or self._last_frame_id
        if self._fields is None:
            # Preserve the original point layout (x/y/z/intensity/...) so downstream nodes stay compatible.
            self._fields = list(cloud.fields)

        arrival_stamp = now
        self._history.append((arrival_stamp, raw_points))
        self._history_point_count += len(raw_points)
        while len(self._history) > self.max_clouds:
            _, removed = self._history.popleft()
            self._history_point_count -= len(removed)
        self._prune_history(now)
        self._total_received += 1
        self._last_input_stamp = arrival_stamp
        self._maybe_publish_debug(now, event="scan")

    def _prune_history(self, reference_time: Time) -> None:
        """Drop cached scans that fall outside the configured window."""
        while self._history:
            stamp, points = self._history[0]
            if reference_time - stamp <= self.history_duration:
                break
            self._history.popleft()
            self._history_point_count -= len(points)

    def _on_timer(self) -> None:
        now = self.get_clock().now()
        self._prune_history(now)

        accumulated: List[Tuple[float, ...]] = []
        for _, points in self._history:
            accumulated.extend(points)

        header = Header()
        header.stamp = now.to_msg()
        header.frame_id = self._last_frame_id or ""

        if accumulated:
            if self._fields:
                msg = point_cloud2.create_cloud(header, self._fields, accumulated)
            else:
                msg = point_cloud2.create_cloud_xyz32(header, accumulated)
        else:
            msg = PointCloud2()
            msg.header = header

        self._publisher.publish(msg)
        self._last_publish_stamp = now
        self._total_published += 1
        self._maybe_publish_debug(now, event="publish")

    # ------------------------------------------------------------------ Debug helpers
    def _summary_string(self, current_time: Optional[Time] = None) -> str:
        now = current_time or self.get_clock().now()
        point_count = self._history_point_count
        history_age = (
            (now - self._history[0][0]).nanoseconds / 1e9 if self._history else 0.0
        )
        last_input_age = (
            (now - self._last_input_stamp).nanoseconds / 1e9
            if self._last_input_stamp is not None
            else None
        )
        last_publish_age = (
            (now - self._last_publish_stamp).nanoseconds / 1e9
            if self._last_publish_stamp is not None
            else None
        )
        return (
            f"history_len={len(self._history)}, points={point_count}, "
            f"window={self.history_duration.nanoseconds / 1e9:.2f}s, "
            f"oldest_age={history_age:.3f}s, "
            f"last_input_age={'n/a' if last_input_age is None else f'{last_input_age:.3f}s'}, "
            f"last_publish_age={'n/a' if last_publish_age is None else f'{last_publish_age:.3f}s'}, "
            f"received={self._total_received}, published={self._total_published}"
        )

    def _maybe_publish_debug(self, now: Time, *, event: str) -> None:
        if (now - self._last_debug_time).nanoseconds < 200_000_000:
            return
        self._last_debug_time = now
        summary = f"[{event}] " + self._summary_string(now)
        self._debug_pub.publish(String(data=summary))
        self.get_logger().debug(summary)

    def _on_dump_status(self, request, response):
        summary = self._summary_string()
        response.success = True
        response.message = summary
        self.get_logger().info(f"Status requested via CLI: {summary}")
        return response


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = SlidingWindowLidarAccumulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
