#!/usr/bin/env python3
"""Re-publish Unitree LiDAR points in the robot frame.

The LiDAR on Go2 is mounted with a fixed tilt relative to the body frame.
In the larger autonomy stack this correction is done by the ``transform_sensors``
utility.  This lightweight clone applies the same hard-coded transform before
the local planner consumes the scan data.
"""

from __future__ import annotations

from typing import List, Tuple

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf_transformations import euler_matrix, quaternion_from_euler


class UTLidarTransformer(Node):
    """Rotate/translate incoming clouds so they align with ``path_frame``."""

    def __init__(self) -> None:
        super().__init__("utlidar_transformer")
        self.declare_parameter("input_topic", "/utlidar/cloud")
        self.declare_parameter("output_topic", "/utlidar/transformed_cloud")
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("lidar_frame", "utlidar_lidar")
        self.declare_parameter("lidar_x", 0.0)
        self.declare_parameter("lidar_y", 0.0)
        self.declare_parameter("lidar_z", 0.0)
        self.declare_parameter("lidar_roll", 0.0)
        self.declare_parameter("lidar_pitch", 0.0)
        self.declare_parameter("lidar_yaw", 0.0)

        self.input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        self.output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self.target_frame = self.get_parameter("target_frame").get_parameter_value().string_value
        self.lidar_frame = self.get_parameter("lidar_frame").get_parameter_value().string_value
        self.translation = (
            self.get_parameter("lidar_x").get_parameter_value().double_value,
            self.get_parameter("lidar_y").get_parameter_value().double_value,
            self.get_parameter("lidar_z").get_parameter_value().double_value,
        )
        roll = self.get_parameter("lidar_roll").get_parameter_value().double_value
        pitch = self.get_parameter("lidar_pitch").get_parameter_value().double_value
        yaw = self.get_parameter("lidar_yaw").get_parameter_value().double_value
        self.rotation = euler_matrix(roll, pitch, yaw)[:3, :3]

        self.publisher = self.create_publisher(PointCloud2, self.output_topic, 10)
        self.subscription = self.create_subscription(
            PointCloud2, self.input_topic, self._cloud_callback, 10
        )
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self._broadcast_static_tf(roll, pitch, yaw)

        self.get_logger().info(
            f"UTLiDAR transformer publishing {self.input_topic} -> {self.output_topic} "
            f"with RPY=({roll:.3f}, {pitch:.3f}, {yaw:.3f})"
        )

    # ------------------------------------------------------------------ Callbacks
    def _cloud_callback(self, cloud: PointCloud2) -> None:
        points = list(point_cloud2.read_points(cloud, skip_nans=True))
        if not points:
            return

        transformed_points: List[Tuple[float, ...]] = []
        rot = self.rotation
        tx, ty, tz = self.translation

        for entry in points:
            x, y, z, *rest = entry
            x_new = rot[0, 0] * x + rot[0, 1] * y + rot[0, 2] * z + tx
            y_new = rot[1, 0] * x + rot[1, 1] * y + rot[1, 2] * z + ty
            z_new = rot[2, 0] * x + rot[2, 1] * y + rot[2, 2] * z + tz
            transformed_points.append((x_new, y_new, z_new, *rest))

        msg = point_cloud2.create_cloud(cloud.header, cloud.fields, transformed_points)
        msg.header.frame_id = self.target_frame
        self.publisher.publish(msg)

    def _broadcast_static_tf(self, roll: float, pitch: float, yaw: float) -> None:
        quat = quaternion_from_euler(roll, pitch, yaw)
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.target_frame
        transform.child_frame_id = self.lidar_frame
        transform.transform.translation.x = self.translation[0]
        transform.transform.translation.y = self.translation[1]
        transform.transform.translation.z = self.translation[2]
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]
        self.tf_broadcaster.sendTransform(transform)
        self.get_logger().info(
            f"Broadcasting static TF {self.target_frame} -> {self.lidar_frame} "
            f"with translation=({self.translation[0]:.3f}, {self.translation[1]:.3f}, {self.translation[2]:.3f}) "
            f"and RPY=({roll:.3f}, {pitch:.3f}, {yaw:.3f})"
        )



def main() -> None:
    rclpy.init()
    try:
        transformer = UTLidarTransformer()
        rclpy.spin(transformer)
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
