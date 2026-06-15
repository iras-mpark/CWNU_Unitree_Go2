#!/usr/bin/env python3
"""Fixed transform for Unitree LiDAR point clouds into the local planning frame."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf_transformations import euler_matrix, quaternion_from_euler


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

        self.pub = self.create_publisher(PointCloud2, self.output_topic, 10)
        self.create_subscription(PointCloud2, self.input_topic, self._cloud_cb, 10)
        self.tf_pub = StaticTransformBroadcaster(self)
        self._broadcast_static_tf()
        self.get_logger().info(
            f"LiDAR transformer: {self.input_topic} -> {self.output_topic}, "
            f"frame={self.target_frame}, rpy=({self.roll:.3f},{self.pitch:.3f},{self.yaw:.3f})"
        )

    def _cloud_cb(self, cloud: PointCloud2) -> None:
        points = list(point_cloud2.read_points(cloud, skip_nans=True))
        if not points:
            out = PointCloud2()
            out.header = cloud.header
            out.header.frame_id = self.target_frame
            self.pub.publish(out)
            return

        transformed: List[Tuple[float, ...]] = []
        for p in points:
            xyz = np.array([p[0], p[1], p[2]], dtype=np.float64)
            xyz_new = self.rotation.dot(xyz) + self.translation
            transformed.append((float(xyz_new[0]), float(xyz_new[1]), float(xyz_new[2]), *p[3:]))

        msg = point_cloud2.create_cloud(cloud.header, cloud.fields, transformed)
        msg.header.frame_id = self.target_frame
        self.pub.publish(msg)

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
