#!/usr/bin/env python3
"""Path follower that triggers Unitree Go2 sport API only while a target is active."""

from __future__ import annotations

import json
import math
from typing import Optional, Sequence, Tuple

import rclpy
from geometry_msgs.msg import PointStamped, TwistStamped
from nav_msgs.msg import Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from unitree_api.msg import Request


class Go2PathFollowerNode(Node):
    def __init__(self) -> None:
        super().__init__("go2_path_follower")
        self.declare_parameter("path_topic", "/path")
        self.declare_parameter("target_topic", "/local_goal_point")
        self.declare_parameter("target_status_topic", "/target/status")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("sport_request_topic", "/api/sport/request")
        self.declare_parameter("cmd_frame", "base_link")
        self.declare_parameter("publish_rate_hz", 30.0)

        self.declare_parameter("api_control_enabled", True)
        self.declare_parameter("manual_release_when_no_target", True)
        self.declare_parameter("send_stop_on_release", True)
        self.declare_parameter("target_stale_timeout_s", 0.6)
        self.declare_parameter("path_stale_timeout_s", 0.6)
        self.declare_parameter("follow_distance_m", 2.0)
        self.declare_parameter("goal_tolerance_m", 0.18)
        self.declare_parameter("lookahead_distance_m", 0.45)

        self.declare_parameter("max_forward_speed_mps", 0.60)
        self.declare_parameter("max_lateral_speed_mps", 0.35)
        self.declare_parameter("speed_gain", 0.75)
        self.declare_parameter("slowdown_yaw_error_rad", 0.65)
        self.declare_parameter("yaw_gain", 1.6)
        self.declare_parameter("max_yaw_rate_rps", 1.0)
        self.declare_parameter("yaw_deadband_rad", 0.03)

        self.path: Optional[Path] = None
        self.path_time: Optional[Time] = None
        self.target_xy: Optional[Tuple[float, float]] = None
        self.target_time: Optional[Time] = None
        self.status_tracked: bool = False
        self.status_time: Optional[Time] = None
        self.autonomy_active: bool = False
        self.seq = 0

        self.create_subscription(Path, str(self.get_parameter("path_topic").value), self._path_cb, 5)
        self.create_subscription(PointStamped, str(self.get_parameter("target_topic").value), self._target_cb, 5)
        self.create_subscription(String, str(self.get_parameter("target_status_topic").value), self._status_cb, 10)
        self.cmd_pub = self.create_publisher(TwistStamped, str(self.get_parameter("cmd_vel_topic").value), 10)
        self.req_pub = self.create_publisher(Request, str(self.get_parameter("sport_request_topic").value), 10)
        rate = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self.create_timer(1.0 / rate, self._timer_cb)
        self.get_logger().info("Go2 path follower ready. API commands are gated by active target status.")

    # ---------------------------------------------------------------- callbacks
    def _path_cb(self, msg: Path) -> None:
        self.path = msg
        self.path_time = self.get_clock().now()

    def _target_cb(self, msg: PointStamped) -> None:
        self.target_xy = (float(msg.point.x), float(msg.point.y))
        self.target_time = self.get_clock().now()

    def _status_cb(self, msg: String) -> None:
        self.status_time = self.get_clock().now()
        try:
            payload = json.loads(msg.data)
            self.status_tracked = bool(payload.get("tracked", False))
        except Exception:
            self.status_tracked = False

    # ---------------------------------------------------------------- timer
    def _timer_cb(self) -> None:
        now = self.get_clock().now()
        active = self._target_is_active(now)
        if not active:
            if self.autonomy_active:
                self.get_logger().warn("Autonomy released: target/path inactive. Manual control is no longer overwritten.")
                if bool(self.get_parameter("send_stop_on_release").value):
                    self._publish_zero(stop_api_once=True)
            self.autonomy_active = False
            return

        self.autonomy_active = True
        cmd = self._compute_cmd(now)
        self.cmd_pub.publish(cmd)
        if bool(self.get_parameter("api_control_enabled").value):
            self._publish_sport_request(cmd)

    def _target_is_active(self, now: Time) -> bool:
        if not self.status_tracked:
            return False
        target_timeout = Duration(seconds=float(self.get_parameter("target_stale_timeout_s").value))
        path_timeout = Duration(seconds=float(self.get_parameter("path_stale_timeout_s").value))
        if self.status_time is None or now - self.status_time > target_timeout:
            return False
        if self.target_time is None or now - self.target_time > target_timeout:
            return False
        if self.path_time is None or now - self.path_time > path_timeout:
            return False
        if self.path is None or len(self.path.poses) == 0:
            return False
        return True

    def _compute_cmd(self, now: Time) -> TwistStamped:
        cmd = TwistStamped()
        cmd.header.stamp = now.to_msg()
        cmd.header.frame_id = str(self.get_parameter("cmd_frame").value)

        path = self.path
        target_xy = self.target_xy
        if path is None or target_xy is None:
            return cmd

        target_distance = math.hypot(target_xy[0], target_xy[1])
        follow_distance = float(self.get_parameter("follow_distance_m").value)
        heading_error = math.atan2(target_xy[1], target_xy[0])
        if abs(heading_error) <= float(self.get_parameter("yaw_deadband_rad").value):
            yaw_rate = 0.0
        else:
            yaw_rate = self._clip(
                float(self.get_parameter("yaw_gain").value) * heading_error,
                -float(self.get_parameter("max_yaw_rate_rps").value),
                float(self.get_parameter("max_yaw_rate_rps").value),
            )
        cmd.twist.angular.z = yaw_rate

        if target_distance <= follow_distance + float(self.get_parameter("goal_tolerance_m").value):
            return cmd

        lookahead = float(self.get_parameter("lookahead_distance_m").value)
        goal_xy = self._select_lookahead_point(path, lookahead)
        path_distance = math.hypot(goal_xy[0], goal_xy[1])
        if path_distance <= float(self.get_parameter("goal_tolerance_m").value):
            return cmd

        speed = min(float(self.get_parameter("max_forward_speed_mps").value), float(self.get_parameter("speed_gain").value) * path_distance)
        slowdown_error = max(1e-3, float(self.get_parameter("slowdown_yaw_error_rad").value))
        yaw_slowdown = max(0.15, 1.0 - min(1.0, abs(heading_error) / slowdown_error) * 0.65)
        speed *= yaw_slowdown

        path_heading = math.atan2(goal_xy[1], goal_xy[0])
        vx = speed * math.cos(path_heading)
        vy = speed * math.sin(path_heading)
        max_lat = float(self.get_parameter("max_lateral_speed_mps").value)
        cmd.twist.linear.x = self._clip(vx, -float(self.get_parameter("max_forward_speed_mps").value), float(self.get_parameter("max_forward_speed_mps").value))
        cmd.twist.linear.y = self._clip(vy, -max_lat, max_lat)
        return cmd

    def _select_lookahead_point(self, path: Path, lookahead: float) -> Tuple[float, float]:
        if not path.poses:
            return (0.0, 0.0)
        for pose in path.poses:
            p = pose.pose.position
            if math.hypot(p.x, p.y) >= lookahead:
                return (float(p.x), float(p.y))
        p = path.poses[-1].pose.position
        return (float(p.x), float(p.y))

    # ---------------------------------------------------------------- publish
    def _publish_zero(self, *, stop_api_once: bool = False) -> None:
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = str(self.get_parameter("cmd_frame").value)
        self.cmd_pub.publish(cmd)
        if stop_api_once and bool(self.get_parameter("api_control_enabled").value):
            req = Request()
            req.header.identity.id = self.seq
            self.seq += 1
            req.header.identity.api_id = 1003  # ROBOT_SPORT_API_ID_STOPMOVE
            req.parameter = ""
            self.req_pub.publish(req)

    def _publish_sport_request(self, cmd: TwistStamped) -> None:
        req = Request()
        req.header.identity.id = self.seq
        self.seq += 1
        is_zero = (
            abs(cmd.twist.linear.x) < 1e-4
            and abs(cmd.twist.linear.y) < 1e-4
            and abs(cmd.twist.angular.z) < 1e-4
        )
        if is_zero:
            req.header.identity.api_id = 1003  # ROBOT_SPORT_API_ID_STOPMOVE
            req.parameter = ""
        else:
            req.header.identity.api_id = 1008  # ROBOT_SPORT_API_ID_MOVE
            req.parameter = json.dumps(
                {
                    "x": float(cmd.twist.linear.x),
                    "y": float(cmd.twist.linear.y),
                    "z": float(cmd.twist.angular.z),
                }
            )
        self.req_pub.publish(req)

    @staticmethod
    def _clip(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = Go2PathFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
