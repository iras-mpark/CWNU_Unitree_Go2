#!/usr/bin/env python3
"""Path follower for Go2 sport API.

This node intentionally keeps the Unitree sport Request message format compatible
with the original CWNU package.  Autonomy is enabled by a fresh target point and
fresh path.  /target/status can be used as an additional gate, but it is disabled
by default because the original working package did not require it and multi-PC
ROS networks often miss auxiliary debug/status topics during early integration.
"""

from __future__ import annotations

import json
import math
from typing import List, Optional, Sequence, Tuple

import rclpy
from geometry_msgs.msg import PointStamped, TwistStamped
from nav_msgs.msg import Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String

try:
    from unitree_api.msg import Request
    HAS_UNITREE_API = True
except ModuleNotFoundError:
    Request = None
    HAS_UNITREE_API = False


class Go2PathFollowerNode(Node):
    def __init__(self) -> None:
        super().__init__("go2_path_follower")
        self.declare_parameter("path_topic", "/path")
        self.declare_parameter("target_topic", "/local_goal_point")
        self.declare_parameter("target_status_topic", "/target/status")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("sport_request_topic", "/api/sport/request")
        self.declare_parameter("follower_status_topic", "/go2_follower/status")
        self.declare_parameter("safety_scan_topic", "/utlidar/transformed_cloud")
        self.declare_parameter("cmd_frame", "base_link")
        self.declare_parameter("publish_rate_hz", 30.0)

        self.declare_parameter("api_control_enabled", True)
        # Keep this false by default.  The target PointStamped itself is the
        # control trigger.  Requiring /target/status made the robot refuse to
        # move even when /local_goal_point and /path were valid.
        self.declare_parameter("require_target_status", False)
        self.declare_parameter("manual_release_when_no_target", True)
        self.declare_parameter("send_stop_on_release", True)
        self.declare_parameter("target_stale_timeout_s", 0.8)
        self.declare_parameter("status_stale_timeout_s", 1.0)
        self.declare_parameter("path_stale_timeout_s", 0.8)
        self.declare_parameter("follow_distance_m", 2.0)
        self.declare_parameter("goal_tolerance_m", 0.18)
        self.declare_parameter("lookahead_distance_m", 0.45)

        self.declare_parameter("max_forward_speed_mps", 3.7)
        self.declare_parameter("max_lateral_speed_mps", 0.30)
        self.declare_parameter("distance_pid_kp", 0.70)
        self.declare_parameter("distance_pid_ki", 0.0)
        self.declare_parameter("distance_pid_kd", 0.08)
        self.declare_parameter("distance_integral_limit", 1.0)
        self.declare_parameter("min_tracking_speed_mps", 0.12)
        self.declare_parameter("linear_accel_limit_mps2", 1.5)
        self.declare_parameter("linear_decel_limit_mps2", 4.0)
        self.declare_parameter("high_speed_safety_enabled", True)
        self.declare_parameter("safety_scan_timeout_s", 0.15)
        self.declare_parameter("safety_obstacle_z_min_m", 0.10)
        self.declare_parameter("safety_obstacle_z_min_negative_x_m", 0.10)
        self.declare_parameter("safety_obstacle_z_max_m", 1.20)
        self.declare_parameter("robot_collision_radius_m", 0.32)
        self.declare_parameter("obstacle_stop_margin_m", 0.45)
        self.declare_parameter("control_reaction_time_s", 0.20)
        self.declare_parameter("max_braking_decel_mps2", 4.0)
        self.declare_parameter("ttc_emergency_stop_s", 0.55)
        self.declare_parameter("data_age_slowdown_s", 0.20)
        self.declare_parameter("stale_data_speed_limit_mps", 0.80)
        self.declare_parameter("turn_slowdown_start_rad", 0.08)
        self.declare_parameter("turn_speed_limit_heading_rad", 0.25)
        self.declare_parameter("max_turning_speed_mps", 1.20)
        self.declare_parameter("slowdown_yaw_error_rad", 0.65)
        # Yaw control is deliberately damped.  Person detections and depth-derived
        # target points jitter frame-to-frame; without a deadband/filter/slew limit
        # the Go2 keeps twisting left and right while trying to keep the person at
        # the exact image center.
        self.declare_parameter("yaw_gain", 0.9)
        self.declare_parameter("max_yaw_rate_rps", 0.55)
        self.declare_parameter("yaw_deadband_rad", 0.12)
        self.declare_parameter("yaw_filter_alpha", 0.25)
        self.declare_parameter("yaw_slew_rate_limit_rps2", 1.2)
        self.declare_parameter("yaw_hold_when_close", True)

        self.path: Optional[Path] = None
        self.path_time: Optional[Time] = None
        self.target_xyz: Optional[Tuple[float, float, float]] = None
        self.target_time: Optional[Time] = None
        self.status_tracked: bool = False
        self.status_time: Optional[Time] = None
        self.autonomy_active: bool = False
        self.seq = 0
        self._last_inactive_reason = "initializing"
        self._last_diag_time = self.get_clock().now() - Duration(seconds=10.0)
        self._filtered_heading_error: Optional[float] = None
        self._last_yaw_rate: float = 0.0
        self._last_cmd_time: Optional[Time] = None
        self._distance_integral: float = 0.0
        self._last_distance_error: Optional[float] = None
        self._last_distance_pid_time: Optional[Time] = None
        self._last_linear_cmd: Tuple[float, float] = (0.0, 0.0)
        self._last_linear_cmd_time: Optional[Time] = None
        self.safety_obstacles_xy: List[Tuple[float, float]] = []
        self.safety_scan_time: Optional[Time] = None
        self._safety_debug = {}

        self.create_subscription(Path, str(self.get_parameter("path_topic").value), self._path_cb, 5)
        self.create_subscription(PointStamped, str(self.get_parameter("target_topic").value), self._target_cb, 5)
        self.create_subscription(String, str(self.get_parameter("target_status_topic").value), self._status_cb, 10)
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("safety_scan_topic").value),
            self._safety_scan_cb,
            10,
        )
        self.cmd_pub = self.create_publisher(TwistStamped, str(self.get_parameter("cmd_vel_topic").value), 10)
        self.status_pub = self.create_publisher(String, str(self.get_parameter("follower_status_topic").value), 10)
        self.req_pub = None
        if HAS_UNITREE_API:
            self.req_pub = self.create_publisher(Request, str(self.get_parameter("sport_request_topic").value), 10)
        elif bool(self.get_parameter("api_control_enabled").value):
            self.get_logger().error(
                "unitree_api.msg.Request is not importable. /cmd_vel will still be published, "
                "but /api/sport/request cannot be used. Source/build the Unitree ROS2 interface "
                "package, or run with api_control_enabled:=false for planner-only debugging."
            )

        rate = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self.create_timer(1.0 / rate, self._timer_cb)
        self.get_logger().info(
            "Go2 path follower ready. Control is gated by fresh /local_goal_point and /path; "
            f"require_target_status={bool(self.get_parameter('require_target_status').value)}, "
            f"unitree_api_available={HAS_UNITREE_API}."
        )

    # ---------------------------------------------------------------- callbacks
    def _path_cb(self, msg: Path) -> None:
        self.path = msg
        self.path_time = self.get_clock().now()

    def _target_cb(self, msg: PointStamped) -> None:
        self.target_xyz = (float(msg.point.x), float(msg.point.y), float(msg.point.z))
        self.target_time = self.get_clock().now()

    def _status_cb(self, msg: String) -> None:
        self.status_time = self.get_clock().now()
        try:
            payload = json.loads(msg.data)
            self.status_tracked = bool(payload.get("tracked", False))
        except Exception:
            self.status_tracked = False

    def _safety_scan_cb(self, cloud: PointCloud2) -> None:
        z_min_pos = float(self.get_parameter("safety_obstacle_z_min_m").value)
        z_min_neg = float(self.get_parameter("safety_obstacle_z_min_negative_x_m").value)
        z_max_param = float(self.get_parameter("safety_obstacle_z_max_m").value)
        obstacles = []
        for x, y, z in point_cloud2.read_points(
            cloud, field_names=("x", "y", "z"), skip_nans=True
        ):
            x = float(x)
            y = float(y)
            z = float(z)
            z_min = z_min_pos if x >= 0.0 else z_min_neg
            z_low, z_high = min(z_min, z_max_param), max(z_min, z_max_param)
            if z_low <= z <= z_high and math.isfinite(x) and math.isfinite(y):
                obstacles.append((x, y))
        self.safety_obstacles_xy = obstacles
        self.safety_scan_time = self.get_clock().now()

    # ---------------------------------------------------------------- timer
    def _timer_cb(self) -> None:
        now = self.get_clock().now()
        active, reason = self._target_is_active(now)
        self._last_inactive_reason = reason
        if not active:
            if self.autonomy_active:
                self.get_logger().warn(f"Autonomy released: {reason}. Manual control is no longer overwritten.")
                if bool(self.get_parameter("send_stop_on_release").value):
                    self._publish_zero(stop_api_once=True)
            self.autonomy_active = False
            self._filtered_heading_error = None
            self._last_yaw_rate = 0.0
            self._last_cmd_time = None
            self._reset_distance_controller()
            self._publish_follower_status(now, False, reason, None)
            self._diagnose_inactive(now, reason)
            return

        self.autonomy_active = True
        cmd = self._compute_cmd(now)
        self.cmd_pub.publish(cmd)
        if bool(self.get_parameter("api_control_enabled").value) and HAS_UNITREE_API and self.req_pub is not None:
            self._publish_sport_request(cmd)
        self._publish_follower_status(now, True, "active", cmd)

    def _target_is_active(self, now: Time) -> Tuple[bool, str]:
        target_timeout = Duration(seconds=float(self.get_parameter("target_stale_timeout_s").value))
        path_timeout = Duration(seconds=float(self.get_parameter("path_stale_timeout_s").value))
        status_timeout = Duration(seconds=float(self.get_parameter("status_stale_timeout_s").value))

        if self.target_time is None or self.target_xyz is None:
            return False, "no target point received"
        if now - self.target_time > target_timeout:
            return False, "target point stale"
        if self.path_time is None or self.path is None:
            return False, "no path received"
        if now - self.path_time > path_timeout:
            return False, "path stale"
        if len(self.path.poses) == 0:
            return False, "empty path"

        if bool(self.get_parameter("high_speed_safety_enabled").value):
            scan_timeout = Duration(seconds=float(self.get_parameter("safety_scan_timeout_s").value))
            if self.safety_scan_time is None:
                return False, "safety lidar missing"
            if now - self.safety_scan_time > scan_timeout:
                return False, "safety lidar stale"

        if bool(self.get_parameter("require_target_status").value):
            if self.status_time is None:
                return False, "target status missing"
            if now - self.status_time > status_timeout:
                return False, "target status stale"
            if not self.status_tracked:
                return False, "target status tracked=false"

        return True, "active"

    def _compute_cmd(self, now: Time) -> TwistStamped:
        cmd = TwistStamped()
        cmd.header.stamp = now.to_msg()
        cmd.header.frame_id = str(self.get_parameter("cmd_frame").value)

        path = self.path
        target_xyz = self.target_xyz
        if path is None or target_xyz is None:
            return cmd

        target_xy = (target_xyz[0], target_xyz[1])
        target_distance = math.hypot(target_xy[0], target_xy[1])
        follow_distance = float(self.get_parameter("follow_distance_m").value)
        heading_error = math.atan2(target_xy[1], target_xy[0])
        cmd.twist.angular.z = self._compute_yaw_rate(now, heading_error, target_distance)

        # If already within the desired person-following distance, only yaw-align.
        # With yaw_hold_when_close=true, _compute_yaw_rate() also applies a wider
        # practical deadband near the target so the robot does not keep wiggling
        # while already standing at the desired following distance.
        if target_distance <= follow_distance + float(self.get_parameter("goal_tolerance_m").value):
            self._reset_distance_controller()
            return cmd

        lookahead = float(self.get_parameter("lookahead_distance_m").value)
        goal_xy = self._select_lookahead_point(path, lookahead)
        path_distance = math.hypot(goal_xy[0], goal_xy[1])
        if path_distance <= float(self.get_parameter("goal_tolerance_m").value):
            return cmd

        distance_error = max(0.0, target_distance - follow_distance)
        speed = self._compute_tracking_speed(now, distance_error)
        slowdown_error = max(1e-3, float(self.get_parameter("slowdown_yaw_error_rad").value))
        yaw_slowdown = max(0.15, 1.0 - min(1.0, abs(heading_error) / slowdown_error) * 0.65)
        speed *= yaw_slowdown

        path_heading = math.atan2(goal_xy[1], goal_xy[0])
        vx = speed * math.cos(path_heading)
        vy = speed * math.sin(path_heading)
        max_lat = float(self.get_parameter("max_lateral_speed_mps").value)
        vx = self._clip(
            vx,
            -float(self.get_parameter("max_forward_speed_mps").value),
            float(self.get_parameter("max_forward_speed_mps").value),
        )
        vy = self._clip(vy, -max_lat, max_lat)
        vx, vy = self._limit_linear_acceleration(now, vx, vy)
        cmd.twist.linear.x = vx
        cmd.twist.linear.y = vy
        self._apply_safety_limits(now, cmd, distance_error, path_heading)
        return cmd

    def _apply_safety_limits(
        self, now: Time, cmd: TwistStamped, distance_error: float, path_heading: float
    ) -> None:
        if not bool(self.get_parameter("high_speed_safety_enabled").value):
            self._safety_debug = {"enabled": False}
            return

        vx = float(cmd.twist.linear.x)
        vy = float(cmd.twist.linear.y)
        requested_speed = math.hypot(vx, vy)
        if requested_speed <= 1e-6:
            self._safety_debug = {"enabled": True, "requested_speed_mps": 0.0}
            return

        speed_cap = float(self.get_parameter("max_forward_speed_mps").value)
        cap_reasons = []
        braking_decel = max(0.1, float(self.get_parameter("max_braking_decel_mps2").value))
        reaction_time = max(0.0, float(self.get_parameter("control_reaction_time_s").value))

        target_available = max(0.0, distance_error - float(self.get_parameter("goal_tolerance_m").value))
        target_cap = self._stopping_speed_cap(target_available, braking_decel, reaction_time)
        if target_cap < speed_cap:
            speed_cap = target_cap
            cap_reasons.append("target_stopping_distance")

        heading = abs(path_heading)
        turn_start = max(0.0, float(self.get_parameter("turn_slowdown_start_rad").value))
        turn_full = max(turn_start + 1e-3, float(self.get_parameter("turn_speed_limit_heading_rad").value))
        turn_limit = max(0.0, float(self.get_parameter("max_turning_speed_mps").value))
        if heading > turn_start:
            ratio = self._clip((heading - turn_start) / (turn_full - turn_start), 0.0, 1.0)
            turn_cap = (1.0 - ratio) * speed_cap + ratio * min(speed_cap, turn_limit)
            if turn_cap < speed_cap:
                speed_cap = turn_cap
                cap_reasons.append("path_heading")

        age_slowdown = max(0.0, float(self.get_parameter("data_age_slowdown_s").value))
        target_age = self._age_s(now, self.target_time) or 0.0
        path_age = self._age_s(now, self.path_time) or 0.0
        if max(target_age, path_age) > age_slowdown:
            stale_cap = max(0.0, float(self.get_parameter("stale_data_speed_limit_mps").value))
            if stale_cap < speed_cap:
                speed_cap = stale_cap
                cap_reasons.append("data_age")

        ux = vx / requested_speed
        uy = vy / requested_speed
        corridor_radius = max(0.05, float(self.get_parameter("robot_collision_radius_m").value))
        clearance = float("inf")
        for ox, oy in self.safety_obstacles_xy:
            longitudinal = ox * ux + oy * uy
            lateral = abs(-ox * uy + oy * ux)
            if longitudinal > 0.0 and lateral <= corridor_radius:
                clearance = min(clearance, longitudinal)

        stop_margin = max(0.0, float(self.get_parameter("obstacle_stop_margin_m").value))
        current_speed = max(requested_speed, math.hypot(*self._last_linear_cmd))
        ttc = float("inf") if not math.isfinite(clearance) else clearance / max(current_speed, 1e-3)
        emergency_ttc = max(0.0, float(self.get_parameter("ttc_emergency_stop_s").value))
        emergency = math.isfinite(clearance) and (clearance <= stop_margin or ttc <= emergency_ttc)

        if math.isfinite(clearance):
            obstacle_available = max(0.0, clearance - stop_margin)
            obstacle_cap = self._stopping_speed_cap(obstacle_available, braking_decel, reaction_time)
            if obstacle_cap < speed_cap:
                speed_cap = obstacle_cap
                cap_reasons.append("obstacle_stopping_distance")

        if emergency:
            cmd.twist.linear.x = 0.0
            cmd.twist.linear.y = 0.0
            cmd.twist.angular.z = 0.0
            self._last_linear_cmd = (0.0, 0.0)
            cap_reasons.append("ttc_emergency_stop")
        elif requested_speed > speed_cap:
            scale = max(0.0, speed_cap) / requested_speed
            cmd.twist.linear.x *= scale
            cmd.twist.linear.y *= scale
            self._last_linear_cmd = (float(cmd.twist.linear.x), float(cmd.twist.linear.y))

        self._safety_debug = {
            "enabled": True,
            "requested_speed_mps": requested_speed,
            "speed_cap_mps": speed_cap,
            "clearance_m": None if not math.isfinite(clearance) else clearance,
            "ttc_s": None if not math.isfinite(ttc) else ttc,
            "emergency_stop": emergency,
            "cap_reasons": cap_reasons,
            "scan_age_s": self._age_s(now, self.safety_scan_time),
        }

    @staticmethod
    def _stopping_speed_cap(available_distance: float, decel: float, reaction_time: float) -> float:
        # Solve v * reaction_time + v^2 / (2 * decel) <= available_distance.
        return max(0.0, -decel * reaction_time + math.sqrt(
            (decel * reaction_time) ** 2 + 2.0 * decel * max(0.0, available_distance)
        ))

    def _compute_tracking_speed(self, now: Time, distance_error: float) -> float:
        """Compute translational speed from distance-to-follow error."""
        if self._last_distance_pid_time is None or self._last_distance_error is None:
            dt = 0.0
            derivative = 0.0
        else:
            dt = max(1e-3, float((now - self._last_distance_pid_time).nanoseconds) * 1e-9)
            derivative = (distance_error - self._last_distance_error) / dt

        if dt > 0.0:
            self._distance_integral += distance_error * dt
            integral_limit = max(0.0, float(self.get_parameter("distance_integral_limit").value))
            self._distance_integral = self._clip(
                self._distance_integral, -integral_limit, integral_limit
            )

        speed = (
            float(self.get_parameter("distance_pid_kp").value) * distance_error
            + float(self.get_parameter("distance_pid_ki").value) * self._distance_integral
            + float(self.get_parameter("distance_pid_kd").value) * derivative
        )
        max_speed = max(0.0, float(self.get_parameter("max_forward_speed_mps").value))
        min_speed = self._clip(
            float(self.get_parameter("min_tracking_speed_mps").value), 0.0, max_speed
        )

        self._last_distance_error = distance_error
        self._last_distance_pid_time = now
        return self._clip(speed, min_speed, max_speed)

    def _limit_linear_acceleration(self, now: Time, vx: float, vy: float) -> Tuple[float, float]:
        accel_limit = max(0.0, float(self.get_parameter("linear_accel_limit_mps2").value))
        decel_limit = max(accel_limit, float(self.get_parameter("linear_decel_limit_mps2").value))
        desired_speed = math.hypot(vx, vy)
        previous_speed = math.hypot(*self._last_linear_cmd)
        rate_limit = decel_limit if desired_speed < previous_speed else accel_limit
        if self._last_linear_cmd_time is None:
            dt = 1.0 / max(1.0, float(self.get_parameter("publish_rate_hz").value))
        else:
            dt = max(1e-3, float((now - self._last_linear_cmd_time).nanoseconds) * 1e-9)
        if rate_limit <= 0.0:
            limited = (vx, vy)
        else:
            max_delta = rate_limit * dt
            limited = (
                self._clip(vx, self._last_linear_cmd[0] - max_delta, self._last_linear_cmd[0] + max_delta),
                self._clip(vy, self._last_linear_cmd[1] - max_delta, self._last_linear_cmd[1] + max_delta),
            )
        self._last_linear_cmd = limited
        self._last_linear_cmd_time = now
        return limited

    def _reset_distance_controller(self) -> None:
        self._distance_integral = 0.0
        self._last_distance_error = None
        self._last_distance_pid_time = None
        self._last_linear_cmd = (0.0, 0.0)
        self._last_linear_cmd_time = None

    def _compute_yaw_rate(self, now: Time, heading_error: float, target_distance: float) -> float:
        """Return a damped yaw-rate command from target bearing.

        The raw target bearing can change abruptly because the selected person box,
        depth quantile, and ROS network timing all jitter.  This method uses:
        1) an angular deadband, 2) EMA filtering of heading error, and
        3) a yaw-rate slew limit.
        """
        deadband = float(self.get_parameter("yaw_deadband_rad").value)
        if bool(self.get_parameter("yaw_hold_when_close").value):
            follow_distance = float(self.get_parameter("follow_distance_m").value)
            goal_tolerance = float(self.get_parameter("goal_tolerance_m").value)
            if target_distance <= follow_distance + goal_tolerance:
                # Close-range depth/detection jitter is visually amplified.  Use a
                # slightly wider deadband while already within the desired distance.
                deadband = max(deadband, 0.16)

        alpha = self._clip(float(self.get_parameter("yaw_filter_alpha").value), 0.0, 1.0)
        if self._filtered_heading_error is None:
            self._filtered_heading_error = heading_error
        else:
            self._filtered_heading_error = (1.0 - alpha) * self._filtered_heading_error + alpha * heading_error

        err = self._filtered_heading_error
        if abs(err) <= deadband:
            desired = 0.0
        else:
            # Remove the deadband offset to avoid a sudden jump when leaving it.
            effective_err = math.copysign(abs(err) - deadband, err)
            desired = float(self.get_parameter("yaw_gain").value) * effective_err

        max_rate = float(self.get_parameter("max_yaw_rate_rps").value)
        desired = self._clip(desired, -max_rate, max_rate)

        slew = max(0.0, float(self.get_parameter("yaw_slew_rate_limit_rps2").value))
        if self._last_cmd_time is None or slew <= 0.0:
            limited = desired
        else:
            dt = max(1e-3, float((now - self._last_cmd_time).nanoseconds) * 1e-9)
            max_delta = slew * dt
            limited = self._clip(desired, self._last_yaw_rate - max_delta, self._last_yaw_rate + max_delta)

        self._last_cmd_time = now
        self._last_yaw_rate = limited
        return limited

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
        if stop_api_once and bool(self.get_parameter("api_control_enabled").value) and HAS_UNITREE_API and self.req_pub is not None:
            req = Request()
            req.header.identity.id = self.seq
            self.seq += 1
            req.header.identity.api_id = 1003  # ROBOT_SPORT_API_ID_STOPMOVE
            req.parameter = ""
            self.req_pub.publish(req)

    def _publish_sport_request(self, cmd: TwistStamped) -> None:
        if not HAS_UNITREE_API or self.req_pub is None or Request is None:
            return
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
            # Same parameter schema as the original working package.
            req.parameter = json.dumps(
                {
                    "x": float(cmd.twist.linear.x),
                    "y": float(cmd.twist.linear.y),
                    "z": float(cmd.twist.angular.z),
                }
            )
        self.req_pub.publish(req)

    def _publish_follower_status(self, now: Time, active: bool, reason: str, cmd: Optional[TwistStamped]) -> None:
        payload = {
            "active": bool(active),
            "reason": reason,
            "unitree_api_available": HAS_UNITREE_API,
            "api_control_enabled": bool(self.get_parameter("api_control_enabled").value),
            "request_publisher_ready": self.req_pub is not None,
            "target_xy": None if self.target_xyz is None else {
                "x": self.target_xyz[0],
                "y": self.target_xyz[1],
            },
            "target_xyz": None if self.target_xyz is None else {
                "x": self.target_xyz[0],
                "y": self.target_xyz[1],
                "z": self.target_xyz[2],
            },
            "path_len": 0 if self.path is None else len(self.path.poses),
            "target_age_s": self._age_s(now, self.target_time),
            "path_age_s": self._age_s(now, self.path_time),
            "target_timeout_s": float(self.get_parameter("target_stale_timeout_s").value),
            "path_timeout_s": float(self.get_parameter("path_stale_timeout_s").value),
            "safety_scan_age_s": self._age_s(now, self.safety_scan_time),
            "safety": self._safety_debug,
        }
        if cmd is not None:
            payload["cmd"] = {
                "vx": float(cmd.twist.linear.x),
                "vy": float(cmd.twist.linear.y),
                "yaw_rate": float(cmd.twist.angular.z),
            }
            payload["yaw_debug"] = {
                "filtered_heading_error_rad": self._filtered_heading_error,
                "deadband_rad": float(self.get_parameter("yaw_deadband_rad").value),
                "filter_alpha": float(self.get_parameter("yaw_filter_alpha").value),
                "slew_rate_limit_rps2": float(self.get_parameter("yaw_slew_rate_limit_rps2").value),
            }
            payload["distance_pid_debug"] = {
                "error_m": self._last_distance_error,
                "integral": self._distance_integral,
                "kp": float(self.get_parameter("distance_pid_kp").value),
                "ki": float(self.get_parameter("distance_pid_ki").value),
                "kd": float(self.get_parameter("distance_pid_kd").value),
            }
        self.status_pub.publish(String(data=json.dumps(payload)))

    def _age_s(self, now: Time, t: Optional[Time]) -> Optional[float]:
        if t is None:
            return None
        return float((now - t).nanoseconds) * 1e-9

    def _diagnose_inactive(self, now: Time, reason: str) -> None:
        if now - self._last_diag_time < Duration(seconds=2.0):
            return
        self._last_diag_time = now
        target_age = self._age_s(now, self.target_time)
        path_age = self._age_s(now, self.path_time)
        status_age = self._age_s(now, self.status_time)
        safety_scan_age = self._age_s(now, self.safety_scan_time)
        self.get_logger().warn(
            f"Follower inactive: {reason}. target_received={self.target_time is not None}, "
            f"target_age_s={target_age}, target_timeout_s={float(self.get_parameter('target_stale_timeout_s').value):.2f}, "
            f"path_received={self.path_time is not None}, path_age_s={path_age}, "
            f"path_timeout_s={float(self.get_parameter('path_stale_timeout_s').value):.2f}, "
            f"status_tracked={self.status_tracked}, status_age_s={status_age}, "
            f"require_target_status={bool(self.get_parameter('require_target_status').value)}, "
            f"safety_scan_age_s={safety_scan_age}, "
            f"unitree_api_available={HAS_UNITREE_API}"
        )

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
