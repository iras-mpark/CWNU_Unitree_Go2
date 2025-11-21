#!/usr/bin/env python3
"""Lightweight LiDAR-only local planner.

Each scan is voxelized into a local occupancy grid, obstacles are inflated
by the configured safety radius, and an A* search produces a short collision-
free path toward the goal expressed in `path_frame`. The resulting plan plus
visual aids (`/goal_target`, `/goal_preview`, `/local_obstacles`) make it easy
to debug in RViz while remaining standalone.
"""

from __future__ import annotations

import math
import heapq
from typing import Dict, List, Optional, Sequence, Tuple, Set

import rclpy
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.node import Node

from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformException, TransformListener
from tf2_geometry_msgs import do_transform_point

class SimpleLocalPlanner(Node):
    """Generate a short collision-free path directly toward the waypoint."""

    def __init__(self) -> None:
        super().__init__("local_lidar_planner_simple")

        # Path construction parameters
        self.declare_parameter("path_frame", "vehicle")
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("path_resolution", 0.25)
        self.declare_parameter("max_path_length", 3.0)
        self.declare_parameter("goal_tolerance", 0.3)
        self.declare_parameter("goal_offset", 0.2)

        # Obstacle handling parameters
        self.declare_parameter("safety_radius", 0.45)
        self.declare_parameter("max_considered_range", 4.0)
        self.declare_parameter("grid_resolution", 0.2)
        self.declare_parameter("costmap_inflation_radius", 0.8)
        self.declare_parameter("costmap_inflation_weight", 3.0)
        self.declare_parameter("obstacle_z_min", -1.0)
        self.declare_parameter("obstacle_z_max", 1.0)
        self.declare_parameter("scan_topic", "/utlidar/cloud")
        self.declare_parameter("goal_topic", "/local_goal_point")
        self.declare_parameter("goal_tf_timeout", 0.5)
        self.declare_parameter("obstacle_topic", "/local_obstacles")
        self.declare_parameter("plan_forward_min_x", 0.0)
        self.declare_parameter("goal_stale_timeout", 0.2)

        self.path_frame: str = self.get_parameter("path_frame").get_parameter_value().string_value
        publish_rate_hz = self.get_parameter("publish_rate_hz").get_parameter_value().double_value
        self.path_resolution = self.get_parameter("path_resolution").get_parameter_value().double_value
        self.max_path_length = self.get_parameter("max_path_length").get_parameter_value().double_value
        self.goal_tolerance = self.get_parameter("goal_tolerance").get_parameter_value().double_value
        self.goal_offset = self.get_parameter("goal_offset").get_parameter_value().double_value
        self.safety_radius = self.get_parameter("safety_radius").get_parameter_value().double_value
        self.max_considered_range = self.get_parameter("max_considered_range").get_parameter_value().double_value
        self.grid_resolution = max(
            0.05, self.get_parameter("grid_resolution").get_parameter_value().double_value
        )
        self.costmap_inflation_radius = max(
            0.0, self.get_parameter("costmap_inflation_radius").get_parameter_value().double_value
        )
        self.costmap_inflation_weight = max(
            0.0, self.get_parameter("costmap_inflation_weight").get_parameter_value().double_value
        )
        self.grid_radius_cells = max(1, int(math.ceil(self.max_considered_range / self.grid_resolution)))
        self.inflation_cells = max(0, int(math.ceil(self.safety_radius / self.grid_resolution)))
        self.costmap_inflation_cells = (
            0
            if self.costmap_inflation_radius <= 0.0
            else max(1, int(math.ceil(self.costmap_inflation_radius / self.grid_resolution)))
        )
        self.obstacle_z_min = self.get_parameter("obstacle_z_min").get_parameter_value().double_value
        self.obstacle_z_max = self.get_parameter("obstacle_z_max").get_parameter_value().double_value
        self.scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        self.goal_topic = self.get_parameter("goal_topic").get_parameter_value().string_value
        if not self.goal_topic:
            raise ValueError("goal_topic parameter must be set (e.g., '/local_goal_point').")
        self.goal_tf_timeout = self.get_parameter("goal_tf_timeout").get_parameter_value().double_value
        self.plan_forward_min_x = self.get_parameter("plan_forward_min_x").get_parameter_value().double_value
        self.goal_stale_timeout = max(
            0.0, self.get_parameter("goal_stale_timeout").get_parameter_value().double_value
        )

        self.latest_obstacles: List[Tuple[float, float]] = []
        self.latest_cost_penalties: Dict[Tuple[int, int], float] = {}

        # TF tracking
        self.tf_buffer = Buffer()
        TransformListener(self.tf_buffer, self, qos=5)
        self._last_tf_warn_time = 0.0
        self._last_goal_stale_warn_time = 0.0
        self._latest_goal_msg: Optional[PointStamped] = None
        self._last_goal_update: Optional[Time] = None

        # Interfaces
        self.create_subscription(PointCloud2, self.scan_topic, self._scan_callback, 5)
        self.create_subscription(PointStamped, self.goal_topic, self._goal_callback, 5)
        self.path_pub = self.create_publisher(Path, "/path", 10)
        self.goal_pub = self.create_publisher(PointStamped, "/goal_preview", 5)
        self.goal_target_pub = self.create_publisher(PointStamped, "/goal_target", 5)
        obstacle_topic = self.get_parameter("obstacle_topic").get_parameter_value().string_value
        self.obstacle_pub = self.create_publisher(PointCloud2, obstacle_topic, 5)
        self.grid_pub = self.create_publisher(OccupancyGrid, "/local_obstacles_grid", 5)
        self.potential_pub = self.create_publisher(OccupancyGrid, "/local_potential_grid", 5)

        self.timer = self.create_timer(1.0 / max(publish_rate_hz, 1e-3), self._on_timer)
        self.get_logger().info("Simple local planner ready (LiDAR-only, joystick-free).")

    # ------------------------------------------------------------------ Callbacks
    def _scan_callback(self, cloud: PointCloud2) -> None:
        """Convert the incoming point cloud into a filtered obstacle list."""
        obstacle_points: List[Tuple[float, float, float]] = []
        xy_points: List[Tuple[float, float]] = []

        for point in point_cloud2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = point
            if not (self.obstacle_z_min <= z <= self.obstacle_z_max):
                continue

            distance = math.hypot(x, y)
            if distance < 1e-3 or distance > self.max_considered_range:
                continue
            heading = math.atan2(y, x)
            obstacle_points.append((x, y, z))
            xy_points.append((x, y))

        now_msg = self.get_clock().now().to_msg()
        obstacle_header = cloud.header
        obstacle_header.frame_id = self.path_frame
        if obstacle_points:
            obstacle_msg = point_cloud2.create_cloud_xyz32(obstacle_header, obstacle_points)
        else:
            obstacle_msg = PointCloud2()
            obstacle_msg.header = obstacle_header
        obstacle_msg.header.stamp = now_msg
        inflated_cells = self._build_occupancy(points=xy_points)
        cost_penalties = self._compute_cost_penalties(xy_points)
        self.obstacle_pub.publish(obstacle_msg)
        self.grid_pub.publish(self._build_grid_map(xy_points, inflated=False, stamp=now_msg))
        self.potential_pub.publish(
            self._build_cost_grid(inflated_cells, cost_penalties, stamp=now_msg)
        )
        self.latest_obstacles = xy_points
        self.latest_cost_penalties = cost_penalties

    # ------------------------------------------------------------------ Timer
    def _on_timer(self) -> None:
        path_msg = self._build_path()
        self.path_pub.publish(path_msg)

    # ------------------------------------------------------------------ Helpers
    def _build_path(self) -> Path:
        """Return a short path that respects the latest goal and LiDAR data."""
        now = self.get_clock().now()
        path = Path()
        path.header.stamp = now.to_msg()
        path.header.frame_id = self.path_frame

        rel_goal = self._lookup_goal_in_vehicle(now)
        if rel_goal is None:
            self._publish_goal_marker(0.0, 0.0, now)
            path.poses.append(self._pose_at(0.0, 0.0, 0.0, now))
            return path

        raw_distance = math.hypot(rel_goal[0], rel_goal[1])
        desired_heading = math.atan2(rel_goal[1], rel_goal[0])

        if raw_distance <= self.goal_offset:
            path.poses.append(self._pose_at(0.0, 0.0, desired_heading, now))
            self._publish_goal_marker(0.0, 0.0, now)
            return path

        # Clamp the goal so we always stop goal_offset meters short of the target.
        goal_distance = max(0.0, raw_distance - self.goal_offset)
        if raw_distance > 1e-6:
            scale = goal_distance / raw_distance
            rel_goal = (rel_goal[0] * scale, rel_goal[1] * scale)

        if goal_distance < self.goal_tolerance:
            path.poses.append(self._pose_at(0.0, 0.0, desired_heading, now))
            self._publish_goal_marker(0.0, 0.0, now)
            return path

        max_travel = self.max_path_length
        if self.max_considered_range > 0:
            max_travel = min(self.max_path_length, self.max_considered_range)
        if max_travel <= 0:
            max_travel = goal_distance
        if goal_distance > max_travel and goal_distance > 1e-6:
            scale = max_travel / goal_distance
            goal_distance = max_travel
            rel_goal = (rel_goal[0] * scale, rel_goal[1] * scale)

        if rel_goal[0] < self.plan_forward_min_x:
            rel_goal = (self.plan_forward_min_x, rel_goal[1])

        path_points = self._plan_path_astar(rel_goal[0], rel_goal[1])
        if not path_points:
            path.poses.append(self._pose_at(0.0, 0.0, desired_heading, now))
            self._publish_goal_marker(0.0, 0.0, now)
            return path

        prev = path_points[0]
        for current in path_points[1:]:
            heading = math.atan2(current[1] - prev[1], current[0] - prev[0])
            path.poses.append(self._pose_at(current[0], current[1], heading, now))
            prev = current

        if path.poses:
            last_pose = path.poses[-1].pose.position
            self._publish_goal_marker(last_pose.x, last_pose.y, now)
        else:
            self._publish_goal_marker(0.0, 0.0, now)
        return path

    def _goal_callback(self, msg: PointStamped) -> None:
        self._latest_goal_msg = msg
        self._last_goal_update = self.get_clock().now()

    def _lookup_goal_in_vehicle(self, stamp: Time) -> Optional[Tuple[float, float]]:
        goal_msg = self._latest_goal_msg
        last_update = self._last_goal_update
        if goal_msg is None or last_update is None:
            return None

        if self.goal_stale_timeout > 0.0:
            age = (stamp - last_update).nanoseconds / 1e9
            if age > self.goal_stale_timeout:
                now_sec = stamp.nanoseconds / 1e9
                if now_sec - self._last_goal_stale_warn_time > 2.0:
                    frame = goal_msg.header.frame_id or self.path_frame
                    self.get_logger().warn(
                        f"Goal topic '{self.goal_topic}' stale for {age:.2f}s (frame={frame}, limit={self.goal_stale_timeout:.2f}s); "
                        "treating goal as lost."
                    )
                    self._last_goal_stale_warn_time = now_sec
                return None

        goal_point = goal_msg
        source_frame = goal_msg.header.frame_id or self.path_frame
        if source_frame != self.path_frame:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.path_frame,
                    source_frame,
                    Time(),
                    timeout=Duration(seconds=self.goal_tf_timeout),
                )
                goal_point = do_transform_point(goal_msg, transform)
            except TransformException as exc:
                now_sec = self.get_clock().now().nanoseconds / 1e9
                if now_sec - self._last_tf_warn_time > 2.0:
                    self.get_logger().warn(
                        f"Transform from '{source_frame}' to '{self.path_frame}' failed for goal topic '{self.goal_topic}': {exc}"
                    )
                    self._last_tf_warn_time = now_sec
                return None

        rel_x = goal_point.point.x
        rel_y = goal_point.point.y

        target_msg = PointStamped()
        target_msg.header.stamp = stamp.to_msg()
        target_msg.header.frame_id = self.path_frame
        target_msg.point.x = rel_x
        target_msg.point.y = rel_y
        target_msg.point.z = goal_point.point.z
        self.goal_target_pub.publish(target_msg)

        return rel_x, rel_y

    def _build_occupancy(
        self, points: Optional[List[Tuple[float, float]]] = None
    ) -> Set[Tuple[int, int]]:
        occupancy: Set[Tuple[int, int]] = set()
        if points is None:
            points = self.latest_obstacles
        if not points:
            return occupancy

        max_cells = self.grid_radius_cells
        for cell in self._cells_from_points(points):
            for dx in range(-self.inflation_cells, self.inflation_cells + 1):
                for dy in range(-self.inflation_cells, self.inflation_cells + 1):
                    if dx * dx + dy * dy > self.inflation_cells * self.inflation_cells:
                        continue
                    occ = (cell[0] + dx, cell[1] + dy)
                    if abs(occ[0]) <= max_cells and abs(occ[1]) <= max_cells:
                        occupancy.add(occ)
        return occupancy

    def _plan_path_astar(self, goal_x: float, goal_y: float) -> Optional[List[Tuple[float, float]]]:
        occupancy = self._build_occupancy()
        penalties = self.latest_cost_penalties
        grid_limit = self.grid_radius_cells * self.grid_resolution
        goal_distance = math.hypot(goal_x, goal_y)
        if goal_distance > grid_limit and goal_distance > 1e-6:
            scale = grid_limit / goal_distance
            goal_x *= scale
            goal_y *= scale
        start_cell = (0, 0)
        goal_cell = self._world_to_cell(goal_x, goal_y)

        if goal_cell == start_cell:
            return [(0.0, 0.0)]

        if goal_cell in occupancy:
            goal_cell = self._find_nearest_free(goal_cell, occupancy)
            if goal_cell is None:
                return None

        def heuristic(cell: Tuple[int, int]) -> float:
            return math.hypot(goal_cell[0] - cell[0], goal_cell[1] - cell[1])

        open_heap: List[Tuple[float, float, Tuple[int, int]]] = []
        heapq.heappush(open_heap, (heuristic(start_cell), 0.0, start_cell))
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_cost: Dict[Tuple[int, int], float] = {start_cell: 0.0}
        max_cells = self.grid_radius_cells
        directions = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]

        while open_heap:
            _, current_cost, current = heapq.heappop(open_heap)
            if current == goal_cell:
                return self._reconstruct_path(came_from, current)

            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if abs(neighbor[0]) > max_cells or abs(neighbor[1]) > max_cells:
                    continue
                if not self._cell_within_forward_limit(neighbor):
                    continue
                if neighbor != start_cell and neighbor in occupancy:
                    continue
                step = math.hypot(dx, dy)
                penalty = penalties.get(neighbor, 0.0)
                tentative_cost = current_cost + step * (1.0 + penalty)
                if tentative_cost < g_cost.get(neighbor, math.inf):
                    came_from[neighbor] = current
                    g_cost[neighbor] = tentative_cost
                    priority = tentative_cost + heuristic(neighbor)
                    heapq.heappush(open_heap, (priority, tentative_cost, neighbor))

        return None

    def _reconstruct_path(
        self, came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]
    ) -> List[Tuple[float, float]]:
        cells = [current]
        while current in came_from:
            current = came_from[current]
            cells.append(current)
        cells.reverse()
        points = [self._cell_to_world(cell) for cell in cells]
        # drop the duplicated start point to avoid zero-length segment
        if len(points) > 1 and math.hypot(points[1][0], points[1][1]) < 1e-6:
            points = points[1:]
        return points

    def _find_nearest_free(
        self, start_cell: Tuple[int, int], occupancy: Set[Tuple[int, int]]
    ) -> Optional[Tuple[int, int]]:
        if start_cell not in occupancy:
            return start_cell
        max_cells = self.grid_radius_cells
        max_radius = max_cells
        for radius in range(1, max_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue
                    candidate = (start_cell[0] + dx, start_cell[1] + dy)
                    if abs(candidate[0]) > max_cells or abs(candidate[1]) > max_cells:
                        continue
                    if candidate not in occupancy:
                        return candidate
        return None

    def _world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        return (
            int(round(x / self.grid_resolution)),
            int(round(y / self.grid_resolution)),
        )

    def _cell_to_world(self, cell: Tuple[int, int]) -> Tuple[float, float]:
        return (cell[0] * self.grid_resolution, cell[1] * self.grid_resolution)

    def _cell_within_forward_limit(self, cell: Tuple[int, int]) -> bool:
        if cell == (0, 0):
            return True
        if not math.isfinite(self.plan_forward_min_x):
            return True
        world_x = cell[0] * self.grid_resolution
        return world_x >= self.plan_forward_min_x - 1e-6

    def _cells_from_points(self, points: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
        cells: List[Tuple[int, int]] = []
        max_range = self.max_considered_range
        for x, y in points:
            if math.hypot(x, y) > max_range:
                continue
            cells.append(self._world_to_cell(x, y))
        return cells

    def _compute_cost_penalties(self, points: List[Tuple[float, float]]) -> Dict[Tuple[int, int], float]:
        penalties: Dict[Tuple[int, int], float] = {}
        if (
            not points
            or self.costmap_inflation_radius <= 0.0
            or self.costmap_inflation_weight <= 0.0
        ):
            return penalties

        inflation_cells = self.costmap_inflation_cells
        if inflation_cells <= 0:
            return penalties
        inflation_radius = max(self.costmap_inflation_radius, self.grid_resolution)
        base_cells = set(self._cells_from_points(points))
        if not base_cells:
            return penalties

        max_cells = self.grid_radius_cells
        for cell in base_cells:
            cx, cy = cell
            for dx in range(-inflation_cells, inflation_cells + 1):
                for dy in range(-inflation_cells, inflation_cells + 1):
                    distance = math.hypot(dx, dy) * self.grid_resolution
                    if distance > inflation_radius:
                        continue
                    target = (cx + dx, cy + dy)
                    if abs(target[0]) > max_cells or abs(target[1]) > max_cells:
                        continue
                    if target in base_cells and distance > 0.0:
                        # Base obstacle cells are handled separately for occupancy; treat them as max penalty.
                        pass
                    ratio = 1.0 - (distance / inflation_radius if inflation_radius > 0 else 0.0)
                    if ratio <= 0.0:
                        continue
                    penalty = self.costmap_inflation_weight * ratio
                    prev = penalties.get(target)
                    if prev is None or penalty > prev:
                        penalties[target] = penalty
        return penalties

    def _build_grid_map(self, points: List[Tuple[float, float]], inflated: bool, stamp) -> OccupancyGrid:
        grid = OccupancyGrid()
        grid.header.stamp = stamp
        grid.header.frame_id = self.path_frame
        grid.info.resolution = self.grid_resolution
        width = height = self.grid_radius_cells * 2 + 1
        grid.info.width = width
        grid.info.height = height
        grid.info.origin.position.x = -self.grid_radius_cells * self.grid_resolution
        grid.info.origin.position.y = -self.grid_radius_cells * self.grid_resolution
        grid.info.origin.orientation.w = 1.0

        total_cells = width * height
        data = [-1] * total_cells

        cells = set(self._cells_from_points(points))
        if inflated and self.inflation_cells > 0:
            inflated_cells: Set[Tuple[int, int]] = set()
            max_cells = self.grid_radius_cells
            for cell in cells:
                for dx in range(-self.inflation_cells, self.inflation_cells + 1):
                    for dy in range(-self.inflation_cells, self.inflation_cells + 1):
                        if dx * dx + dy * dy > self.inflation_cells * self.inflation_cells:
                            continue
                        occ = (cell[0] + dx, cell[1] + dy)
                        if abs(occ[0]) <= max_cells and abs(occ[1]) <= max_cells:
                            inflated_cells.add(occ)
            cells = inflated_cells

        for cell in cells:
            idx = self._cell_to_grid_index(cell, width)
            if 0 <= idx < total_cells:
                data[idx] = 100 if inflated else 75

        grid.data = data
        return grid

    def _build_cost_grid(
        self,
        inflated_cells: Set[Tuple[int, int]],
        penalties: Dict[Tuple[int, int], float],
        stamp,
    ) -> OccupancyGrid:
        grid = OccupancyGrid()
        grid.header.stamp = stamp
        grid.header.frame_id = self.path_frame
        grid.info.resolution = self.grid_resolution
        width = height = self.grid_radius_cells * 2 + 1
        grid.info.width = width
        grid.info.height = height
        grid.info.origin.position.x = -self.grid_radius_cells * self.grid_resolution
        grid.info.origin.position.y = -self.grid_radius_cells * self.grid_resolution
        grid.info.origin.orientation.w = 1.0

        total_cells = width * height
        data = [-1] * total_cells

        for cell in inflated_cells:
            idx = self._cell_to_grid_index(cell, width)
            if 0 <= idx < total_cells:
                data[idx] = 100

        if penalties:
            max_penalty = max(self.costmap_inflation_weight, 1e-3)
            scale = 100.0 / max_penalty
            for cell, penalty in penalties.items():
                idx = self._cell_to_grid_index(cell, width)
                if idx < 0 or idx >= total_cells:
                    continue
                value = min(100, int(round(penalty * scale)))
                data[idx] = max(data[idx], value)

        grid.data = data
        return grid

    def _cell_to_grid_index(self, cell: Tuple[int, int], width: int) -> int:
        x_idx = cell[0] + self.grid_radius_cells
        y_idx = cell[1] + self.grid_radius_cells
        return y_idx * width + x_idx

    def _pose_at(self, x: float, y: float, heading: float, stamp) -> PoseStamped:
        pose = PoseStamped()
        pose.header.stamp = stamp.to_msg()
        pose.header.frame_id = self.path_frame
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z = math.sin(heading / 2.0)
        pose.pose.orientation.w = math.cos(heading / 2.0)
        return pose

    def _publish_goal_marker(self, x: float, y: float, stamp) -> None:
        goal = PointStamped()
        goal.header.stamp = stamp.to_msg()
        goal.header.frame_id = self.path_frame
        goal.point.x = x
        goal.point.y = y
        self.goal_pub.publish(goal)

def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = SimpleLocalPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
