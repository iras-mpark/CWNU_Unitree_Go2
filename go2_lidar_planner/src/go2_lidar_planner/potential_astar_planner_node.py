#!/usr/bin/env python3
"""Local potential-field weighted A* planner for Go2 person following."""

from __future__ import annotations

import heapq
import math
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

Cell = Tuple[int, int]
Point2 = Tuple[float, float]


class PotentialAStarPlannerNode(Node):
    """Build a local cost map and run A* toward a waypoint before the person."""

    def __init__(self) -> None:
        super().__init__("go2_potential_astar_planner")
        self.declare_parameter("path_frame", "base_link")
        self.declare_parameter("occupancy_input_topic", "/utlidar/accumulated_obstacle_grid")
        self.declare_parameter("occupied_threshold", 50)
        self.declare_parameter("target_topic", "/local_goal_point")
        self.declare_parameter("path_topic", "/path")
        self.declare_parameter("goal_waypoint_topic", "/goal_waypoint")
        self.declare_parameter("goal_target_topic", "/goal_target")
        self.declare_parameter("obstacle_cloud_topic", "/local_obstacles")
        self.declare_parameter("occupancy_grid_topic", "/local_obstacle_grid")
        self.declare_parameter("potential_grid_topic", "/local_potential_grid")
        self.declare_parameter("publish_rate_hz", 10.0)

        self.declare_parameter("follow_distance_m", 2.0)
        self.declare_parameter("target_stale_timeout_s", 0.5)
        self.declare_parameter("max_path_length_m", 4.0)
        self.declare_parameter("path_resolution_m", 0.20)
        self.declare_parameter("grid_resolution_m", 0.12)
        self.declare_parameter("grid_x_min_m", 0.0)
        self.declare_parameter("grid_x_max_m", 3.0)
        self.declare_parameter("grid_y_min_m", -1.5)
        self.declare_parameter("grid_y_max_m", 1.5)

        self.declare_parameter("safety_radius_m", 0.45)
        self.declare_parameter("potential_radius_m", 0.95)
        self.declare_parameter("potential_weight", 4.0)
        self.declare_parameter("waypoint_potential_threshold", 0.35)
        self.declare_parameter("astar_max_expansions", 12000)
        self.declare_parameter("prefer_forward_motion", True)
        # Keep this below the normal inflated-map avoidance distance. A larger
        # hard stop prevents the holonomic planner from producing a side-step.
        self.declare_parameter("emergency_stop_distance_m", 0.18)

        self.frame = str(self.get_parameter("path_frame").value)
        self.occupancy_input_topic = str(self.get_parameter("occupancy_input_topic").value)
        self.target_topic = str(self.get_parameter("target_topic").value)
        self.resolution = float(self.get_parameter("grid_resolution_m").value)
        self.x_min = float(self.get_parameter("grid_x_min_m").value)
        self.x_max = float(self.get_parameter("grid_x_max_m").value)
        self.y_min = float(self.get_parameter("grid_y_min_m").value)
        self.y_max = float(self.get_parameter("grid_y_max_m").value)
        self.width = int(math.ceil((self.x_max - self.x_min) / self.resolution)) + 1
        self.height = int(math.ceil((self.y_max - self.y_min) / self.resolution)) + 1

        self.obstacles_xy: List[Point2] = []
        self.obstacle_cells: Set[Cell] = set()
        self._last_emergency_warn_time = self.get_clock().now() - Duration(seconds=10.0)
        self.latest_target: Optional[PointStamped] = None
        self.last_target_time: Optional[Time] = None

        self.create_subscription(OccupancyGrid, self.occupancy_input_topic, self._occupancy_cb, 5)
        self.create_subscription(PointStamped, self.target_topic, self._target_cb, 5)
        self.path_pub = self.create_publisher(Path, str(self.get_parameter("path_topic").value), 10)
        self.waypoint_pub = self.create_publisher(PointStamped, str(self.get_parameter("goal_waypoint_topic").value), 5)
        self.target_pub = self.create_publisher(PointStamped, str(self.get_parameter("goal_target_topic").value), 5)
        self.obstacle_pub = self.create_publisher(PointCloud2, str(self.get_parameter("obstacle_cloud_topic").value), 5)
        self.occupancy_pub = self.create_publisher(OccupancyGrid, str(self.get_parameter("occupancy_grid_topic").value), 5)
        self.potential_pub = self.create_publisher(OccupancyGrid, str(self.get_parameter("potential_grid_topic").value), 5)
        rate = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self.create_timer(1.0 / rate, self._timer_cb)
        self.get_logger().info(
            f"Potential A* planner ready: ogm={self.occupancy_input_topic}, target={self.target_topic}, "
            f"grid={self.width}x{self.height}@{self.resolution:.2f}m"
        )

    # ---------------------------------------------------------------- callbacks
    def _occupancy_cb(self, grid: OccupancyGrid) -> None:
        threshold = int(self.get_parameter("occupied_threshold").value)
        resolution = float(grid.info.resolution)
        origin_x = float(grid.info.origin.position.x)
        origin_y = float(grid.info.origin.position.y)
        cells: Set[Cell] = set()
        pts2: List[Point2] = []
        pts3 = []
        for index, value in enumerate(grid.data):
            if value < threshold:
                continue
            source_x = index % grid.info.width
            source_y = index // grid.info.width
            x = origin_x + (source_x + 0.5) * resolution
            y = origin_y + (source_y + 0.5) * resolution
            cell = self._world_to_cell(x, y)
            if cell is None:
                continue
            cells.add(cell)
            point = self._cell_to_world(cell)
            pts2.append(point)
            pts3.append((point[0], point[1], 0.0))
        self.obstacle_cells = cells
        self.obstacles_xy = pts2
        header = grid.header
        header.frame_id = self.frame
        out = point_cloud2.create_cloud_xyz32(header, pts3) if pts3 else PointCloud2()
        out.header = header
        out.header.stamp = self.get_clock().now().to_msg()
        self.obstacle_pub.publish(out)

    def _target_cb(self, msg: PointStamped) -> None:
        # The perception node already publishes target in base_link by design.
        self.latest_target = msg
        self.last_target_time = self.get_clock().now()

    def _timer_cb(self) -> None:
        now = self.get_clock().now()
        inflated, potential = self._build_maps()
        self.occupancy_pub.publish(self._to_grid_msg(inflated, None, now, as_potential=False))
        self.potential_pub.publish(self._to_grid_msg(inflated, potential, now, as_potential=True))

        target_xy = self._fresh_target_xy(now)
        if target_xy is None:
            self.path_pub.publish(self._path_msg([(0.0, 0.0)], now))
            return
        self._publish_point(self.target_pub, target_xy, now)

        if self._front_obstacle_too_close():
            if now - self._last_emergency_warn_time > Duration(seconds=1.0):
                self._last_emergency_warn_time = now
                self.get_logger().warn("Emergency stop: obstacle too close in front.")
            self.path_pub.publish(self._path_msg([(0.0, 0.0)], now))
            self._publish_point(self.waypoint_pub, (0.0, 0.0), now)
            return

        desired = self._desired_waypoint(target_xy)
        if desired is None:
            self.path_pub.publish(self._path_msg([(0.0, 0.0)], now))
            self._publish_point(self.waypoint_pub, (0.0, 0.0), now)
            return

        adjusted = self._adjust_waypoint_to_free_boundary(desired, inflated, potential, target_xy=target_xy)
        self._publish_point(self.waypoint_pub, adjusted, now)
        path_points = self._astar((0.0, 0.0), adjusted, inflated, potential)
        if not path_points:
            self.path_pub.publish(self._path_msg([(0.0, 0.0)], now))
            return
        self.path_pub.publish(self._path_msg(path_points, now))

    # ---------------------------------------------------------------- target / waypoint
    def _fresh_target_xy(self, now: Time) -> Optional[Point2]:
        if self.latest_target is None or self.last_target_time is None:
            return None
        timeout = float(self.get_parameter("target_stale_timeout_s").value)
        if timeout > 0.0 and now - self.last_target_time > Duration(seconds=timeout):
            return None
        p = self.latest_target.point
        if not (math.isfinite(p.x) and math.isfinite(p.y)):
            return None
        return (float(p.x), float(p.y))

    def _desired_waypoint(self, target_xy: Point2) -> Optional[Point2]:
        tx, ty = target_xy
        dist = math.hypot(tx, ty)
        follow_dist = float(self.get_parameter("follow_distance_m").value)
        if dist <= follow_dist:
            return None
        travel = min(dist - follow_dist, float(self.get_parameter("max_path_length_m").value))
        scale = travel / max(dist, 1e-6)
        gx = tx * scale
        gy = ty * scale
        gx = min(max(gx, self.x_min + self.resolution), self.x_max - self.resolution)
        gy = min(max(gy, self.y_min + self.resolution), self.y_max - self.resolution)
        return (gx, gy)

    def _front_obstacle_too_close(self) -> bool:
        threshold = float(self.get_parameter("emergency_stop_distance_m").value)
        if threshold <= 0:
            return False
        for x, y in self.obstacles_xy:
            if 0.0 < x < threshold and abs(y) < 0.35:
                return True
        return False

    # ---------------------------------------------------------------- map building
    def _build_maps(self) -> Tuple[Set[Cell], Dict[Cell, float]]:
        obstacle_cells = self.obstacle_cells.copy()
        inflated: Set[Cell] = set()
        safety_cells = max(0, int(math.ceil(float(self.get_parameter("safety_radius_m").value) / self.resolution)))
        for c in obstacle_cells:
            for nb, d_cells in self._cells_in_radius(c, safety_cells):
                if self._cell_in_bounds(nb):
                    inflated.add(nb)

        potential: Dict[Cell, float] = {}
        pot_radius = float(self.get_parameter("potential_radius_m").value)
        pot_cells = max(0, int(math.ceil(pot_radius / self.resolution)))
        weight = float(self.get_parameter("potential_weight").value)
        if pot_cells > 0 and weight > 0.0:
            for c in obstacle_cells:
                for nb, d_cells in self._cells_in_radius(c, pot_cells):
                    if not self._cell_in_bounds(nb):
                        continue
                    d_m = d_cells * self.resolution
                    if d_m > pot_radius:
                        continue
                    # Normalized exponential potential in [0, weight].
                    val = weight * math.exp(-2.8 * d_m / max(pot_radius, 1e-6))
                    if val > potential.get(nb, 0.0):
                        potential[nb] = val
        return inflated, potential

    def _cells_in_radius(self, center: Cell, radius_cells: int) -> Iterable[Tuple[Cell, float]]:
        cx, cy = center
        if radius_cells <= 0:
            yield center, 0.0
            return
        r2 = radius_cells * radius_cells
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                d2 = dx * dx + dy * dy
                if d2 <= r2:
                    yield (cx + dx, cy + dy), math.sqrt(d2)

    def _adjust_waypoint_to_free_boundary(
        self,
        waypoint: Point2,
        inflated: Set[Cell],
        potential: Dict[Cell, float],
        target_xy: Optional[Point2] = None,
    ) -> Point2:
        start = self._world_to_cell(*waypoint)
        if start is None:
            return waypoint
        threshold = float(self.get_parameter("waypoint_potential_threshold").value) * float(
            self.get_parameter("potential_weight").value
        )

        def acceptable(c: Cell) -> bool:
            return self._cell_in_bounds(c) and c not in inflated and potential.get(c, 0.0) <= threshold

        if target_xy is not None:
            boundary_waypoint = self._target_component_boundary_waypoint(target_xy, inflated, potential, threshold)
            if boundary_waypoint is not None:
                return boundary_waypoint

        if acceptable(start):
            return waypoint

        # Ring/BFS expansion to nearest free cell outside inflated/high-potential zone.
        visited = {start}
        queue: List[Tuple[int, Cell]] = [(0, start)]
        while queue:
            dist_cells, c = heapq.heappop(queue)
            if dist_cells > int(math.ceil(2.0 / self.resolution)):
                break
            if acceptable(c):
                return self._cell_to_world(c)
            for nb in self._neighbors8(c):
                if nb in visited or not self._cell_in_bounds(nb):
                    continue
                visited.add(nb)
                heapq.heappush(queue, (dist_cells + 1, nb))
        return waypoint

    def _target_component_boundary_waypoint(
        self, target_xy: Point2, inflated: Set[Cell], potential: Dict[Cell, float], threshold: float
    ) -> Optional[Point2]:
        target_cell = self._world_to_cell(*target_xy)
        if target_cell is None or target_cell not in inflated:
            return None

        component = self._connected_component(target_cell, inflated)
        if not component:
            return None

        tx, ty = target_xy
        target_dist = math.hypot(tx, ty)
        if target_dist < 1e-6:
            return None

        step = max(self.resolution * 0.5, 0.01)
        samples = max(1, int(math.ceil(target_dist / step)))
        last_acceptable: Optional[Cell] = None

        for i in range(samples + 1):
            scale = min(1.0, (i * step) / target_dist)
            c = self._world_to_cell(tx * scale, ty * scale)
            if c is None:
                continue
            if c in component:
                return self._cell_to_world(last_acceptable) if last_acceptable is not None else None
            if c in inflated:
                return None
            if potential.get(c, 0.0) <= threshold:
                last_acceptable = c

        return None

    def _connected_component(self, start: Cell, cells: Set[Cell]) -> Set[Cell]:
        if start not in cells:
            return set()
        component: Set[Cell] = set()
        queue = [start]
        while queue:
            c = queue.pop()
            if c in component:
                continue
            component.add(c)
            for nb in self._neighbors8(c):
                if nb in cells and nb not in component:
                    queue.append(nb)
        return component

    # ---------------------------------------------------------------- A*
    def _astar(
        self, start_xy: Point2, goal_xy: Point2, inflated: Set[Cell], potential: Dict[Cell, float]
    ) -> List[Point2]:
        start = self._world_to_cell(*start_xy)
        goal = self._world_to_cell(*goal_xy)
        if start is None or goal is None:
            return []
        if start in inflated:
            inflated = set(inflated)
            inflated.discard(start)
        if goal in inflated:
            goal_xy = self._adjust_waypoint_to_free_boundary(goal_xy, inflated, potential)
            goal = self._world_to_cell(*goal_xy)
            if goal is None:
                return []

        open_heap: List[Tuple[float, int, Cell]] = []
        came_from: Dict[Cell, Cell] = {}
        g_score: Dict[Cell, float] = {start: 0.0}
        counter = 0
        heapq.heappush(open_heap, (self._heuristic(start, goal), counter, start))
        closed: Set[Cell] = set()
        max_exp = int(self.get_parameter("astar_max_expansions").value)
        prefer_forward = bool(self.get_parameter("prefer_forward_motion").value)

        while open_heap and len(closed) < max_exp:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal:
                return self._reconstruct_path(came_from, current)
            closed.add(current)
            for nb in self._neighbors8(current):
                if nb in closed or not self._cell_in_bounds(nb) or nb in inflated:
                    continue
                step = self._cell_distance(current, nb)
                penalty = potential.get(nb, 0.0)
                wx, _ = self._cell_to_world(nb)
                if prefer_forward and wx < -0.05:
                    penalty += 2.0
                tentative = g_score[current] + step + penalty * self.resolution
                if tentative < g_score.get(nb, float("inf")):
                    came_from[nb] = current
                    g_score[nb] = tentative
                    counter += 1
                    f = tentative + self._heuristic(nb, goal)
                    heapq.heappush(open_heap, (f, counter, nb))
        return []

    def _reconstruct_path(self, came_from: Dict[Cell, Cell], current: Cell) -> List[Point2]:
        cells = [current]
        while current in came_from:
            current = came_from[current]
            cells.append(current)
        cells.reverse()
        points = [self._cell_to_world(c) for c in cells]
        return self._downsample_path(points)

    def _downsample_path(self, points: List[Point2]) -> List[Point2]:
        if not points:
            return []
        spacing = max(self.resolution, float(self.get_parameter("path_resolution_m").value))
        out = [points[0]]
        last = points[0]
        for p in points[1:]:
            if math.hypot(p[0] - last[0], p[1] - last[1]) >= spacing:
                out.append(p)
                last = p
        if out[-1] != points[-1]:
            out.append(points[-1])
        return out

    # ---------------------------------------------------------------- coordinates / messages
    def _world_to_cell(self, x: float, y: float) -> Optional[Cell]:
        if not (self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max):
            return None
        ix = int(round((x - self.x_min) / self.resolution))
        iy = int(round((y - self.y_min) / self.resolution))
        c = (ix, iy)
        return c if self._cell_in_bounds(c) else None

    def _cell_to_world(self, cell: Cell) -> Point2:
        return (self.x_min + cell[0] * self.resolution, self.y_min + cell[1] * self.resolution)

    def _cell_in_bounds(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height

    def _neighbors8(self, cell: Cell) -> Iterable[Cell]:
        x, y = cell
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                yield (x + dx, y + dy)

    def _cell_distance(self, a: Cell, b: Cell) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1]) * self.resolution

    def _heuristic(self, a: Cell, b: Cell) -> float:
        return self._cell_distance(a, b)

    def _path_msg(self, points: List[Point2], now: Time) -> Path:
        path = Path()
        path.header.stamp = now.to_msg()
        path.header.frame_id = self.frame
        for i, (x, y) in enumerate(points):
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            if i + 1 < len(points):
                yaw = math.atan2(points[i + 1][1] - y, points[i + 1][0] - x)
            elif i > 0:
                yaw = math.atan2(y - points[i - 1][1], x - points[i - 1][0])
            else:
                yaw = 0.0
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            path.poses.append(pose)
        return path

    def _publish_point(self, pub, point: Point2, now: Time) -> None:
        msg = PointStamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = self.frame
        msg.point.x = float(point[0])
        msg.point.y = float(point[1])
        msg.point.z = 0.0
        pub.publish(msg)

    def _to_grid_msg(
        self, inflated: Set[Cell], potential: Optional[Dict[Cell, float]], now: Time, *, as_potential: bool
    ) -> OccupancyGrid:
        grid = OccupancyGrid()
        grid.header.stamp = now.to_msg()
        grid.header.frame_id = self.frame
        grid.info.resolution = float(self.resolution)
        grid.info.width = self.width
        grid.info.height = self.height
        grid.info.origin.position.x = self.x_min
        grid.info.origin.position.y = self.y_min
        grid.info.origin.orientation.w = 1.0
        data = np.zeros((self.height, self.width), dtype=np.int8)
        if as_potential and potential is not None:
            max_weight = max(1e-6, float(self.get_parameter("potential_weight").value))
            for (ix, iy), val in potential.items():
                if self._cell_in_bounds((ix, iy)):
                    data[iy, ix] = int(np.clip(100.0 * val / max_weight, 0, 100))
            for ix, iy in inflated:
                if self._cell_in_bounds((ix, iy)):
                    data[iy, ix] = 100
        else:
            for ix, iy in inflated:
                if self._cell_in_bounds((ix, iy)):
                    data[iy, ix] = 100
        grid.data = data.flatten(order="C").tolist()
        return grid


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = PotentialAStarPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
