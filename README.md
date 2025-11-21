# IRAS Unitree Go2

ROS 2 workspace for the Unitree Go2 that ties a lightweight 3D object detector
to a LiDAR-only local navigation stack. Each node keeps its own scope small and
communicates only with topics/TF primitives so that perception and planning can
be mixed or swapped without touching the rest of the system.

## Packages

### `detect_3d_object`
Consumes RGB, depth, and detection topics (from an OAK-D + Darknet pipeline by
default), renders overlays, and exports per-object TF frames plus local goal
points for downstream planners.

#### Nodes
- `detect_3d_object_node`
  - Intent: Fuse `/color/image`, `/depth/image`, `/depth/camera_info`, and
    `detector_node/detections` to keep an overlay stream (`image_containing_object`)
    in sync with what the planner sees while emitting `PointStamped` targets.
  - Publishes:
    - `image_containing_object/eighth` + `/compressed` plus
      `image_containing_object/camera_info` so RViz can visualize detections.
    - `local_goal_point` in `parent_frame` (`base_link` by default) when the
      detection class matches `goal_object_class` and, optionally,
      `goal_object_identifier` (e.g., `person_2`).
    - `local_goal_point/<class>_frame_<idx>` alongside TF frames
      `<class>_frame[_<idx>]` for every detection so planners can latch onto a
      specific instance.
  - Notable parameters: `overlay_color_mode` (`rgb`/`gray`), `parent_frame`,
    `goal_object_class`, `goal_object_identifier`, and the usual `use_sim_time`
    (forced to `true` unless overridden earlier).
- `local_clock_publisher`
  - Intent: Provide a deterministic `/clock` when running log playback or real
    hardware without Gazebo.
  - Parameters: `clock_frequency` (Hz) and `initial_time` (sec). Publishes to
    `/clock`; no subscriptions.

#### Launch files
- `ros2 launch detect_3d_object oak_d_darknet_3d_detection.launch.xml`
  brings up the OAK-D driver, Darknet detector, and `detect_3d_object_node`
  together. Override `goal_object_class`, `goal_object_identifier`,
  `parent_frame`, etc., via launch arguments.
- Alternatively launch each stage manually via
  `1_oak_cam.launch.xml`, `2_darknet.launch.xml`, and `3_detect_3d.launch.xml`
  (found in `launch/`) when you want tighter control or to swap in simulated
  inputs. `launch/legacy` contains older setups kept for reference.

### `local_lidar_planner`
Self-contained LiDAR stack that aligns the stock Unitree LiDAR, aggregates
points into a short-term map, generates an A*-style local plan, and commands
the robot (or simulator) without a joystick.

#### Nodes
- `utlidar_transformer.py`
  - Intent: Apply a fixed roll/pitch/yaw plus translation offset so incoming
    `PointCloud2` messages align with the chassis frame while broadcasting the
    static TF between `target_frame` (default `base_link`) and `lidar_frame`.
  - Subscribes: `/utlidar/cloud` (configurable via `input_topic`).
  - Publishes: `/utlidar/transformed_cloud` and a static TF tree entry.
- `lidar_accumulator.py`
  - Intent: Maintain a sliding window (duration + max cloud count) of LiDAR
    scans so sparse data becomes dense enough for local planning.
  - Subscribes: `/utlidar/transformed_cloud`.
  - Publishes: `/utlidar/accumulated_cloud`, `~/debug` (String), and exposes
    `~/dump_status` (Trigger service) for quick CLI status checks.
- `local_lidar_planner_simple.py`
  - Intent: Convert recent clouds into an occupancy/cost grid, inflate
    obstacles by `safety_radius`, and run a short A* search toward the most
    recent goal expressed in `path_frame`.
  - Subscribes: `scan_topic` (defaults to `/utlidar/cloud`, overridden to the
    accumulator output in the launch file) and `goal_topic`
    (defaults to `/local_goal_point` from `detect_3d_object_node`).
  - Publishes: `/path`, `/goal_preview`, `/goal_target`,
    `/local_obstacles` (PointCloud2), `/local_obstacles_grid`, and
    `/local_potential_grid` for RViz debugging.
- `path_follower_simple.py`
  - Intent: Run a pure-pursuit style follower that keeps heading aligned with
    the latest goal while regulating speed, optionally mirroring commands onto
    Unitree's sport API.
  - Subscribes: `/path` and `/goal_target`.
  - Publishes: `/cmd_vel` plus `/api/sport/request` when
    `is_real_robot:=true`. Parameters such as `max_speed`, `lookahead_distance`,
    `translation_stop_distance`, and `heading_tolerance` expose the controller
    gains.

#### Launch
- `ros2 launch local_lidar_planner local_lidar_planner.launch.xml`
  chains the transformer → accumulator → planner → follower nodes. Common
  arguments include:
  - `path_frame`/`cmd_frame` (usually `base_link`),
  - LiDAR pose (`lidar_*`) so the static TF matches your mounting point,
  - Planner knobs such as `safety_radius`, `goal_topic`, `grid_resolution`,
    `plan_forward_min_x`, and goal-staleness timeouts, and
  - Follower gains (`max_speed`, `lookahead_distance`, `angular_gain`,
    `translation_stop_distance`, etc.).
  Feed `/local_goal_point` from `detect_3d_object_node` (or another source) and
  the launch file will propagate it through the rest of the stack.

## End-to-end flow
1. Launch perception: either the monolithic
   `ros2 launch detect_3d_object oak_d_darknet_3d_detection.launch.xml`
   or the three individual launch files if you need finer control. Confirm
   `image_containing_object/eighth` renders detections and
   `local_goal_point` is being updated for your chosen class.
2. Launch the LiDAR stack:
   `ros2 launch local_lidar_planner local_lidar_planner.launch.xml
   goal_topic:=/local_goal_point path_frame:=base_link`.
   Tailor LiDAR pose offsets and planner/follower parameters to your robot.
3. In RViz, monitor `/path`, `/cmd_vel`, the `/local_*` debug grids, and the TF
   tree to ensure the planner respects detected goals before switching to a
   real robot or enabling sport requests.
