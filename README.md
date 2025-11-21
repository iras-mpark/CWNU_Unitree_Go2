# IRAS Unitree Go2

ROS 2 workspace that pairs a lightweight 3D object detector with a LiDAR-only
local planner for the Unitree Go2 platform. The focus is on keeping each node
simple and exposing only the topics you need to connect the perception and
navigation stacks.

## Package overview

### `detect_3d_object`
DepthAI RGB-D inputs plus Darknet detections are fused into goal points or TF
frames that downstream planners can consume.

Nodes:
- `detect_3d_object_node`
  - Intent: Overlay detections on the RGB feed, back-project detections into 3D
    using the depth stream, and publish per-object goal points and TF frames.
  - Subscribes: `/color/image`, `/depth/image`, `/depth/camera_info`,
    `detector_node/detections`
  - Publishes: `image_containing_object/eighth[/compressed]`,
    `image_containing_object/camera_info`, `local_goal_point`,
    `local_goal_point/<class>_frame_<idx>`, TF frames `<class>_frame[_idx]`
- `move_to_object_node`
  - Intent: Follow a TF frame produced by the detector and expose it as a
    `PoseStamped` goal.
  - Subscribes: TF buffer lookup between `source_frame` (default `map`) and
    `target_frame` (default `chair_frame`)
  - Publishes: `/goal_pose` (PoseStamped)
- `local_clock_publisher`
  - Intent: Provide a deterministic `/clock` when running without Gazebo.
  - Publishes: `/clock` (Clock); no subscriptions

Launch:
- `ros2 launch detect_3d_object oak_d_darknet_3d_detection.launch.xml`
  starts the OAK-D camera driver, Darknet detector, and
  `detect_3d_object_node` in one shot.
- If you prefer step-by-step control, run
  `1_oak_cam.launch.xml`, `2_darknet.launch.xml`, and `3_detect_3d.launch.xml`
  in separate terminals. Adjust `goal_object_class`, `goal_object_identifier`,
  or frame remappings via launch arguments.

### `local_lidar_planner`
Provides a self-contained stack that aligns the stock Unitree LiDAR, fuses
scans, generates short collision-free paths, and commands the robot without a
joystick.

Nodes:
- `utlidar_transformer.py`
  - Intent: Apply a fixed transform so LiDAR points align with the chassis frame
    and broadcast the corresponding static TF.
  - Subscribes: `/utlidar/cloud` (configurable `input_topic`)
  - Publishes: `/utlidar/transformed_cloud` (configurable `output_topic`), static
    TF between `target_frame` and `lidar_frame`
- `lidar_accumulator.py`
  - Intent: Maintain a sliding window of scans to densify local observations.
  - Subscribes: `/utlidar/transformed_cloud`
  - Publishes: `/utlidar/accumulated_cloud`, `~/debug` (String); exposes
    `~/dump_status` (Trigger service) for quick health checks
- `local_lidar_planner_simple.py`
  - Intent: Build a small occupancy grid from the accumulated clouds and create
    a short A*-based plan toward the most recent goal.
  - Subscribes: `/utlidar/accumulated_cloud` (or any `scan_topic` you set),
    `/local_goal_point`
  - Publishes: `/path`, `/goal_preview`, `/goal_target`, `/local_obstacles`,
    `/local_obstacles_grid`, `/local_potential_grid`
- `path_follower_simple.py`
  - Intent: Run a pure-pursuit style controller that keeps heading aligned with
    the planner’s goal while regulating speed.
  - Subscribes: `/path`, `/goal_target`
  - Publishes: `/cmd_vel`, `/api/sport/request` (when `is_real_robot:=true`)

Launch:
- `ros2 launch local_lidar_planner local_lidar_planner.launch.xml`
  brings up the full LiDAR stack. Common tweaks include
  `path_frame:=base_link`, `goal_topic:=/local_goal_point`, and the LiDAR pose
  arguments (`lidar_x`, `lidar_y`, `lidar_z`, `lidar_roll`, `lidar_pitch`,
  `lidar_yaw`). The launch file wires the transformer → accumulator →
  planner → follower chain automatically, so once `local_goal_point` is fed
  (e.g., from `detect_3d_object_node`) the robot will start moving.

## Typical flow
1. Start the perception pipeline (`detect_3d_object` launch of your choice) so
   `local_goal_point` is produced for your target class.
2. Launch the LiDAR planner: `ros2 launch local_lidar_planner local_lidar_planner.launch.xml`.
3. Monitor `/path`, `/cmd_vel`, and the TF tree in RViz to confirm the planner
   is respecting detected goals before switching to the real robot.
