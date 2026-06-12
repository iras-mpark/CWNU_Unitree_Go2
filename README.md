# Go2 OAK YOLO + LiDAR Person-Following Redesign

This rewrite keeps the same hardware assumptions as the original project:

- Luxonis OAK RGB-D camera
- NVIDIA Jetson Orin for native YOLO inference / optional TensorRT engine
- Unitree Go2 + Unitree ROS2 API bridge
- Unitree LiDAR point cloud topic

The system is intentionally split into **two ROS2 Python packages** because the camera/YOLO side and the Go2/LiDAR/control side may run on different devices.

```text
oak_yolo_target       # OAK RGB-D + native Ultralytics YOLO detection -> nearest 3D person target
go2_lidar_planner     # LiDAR accumulation + potential-field A* + Go2 sport API follower
```

## Device-level architecture

```text
[Device A: Jetson + OAK camera]
  ros2 launch oak_yolo_target oak_yolo_target.launch.py

  Publishes over ROS 2 network:
    /local_goal_point
    /target/status
    /target/debug_image/compressed
    /target/debug_image

                 ROS 2 DDS network
                       ↓

[Device B: Go2 / LiDAR / Unitree API control computer]
  ros2 launch go2_lidar_planner go2_lidar_planner.launch.py

  Subscribes:
    /local_goal_point
    /target/status

  Publishes:
    /api/sport/request
```

Both devices must be on the same ROS 2 network, with matching `ROS_DOMAIN_ID` and compatible DDS/multicast configuration.

## Data flow

```text
/oak/rgb/image_raw
  -> Ultralytics YOLO11 detect mode, person class only
  -> nearest valid person selected every frame; tracking IDs are ignored

/oak/stereo/image_raw + /oak/stereo/camera_info
  -> robust depth from box ROI quantiles
  -> /local_goal_point geometry_msgs/PointStamped in base_link
  -> /target/status std_msgs/String JSON
  -> /target/debug_image/compressed sensor_msgs/CompressedImage
  -> /target/debug_image sensor_msgs/Image

/utlidar/cloud
  -> fixed mounting transform
  -> /utlidar/transformed_cloud
  -> sliding time-window accumulation
  -> /utlidar/accumulated_cloud
  -> potential-field cost map + A*
  -> /path nav_msgs/Path
  -> /goal_waypoint and /goal_target
  -> Go2 path follower
  -> /cmd_vel TwistStamped and /api/sport/request unitree_api/Request
```

## Target selection policy

The perception node now ignores YOLO tracking IDs by default because ID switching caused unstable following.

1. Detect only class `person`.
2. Reject invalid or out-of-range depth.
3. Reject candidates behind the robot or beyond `acquire_max_distance_m`.
4. Select the nearest valid person every RGB frame.
5. Publish that person's 3D point immediately to `/local_goal_point`.

The optional `nearest_person_use_center_gate:=true` parameter can re-enable a camera-center gate, but the default is `false` so the closest detected person is used directly.

## Depth estimation

For each YOLO box, the node samples a center crop inside the box rather than a single center pixel. It then:

1. removes NaN/zero/out-of-range depth values,
2. computes depth quantiles,
3. averages only the quantile band.

Default values are closer-biased:

```text
depth_quantile_low  = 0.25
depth_quantile_high = 0.50
```

Use `0.25` to `0.75` if you want a more neutral robust average.

## Planner behavior

The LiDAR planner accumulates point clouds over a short moving time window. It then builds:

- hard inflated obstacle cells using `safety_radius_m`,
- soft potential cost around obstacles using `potential_radius_m`,
- an A* path weighted by the potential field.

The desired waypoint is placed on the line to the target but `follow_distance_m` short of the person. The default is 2.0 m. If the waypoint lands inside an obstacle or high-potential region, it is moved to the nearest free boundary cell.

## Manual control safety

The follower sends Unitree sport API commands only while target status and path are fresh. If the target disappears, it sends one STOP command, then stops publishing Go2 sport requests so manual control is not continuously overwritten.

## Build on Device A: Jetson + OAK

Copy only the OAK/YOLO package to the Jetson workspace:

```bash
cd ~/ros2_ws/src
cp -r /path/to/oak_yolo_target .
cd ~/ros2_ws
rosdep install --from-paths src -y --ignore-src
colcon build --symlink-install --packages-select oak_yolo_target
source install/setup.bash
```

Python dependencies on Jetson:

```bash
pip3 install ultralytics
sudo apt install -y ros-humble-cv-bridge ros-humble-depthai-ros-driver
```

Launch Device A:

```bash
ros2 launch oak_yolo_target oak_yolo_target.launch.py
```

This starts `depthai_ros_driver` by default with:

```text
camera_model:=OAK-D-PRO-W
parent_frame:=base_link
rectify_rgb:=false
```

If the OAK driver is already running separately:

```bash
ros2 launch oak_yolo_target oak_yolo_target.launch.py launch_oak_driver:=false
```

Common Device A overrides:

```bash
ros2 launch oak_yolo_target oak_yolo_target.launch.py \
  model_path:=yolo11n.pt \
  rgb_topic:=/oak/rgb/image_raw \
  depth_topic:=/oak/stereo/image_raw \
  camera_info_topic:=/oak/stereo/camera_info \
  depth_quantile_low:=0.25 \
  depth_quantile_high:=0.50 \
  debug_jpeg_quality:=35 \
  use_yolo_tracking:=false \
  nearest_person_use_center_gate:=false
```

## Jetson TensorRT model export

On the Jetson itself, export the model to TensorRT engine format:

```bash
yolo export model=yolo11n.pt format=engine imgsz=640 half=True device=0
```

Then launch the OAK/YOLO package with the engine:

```bash
ros2 launch oak_yolo_target oak_yolo_target.launch.py \
  model_path:=/absolute/path/to/yolo11n.engine \
  fallback_model_path:=
```

The default model is `yolo11n.pt`. On Jetson, use a YOLO11 TensorRT `.engine` as `model_path` after export.

## Build on Device B: Go2 / LiDAR / Unitree API

Copy only the Go2/LiDAR package to the Go2/control workspace:

```bash
cd ~/ros2_ws/src
cp -r /path/to/go2_lidar_planner .
cd ~/ros2_ws
rosdep install --from-paths src -y --ignore-src
colcon build --symlink-install --packages-select go2_lidar_planner
source install/setup.bash
```

Launch Device B:

```bash
ros2 launch go2_lidar_planner go2_lidar_planner.launch.py
```

Common Device B overrides:

```bash
ros2 launch go2_lidar_planner go2_lidar_planner.launch.py \
  raw_lidar_topic:=/utlidar/cloud \
  target_topic:=/local_goal_point \
  target_status_topic:=/target/status \
  follow_distance_m:=2.0 \
  api_control_enabled:=true
```

For safe debug without sending Go2 sport API requests:

```bash
ros2 launch go2_lidar_planner go2_lidar_planner.launch.py api_control_enabled:=false
```

## ROS 2 network check between the two devices

On Device B, verify that the target topics from Device A are visible:

```bash
ros2 topic list | grep target
ros2 topic echo /local_goal_point
ros2 topic echo /target/status
```

If they are not visible, check both devices:

```bash
echo $ROS_DOMAIN_ID
echo $RMW_IMPLEMENTATION
```

The two devices should use the same `ROS_DOMAIN_ID`. DDS discovery must also be allowed through the wireless network.

## Debug topics

```text
/local_goal_point                  geometry_msgs/PointStamped
/target/status                     std_msgs/String JSON
/target/debug_image/compressed     sensor_msgs/CompressedImage, JPEG compressed
/target/debug_image                sensor_msgs/Image, resized raw fallback for RViz without compressed plugin
/utlidar/transformed_cloud         sensor_msgs/PointCloud2
/utlidar/accumulated_cloud         sensor_msgs/PointCloud2
/local_obstacle_grid               nav_msgs/OccupancyGrid
/local_potential_grid              nav_msgs/OccupancyGrid
/path                              nav_msgs/Path
/goal_waypoint                     geometry_msgs/PointStamped
/cmd_vel                           geometry_msgs/TwistStamped
/api/sport/request                 unitree_api/Request
```


## v4 note

`oak_yolo_target.launch.py` now forces the YOLO `device` launch argument to ROS string type. This prevents ROS 2 Humble from coercing `device:=0` into an integer and crashing the node with `InvalidParameterTypeException`.

## v5 note

RViz2 can display `/target/debug_image/compressed` only if the monitoring computer has the compressed image transport subscriber plugin installed. On ROS 2 Humble, install it on the computer running RViz2:

```bash
sudo apt update
sudo apt install -y ros-humble-compressed-image-transport
source /opt/ros/humble/setup.bash
```

This version also publishes `/target/debug_image` as a resized raw `sensor_msgs/Image` fallback. In RViz2, use the raw topic if the compressed transport plugin is not installed. The raw fallback can be disabled with:

```bash
ros2 launch oak_yolo_target oak_yolo_target.launch.py publish_debug_raw_image:=false
```


## v6 note: YOLO CUDA fallback

`oak_yolo_target.launch.py` now defaults to `device:=auto`. If the active PyTorch install reports `torch.cuda.is_available() == False`, the node uses `device=cpu` instead of failing every frame with `Invalid CUDA device=0`. To force CPU explicitly:

```bash
ros2 launch oak_yolo_target oak_yolo_target.launch.py device:=cpu
```

For Jetson GPU acceleration, fix the Jetson PyTorch/CUDA environment or export a YOLO11 TensorRT engine and pass it as `model_path`.

### Floor / ground filtering for 3D LiDAR

The local planner treats only points in this vertical band as obstacles:

```bash
obstacle_z_min_m:=0.05
obstacle_z_max_m:=1.20
```

This keeps floor returns from the 3D LiDAR out of the 2D potential field.
If the floor still appears in `/local_obstacles` or `/local_obstacle_grid`, raise
`obstacle_z_min_m` to `0.08` or `0.10`. If low obstacles must be detected, lower it
carefully while checking the debug grids.



## v8 notes

- Fixed a Python f-string syntax error in `potential_astar_planner_node.py`.
- `go2_path_follower_node.py` now starts even if `unitree_api` is not importable. In that case it still publishes `/cmd_vel` for debugging, but it cannot publish `/api/sport/request`.
- For planner-only debugging without Unitree ROS2 API messages, run:

```bash
ros2 launch go2_lidar_planner go2_lidar_planner.launch.py api_control_enabled:=false
```

- For real Go2 sport API control, source/build the Unitree ROS2 interface workspace so that this works:

```bash
python3 - <<'PY'
from unitree_api.msg import Request
print('unitree_api OK')
PY
```


## v12 integration note

The Go2-side launch uses `target_stale_timeout_s:=2.0` and `path_stale_timeout_s:=1.5` by default because the OAK/YOLO node can run on a separate device and may publish `/local_goal_point` slower than the local LiDAR/control loop, especially when YOLO falls back to CPU. The planner and follower now use the same target timeout. Check `/go2_follower/status` for `target_age_s`, `path_age_s`, `target_timeout_s`, and `unitree_api_available`. If `unitree_api_available` is false, source the Unitree ROS2 interface workspace before launching this package; otherwise `/api/sport/request` cannot be published.
