# Go2 OAK YOLO + LiDAR Person-Following Redesign

This rewrite keeps the same hardware assumptions as the original project:

- Luxonis OAK RGB-D camera
- NVIDIA Jetson Orin for native YOLO inference / optional TensorRT engine
- Unitree Go2 + Unitree ROS2 API bridge
- Unitree LiDAR point cloud topic

The system is intentionally split into **two ROS2 Python packages** because the camera/YOLO side and the Go2/LiDAR/control side may run on different devices.

```text
oak_yolo_target       # OAK RGB-D + native Ultralytics YOLO tracking -> 3D person target
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
  -> Ultralytics YOLO11 track mode, person class only
  -> chosen target track ID

/oak/stereo/image_raw + /oak/stereo/camera_info
  -> robust depth from box ROI quantiles
  -> /local_goal_point geometry_msgs/PointStamped in base_link
  -> /target/status std_msgs/String JSON
  -> /target/debug_image/compressed sensor_msgs/CompressedImage

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

## Target acquisition policy

For startup without a monitor/UI, the perception node uses an automatic lock rule:

1. Detect only class `person`.
2. Reject invalid depth.
3. Prefer the nearest person inside a configurable center gate.
4. Lock only after the same candidate is stable for `acquire_stable_frames` frames.
5. After lock, follow the same YOLO tracking ID when available.
6. If YOLO ID is temporarily missing, short-term reacquisition uses 3D proximity.
7. If the target is lost beyond `lost_timeout_s`, the lock is cleared.

This is intentionally conservative. It avoids suddenly switching to a different person every frame.

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
  debug_jpeg_quality:=35
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
/utlidar/transformed_cloud         sensor_msgs/PointCloud2
/utlidar/accumulated_cloud         sensor_msgs/PointCloud2
/local_obstacle_grid               nav_msgs/OccupancyGrid
/local_potential_grid              nav_msgs/OccupancyGrid
/path                              nav_msgs/Path
/goal_waypoint                     geometry_msgs/PointStamped
/cmd_vel                           geometry_msgs/TwistStamped
/api/sport/request                 unitree_api/Request
```
