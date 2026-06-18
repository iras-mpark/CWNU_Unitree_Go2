from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    args = [
        # This launch is intended for the OAK / YOLO / Jetson computer only.
        # The Go2 LiDAR planner should run on the robot/control computer and
        # subscribe to /local_goal_point and /target/status over the ROS 2 network.
        DeclareLaunchArgument('launch_oak_driver', default_value='true'),
        DeclareLaunchArgument('oak_camera_model', default_value='OAK-D-PRO-W'),
        DeclareLaunchArgument('oak_parent_frame', default_value='base_link'),
        DeclareLaunchArgument('rectify_rgb', default_value='false'),

        # Topics produced by depthai_ros_driver for the selected OAK model.
        DeclareLaunchArgument('rgb_topic', default_value='/oak/rgb/image_raw'),
        DeclareLaunchArgument('depth_topic', default_value='/oak/stereo/image_raw'),
        DeclareLaunchArgument('camera_info_topic', default_value='/oak/stereo/camera_info'),

        # Topics consumed by the Go2-side package over the ROS 2 network.
        DeclareLaunchArgument('target_point_topic', default_value='/local_goal_point'),
        DeclareLaunchArgument('target_status_topic', default_value='/target/status'),
        DeclareLaunchArgument('debug_image_topic', default_value='/target/debug_image/compressed'),
        DeclareLaunchArgument('debug_raw_image_topic', default_value='/target/debug_image'),
        DeclareLaunchArgument('target_camera_info_topic', default_value='/target/camera_info'),
        DeclareLaunchArgument('target_frame', default_value='base_link'),

        # YOLO / tracking. For Jetson TensorRT, set model_path to a YOLO11 .engine file.
        DeclareLaunchArgument('model_path', default_value='yolo11n.pt'),
        DeclareLaunchArgument('fallback_model_path', default_value=''),
        DeclareLaunchArgument('device', default_value='auto'),
        DeclareLaunchArgument('imgsz', default_value='640'),
        DeclareLaunchArgument('tracker', default_value='bytetrack.yaml'),
        DeclareLaunchArgument('use_yolo_tracking', default_value='false'),
        DeclareLaunchArgument('conf_threshold', default_value='0.35'),
        DeclareLaunchArgument('iou_threshold', default_value='0.50'),
        DeclareLaunchArgument('half', default_value='true'),

        # Target acquisition / depth robustness.
        DeclareLaunchArgument('acquire_stable_frames', default_value='1'),
        DeclareLaunchArgument('acquire_max_distance_m', default_value='6.0'),
        DeclareLaunchArgument('nearest_person_use_center_gate', default_value='false'),
        DeclareLaunchArgument('lost_timeout_s', default_value='1.0'),
        DeclareLaunchArgument('depth_quantile_low', default_value='0.25'),
        DeclareLaunchArgument('depth_quantile_high', default_value='0.50'),
        DeclareLaunchArgument('depth_roi_scale', default_value='0.60'),
        DeclareLaunchArgument('depth_roi_max_pixels', default_value='180'),
        DeclareLaunchArgument('depth_sample_stride', default_value='2'),

        # Kalman filter before publishing /local_goal_point.
        # The KF smooths noisy depth and associates multiple person detections
        # by nearest distance to the predicted filter estimate.
        DeclareLaunchArgument('use_kalman_filter', default_value='true'),
        DeclareLaunchArgument('kalman_process_noise_std_m', default_value='0.18'),
        DeclareLaunchArgument('kalman_measurement_noise_x_m', default_value='0.35'),
        DeclareLaunchArgument('kalman_measurement_noise_y_m', default_value='0.18'),
        DeclareLaunchArgument('kalman_measurement_noise_z_m', default_value='0.25'),
        DeclareLaunchArgument('kalman_initial_variance_m2', default_value='0.25'),
        DeclareLaunchArgument('kalman_association_gate_m', default_value='1.20'),
        DeclareLaunchArgument('kalman_reset_timeout_s', default_value='1.50'),
        DeclareLaunchArgument('kalman_publish_prediction_without_measurement', default_value='false'),
        DeclareLaunchArgument('kalman_strict_measurement_required', default_value='true'),
        DeclareLaunchArgument('kalman_missed_frames_to_reset', default_value='1'),

        # Compressed debug image for wireless monitoring.
        DeclareLaunchArgument('publish_debug_image', default_value='true'),
        DeclareLaunchArgument('publish_debug_raw_image', default_value='false'),
        DeclareLaunchArgument('publish_target_camera_info', default_value='true'),
        DeclareLaunchArgument('debug_image_rate_hz', default_value='5.0'),
        DeclareLaunchArgument('debug_jpeg_quality', default_value='35'),
        DeclareLaunchArgument('debug_resize_width', default_value='640'),
    ]

    oak_driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('depthai_ros_driver'),
                'launch',
                'camera.launch.py',
            ])
        ),
        condition=IfCondition(LaunchConfiguration('launch_oak_driver')),
        launch_arguments={
            'camera_model': LaunchConfiguration('oak_camera_model'),
            'parent_frame': LaunchConfiguration('oak_parent_frame'),
            'rectify_rgb': LaunchConfiguration('rectify_rgb'),
        }.items(),
    )

    node = Node(
        package='oak_yolo_target',
        executable='yolo_target_node',
        name='oak_yolo_target_node',
        output='screen',
        parameters=[{
            'rgb_topic': LaunchConfiguration('rgb_topic'),
            'depth_topic': LaunchConfiguration('depth_topic'),
            'camera_info_topic': LaunchConfiguration('camera_info_topic'),
            'target_point_topic': LaunchConfiguration('target_point_topic'),
            'target_status_topic': LaunchConfiguration('target_status_topic'),
            'debug_image_topic': LaunchConfiguration('debug_image_topic'),
            'debug_raw_image_topic': LaunchConfiguration('debug_raw_image_topic'),
            'target_camera_info_topic': LaunchConfiguration('target_camera_info_topic'),
            'target_frame': LaunchConfiguration('target_frame'),
            'model_path': LaunchConfiguration('model_path'),
            'fallback_model_path': LaunchConfiguration('fallback_model_path'),
            # Force string type. Without this, ROS 2 may parse the default '0' as integer
            # and reject it because yolo_target_node declares 'device' as a string.
            'device': ParameterValue(LaunchConfiguration('device'), value_type=str),
            'imgsz': ParameterValue(LaunchConfiguration('imgsz'), value_type=int),
            'tracker': LaunchConfiguration('tracker'),
            'use_yolo_tracking': ParameterValue(LaunchConfiguration('use_yolo_tracking'), value_type=bool),
            'conf_threshold': ParameterValue(LaunchConfiguration('conf_threshold'), value_type=float),
            'iou_threshold': ParameterValue(LaunchConfiguration('iou_threshold'), value_type=float),
            'half': ParameterValue(LaunchConfiguration('half'), value_type=bool),
            'depth_quantile_low': ParameterValue(LaunchConfiguration('depth_quantile_low'), value_type=float),
            'depth_quantile_high': ParameterValue(LaunchConfiguration('depth_quantile_high'), value_type=float),
            'depth_roi_scale': ParameterValue(LaunchConfiguration('depth_roi_scale'), value_type=float),
            'depth_roi_max_pixels': ParameterValue(LaunchConfiguration('depth_roi_max_pixels'), value_type=int),
            'depth_sample_stride': ParameterValue(LaunchConfiguration('depth_sample_stride'), value_type=int),
            'acquire_stable_frames': ParameterValue(LaunchConfiguration('acquire_stable_frames'), value_type=int),
            'acquire_max_distance_m': ParameterValue(LaunchConfiguration('acquire_max_distance_m'), value_type=float),
            'nearest_person_use_center_gate': ParameterValue(LaunchConfiguration('nearest_person_use_center_gate'), value_type=bool),
            'lost_timeout_s': ParameterValue(LaunchConfiguration('lost_timeout_s'), value_type=float),
            'use_kalman_filter': ParameterValue(LaunchConfiguration('use_kalman_filter'), value_type=bool),
            'kalman_process_noise_std_m': ParameterValue(LaunchConfiguration('kalman_process_noise_std_m'), value_type=float),
            'kalman_measurement_noise_x_m': ParameterValue(LaunchConfiguration('kalman_measurement_noise_x_m'), value_type=float),
            'kalman_measurement_noise_y_m': ParameterValue(LaunchConfiguration('kalman_measurement_noise_y_m'), value_type=float),
            'kalman_measurement_noise_z_m': ParameterValue(LaunchConfiguration('kalman_measurement_noise_z_m'), value_type=float),
            'kalman_initial_variance_m2': ParameterValue(LaunchConfiguration('kalman_initial_variance_m2'), value_type=float),
            'kalman_association_gate_m': ParameterValue(LaunchConfiguration('kalman_association_gate_m'), value_type=float),
            'kalman_reset_timeout_s': ParameterValue(LaunchConfiguration('kalman_reset_timeout_s'), value_type=float),
            'kalman_publish_prediction_without_measurement': ParameterValue(LaunchConfiguration('kalman_publish_prediction_without_measurement'), value_type=bool),
            'kalman_strict_measurement_required': ParameterValue(LaunchConfiguration('kalman_strict_measurement_required'), value_type=bool),
            'kalman_missed_frames_to_reset': ParameterValue(LaunchConfiguration('kalman_missed_frames_to_reset'), value_type=int),
            'publish_debug_image': ParameterValue(LaunchConfiguration('publish_debug_image'), value_type=bool),
            'publish_debug_raw_image': ParameterValue(LaunchConfiguration('publish_debug_raw_image'), value_type=bool),
            'publish_target_camera_info': ParameterValue(LaunchConfiguration('publish_target_camera_info'), value_type=bool),
            'debug_image_rate_hz': ParameterValue(LaunchConfiguration('debug_image_rate_hz'), value_type=float),
            'debug_jpeg_quality': ParameterValue(LaunchConfiguration('debug_jpeg_quality'), value_type=int),
            'debug_resize_width': ParameterValue(LaunchConfiguration('debug_resize_width'), value_type=int),
        }],
    )

    return LaunchDescription(args + [oak_driver, node])
