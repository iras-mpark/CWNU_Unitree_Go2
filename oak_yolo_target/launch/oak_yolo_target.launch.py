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
        DeclareLaunchArgument('target_frame', default_value='base_link'),

        # YOLO / tracking. For Jetson TensorRT, set model_path to a YOLO11 .engine file.
        DeclareLaunchArgument('model_path', default_value='yolo11n.pt'),
        DeclareLaunchArgument('fallback_model_path', default_value=''),
        DeclareLaunchArgument('device', default_value='0'),
        DeclareLaunchArgument('imgsz', default_value='640'),
        DeclareLaunchArgument('tracker', default_value='bytetrack.yaml'),
        DeclareLaunchArgument('conf_threshold', default_value='0.35'),
        DeclareLaunchArgument('iou_threshold', default_value='0.50'),
        DeclareLaunchArgument('half', default_value='true'),

        # Target acquisition / depth robustness.
        DeclareLaunchArgument('acquire_stable_frames', default_value='8'),
        DeclareLaunchArgument('acquire_max_distance_m', default_value='6.0'),
        DeclareLaunchArgument('lost_timeout_s', default_value='1.0'),
        DeclareLaunchArgument('depth_quantile_low', default_value='0.25'),
        DeclareLaunchArgument('depth_quantile_high', default_value='0.50'),
        DeclareLaunchArgument('depth_roi_scale', default_value='0.60'),
        DeclareLaunchArgument('depth_roi_max_pixels', default_value='180'),
        DeclareLaunchArgument('depth_sample_stride', default_value='2'),

        # Compressed debug image for wireless monitoring.
        DeclareLaunchArgument('publish_debug_image', default_value='true'),
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
            'target_frame': LaunchConfiguration('target_frame'),
            'model_path': LaunchConfiguration('model_path'),
            'fallback_model_path': LaunchConfiguration('fallback_model_path'),
            'device': ParameterValue(LaunchConfiguration('device'), value_type=str),
            'imgsz': ParameterValue(LaunchConfiguration('imgsz'), value_type=int),
            'tracker': LaunchConfiguration('tracker'),
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
            'lost_timeout_s': ParameterValue(LaunchConfiguration('lost_timeout_s'), value_type=float),
            'publish_debug_image': ParameterValue(LaunchConfiguration('publish_debug_image'), value_type=bool),
            'debug_image_rate_hz': ParameterValue(LaunchConfiguration('debug_image_rate_hz'), value_type=float),
            'debug_jpeg_quality': ParameterValue(LaunchConfiguration('debug_jpeg_quality'), value_type=int),
            'debug_resize_width': ParameterValue(LaunchConfiguration('debug_resize_width'), value_type=int),
        }],
    )

    return LaunchDescription(args + [oak_driver, node])
