from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = [
        # This launch is intended for the Go2 / LiDAR / control computer only.
        # The OAK + YOLO package should run on the camera/Jetson computer and
        # publish /local_goal_point and /target/status over the ROS 2 network.
        DeclareLaunchArgument('target_frame', default_value='base_link'),
        DeclareLaunchArgument('raw_lidar_topic', default_value='/utlidar/cloud'),
        DeclareLaunchArgument('target_topic', default_value='/local_goal_point'),
        DeclareLaunchArgument('target_status_topic', default_value='/target/status'),
        DeclareLaunchArgument('path_topic', default_value='/path'),
        DeclareLaunchArgument('cmd_vel_topic', default_value='/cmd_vel'),
        DeclareLaunchArgument('sport_request_topic', default_value='/api/sport/request'),
        DeclareLaunchArgument('follower_status_topic', default_value='/go2_follower/status'),

        # LiDAR mounting correction. Defaults preserve the original package's pitch.
        DeclareLaunchArgument('lidar_roll', default_value='0.0'),
        DeclareLaunchArgument('lidar_pitch', default_value='2.8782025850555556'),
        DeclareLaunchArgument('lidar_yaw', default_value='0.0'),
        DeclareLaunchArgument('lidar_x', default_value='0.0'),
        DeclareLaunchArgument('lidar_y', default_value='0.0'),
        DeclareLaunchArgument('lidar_z', default_value='0.0'),

        # LiDAR moving-window accumulation.
        DeclareLaunchArgument('history_duration', default_value='0.8'),
        DeclareLaunchArgument('max_clouds', default_value='10'),
        DeclareLaunchArgument('accumulator_publish_rate_hz', default_value='15.0'),

        # Planner / follower.
        DeclareLaunchArgument('follow_distance_m', default_value='2.0'),
        DeclareLaunchArgument('grid_resolution_m', default_value='0.12'),
        DeclareLaunchArgument('safety_radius_m', default_value='0.35'),
        DeclareLaunchArgument('potential_radius_m', default_value='0.95'),
        DeclareLaunchArgument('potential_weight', default_value='4.0'),
        DeclareLaunchArgument('waypoint_potential_threshold', default_value='0.35'),
        # Only points inside this height band are treated as obstacles.
        # The lower bound intentionally removes floor/ground returns from the 3D LiDAR.
        DeclareLaunchArgument('obstacle_z_min_m', default_value='0.1'),
        DeclareLaunchArgument('obstacle_z_max_m', default_value='1.20'),
        DeclareLaunchArgument('api_control_enabled', default_value='true'),
        DeclareLaunchArgument('require_target_status', default_value='false'),
        DeclareLaunchArgument('max_forward_speed_mps', default_value='1.0'),
        DeclareLaunchArgument('max_lateral_speed_mps', default_value='0.5'),
        # Yaw damping. These defaults intentionally prioritize stable following
        # over exact pixel-center alignment because YOLO/depth target points jitter.
        DeclareLaunchArgument('yaw_gain', default_value='0.9'),
        DeclareLaunchArgument('max_yaw_rate_rps', default_value='0.55'),
        DeclareLaunchArgument('yaw_deadband_rad', default_value='0.12'),
        DeclareLaunchArgument('yaw_filter_alpha', default_value='0.25'),
        DeclareLaunchArgument('yaw_slew_rate_limit_rps2', default_value='1.2'),
        DeclareLaunchArgument('yaw_hold_when_close', default_value='true'),
        DeclareLaunchArgument('target_stale_timeout_s', default_value='2.0'),
        DeclareLaunchArgument('path_stale_timeout_s', default_value='1.5'),
    ]

    lidar_transformer = Node(
        package='go2_lidar_planner',
        executable='lidar_transformer_node',
        name='go2_lidar_transformer',
        output='screen',
        parameters=[{
            'input_topic': LaunchConfiguration('raw_lidar_topic'),
            'output_topic': '/utlidar/transformed_cloud',
            'target_frame': LaunchConfiguration('target_frame'),
            'lidar_roll': ParameterValue(LaunchConfiguration('lidar_roll'), value_type=float),
            'lidar_pitch': ParameterValue(LaunchConfiguration('lidar_pitch'), value_type=float),
            'lidar_yaw': ParameterValue(LaunchConfiguration('lidar_yaw'), value_type=float),
            'lidar_x': ParameterValue(LaunchConfiguration('lidar_x'), value_type=float),
            'lidar_y': ParameterValue(LaunchConfiguration('lidar_y'), value_type=float),
            'lidar_z': ParameterValue(LaunchConfiguration('lidar_z'), value_type=float),
        }],
    )

    accumulator = Node(
        package='go2_lidar_planner',
        executable='lidar_accumulator_node',
        name='go2_lidar_accumulator',
        output='screen',
        parameters=[{
            'input_topic': '/utlidar/transformed_cloud',
            'output_topic': '/utlidar/accumulated_cloud',
            'history_duration': ParameterValue(LaunchConfiguration('history_duration'), value_type=float),
            'max_clouds': ParameterValue(LaunchConfiguration('max_clouds'), value_type=int),
            'publish_rate_hz': ParameterValue(LaunchConfiguration('accumulator_publish_rate_hz'), value_type=float),
        }],
    )

    planner = Node(
        package='go2_lidar_planner',
        executable='potential_astar_planner_node',
        name='go2_potential_astar_planner',
        output='screen',
        parameters=[{
            'path_frame': LaunchConfiguration('target_frame'),
            'scan_topic': '/utlidar/accumulated_cloud',
            'target_topic': LaunchConfiguration('target_topic'),
            'follow_distance_m': ParameterValue(LaunchConfiguration('follow_distance_m'), value_type=float),
            'grid_resolution_m': ParameterValue(LaunchConfiguration('grid_resolution_m'), value_type=float),
            'safety_radius_m': ParameterValue(LaunchConfiguration('safety_radius_m'), value_type=float),
            'potential_radius_m': ParameterValue(LaunchConfiguration('potential_radius_m'), value_type=float),
            'potential_weight': ParameterValue(LaunchConfiguration('potential_weight'), value_type=float),
            'waypoint_potential_threshold': ParameterValue(LaunchConfiguration('waypoint_potential_threshold'), value_type=float),
            'obstacle_z_min_m': ParameterValue(LaunchConfiguration('obstacle_z_min_m'), value_type=float),
            'obstacle_z_max_m': ParameterValue(LaunchConfiguration('obstacle_z_max_m'), value_type=float),
            # OAK/YOLO may run on another device and can be slower than LiDAR/control.
            # Use the same target timeout in planner and follower.
            'target_stale_timeout_s': ParameterValue(LaunchConfiguration('target_stale_timeout_s'), value_type=float),
        }],
    )

    follower = Node(
        package='go2_lidar_planner',
        executable='go2_path_follower_node',
        name='go2_path_follower',
        output='screen',
        parameters=[{
            'follow_distance_m': ParameterValue(LaunchConfiguration('follow_distance_m'), value_type=float),
            'api_control_enabled': ParameterValue(LaunchConfiguration('api_control_enabled'), value_type=bool),
            'max_forward_speed_mps': ParameterValue(LaunchConfiguration('max_forward_speed_mps'), value_type=float),
            'max_lateral_speed_mps': ParameterValue(LaunchConfiguration('max_lateral_speed_mps'), value_type=float),
            'yaw_gain': ParameterValue(LaunchConfiguration('yaw_gain'), value_type=float),
            'max_yaw_rate_rps': ParameterValue(LaunchConfiguration('max_yaw_rate_rps'), value_type=float),
            'yaw_deadband_rad': ParameterValue(LaunchConfiguration('yaw_deadband_rad'), value_type=float),
            'yaw_filter_alpha': ParameterValue(LaunchConfiguration('yaw_filter_alpha'), value_type=float),
            'yaw_slew_rate_limit_rps2': ParameterValue(LaunchConfiguration('yaw_slew_rate_limit_rps2'), value_type=float),
            'yaw_hold_when_close': ParameterValue(LaunchConfiguration('yaw_hold_when_close'), value_type=bool),
            'target_stale_timeout_s': ParameterValue(LaunchConfiguration('target_stale_timeout_s'), value_type=float),
            'path_stale_timeout_s': ParameterValue(LaunchConfiguration('path_stale_timeout_s'), value_type=float),
            'target_topic': LaunchConfiguration('target_topic'),
            'target_status_topic': LaunchConfiguration('target_status_topic'),
            'path_topic': LaunchConfiguration('path_topic'),
            'cmd_vel_topic': LaunchConfiguration('cmd_vel_topic'),
            'sport_request_topic': LaunchConfiguration('sport_request_topic'),
            'follower_status_topic': LaunchConfiguration('follower_status_topic'),
            'require_target_status': ParameterValue(LaunchConfiguration('require_target_status'), value_type=bool),
            'cmd_frame': LaunchConfiguration('target_frame'),
        }],
    )

    return LaunchDescription(args + [lidar_transformer, accumulator, planner, follower])
