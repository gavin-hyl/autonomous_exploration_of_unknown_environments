from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """Generate launch description for RRT-based autonomous driving."""
    
    package_dir = get_package_share_directory('multi_slam')
    
    # Start the physics simulation node
    physics_sim_node = Node(
        package='multi_slam',
        executable='physics_sim',
        name='physics_sim_node',
        parameters=[{
            'update_rate': 50.0,  # Hz
            'use_sim_time': False,
            'beacon_count': 10,
            'world_size': 50.0,
            'map_origin_x': -25.0,
            'map_origin_y': -25.0,
            'lidar_max_range': 10.0,
            'lidar_beam_count': 360,
            'beacon_observation_probability': 0.7,
            'random_seed': 42
        }]
    )
    
    # Start the SLAM node with RRT planner enabled
    slam_node = Node(
        package='multi_slam',
        executable='planner_slam_node',
        name='planner_slam_node',
        parameters=[{
            'map_size_x': 50.0,
            'map_size_y': 50.0,
            'map_origin_x': -25.0,
            'map_origin_y': -25.0,
            'grid_size': 0.1,
            'num_particles': 1000,
            'position_std_dev': 0.1,
            'initial_noise': 0.5,
            # Planner parameters
            'use_planner': True,
            'rrt_step_size': 0.5,
            'rrt_max_iter': 500,
            'rrt_goal_sample_rate': 5,
            'rrt_connect_circle_dist': 0.5,
            'pd_p_gain': 1.0,
            'pd_d_gain': 0.1,
            'entropy_weight': 1.0,
            'beacon_weight': 2.0,
            'beacon_attraction_radius': 3.0
        }]
    )
    
    # Start the planner controller node
    controller_node = Node(
        package='multi_slam',
        executable='planner_controller_node',
        name='planner_controller_node',
        parameters=[{
            'max_speed': 1.0,
            'control_frequency': 10.0,
            'use_manual_fallback': True,
            'fallback_control_x': 0.0,
            'fallback_control_y': 0.5
        }]
    )
    
    # Start RViz for visualization
    rviz_config_path = os.path.join(package_dir, 'rviz', 'config.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        arguments=['-d', rviz_config_path]
    )

    # Return the launch description
    return LaunchDescription([
        physics_sim_node,
        slam_node,
        controller_node,
        rviz_node
    ]) 