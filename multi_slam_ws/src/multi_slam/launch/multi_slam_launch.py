from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the package share directory
    pkg_share = get_package_share_directory('multi_slam')
    
    # Create path to RViz config file
    rviz_config_path = os.path.join(pkg_share, 'rviz', 'config.rviz')
    
    return LaunchDescription([
        Node(
            package='multi_slam',
            executable='physics_sim',
            name='physics_sim',
            output='screen'
        ),
        Node(
            package='multi_slam',
            executable='controller_node',
            name='controller_node',
            output='screen'
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_path]
        )
    ])