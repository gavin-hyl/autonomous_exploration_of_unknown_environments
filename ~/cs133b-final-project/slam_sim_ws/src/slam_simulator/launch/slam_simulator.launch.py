from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    """
    SLAM 시뮬레이터를 위한 런치 파일입니다.
    SLAM 노드와 선택적으로 RViz를 실행합니다.
    """
    pkg_slam_simulator = get_package_share_directory('slam_simulator')
    
    # RViz 설정 파일 경로
    rviz_config_dir = os.path.join(pkg_slam_simulator, 'rviz', 'slam.rviz')
    
    # 런치 파라미터
    use_rviz = LaunchConfiguration('use_rviz')
    declare_use_rviz = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='True to start RViz'
    )
    
    # SLAM 노드
    slam_node = Node(
        package='slam_simulator',
        executable='slam_node',
        name='slam_node',
        output='screen',
        parameters=[
            {'map_resolution': 0.05},
            {'map_width': 20.0},
            {'map_height': 20.0},
            {'update_frequency': 10.0},
            {'exploration_weight': 0.7}
        ]
    )
    
    # RViz 노드
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_dir],
        condition=LaunchConfiguration('use_rviz')
    )
    
    return LaunchDescription([
        declare_use_rviz,
        slam_node,
        rviz_node
    ]) 