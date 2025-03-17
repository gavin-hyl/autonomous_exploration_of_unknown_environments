from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the package share directory
    pkg_share = get_package_share_directory('multi_slam')
    
    # Create path to RViz config file
    rviz_config_path = os.path.join(pkg_share, 'rviz', 'config.rviz')
    
    # Get the absolute path to the script
    script_path = '/home/tkleeneuron/cs133b-final-project/multi_slam_ws/src/multi_slam/multi_slam/autonomous_control/rl_approach.py'
    
    return LaunchDescription([
        Node(
            package='multi_slam',
            executable='physics_sim',
            name='physics_sim',
            output='screen'
        ),
        # SLAM node for simultaneous localization and mapping
        Node(
            package='multi_slam',
            executable='slam_node',
            name='slam_node',
            output='screen',
        ),
        # Reinforcement Learning autonomous exploration controller
        ExecuteProcess(
            cmd=['python3', script_path,
                 '--ros-args',
                 '-p', 'state_size:=32',
                 '-p', 'action_size:=8',
                 '-p', 'learning_rate:=0.001',
                 '-p', 'gamma:=0.99',
                 '-p', 'epsilon_initial:=1.0',
                 '-p', 'epsilon_min:=0.1',
                 '-p', 'epsilon_decay:=0.995',
                 '-p', 'batch_size:=32',
                 '-p', 'update_frequency:=10',
                 '-p', 'train:=false',
                 '-p', 'model_path:=rl_explorer_model',
                 '-p', 'occupancy_threshold:=50',
                 '-p', 'move_step:=0.5',
                 '-p', 'reward_complete_map:=100.0',
                 '-p', 'reward_discover_beacon:=50.0',
                 '-p', 'reward_collision:=-20.0',
                 '-p', 'reward_explore_unknown:=1.0',
            ],
            name='rl_exploration_controller',
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