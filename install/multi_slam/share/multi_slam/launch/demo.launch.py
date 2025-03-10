### Yeah this launch is not working

import launch
import launch_ros.actions
from multi_slam.Map import Map

def generate_launch_description():
    # Create shared world map
    world_map = Map()

    # Create nodes with shared world map
    controller_node = launch_ros.actions.Node(
        package='multi_slam',
        executable='controller_node',
        name='controller_node',
        parameters=[{'world_map': world_map}]
    )

    sim_node = launch_ros.actions.Node(
        package='multi_slam',
        executable='sim_node',
        name='sim_node',
        parameters=[{'world_map': world_map}]
    )

    # Add other nodes similarly...

    return launch.LaunchDescription([
        controller_node,
        sim_node,
    ])