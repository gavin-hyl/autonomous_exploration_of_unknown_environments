# Autonomous Exploration of Unknown Environments (SLAM + Entropy-Informed A* Exploration)

## Contributors
Gio Huh, Taekyung Lee, Gavin Hua

## Instructions
After cloning the repo, please
```sh
pip install -r requirements.txt
```

## ROS2 Node/Topic Structure
Please keep this updated as we code!

### Topics
- `/control_signal`
    - `geometry_msgs.msg.Vector3` with `z` component set to $0$
    - Represents the desired acceleration.
- `/lidar_point_cloud`
    - `sensor_msgs` package, `PointCloud2`.
    - Note that this message must be converted from python format to ROS format using `sensor_msgs_py.point_cloud2`
- `/beacon_pos`
    - `sensor_msgs.PointCloud2`. length equal to the number of beacons currently visible beacons.
    - Represents the positions of the beacons from the frame of the robot.
- `/pos_hat`
    - `Vector3`, with `z = 0`
    - Represents the position if the control signal was perfectly executed

### Nodes
Nodes may publish other topics to RVIZ for visualization purposes. The topics listed here relate to core functionality.
- `ControllerNode`
    - Description:
    - Publish: `/control_signal`
    - Subscribe: `/lidar_point_cloud`, `/beacon_pos`, `/pos_hat`
    - Maintained by: TK
- `PhysicsSimNode`
    - Description:
    - Publish: `/lidar_point_cloud`, `/beacon_pos`, `/pos_hat`
    - Subscribe: `/control_signal`
    - Maintained by: Gavin, Gio
