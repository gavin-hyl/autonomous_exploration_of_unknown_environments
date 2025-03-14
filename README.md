# Robotics (Caltech ME/EE/CS 133b) Final Project - Multi-SLAM

## Contributors
Gio Huh, Taekyung Lee, Gavin Hua

## Overview
This project implements a 2D Simultaneous Localization and Mapping (SLAM) system using ROS 2. It features robot simulation, sensor data generation, particle filter-based localization, and grid-based mapping. The system allows both autonomous navigation and teleoperation using keyboard controls.

## Architecture
The system consists of four main components:

1. **Physics Simulation Node**: Simulates robot movement and sensor data generation
2. **SLAM Node**: Performs simultaneous localization and mapping using sensor data
3. **Controller Node**: Handles robot movement commands with autonomous and teleoperation modes
4. **Teleop Keyboard Node**: Provides keyboard-based teleoperation

## Installation

### Prerequisites
- ROS 2 (Foxy or later)
- Python 3.6+
- NumPy
- Shapely (for collision detection)

### Setup
After cloning the repository:
```sh
# Install required Python packages
pip install -r requirements.txt

# Build the ROS 2 package
cd multi_slam_ws
colcon build
source install/setup.bash
```

## Usage

### Running the System
```sh
# Launch the complete system
ros2 launch multi_slam multi_slam_launch.py
```

### Parameter Configuration
Parameters can be configured at launch time:
```sh
ros2 launch multi_slam multi_slam_launch.py max_speed:=2.0 teleop_enabled:=true
```

### Teleoperation
When teleoperation is enabled, use the following keys:
- `w`: Forward
- `s`: Backward
- `a`: Left
- `d`: Right
- `x`: Stop
- `q`: Quit

## ROS 2 Topic Structure

### Main Topics
- `control_signal`: Robot control commands (`Vector3`)
- `/lidar`: LiDAR sensor data (`PointCloud2`)
- `/beacon`: Beacon sensor data (`PointCloud2`)
- `/estimated_pose`: Estimated robot pose (`PoseStamped`)
- `/occupancy_grid`: Occupancy grid map (`OccupancyGrid`)

### Synchronization Topics
- `/slam_done`: Signal indicating SLAM processing completion (`Bool`)
- `/sim_done`: Signal indicating simulation step completion (`Bool`)

### Visualization Topics
- `visualization_marker_true`: True robot position (`Marker`)
- `/pos_hat_viz`: Estimated robot position (`Marker`)
- `/estimated_beacons`: Estimated beacon positions (`MarkerArray`)
- `beacon_viz`, `lidar_viz`: Sensor data visualization (`PointCloud2`)

## Nodes and Parameters

### Physics Simulation Node
Simulates robot movement and sensor data generation.

#### Parameters:
- `lidar_r_max`: Maximum LiDAR range (10.0m)
- `lidar_r_min`: Minimum LiDAR range (0.1m)
- `lidar_delta_theta`: Angular resolution (3 degrees)
- `lidar_std_dev`: LiDAR noise (0.1m)
- `beacon_std_dev`: Beacon noise (0.1m)
- `vel_std_dev`: Velocity noise (0.4)
- `collision_buffer`: Collision detection buffer (0.1m)
- `collision_increment`: Collision checking increment (0.02m)
- `sim_dt`: Simulation time step (0.1s)

### SLAM Node
Performs simultaneous localization and mapping.

#### Parameters:
- `map_size_x`, `map_size_y`: Map dimensions (100.0m)
- `map_origin_x`, `map_origin_y`: Map origin (-50.0m)
- `grid_size`: Grid cell resolution (0.1m)
- `num_particles`: Particle filter count (1000)
- `position_std_dev`: Position noise (0.1m)
- `initial_noise`: Initial particle spread (0.5)

### Controller Node
Handles robot movement commands.

#### Parameters:
- `max_speed`: Maximum robot speed (1.0m/s)
- `control_frequency`: Control loop frequency (10.0Hz)
- `teleop_enabled`: Enable teleoperation (True)

### Teleop Keyboard Node
Provides keyboard-based teleoperation.

#### Parameters:
- `max_speed`: Maximum robot speed (1.0m/s)
- `publish_rate`: Command publish rate (20.0Hz)

