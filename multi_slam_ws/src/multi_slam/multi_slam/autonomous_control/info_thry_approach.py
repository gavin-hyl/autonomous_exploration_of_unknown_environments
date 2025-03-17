#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool
import numpy as np
import math
import sys
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points
import random

class InformationTheoreticController(Node):
    """
    Information-theoretic exploration controller for autonomous robot navigation.
    
    This controller uses information theory principles to guide the robot towards
    areas that maximize information gain, focusing on discovering beacons and
    building a complete map as efficiently as possible.
    """
    
    def __init__(self):
        """Initialize the exploration controller node."""
        super().__init__('info_theoretic_controller')
        
        # Parameters
        self.declare_parameter('exploration_radius', 2.0)  # Radius for candidate move generation
        self.declare_parameter('num_candidates', 8)  # Number of candidate directions to evaluate
        self.declare_parameter('move_step', 0.5)  # Step size for each move
        self.declare_parameter('occupancy_threshold', 50)  # Threshold for considering a cell as occupied
        self.declare_parameter('unknown_weight', 1.0)  # Weight for unknown area exploration
        self.declare_parameter('beacon_weight', 3.0)  # Weight for beacon discovery
        self.declare_parameter('beacon_proximity_threshold', 5.0)  # Distance threshold for beacon proximity
        
        # Get parameters
        self.exploration_radius = self.get_parameter('exploration_radius').value
        self.num_candidates = self.get_parameter('num_candidates').value
        self.move_step = self.get_parameter('move_step').value
        self.occupancy_threshold = self.get_parameter('occupancy_threshold').value
        self.unknown_weight = self.get_parameter('unknown_weight').value
        self.beacon_weight = self.get_parameter('beacon_weight').value
        self.beacon_proximity_threshold = self.get_parameter('beacon_proximity_threshold').value
        
        # Robot state
        self.position = np.array([0.0, 0.0, 0.0])  # Current position [x, y, z]
        self.occupancy_grid = None  # Occupancy grid map
        self.grid_info = None  # Map metadata
        self.known_beacons = []  # List of known beacon positions
        self.lidar_data = []  # Current LiDAR readings
        self.moving = False  # Flag to indicate if robot is currently moving
        
        # Exploration state
        self.exploration_complete = False
        self.coverage_threshold = 0.95  # Consider exploration complete when 95% of the map is explored
        self.exploration_steps = 0
        self.max_exploration_steps = 1000  # Safeguard to prevent infinite exploration
        
        # Subscribers
        self.create_subscription(
            Vector3, '/pos_hat', self.position_callback, 10
        )
        self.create_subscription(
            OccupancyGrid, '/occupancy_grid', self.map_callback, 10
        )
        self.create_subscription(
            MarkerArray, '/estimated_beacons', self.beacons_callback, 10
        )
        self.create_subscription(
            PointCloud2, '/lidar', self.lidar_callback, 10
        )
        self.create_subscription(
            Bool, '/slam_done', self.slam_done_callback, 10
        )
        
        # Publishers
        self.control_pub = self.create_publisher(Vector3, '/control_signal', 10)
        
        # Timer
        self.create_timer(1.0, self.exploration_loop)  # Run exploration loop every second
        
        self.get_logger().info('Information Theoretic Controller initialized')
    
    def position_callback(self, msg):
        """Update the robot's current position."""
        self.position = np.array([msg.x, msg.y, msg.z])
    
    def map_callback(self, msg):
        """Update the occupancy grid map."""
        self.occupancy_grid = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        self.grid_info = msg.info
    
    def beacons_callback(self, msg):
        """Update the list of known beacons."""
        self.known_beacons = []
        for marker in msg.markers:
            beacon_pos = np.array([
                marker.pose.position.x,
                marker.pose.position.y,
                marker.pose.position.z
            ])
            self.known_beacons.append(beacon_pos)
    
    def lidar_callback(self, msg):
        """Update the current LiDAR readings."""
        points = list(read_points(msg, field_names=("x", "y", "z")))
        self.lidar_data = [np.array([p[0], p[1], p[2]]) for p in points]
    
    def slam_done_callback(self, msg):
        """Receive signal that SLAM processing is done."""
        if msg.data and self.moving:
            self.moving = False
            self.get_logger().info(f'Reached position: [{self.position[0]:.2f}, {self.position[1]:.2f}]')
    
    def exploration_loop(self):
        """Main loop for autonomous exploration."""
        if self.exploration_complete:
            self.get_logger().info('Exploration complete.')
            return
        
        if self.moving:
            self.get_logger().debug('Still moving, waiting for completion...')
            return
        
        if self.occupancy_grid is None:
            self.get_logger().info('Waiting for map data...')
            return
        
        self.exploration_steps += 1
        if self.exploration_steps >= self.max_exploration_steps:
            self.exploration_complete = True
            self.get_logger().info(f'Reached maximum exploration steps ({self.max_exploration_steps}).')
            return
        
        # Check if exploration is complete based on map coverage
        if self.is_exploration_complete():
            self.exploration_complete = True
            self.get_logger().info('Exploration complete: Map sufficiently explored.')
            return
        
        # Execute exploration step
        self.execute_exploration_step()
    
    def is_exploration_complete(self):
        """Check if exploration is complete based on map coverage."""
        if self.occupancy_grid is None:
            return False
        
        # Count cells that are not unknown (-1 in ROS occupancy grid)
        known_cells = np.sum(self.occupancy_grid >= 0)
        total_cells = self.occupancy_grid.size
        coverage = known_cells / total_cells
        
        self.get_logger().info(f'Map coverage: {coverage:.2f}')
        return coverage >= self.coverage_threshold
    
    def execute_exploration_step(self):
        """Execute one step of information-theoretic exploration."""
        # Generate candidate moves
        candidate_moves = self.generate_candidate_moves()
        
        # Evaluate information gain for each candidate
        best_move = None
        max_information_gain = float('-inf')
        
        for move in candidate_moves:
            # Simulate move
            new_position = self.simulate_move(move)
            
            # Compute information gain
            info_gain = self.compute_information_gain(new_position)
            
            if info_gain > max_information_gain:
                max_information_gain = info_gain
                best_move = move
        
        # Execute best move
        if best_move is not None:
            self.get_logger().info(f'Executing move: [{best_move[0]:.2f}, {best_move[1]:.2f}]')
            self.execute_move(best_move)
        else:
            self.get_logger().warn('No valid move found.')
            # If stuck, make a random move
            random_move = np.array([
                random.uniform(-1.0, 1.0),
                random.uniform(-1.0, 1.0),
                0.0
            ])
            self.execute_move(random_move)
    
    def generate_candidate_moves(self):
        """Generate candidate moves around the current position."""
        candidates = []
        
        # Generate moves in evenly spaced directions
        for i in range(self.num_candidates):
            angle = 2 * math.pi * i / self.num_candidates
            dx = self.move_step * math.cos(angle)
            dy = self.move_step * math.sin(angle)
            candidates.append(np.array([dx, dy, 0.0]))
        
        # Also consider staying in place
        candidates.append(np.array([0.0, 0.0, 0.0]))
        
        return candidates
    
    def simulate_move(self, move):
        """Simulate making a move from the current position."""
        return self.position + move
    
    def compute_information_gain(self, position):
        """
        Compute the expected information gain from a candidate position.
        
        Information gain considers:
        1. Unexplored regions that would be visible
        2. Proximity to potential beacons
        3. Avoiding obstacles
        """
        if self.occupancy_grid is None or self.grid_info is None:
            return 0.0
        
        # Convert position to grid coordinates
        grid_x, grid_y = self.world_to_grid(position[0], position[1])
        
        # Check if position is within grid bounds and not occupied
        if not self.is_valid_position(grid_x, grid_y):
            return float('-inf')  # Invalid position
        
        # Calculate information gain
        info_gain = 0.0
        
        # 1. Gain from exploring unknown areas
        unknown_gain = self.calculate_unknown_cell_gain(grid_x, grid_y)
        info_gain += self.unknown_weight * unknown_gain
        
        # 2. Gain from proximity to undiscovered beacons
        beacon_gain = self.calculate_beacon_gain(position)
        info_gain += self.beacon_weight * beacon_gain
        
        return info_gain
    
    def world_to_grid(self, world_x, world_y):
        """Convert world coordinates to grid coordinates."""
        if self.grid_info is None:
            return 0, 0
            
        grid_x = int((world_x - self.grid_info.origin.position.x) / self.grid_info.resolution)
        grid_y = int((world_y - self.grid_info.origin.position.y) / self.grid_info.resolution)
        return grid_x, grid_y
    
    def is_valid_position(self, grid_x, grid_y):
        """Check if a grid position is valid (within bounds and not occupied)."""
        if self.occupancy_grid is None:
            return False
            
        # Check bounds
        if not (0 <= grid_x < self.grid_info.width and 0 <= grid_y < self.grid_info.height):
            return False
            
        # Check if occupied
        if self.occupancy_grid[grid_y][grid_x] >= self.occupancy_threshold:
            return False
            
        return True
    
    def calculate_unknown_cell_gain(self, grid_x, grid_y):
        """Calculate information gain from exploring unknown cells."""
        if self.occupancy_grid is None:
            return 0.0
            
        # Define a radius to check for unknown cells
        radius = int(self.exploration_radius / self.grid_info.resolution)
        
        # Count unknown cells in the area
        unknown_count = 0
        total_cells = 0
        
        for y in range(max(0, grid_y - radius), min(self.grid_info.height, grid_y + radius + 1)):
            for x in range(max(0, grid_x - radius), min(self.grid_info.width, grid_x + radius + 1)):
                # Check if the cell is within the circular radius
                if math.sqrt((x - grid_x)**2 + (y - grid_y)**2) <= radius:
                    total_cells += 1
                    if self.occupancy_grid[y][x] == -1:  # -1 is unknown in ROS
                        unknown_count += 1
        
        # Return proportion of unknown cells
        return unknown_count / max(1, total_cells)
    
    def calculate_beacon_gain(self, position):
        """Calculate information gain from proximity to potential beacons."""
        if not self.lidar_data:
            return 0.0
            
        # Check if we already have beacons
        if len(self.known_beacons) > 0:
            # For each known beacon, calculate the distance
            min_distance = float('inf')
            for beacon in self.known_beacons:
                distance = np.linalg.norm(position[:2] - beacon[:2])
                min_distance = min(min_distance, distance)
            
            # If we're too close to a known beacon, reduce the gain
            if min_distance < self.beacon_proximity_threshold:
                return 0.0  # No need to go near known beacons
        
        # Check lidar data for potential beacons
        # In a real implementation, you'd analyze patterns in the lidar data
        # to predict where beacons might be
        # For simplicity, we'll just look for discontinuities in lidar readings
        
        # Convert LIDAR points to world frame
        world_lidar = [point + self.position for point in self.lidar_data]
        
        # Sort points by distance from the robot
        distances = [np.linalg.norm(point[:2]) for point in world_lidar]
        
        # Look for gaps in distances which might indicate beacons
        gain = 0.0
        for i in range(1, len(distances)):
            if abs(distances[i] - distances[i-1]) > 0.5:  # Threshold for discontinuity
                gain += 1.0
        
        return gain / max(1, len(distances))
    
    def execute_move(self, move):
        """Execute a move by publishing control commands."""
        self.moving = True
        
        # Create and publish control message
        msg = Vector3()
        msg.x = float(move[0])
        msg.y = float(move[1])
        msg.z = 0.0
        self.control_pub.publish(msg)
        
        self.get_logger().info(f'Published control: [{msg.x:.2f}, {msg.y:.2f}]')


def main(args=None):
    """Main function for the information-theoretic controller."""
    rclpy.init(args=args)
    controller = InformationTheoreticController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
