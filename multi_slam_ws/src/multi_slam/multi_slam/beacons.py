#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
from shapely.geometry import Point, LineString
import math
from map import Map, MAP  # Import Map and MAP from map.py

class BeaconsNode(Node):
    def __init__(self):
        super().__init__('beacons_node')
        
        # Parameters
        self.declare_parameter('range_noise_std', 0.1)
        
        # Use predefined MAP
        self.map = MAP
        
        # Noise parameters for range measurements
        self.range_noise_std = self.get_parameter('range_noise_std').value
        
        # Define beacon positions (x, y)
        self.beacon_positions = [
            (2.0, 2.0),   # Beacon 1
            (8.0, 2.0),   # Beacon 2
            (8.0, 8.0),   # Beacon 3
            (2.0, 8.0),   # Beacon 4
        ]
        
        # Publisher for beacon measurements
        self.beacon_publisher = self.create_publisher(
            Float32MultiArray, '/beacon_measurements', 10)
        
        # Subscribe to robot's true position
        self.create_subscription(
            Float32MultiArray, '/true_position', self.position_callback, 10)
        
        self.get_logger().info(f'Beacons node initialized with {len(self.beacon_positions)} beacons')
    
    def has_line_of_sight(self, robot_x, robot_y, beacon_x, beacon_y):
        """Check if there's a clear line of sight between robot and beacon
        
        Args:
            robot_x, robot_y: Robot position
            beacon_x, beacon_y: Beacon position
            
        Returns:
            bool: True if there is a line of sight, False otherwise
        """
        # Create a line from robot to beacon
        line = LineString([(robot_x, robot_y), (beacon_x, beacon_y)])
        
        # Check if this line intersects with any obstacle in the map
        return not self.map.intersects(line)
    
    def add_noise_to_measurement(self, dx, dy):
        """Add Gaussian noise to delta measurements
        
        Args:
            dx, dy: True delta values from robot to beacon
            
        Returns:
            tuple: Noisy delta values (dx_noisy, dy_noisy)
        """
        # Calculate true range
        true_range = np.sqrt(dx**2 + dy**2)
        
        # Add noise to range
        noisy_range = true_range + np.random.normal(0, self.range_noise_std)
        
        # Calculate angle (no noise added to angle for simplicity)
        angle = np.arctan2(dy, dx)
        
        # Convert back to cartesian coordinates
        dx_noisy = noisy_range * np.cos(angle)
        dy_noisy = noisy_range * np.sin(angle)
        
        return dx_noisy, dy_noisy
    
    def position_callback(self, msg):
        """Callback for receiving robot's true position
        
        Args:
            msg: Message containing robot position [x, y, theta]
        """
        if len(msg.data) >= 3:
            robot_x, robot_y, _ = msg.data[0], msg.data[1], msg.data[2]
            self.update_and_publish(robot_x, robot_y)
    
    def update_and_publish(self, robot_x, robot_y):
        """Update visible beacons and publish measurements
        
        Args:
            robot_x, robot_y: Current robot position
        """
        visible_beacons = []
        
        for beacon_id, (beacon_x, beacon_y) in enumerate(self.beacon_positions):
            if self.has_line_of_sight(robot_x, robot_y, beacon_x, beacon_y):
                # Calculate true deltas
                dx = beacon_x - robot_x
                dy = beacon_y - robot_y
                
                # Add noise to measurements
                dx_noisy, dy_noisy = self.add_noise_to_measurement(dx, dy)
                
                # Store beacon info: [beacon_id, dx_noisy, dy_noisy]
                visible_beacons.extend([float(beacon_id), dx_noisy, dy_noisy])
        
        # Publish beacon measurements
        if visible_beacons:
            msg = Float32MultiArray()
            msg.data = visible_beacons
            self.beacon_publisher.publish(msg)
            
            num_visible = len(visible_beacons) // 3
            self.get_logger().info(f'Published {num_visible} visible beacon measurements')
        else:
            self.get_logger().info('No beacons currently visible')

def main(args=None):
    rclpy.init(args=args)
    node = BeaconsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()