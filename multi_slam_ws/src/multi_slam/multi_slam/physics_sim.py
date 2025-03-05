#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
from shapely.geometry import Polygon
import math
from map import Map, MAP  # Import Map and MAP from map.py

class SimplePhysicsSimNode(Node):
    def __init__(self):
        super().__init__('physics_sim')
        
        # Parameters
        self.declare_parameter('robot_width', 1.0)
        self.declare_parameter('robot_length', 2.0)
        self.declare_parameter('delta_noise_std', 0.02)
        
        # Robot dimensions and noise parameters
        self.robot_width = self.get_parameter('robot_width').value
        self.robot_length = self.get_parameter('robot_length').value
        self.delta_noise_std = self.get_parameter('delta_noise_std').value
        
        # Use predefined MAP
        self.map = MAP
        
        # Current true position
        self.true_x = 5.0
        self.true_y = 5.0
        self.true_theta = 0.0
        
        # Publishers
        self.pos_publisher = self.create_publisher(
            Float32MultiArray, '/true_position', 10)
        
        # Subscribers - expects delta values
        self.create_subscription(
            Float32MultiArray, '/control_input', self.action_callback, 10)
        
        self.get_logger().info('Physics simulation node initialized')
    
    def create_robot_polygon(self, x, y, theta):
        """Create a polygon representing the robot"""
        half_length = self.robot_length / 2.0
        half_width = self.robot_width / 2.0
        
        corners = [
            (-half_length, -half_width),
            (half_length, -half_width),
            (half_length, half_width),
            (-half_length, half_width)
        ]
        
        rotated_corners = []
        for corner_x, corner_y in corners:
            rotated_x = corner_x * math.cos(theta) - corner_y * math.sin(theta) + x
            rotated_y = corner_x * math.sin(theta) + corner_y * math.cos(theta) + y
            rotated_corners.append((rotated_x, rotated_y))
        
        return Polygon(rotated_corners)
    
    def add_noise_to_delta(self, dx, dy, dtheta):
        """Add Gaussian noise to delta values"""
        dx_noisy = dx + np.random.normal(0, self.delta_noise_std)
        dy_noisy = dy + np.random.normal(0, self.delta_noise_std)
        dtheta_noisy = dtheta + np.random.normal(0, self.delta_noise_std/5)
        return dx_noisy, dy_noisy, dtheta_noisy
    
    def find_valid_position(self, start_x, start_y, start_theta, end_x, end_y, end_theta):
        """Find position right before collision"""
        # Check if end position is valid
        robot_poly = self.create_robot_polygon(end_x, end_y, end_theta)
        if not self.map.intersects(robot_poly):
            return end_x, end_y, end_theta
        
        # Step backwards from collision point
        step = 0.05  # Step size
        alpha = 1.0
        
        while alpha > 0:
            alpha -= step
            x = start_x + alpha * (end_x - start_x)
            y = start_y + alpha * (end_y - start_y)
            theta = start_theta + alpha * (end_theta - start_theta)
            
            robot_poly = self.create_robot_polygon(x, y, theta)
            if not self.map.intersects(robot_poly):
                return x, y, theta
        
        # Fallback to start position
        return start_x, start_y, start_theta
    
    def action_callback(self, msg):
        """Process incoming delta actions (dx, dy, dtheta)"""
        if len(msg.data) >= 3:
            # Get delta values
            dx, dy, dtheta = msg.data[0], msg.data[1], msg.data[2]
            
            # Add noise to delta values
            dx_noisy, dy_noisy, dtheta_noisy = self.add_noise_to_delta(dx, dy, dtheta)
            
            # Store original position
            start_x, start_y, start_theta = self.true_x, self.true_y, self.true_theta
            
            # Apply noisy deltas to get target position
            target_x = start_x + dx_noisy
            target_y = start_y + dy_noisy
            target_theta = start_theta + dtheta_noisy
            
            # Find valid position (handles collisions)
            valid_x, valid_y, valid_theta = self.find_valid_position(
                start_x, start_y, start_theta, target_x, target_y, target_theta)
            
            # Update true position
            self.true_x, self.true_y, self.true_theta = valid_x, valid_y, valid_theta
            
            # Publish true position directly
            pos_msg = Float32MultiArray()
            pos_msg.data = [float(self.true_x), float(self.true_y), float(self.true_theta)]
            self.pos_publisher.publish(pos_msg)
            
            # Log movement
            self.get_logger().info(f'Applied action: [{dx:.2f}, {dy:.2f}, {dtheta:.2f}]')
            self.get_logger().info(f'True position: [{self.true_x:.2f}, {self.true_y:.2f}, {self.true_theta:.2f}]')

def main(args=None):
    rclpy.init(args=args)
    node = SimplePhysicsSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()