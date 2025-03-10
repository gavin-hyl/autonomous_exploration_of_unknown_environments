#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import numpy as np
from std_msgs.msg import Header
import sys
import tty
import termios
import select
import threading


class ControllerNode(Node):
    def __init__(self):
        super().__init__("controller_node")
        
        # Publishers
        self.control_pub = self.create_publisher(Vector3, "control_signal", 10)
        self.pos_hat_pub = self.create_publisher(Vector3, "pos_hat", 10)
        
        # Subscribers
        self.lidar_sub = self.create_subscription(
            PointCloud2, "lidar", self.lidar_callback, 10
        )
        self.beacon_sub = self.create_subscription(
            PointCloud2, "beacon", self.beacon_callback, 10
        )
        
        # Parameters
        self.declare_parameter("max_acceleration", 1.0)
        self.max_acceleration = self.get_parameter("max_acceleration").value
        
        self.declare_parameter("control_frequency", 10.0)  # Hz
        control_freq = self.get_parameter("control_frequency").value
        self.control_timer = self.create_timer(1.0 / control_freq, self.control_loop)
        
        self.declare_parameter("teleop_enabled", True)
        self.teleop_enabled = self.get_parameter("teleop_enabled").value
        
        # Controller state
        self.lidar_data = []
        self.beacon_data = []
        self.position_estimate = np.array([0.0, 0.0, 0.0])
        self.control_input = np.array([0.0, 0.0, 0.0])
        
        # Teleop setup if enabled
        if self.teleop_enabled:
            self.get_logger().info("Teleop enabled. Use WASD keys to control the robot.")
            self.get_logger().info("Press 'q' to quit teleop mode.")
            self.key_mapping = {
                'w': np.array([1.0, 0.0, 0.0]),   # Forward (+x)
                's': np.array([-1.0, 0.0, 0.0]),  # Backward (-x)
                'a': np.array([0.0, 1.0, 0.0]),   # Left (+y)
                'd': np.array([0.0, -1.0, 0.0]),  # Right (-y)
                'x': np.array([0.0, 0.0, 0.0]),   # Stop
            }
            
            # Start teleop input thread
            self.teleop_thread = threading.Thread(target=self.teleop_input_loop)
            self.teleop_thread.daemon = True
            self.teleop_thread.start()
    
    def lidar_callback(self, msg):
        """Process incoming LiDAR point cloud data"""
        self.lidar_data = list(point_cloud2.read_points(msg, field_names=("x", "y", "z")))
        
        # Log LiDAR data (debugging)
        if len(self.lidar_data) > 0:
            self.get_logger().debug(f"Received {len(self.lidar_data)} LiDAR points")
    
    def beacon_callback(self, msg):
        """Process incoming beacon position data"""
        self.beacon_data = list(point_cloud2.read_points(msg, field_names=("x", "y", "z")))
        
        # Log beacon data (debugging)
        if len(self.beacon_data) > 0:
            self.get_logger().debug(f"Received {len(self.beacon_data)} beacon positions")
    
    def get_key(self):
        """Get keyboard input without requiring Enter"""
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                return key
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        return None
    
    def teleop_input_loop(self):
        """Thread function to continuously check for keyboard input"""
        while rclpy.ok():
            key = self.get_key()
            if key:
                if key == 'q':
                    self.get_logger().info("Exiting teleop mode.")
                    # Reset control before exiting
                    self.control_input = np.array([0.0, 0.0, 0.0])
                    self.publish_control()
                    break
                    
                if key in self.key_mapping:
                    self.control_input = self.key_mapping[key] * self.max_acceleration
                    self.get_logger().info(f"Teleop command: {key}")
                    self.publish_control()
    
    def update_position_estimate(self):
        """Update the robot's position estimate based on lidar and beacon data"""
        # In a real SLAM system, this would implement a complex algorithm
        # to update the robot's position based on sensor data
        
        # For now, assuming perfect state estimation with some noise reduction
        # In a more complex implementation, this would be where your SLAM logic goes
        
        # This is a placeholder - in a real system you'd use the sensor data
        # to update your position estimate
        self.publish_position_estimate()
    
    def control_loop(self):
        """Main control loop - calculate and publish control commands"""
        # If not in teleop mode, calculate control commands based on sensor data
        if not self.teleop_enabled:
            # Implement your autonomous control strategy here
            # This is where you would compute acceleration commands
            # based on your positioning algorithm and desired trajectory
            pass
        
        # Update position estimate based on sensor data
        self.update_position_estimate()
        
        # If not being handled by teleop, publish current control input
        if not self.teleop_enabled:
            self.publish_control()
    
    def publish_control(self):
        """Publish control command to control_signal topic"""
        msg = Vector3()
        msg.x = float(self.control_input[0])
        msg.y = float(self.control_input[1])
        msg.z = float(self.control_input[2])
        self.control_pub.publish(msg)
    
    def publish_position_estimate(self):
        """Publish the current position estimate to pos_hat topic"""
        msg = Vector3()
        msg.x = float(self.position_estimate[0])
        msg.y = float(self.position_estimate[1])
        msg.z = float(self.position_estimate[2])
        self.pos_hat_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    controller_node = ControllerNode()
    
    try:
        rclpy.spin(controller_node)
    except KeyboardInterrupt:
        controller_node.get_logger().info("Controller node stopped by keyboard interrupt")
    except Exception as e:
        controller_node.get_logger().error(f"Error in controller node: {str(e)}")
    finally:
        # Clean shutdown
        controller_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()