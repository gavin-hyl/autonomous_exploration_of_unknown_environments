#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker
import math

class RobotVisualizer(Node):
    def __init__(self):
        super().__init__('robot_visualizer')
        
        # Declare parameters for robot dimensions.
        self.declare_parameter('robot_width', 1.0)
        self.declare_parameter('robot_length', 2.0)
        self.robot_width = self.get_parameter('robot_width').value
        self.robot_length = self.get_parameter('robot_length').value
        
        # Publisher for visualization marker.
        self.marker_pub = self.create_publisher(Marker, 'visualization_marker', 10)
        
        # Subscriber to the true position published by the simulation.
        self.create_subscription(Float32MultiArray, '/true_position', self.position_callback, 10)
        
        self.get_logger().info('Robot visualizer node initialized')

    def position_callback(self, msg):
        # Expecting msg.data = [x, y, theta]
        if len(msg.data) < 3:
            self.get_logger().warn('Received position message with insufficient data.')
            return
        
        x, y, theta = msg.data[0], msg.data[1], msg.data[2]
        
        marker = Marker()
        marker.header.frame_id = "map"  # Ensure this matches your simulation frame.
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "robot"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        # Set the position of the robot marker.
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0  # Keep at zero since this is 2D.
        
        # Convert yaw (theta) to a quaternion (assuming roll=pitch=0).
        half_theta = theta / 2.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = math.sin(half_theta)
        marker.pose.orientation.w = math.cos(half_theta)
        
        # Set the scale of the marker (using robot_length and robot_width).
        marker.scale.x = self.robot_length  # Length in the x-direction.
        marker.scale.y = self.robot_width   # Width in the y-direction.
        marker.scale.z = 0.1  # A small thickness for visualization.
        
        # Set a color for the robot marker (red here, fully opaque).
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        # Optionally, set the lifetime to zero so the marker persists.
        marker.lifetime.sec = 0
        
        # Publish the marker.
        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = RobotVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
