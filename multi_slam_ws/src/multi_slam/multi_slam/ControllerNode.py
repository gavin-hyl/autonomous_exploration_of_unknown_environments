import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point
import numpy as np
import math

class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller_node')
        
       
        self.control_signal_pub = self.create_publisher(Vector3, 'control_signal', 10)
        self.pos_hat_pub = self.create_publisher(Vector3, 'pos_hat', 10)
        
        self.lidar_sub = self.create_subscription(PointCloud, 'lidar_point_cloud', self.lidar_callback, 10)
        self.beacon_sub = self.create_subscription(PointCloud, 'beacon_pos', self.beacon_callback, 10)
        
        self.declare_parameter('control_update_rate', 10.0)  # Hz
        self.declare_parameter('max_acceleration', 1.0)  # m/s²
        self.declare_parameter('kp', 1.0)  # Position gain
        self.declare_parameter('kd', 2.0)  # Velocity gain
        
        self.control_update_rate = self.get_parameter('control_update_rate').value
        self.max_acceleration = self.get_parameter('max_acceleration').value
        self.kp = self.get_parameter('kp').value
        self.kd = self.get_parameter('kd').value
        
        self.position_estimate = np.array([0.0, 0.0, 0.0]) 
        self.velocity_estimate = np.array([0.0, 0.0, 0.0])
        self.target_position = np.array([5.0, 5.0, 0.0])  
        
        self.last_update_time = self.get_clock().now()
        self.last_position = np.array([0.0, 0.0, 0.0])
        
        # Set up timers
        self.control_timer = self.create_timer(1.0/self.control_update_rate, self.control_loop)
        self.state_estimation_timer = self.create_timer(0.05, self.update_state_estimate)  # 20Hz state estimation
        
        # Data storage
        self.lidar_points = []
        self.beacon_positions = []
        
        self.get_logger().info('ControllerNode initialized')
    
    def lidar_callback(self, msg):
        """Process incoming LiDAR data"""
        # Extract points from PointCloud message
        self.lidar_points = []
        for point in msg.points:
            self.lidar_points.append(np.array([point.x, point.y, point.z]))
        
        # Use LiDAR for localization (simplified)
        self.update_position_from_lidar()
    
    def beacon_callback(self, msg):
        """Process incoming beacon data"""
        # Extract points from PointCloud message
        self.beacon_positions = []
        for point in msg.points:
            self.beacon_positions.append(np.array([point.x, point.y, point.z]))
        
        # Use beacons for more accurate positioning
        self.update_position_from_beacons()
    
    def update_position_from_lidar(self):
        """Update position estimate using LiDAR data"""
        # Simple implementation - in a real system, you'd use a SLAM algorithm
        # This is a placeholder for a more sophisticated algorithm
        if len(self.lidar_points) > 0:
            # For simplicity, we're just slightly adjusting our estimate based on LiDAR
            # In a real implementation, you would perform scan matching or other techniques
            pass
    
    def update_position_from_beacons(self):
        """Update position estimate using beacon positions"""
        # This would be a more accurate position update using trilateration or similar
        # For this simplified version, we'll assume beacons provide direct position information
        if len(self.beacon_positions) >= 3:  # Need at least 3 beacons for triangulation
            # In a real implementation, you would perform trilateration
            # For simplicity, we're using beacon data as-is (assuming it's already in robot's frame)
            pass
    
    def update_state_estimate(self):
        """Update the robot's state estimate (position and velocity)"""
        current_time = self.get_clock().now()
        dt = (current_time - self.last_update_time).nanoseconds / 1e9  # Convert to seconds
        
        # Simple sensor fusion using both LiDAR and beacon data
        # In a real implementation, you might use a Kalman filter
        
        # For now, we're assuming position updates come directly from sensors
        # And we're calculating velocity by differentiation
        
        if dt > 0:
            # Estimate velocity using finite difference
            self.velocity_estimate = (self.position_estimate - self.last_position) / dt
            
        # Update tracking variables
        self.last_position = self.position_estimate.copy()
        self.last_update_time = current_time
        
        # Publish the current position estimate
        pos_hat_msg = Vector3()
        pos_hat_msg.x = float(self.position_estimate[0])
        pos_hat_msg.y = float(self.position_estimate[1])
        pos_hat_msg.z = float(self.position_estimate[2])
        self.pos_hat_pub.publish(pos_hat_msg)
    
    def set_target_position(self, x, y, z=0.0):
        """Set a new target position for the robot"""
        self.target_position = np.array([x, y, z])
        self.get_logger().info(f'New target set: ({x}, {y}, {z})')
    
    def control_loop(self):
        """Main control loop implementing double integrator dynamics"""
        # Calculate position error
        position_error = self.target_position - self.position_estimate
        
        # Double integrator control law: u = Kp*e - Kd*v
        # Where u is acceleration command, e is position error, v is velocity
        acceleration_command = self.kp * position_error - self.kd * self.velocity_estimate
        
        # Limit acceleration magnitude
        acceleration_norm = np.linalg.norm(acceleration_command)
        if acceleration_norm > self.max_acceleration:
            acceleration_command = acceleration_command * self.max_acceleration / acceleration_norm
        
        # Set z-component to 0 (robot moves only in x-y plane)
        acceleration_command[2] = 0.0
        
        # Create and publish control message
        control_msg = Vector3()
        control_msg.x = float(acceleration_command[0])
        control_msg.y = float(acceleration_command[1])
        control_msg.z = float(acceleration_command[2])
        self.control_signal_pub.publish(control_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('Controller node shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()