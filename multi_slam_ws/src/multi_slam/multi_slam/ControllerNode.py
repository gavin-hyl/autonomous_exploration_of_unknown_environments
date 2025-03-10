import numpy as np

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
        
        # Kalman Filter state initialization
        self.position_estimate = np.array([0.0, 0.0, 0.0, 0.0])  # [x, y, vx, vy]
        self.P = np.eye(4) * 100 # Covariance matrix, start with high uncertainty
        
        # Kalman Filter matrices (double integrator model)
        self.F = np.eye(4)  # State transition matrix (double integrator model)
        self.F[0, 2] = 1.0  # x = x + vx * dt
        self.F[1, 3] = 1.0  # y = y + vy * dt
        self.dt = 0.1  # Time step (for simplicity, you can adjust this value)
        
        # Measurement model (we can measure the position relative to beacon)
        self.H = np.array([[1, 0, 0, 0],  # Measure x
                           [0, 1, 0, 0]])  # Measure y
        
        self.R = np.eye(2) * 1e-2  # Measurement noise covariance (adjust as necessary)
        self.Q = np.eye(4) * 1e-5  # Process noise covariance (adjust as necessary)
        
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
    
    def beacon_callback(self, msg):
        """Process incoming beacon position data"""
        self.beacon_data = list(point_cloud2.read_points(msg, field_names=("x", "y", "z")))
    
    def update_position_estimate(self):
        """Update the robot's position estimate using a double integrator Kalman filter"""
        # Predict the next state
        self.position_estimate = np.dot(self.F, self.position_estimate)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        # Beacon-based position update (relative position from LiDAR)
        if len(self.lidar_data) > 0 and len(self.beacon_data) > 0:
            # Get the relative position of the robot to the beacon
            # Here we assume the first beacon in the list is the one we want
            beacon_position = np.array(self.beacon_data[0][:2])  # Beacon position [x, y]
            robot_position = np.array(self.lidar_data[0][:2])  # Robot's position detected via LiDAR
            
            # Calculate relative position (robot position w.r.t beacon)
            relative_position = robot_position - beacon_position
            
            # Update the state with the relative position
            z = relative_position  # Measurement vector (relative position)
            
            # Kalman Update
            S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
            K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
            y = z - np.dot(self.H, self.position_estimate[:2])  # Measurement residual
            self.position_estimate[:2] = self.position_estimate[:2] + np.dot(K, y)
            
            # Update the covariance
            self.P = np.dot(np.eye(4) - np.dot(K, self.H), self.P)
        
        # Publish the updated position estimate
        self.publish_position_estimate()
    
    def control_loop(self):
        """Main control loop - calculate and publish control commands"""
        if not self.teleop_enabled:
            # Autonomous control logic (not implemented here)
            pass
        
        # Update position estimate based on sensor data
        self.update_position_estimate()
        
        # Publish control input
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
