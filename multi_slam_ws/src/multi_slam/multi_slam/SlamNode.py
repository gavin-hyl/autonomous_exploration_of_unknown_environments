# slam_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, Vector3
from multi_slam.Map import MAP
import numpy as np
from scipy.linalg import expm
from sensor_msgs_py.point_cloud2 import point_cloud2

class SLAMNode(Node):
    def __init__(self):
        super().__init__("slam_node")
        
        # Subscribers
        self.create_subscription(
            PointCloud2, "/lidar", self.lidar_callback, 10
        )
        self.create_subscription(
            PointCloud2, "/beacon", self.beacon_callback, 10
        )
        self.create_subscription(
            Vector3, "/control_signal", self.control_callback, 10
        )
        
        # Publisher
        self.pose_pub = self.create_publisher(PoseStamped, "/estimated_pose", 10)
        
        # EKF State (x, y, theta, vx, vy)
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.covariance = np.eye(5) * 0.1
        self.dt = 0.1  # Update rate matches control frequency
        
        # Process noise
        self.Q = np.diag([0.1, 0.1, 0.05, 0.1, 0.1])  # Tune these values
        
        # Measurement noise (beacon and LiDAR)
        self.R_beacon = np.eye(2) * 0.1
        self.R_lidar = np.eye(2) * 0.2
        
        # Known beacon positions from MAP
        self.beacon_positions = [np.array([beacon.x, beacon.y]) for beacon in MAP.beacons]

    def control_callback(self, msg: Vector3):
        """Update velocity for EKF prediction"""
        self.state[3] = msg.x  # vx
        self.state[4] = msg.y  # vy

    def predict_step(self):
        """EKF prediction using velocity"""
        x, y, theta, vx, vy = self.state
        
        # State transition matrix (linearized)
        F = np.eye(5)
        F[0, 3] = self.dt
        F[1, 4] = self.dt
        
        # Predict state
        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + self.Q

    def beacon_callback(self, msg: PointCloud2):
        """EKF update using beacon measurements"""
        if not self.beacon_positions:
            return
        
        # Extract relative beacon positions from message
        beacon_measurements = list(point_cloud2.read_points(msg, field_names=("x", "y")))
        
        # For each beacon measurement, update EKF
        for meas in beacon_measurements:
            dx, dy = meas[0], meas[1]
            H = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
            z = np.array([dx, dy])
            
            # Kalman update
            S = H @ self.covariance @ H.T + self.R_beacon
            K = self.covariance @ H.T @ np.linalg.inv(S)
            self.state += K @ (z - H @ self.state[:2])
            self.covariance = (np.eye(5) - K @ H) @ self.covariance
        
        self.publish_pose()

    def lidar_callback(self, msg: PointCloud2):
        """Optional: Use LiDAR for additional updates (e.g., obstacle matching)"""
        pass  # Implement feature extraction/matching logic here

    def publish_pose(self):
        """Publish the estimated pose"""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = self.state[0]
        msg.pose.position.y = self.state[1]
        self.pose_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = SLAMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()