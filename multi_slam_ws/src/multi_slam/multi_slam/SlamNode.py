# slam_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, Vector3, Twist
from nav_msgs.msg import OccupancyGrid, Path, MapMetaData
from visualization_msgs.msg import Marker, MarkerArray
from multi_slam.Localization import Localization
from multi_slam.Mapping import Mapping
import numpy as np
from sensor_msgs_py.point_cloud2 import read_points
import math

class SLAMNode(Node):
    def __init__(self):
        super().__init__("slam_node")
        
        # Parameters
        self.declare_parameter('map_size_x', 100.0)
        self.declare_parameter('map_size_y', 100.0)
        self.declare_parameter('map_origin_x', -50.0)
        self.declare_parameter('map_origin_y', -50.0)
        self.declare_parameter('grid_size', 0.1)
        self.declare_parameter('num_particles', 100)
        self.declare_parameter('position_std_dev', 0.1)
        self.declare_parameter('use_proposed_control', True)  # Flag to switch between control methods
        self.declare_parameter('debug_visualization', True)   # Flag to enable debug visualization
        
        # Get parameters
        map_size_x = self.get_parameter('map_size_x').value
        map_size_y = self.get_parameter('map_size_y').value
        map_origin_x = self.get_parameter('map_origin_x').value
        map_origin_y = self.get_parameter('map_origin_y').value
        grid_size = self.get_parameter('grid_size').value
        num_particles = self.get_parameter('num_particles').value
        position_std_dev = self.get_parameter('position_std_dev').value
        self.use_proposed_control = self.get_parameter('use_proposed_control').value
        self.debug_visualization = self.get_parameter('debug_visualization').value
        
        # Initialize Map and Localization
        self.map = Mapping(
            map_size=(map_size_x, map_size_y),
            map_origin=(map_origin_x, map_origin_y),
            grid_size=grid_size
        )
        
        # Initial position (x,y,0) - 2D position with z=0
        initial_position = np.array([0.0, 0.0, 0.0])
        self.position = initial_position
        self.position_cov = np.eye(3) * 0.1  # Initial covariance
        
        # Update rate
        self.dt = 0.1
        
        # Initialize Localization
        self.localization = Localization(
            initial_location=initial_position, 
            std_dev_noise=position_std_dev,
            num_particles=num_particles,
            dt=self.dt
        )
        
        # Lidar range
        self.lidar_range = (0.1, 10.0)  # min and max range in meters
        
        # Current data
        self.lidar_data = []
        self.beacon_data = []
        self.control_input = np.zeros(3)  # vx, vy, 0

        # Subscribers
        self.create_subscription(
            PointCloud2, "/lidar", self.lidar_callback, 10
        )
        self.create_subscription(
            PointCloud2, "/beacon", self.beacon_callback, 10
        )
        self.create_subscription(
            Twist, "/cmd_vel", self.control_callback, 10
        )
        
        # Publishers
        self.pose_pub = self.create_publisher(Marker, "/pos_hat_viz", 10)
        self.map_pub = self.create_publisher(OccupancyGrid, "/occupancy_grid", 10)
        self.beacon_pub = self.create_publisher(MarkerArray, "/estimated_beacons", 10)
        self.path_pub = self.create_publisher(Path, "/robot_path", 10)
        
        # Publisher for control commands (when using proposed control method)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        
        # Timer for SLAM main loop
        self.create_timer(self.dt, self.slam_loop)

        self.map_pub_timer = self.create_timer(1.0, self.publish_map)
        self.pos_pub_timer = self.create_timer(1.0, self.publish_pos)
        
        # Path for visualization
        self.path = Path()
        self.path.header.frame_id = "map"
        

    def lidar_callback(self, msg: PointCloud2):
        """Process LiDAR data"""
        try:
            points = list(read_points(msg, field_names=("x", "y", "z")))
            self.lidar_data = [np.array([p[0], p[1], p[2]]) for p in points]
        except Exception as e:
            self.get_logger().error(f"Error processing lidar data: {e}")

    def beacon_callback(self, msg: PointCloud2):
        """Process beacon data"""
        try:
            points = list(read_points(msg, field_names=("x", "y", "z")))
            self.beacon_data = [np.array([p[0], p[1], p[2]]) for p in points]
        except Exception as e:
            self.get_logger().error(f"Error processing beacon data: {e}")

    def control_callback(self, msg: Twist):
        """Process control input"""
        self.control_input = np.array([msg.linear.x, msg.linear.y, msg.linear.z])

    def slam_loop(self):
        """Main SLAM loop"""
        # Localization
        self.get_logger().info("Updating position")
        updated_position, updated_cov = self.localization.update_position(
            self.control_input,
            self.beacon_data,
            self.map
        )
        
        updated_position[2] = 0.0
        self.position = updated_position
        self.position_cov = updated_cov
        
        Mapping
        self.map.update(
            robot_pos=self.position,
            robot_cov=self.position_cov,
            lidar_data=self.lidar_data,
            lidar_range=self.lidar_range,
            beacon_data=self.beacon_data
        )
        
        # Visualization
        # self.publish_pose()
        # self.publish_map()
        # self.publish_beacons()
        # self.publish_path()
        
        # # Control update based on selected method
        # if self.use_proposed_control:
        #     self.update_control_proposed()
        # else:
        #     # Using existing teleop code (no changes needed as it comes from /cmd_vel)
        #     pass

    def update_control_proposed(self):
        """Proposed control method: optimize towards finding beacons based on current map and position"""
        # If no beacon data is available, explore randomly
        if len(self.map.beacon_positions) == 0:
            # Random exploration if no beacons detected yet
            angular_z = 0.2  # slow rotation
            linear_x = 0.1   # slow forward movement
            cmd = Twist()
            cmd.linear.x = linear_x
            cmd.angular.z = angular_z
            self.cmd_vel_pub.publish(cmd)
            return
            
        # Find the beacon with highest uncertainty (largest covariance determinant)
        beacon_uncertainties = [np.linalg.det(cov) for cov in self.map.beacon_covariances]
        target_idx = np.argmax(beacon_uncertainties)
        target_beacon = self.map.beacon_positions[target_idx]
        
        # Calculate vector from robot to target beacon (2D only, ignore orientation)
        delta_x = target_beacon[0] - self.position[0]
        delta_y = target_beacon[1] - self.position[1]
        distance = math.sqrt(delta_x**2 + delta_y**2)
        
        # For 2D robot without orientation, just move directly towards target
        # (we ignore orientation/theta since robot frame is aligned with world frame)
        
        # Simple proportional control for movement
        k_linear = 0.3
        linear_x = min(k_linear * distance, 0.2)
        
        # Direction control (choose positive or negative x based on target direction)
        if delta_x < 0:
            linear_x = -linear_x
            
        # Create and publish control command
        cmd = Twist()
        cmd.linear.x = linear_x
        cmd.linear.y = 0.0  # No sideways movement
        cmd.angular.z = 0.0  # No rotation for 2D robot
        self.cmd_vel_pub.publish(cmd)

    def publish_pos(self):
        """Publish the estimated pose"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "pos_hat_viz"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Position
        marker.pose.position.x = self.position[0]
        marker.pose.position.y = self.position[1]
        marker.pose.position.z = 0.0
        
        # Scale
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        
        # Color
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        self.pose_pub.publish(marker)

    def publish_map(self):
        """Publish occupancy grid"""
        self.get_logger().info("Publishing map")

        # Create occupancy grid message
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.info.map_load_time = self.get_clock().now().to_msg()
        msg.info.resolution = self.map.grid_size
        msg.info.width = self.map.grid_width
        msg.info.height = self.map.grid_height
        
        # Set origin (position and orientation)
        msg.info.origin.position.x = self.map.map_origin[0]
        msg.info.origin.position.y = self.map.map_origin[1]
        
        # Convert log-odds to probabilities (0-100)
        probs = 1.0 / (1.0 + np.exp(-self.map.log_odds_grid))
        msg.data = (probs * 100).astype(int).flatten().tolist()
        
        self.map_pub.publish(msg)


    def publish_beacons(self):
        """Publish estimated beacon positions"""
        marker_array = MarkerArray()
        
        for i, (beacon_pos, beacon_cov) in enumerate(zip(self.map.beacon_positions, self.map.beacon_covariances)):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "beacons"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            # Position
            marker.pose.position.x = beacon_pos[0]
            marker.pose.position.y = beacon_pos[1]
            marker.pose.position.z = 0.0
            
            # Scale based on uncertainty
            uncertainty = np.sqrt(np.linalg.det(beacon_cov))
            scale = max(0.1, min(1.0, uncertainty))
            marker.scale.x = scale
            marker.scale.y = scale
            marker.scale.z = scale
            
            # Color (red, becoming more transparent with higher certainty)
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = max(0.1, min(1.0, 1.0 - 0.9 * (1.0 - uncertainty)))
            
            marker_array.markers.append(marker)
            
        self.beacon_pub.publish(marker_array)

    def publish_path(self):
        """Publish robot path"""
        self.path_pub.publish(self.path)


    def create_line(self, x0, y0, x1, y1):
        """Create a line of points using Bresenham's algorithm"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
                
        return points

def main(args=None):
    rclpy.init(args=args)
    node = SLAMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()