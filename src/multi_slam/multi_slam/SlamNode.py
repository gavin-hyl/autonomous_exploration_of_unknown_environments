# slam_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Vector3
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from multi_slam.Localization import Localization
from multi_slam.Mapping import Mapping
import numpy as np
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, Bool
from sensor_msgs_py.point_cloud2 import read_points
from multi_slam.Planner import Planner


class SLAMNode(Node):
    """
    Simultaneous Localization and Mapping (SLAM) node.
    
    Processes sensor data from LiDAR and beacons to perform SLAM,
    maintaining a probabilistic map and robot pose estimate.
    """
    
    def __init__(self):
        """Initialize the SLAM node with publishers, subscribers, and parameters."""
        super().__init__("slam_node")
        
        # Parameters
        self.declare_parameter('map_size_x', 50.0)
        self.declare_parameter('map_size_y', 50.0)
        self.declare_parameter('map_origin_x', -25.0)
        self.declare_parameter('map_origin_y', -25.0)
        self.declare_parameter('grid_size', 0.1)
        self.declare_parameter('num_particles', 1000)
        self.declare_parameter('position_std_dev', 0.1)
        self.declare_parameter('initial_noise', 0.5)

        # Get parameters
        map_size_x = self.get_parameter('map_size_x').value
        map_size_y = self.get_parameter('map_size_y').value
        map_origin_x = self.get_parameter('map_origin_x').value
        map_origin_y = self.get_parameter('map_origin_y').value
        grid_size = self.get_parameter('grid_size').value
        num_particles = self.get_parameter('num_particles').value
        position_std_dev = self.get_parameter('position_std_dev').value
        initial_noise = self.get_parameter('initial_noise').value
    
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
        self.dt = 0.02
        
        # Initialize Localization
        self.localization = Localization(
            initial_location=initial_position, 
            initial_noise=initial_noise,
            std_dev_noise=position_std_dev,
            num_particles=num_particles,
            dt=self.dt
        )

        self.planner = Planner()
        self.planner.update_map(self.map.get_prob_grid(), self.map.map_origin, self.map.grid_size)
        
        # Lidar range
        self.lidar_range = (0.1, 10.0)  # min and max range in meters
        
        # Current data
        self.lidar_data = []
        self.beacon_data = []
        self.control_input = np.zeros(3)  # vx, vy, 0
        self.pos_hat_new = np.array([0.0, 0.0, 0.0])
        self.particles = None
        self.sim_done = True

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
        self.create_subscription(
            PointCloud2, "/particles_pred", self.particles_pred_callback, 10
        )
        self.sim_done_sub = self.create_subscription(
            Bool, "/sim_done", self.sim_done_cb, 10
        )

        # Publishers
        self.particles_pub = self.create_publisher(PointCloud2, "particles", 10)
        self.slam_done_pub = self.create_publisher(Bool, "/slam_done", 10)
        self.pose_pub = self.create_publisher(Marker, "/pos_hat_viz", 10)
        self.map_pub = self.create_publisher(OccupancyGrid, "/occupancy_grid", 10)
        self.beacon_pub = self.create_publisher(MarkerArray, "/estimated_beacons", 10)
        
        # Timer for visualization
        self.create_timer(1, self.publish_viz)
        self.sim_done_cb(Bool(data=True))  # Call once to initialize


    def sim_done_cb(self, msg: Bool):
        """
        Process simulation step completion and update SLAM state.
        
        This is the main SLAM update loop that runs when the physics simulation
        has completed a step.
        """
        # Localization
        particles, cov = self.localization.update_position(
            self.beacon_data,
            self.map
        )

        pos_hat_new = np.average(particles, axis=0)
        pos_hat_new = pos_hat_new.astype(float)
        pos_hat_new[2] = 0.0

        self.pos_hat_new = pos_hat_new
        self.position = self.pos_hat_new
        self.position_cov = cov
        
        self.map.update(
            robot_pos=self.position,
            robot_cov=self.position_cov,
            lidar_data=self.lidar_data,
            lidar_range=self.lidar_range,
            beacon_data=self.beacon_data
        )
        
        self.get_logger().info("Updated map.")

        self.planner.update_map(self.map.get_prob_grid(), self.map.map_origin, self.map.grid_size)
        self.planner.update_beacons(self.map.beacon_positions)
        self.planner.update_position(self.position)
        goal = self.planner.select_goal_point()
        self.get_logger().info(f"Goal: {goal}")

        self.publish_particles()
        self.slam_done_pub.publish(Bool(data=True))

    def particles_callback(self, msg: PointCloud2):
        """Process particles from visualization."""
        points = list(read_points(msg, field_names=("x", "y", "z")))
        self.particles = np.array([np.array([p[0], p[1], p[2]]) for p in points])

    def publish_viz(self):
        """Publish all visualization messages."""
        self.publish_pos_viz()
        self.publish_map_viz()
        self.publish_beacons_viz()

    def lidar_callback(self, msg: PointCloud2):
        """Process LiDAR data."""
        points = list(read_points(msg, field_names=("x", "y", "z")))
        self.lidar_data = [np.array([p[0], p[1], p[2]]) for p in points]

    def beacon_callback(self, msg: PointCloud2):
        """Process beacon data."""
        points = list(read_points(msg, field_names=("x", "y", "z")))
        self.beacon_data = [np.array([p[0], p[1], p[2]]) for p in points]

    def particles_pred_callback(self, msg: PointCloud2):
        """Process predicted particles from motion model."""
        points = list(read_points(msg, field_names=("x", "y", "z")))
        self.localization.particles = [np.array([p[0], p[1], p[2]]) for p in points]

    def control_callback(self, msg: Vector3):
        """Process control input."""
        self.control_input = np.array([msg.x, msg.y, msg.z])

    def publish_particles(self):
        """Publish particle filter state."""
        particles = self.localization.particles

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"
        particles_msg = point_cloud2.create_cloud_xyz32(header, particles)
        self.particles_pub.publish(particles_msg)

    def publish_pos_viz(self):
        """Publish the estimated pose visualization marker."""
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

    def publish_map_viz(self):
        """Publish occupancy grid for visualization."""
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
        probs = self.map.get_prob_grid()
        msg.data = (probs * 100).astype(int).flatten().tolist()
        
        self.map_pub.publish(msg)

    def publish_beacons_viz(self):
        """Publish estimated beacon positions for visualization."""
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


def main(args=None):
    """Entry point for the SLAM node."""
    rclpy.init(args=args)
    node = SLAMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()