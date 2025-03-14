import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from multi_slam.Map import MAP
from std_msgs.msg import Header, Bool
import numpy as np
from visualization_msgs.msg import Marker
import math
from shapely.geometry import Point


class PhysicsSimNode(Node):
    """
    Physics simulation node that handles robot movement and sensor simulation.
    
    Simulates robot movement, collision detection, and generates sensor data (LiDAR, beacons).
    """
    
    def __init__(self):
        """Initialize the physics simulation node with publishers, subscribers, and parameters."""
        super().__init__("physics_sim")
        
        # Subscribers
        self.control_signal_sub = self.create_subscription(
            Vector3, "control_signal", self.control_signal_cb, 10
        )
        
        self.slam_done_sub = self.create_subscription(
            Bool, "/slam_done", self.slam_done_cb, 10
        )
        
        self.create_subscription(
            PointCloud2, "/particles", self.particles_callback, 10
        )

        # Parameters
        self.declare_parameter("lidar_r_max", 10.0)
        self.lidar_r_max = self.get_parameter("lidar_r_max").value

        self.declare_parameter("lidar_r_min", 0.1)
        self.lidar_r_min = self.get_parameter("lidar_r_min").value

        self.declare_parameter("lidar_delta_theta", 3)
        self.lidar_delta_theta = self.get_parameter("lidar_delta_theta").value

        self.declare_parameter("lidar_std_dev", 0.1)
        self.lidar_std_dev = self.get_parameter("lidar_std_dev").value

        self.declare_parameter("beacon_std_dev", 0.1)
        self.beacon_std_dev = self.get_parameter("beacon_std_dev").value

        self.declare_parameter("vel_std_dev", 0.4)
        self.vel_std_dev = self.get_parameter("vel_std_dev").value

        self.declare_parameter("collision_buffer", 0.1)
        self.collision_buffer = self.get_parameter("collision_buffer").value
        
        self.declare_parameter("collision_increment", 0.02)
        self.collision_increment = self.get_parameter("collision_increment").value

        self.declare_parameter("sim_dt", 0.1)
        self.sim_dt = self.get_parameter("sim_dt").value

        # Publishers
        self.lidar_pub = self.create_publisher(PointCloud2, "lidar", 10)
        self.beacon_pub = self.create_publisher(PointCloud2, "beacon", 10)
        self.pos_true_viz_pub = self.create_publisher(
            Marker, "visualization_marker_true", 10
        )
        self.beacon_viz_pub = self.create_publisher(
            PointCloud2, "beacon_viz", 10
        )
        self.lidar_viz_pub = self.create_publisher(
            PointCloud2, "lidar_viz", 10
        )
        self.sim_done_pub = self.create_publisher(Bool, "/sim_done", 10)
        self.pos_baseline_viz_pub = self.create_publisher(Marker, "/pos_baseline_viz", 10)
        self.particles_pred_pub = self.create_publisher(
            PointCloud2, "/particles_pred", 10
        )

        # Timers
        self.lidar_pub_timer = self.create_timer(0.1, self.lidar_publish_cb)
        self.beacon_pub_timer = self.create_timer(0.1, self.beacon_publish_cb)
        self.sim_update_timer = self.create_timer(self.sim_dt, self.sim_update_cb)

        # State variables
        self.pos_true = np.array([0, 0, 0], dtype=float)
        self.vel_true = np.array([0, 0, 0], dtype=float)
        self.vel_ideal = np.array([0, 0, 0], dtype=float)
        self.slam_done = True
        self.pos_baseline = np.array([0, 0, 0], dtype=float)
        self.particles = None

    def sim_update_cb(self):
        """Update simulation state at each time step."""
        # Wait until the SLAM node has published a new position
        if not self.slam_done:
            return

        intended_true_pos = self.pos_true + self.vel_true * self.sim_dt
        self.pos_true = self.check_collision(self.pos_true, intended_true_pos)

        self.pos_baseline += self.vel_ideal * self.sim_dt
        self.pub_pos_true_viz()
        self.pub_pos_baseline_viz()

        if self.particles is not None:
            particles_pred = self.particles + self.vel_ideal * self.sim_dt
            header = Header(stamp=self.get_clock().now().to_msg(), frame_id="map")
            msg = pc2.create_cloud_xyz32(header, particles_pred)
            self.particles_pred_pub.publish(msg)
        
        self.sim_done_pub.publish(Bool(data=True))
        self.slam_done = False

    def particles_callback(self, msg: PointCloud2):
        """Store particle positions from SLAM node."""
        points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        self.particles = np.array([np.array([p[0], p[1], p[2]]) for p in points])

    def slam_done_cb(self, _: Bool):
        """Mark SLAM processing as complete."""
        self.slam_done = True

    def control_signal_cb(self, msg: Vector3):
        """Process control commands and apply noise to simulate real-world conditions."""
        self.vel_ideal = np.array([msg.x, msg.y, 0.0])
        self.vel_true = self._apply_2d_noise([self.vel_ideal], self.vel_std_dev)[0]

    def _apply_2d_noise(self, points: np.array, std_dev: float):
        """Apply Gaussian noise to 2D points to simulate sensor noise."""
        noisy_points = []
        for point in points:
            noisy_point = np.array(
                [
                    point[0] + np.random.normal(0, std_dev),
                    point[1] + np.random.normal(0, std_dev),
                    0,
                ]
            )
            noisy_points.append(noisy_point)
        return noisy_points

    def create_robot_polygon(self, position):
        """Create a circular buffer around the robot for collision detection."""
        return Point(position[0], position[1]).buffer(self.collision_buffer)

    def check_collision(self, current_pos, intended_pos):
        """
        Check for collisions between robot and environment.
        
        Uses incremental movement to detect collisions accurately.
        Returns the farthest safe position without collisions.
        """
        # Create a circular buffer at the intended position
        robot_circle = self.create_robot_polygon(intended_pos)
        
        if not MAP.intersections(robot_circle):
            return intended_pos
            
        # Calculate direction from current to intended position
        direction = np.array([
            intended_pos[0] - current_pos[0],
            intended_pos[1] - current_pos[1],
            intended_pos[2] - current_pos[2]
        ])
        
        # Calculate distance between current and intended position
        distance = math.sqrt(direction[0]**2 + direction[1]**2)
        
        # If no movement, return current position
        if distance < 1e-6:
            return current_pos
            
        # Normalize direction vector (for x, y components)
        if distance > 0:
            direction = direction / distance
        
        # Start from current position and move incrementally
        safe_pos = np.array(current_pos)  # Last known safe position
        test_pos = np.array(current_pos)
        increment_distance = 0.0
        test_collides = False
        
        # Use smaller increments to ensure we don't miss collisions
        while increment_distance < distance:
            # Move forward by increment
            increment_distance += self.collision_increment
            if increment_distance >= distance:
                # We've reached the full distance
                break
                
            # Calculate the new test position
            test_pos = np.array([
                current_pos[0] + direction[0] * increment_distance,
                current_pos[1] + direction[1] * increment_distance,
                current_pos[2] + (intended_pos[2] - current_pos[2]) * (increment_distance / distance)
            ])
            
            # Create robot polygon at test position
            test_poly = self.create_robot_polygon(test_pos)
            
            # Check for collisions at test position
            test_collides = bool(MAP.intersections(test_poly))
                
            # If collision detected, return the last safe position
            if test_collides:
                return safe_pos
            
            # Update the safe position
            safe_pos = np.array(test_pos)
                
        # If we completed the loop without collision, return the intended position
        # This handles the case where the last increment might go past the intended position
        if not test_collides:
            return intended_pos
        else:
            return safe_pos

    def lidar_publish_cb(self):
        """Generate and publish simulated LiDAR data."""
        # Use the true position (with noise) for LiDAR measurements
        points = MAP.calc_lidar_point_cloud(
            self.pos_true,
            self.lidar_delta_theta,
            self.lidar_r_max,
            self.lidar_r_min,
        )
        # Apply additional sensor noise to the points
        noisy_points = self._apply_2d_noise(points, self.lidar_std_dev)
        
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id="map")
        lidar_msg = pc2.create_cloud_xyz32(header, noisy_points)
        self.lidar_pub.publish(lidar_msg)

        lidar_points_world = []
        for point in noisy_points:
            lidar_points_world.append(
                np.array(
                    [
                        point[0] + self.pos_true[0],
                        point[1] + self.pos_true[1],
                        0,
                    ]
                )
            )
        lidar_points_world_msg = pc2.create_cloud_xyz32(
            header, lidar_points_world
        )
        self.lidar_viz_pub.publish(lidar_points_world_msg)

    def beacon_publish_cb(self):
        """Generate and publish simulated beacon data."""
        # Use the true position (with noise) for beacon measurements
        points = MAP.calc_beacon_positions(self.pos_true)
        # Apply sensor noise
        noisy_points = self._apply_2d_noise(points, self.beacon_std_dev)
        
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id="map")
        beacon_msg = pc2.create_cloud_xyz32(header, noisy_points)
        self.beacon_pub.publish(beacon_msg)

        beacon_positions_world = []
        for point in noisy_points:
            beacon_positions_world.append(
                np.array(
                    [
                        point[0] + self.pos_true[0],
                        point[1] + self.pos_true[1],
                        0,
                    ]
                )
            )
        beacon_positions_world_msg = pc2.create_cloud_xyz32(
            header, beacon_positions_world
        )
        self.beacon_viz_pub.publish(beacon_positions_world_msg)

    def pub_pos_true_viz(self):
        """Publish the true position marker (actual physical position with noise)."""
        marker = Marker()
        marker.header = Header(stamp=self.get_clock().now().to_msg(), frame_id="map")
        marker.ns = "true_position"
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Set the position to the true position
        marker.pose.position.x = self.pos_true[0]
        marker.pose.position.y = self.pos_true[1]
        marker.pose.position.z = self.pos_true[2]

        # No rotation needed
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # For a sphere with radius 0.3
        marker.scale.x = 0.6
        marker.scale.y = 0.6
        marker.scale.z = 0.6

        # Green color for true position
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0  # Fully opaque

        self.pos_true_viz_pub.publish(marker)

    def pub_pos_baseline_viz(self):
        """Publish the baseline position marker (idealized position without noise)."""
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "map"
        marker.ns = "baseline_position"
        marker.id = 2
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = self.pos_baseline[0]
        marker.pose.position.y = self.pos_baseline[1]
        marker.pose.position.z = self.pos_baseline[2]
        
        marker.scale.x = 0.6
        marker.scale.y = 0.6
        marker.scale.z = 0.6

        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 0.5

        self.pos_baseline_viz_pub.publish(marker)


def main(args=None):
    """Entry point for the physics simulation node."""
    rclpy.init(args=args)
    node = PhysicsSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()