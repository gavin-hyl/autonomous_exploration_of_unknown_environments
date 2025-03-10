import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from multi_slam.Map import MAP
from std_msgs.msg import Header
import numpy as np
from visualization_msgs.msg import Marker
import math
from shapely.geometry import Polygon


class PhysicsSimNode(Node):
    def __init__(self):
        super().__init__("physics_sim")
        self.control_signal_sub = self.create_subscription(
            Vector3, "control_signal", self.control_signal_cb, 10
        )

        self.declare_parameter("lidar_r_max", 5)
        self.lidar_r_max = self.get_parameter("lidar_r_max").value

        self.declare_parameter("lidar_r_min", 0.2)
        self.lidar_r_min = self.get_parameter("lidar_r_min").value

        self.declare_parameter("lidar_delta_theta", 3)
        self.lidar_delta_theta = self.get_parameter("lidar_delta_theta").value

        self.declare_parameter("lidar_std_dev", 0)
        self.lidar_std_dev = self.get_parameter("lidar_std_dev").value

        self.declare_parameter("beacon_std_dev", 0)
        self.beacon_std_dev = self.get_parameter("beacon_std_dev").value

        self.declare_parameter("pos_std_dev_dist", 0.1)
        self.pos_std_dev_dist = self.get_parameter("pos_std_dev_dist").value

        # Robot dimensions parameters
        self.declare_parameter("robot_length", 0.4)
        self.robot_length = self.get_parameter("robot_length").value
        
        self.declare_parameter("robot_width", 0.3)
        self.robot_width = self.get_parameter("robot_width").value
        
        # Increment for collision avoidance
        self.declare_parameter("collision_increment", 0.05)
        self.collision_increment = self.get_parameter("collision_increment").value

        self.declare_parameter("sim_dt", 0.1)
        self.sim_dt = self.get_parameter("sim_dt").value

        self.lidar_pub = self.create_publisher(PointCloud2, "lidar", 10)
        self.beacon_pub = self.create_publisher(PointCloud2, "beacon", 10)

        self.pos_viz_pub = self.create_publisher(Marker, "visualization_marker", 10)
        self.pos_noisy_viz_pub = self.create_publisher(Marker, "visualization_marker_noisy", 10)
        self.beacon_viz_pub = self.create_publisher(PointCloud2, "beacon_viz", 10)
        self.lidar_viz_pub = self.create_publisher(PointCloud2, "lidar_viz", 10)

        self.lidar_pub_timer = self.create_timer(0.1, self.lidar_publish_cb)
        self.beacon_pub_timer = self.create_timer(0.1, self.beacon_publish_cb)
        self.sim_update_timer = self.create_timer(self.sim_dt, self.sim_update_cb)

        self.pos_true = np.array([2, 2, 0], dtype=float)
        self.pos_noisy = np.array([2, 2, 0], dtype=float)
        self.vel_true = np.array([0, 0, 0], dtype=float)
        self.pos_est = np.array([0, 0, 0], dtype=float)
        self.accel = np.array([0, 0, 0], dtype=float)

    def control_signal_cb(self, msg: Vector3):
        # Update velocity directly instead of using acceleration
        # Set z to 0 to ensure we're only working in 2D
        self.vel_true = np.array([msg.x, msg.y, 0.0])

    def _apply_2d_noise(self, points: np.array, std_dev: float):
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
        """
        Creates a polygon representing the robot at the given position
        """
        # Extract position and orientation
        x, y, theta = position
        
        # Calculate the four corners of the rectangle
        # First, calculate corners for a rectangle centered at origin, aligned with x-axis
        half_length = self.robot_length / 2
        half_width = self.robot_width / 2
        
        corners = [
            [-half_length, -half_width],  # rear left
            [half_length, -half_width],   # front left
            [half_length, half_width],    # front right
            [-half_length, half_width]    # rear right
        ]
        
        # Rotate corners by theta
        rotated_corners = []
        for corner in corners:
            # Rotation matrix
            rotated_x = corner[0] * math.cos(theta) - corner[1] * math.sin(theta)
            rotated_y = corner[0] * math.sin(theta) + corner[1] * math.cos(theta)
            
            # Translate to position
            rotated_corners.append([x + rotated_x, y + rotated_y])
        
        # Create a polygon from the rotated corners
        return Polygon(rotated_corners)

    def check_collision(self, current_pos, intended_pos):
        """
        Check if there would be a collision when moving from current_pos to intended_pos
        Robot is modeled as a point with a circular buffer
        Returns the position right before collision, or intended_pos if no collision
        """
        # Create a point representing the robot at the intended position with a buffer
        robot_radius = max(self.robot_length, self.robot_width) / 2
        robot_point = Point(intended_pos[0], intended_pos[1]).buffer(robot_radius)
        
        # Check if the buffered point intersects with any obstacles or the boundary
        obstacles_intersect = False
        for obstacle in MAP.obstacles:
            if robot_point.intersects(obstacle):
                obstacles_intersect = True
                break
                
        # Check boundary intersection
        boundary_intersect = robot_point.intersects(MAP.boundary)
        
        # If no collision, return the intended position
        if not obstacles_intersect and not boundary_intersect:
            return intended_pos
            
        # Calculate direction from current to intended position (only x and y)
        direction = np.array([
            intended_pos[0] - current_pos[0],
            intended_pos[1] - current_pos[1],
            0.0  # No z direction since we're in 2D
        ])
        
        # Calculate distance between current and intended position
        distance = math.sqrt(direction[0]**2 + direction[1]**2)
        
        # If no movement, return current position
        if distance < 1e-6:
            return current_pos
            
        # Normalize direction vector (for x, y components)
        if distance > 0:
            direction[0] /= distance
            direction[1] /= distance
        
        # Start from current position and move incrementally
        increment_distance = 0.0
        
        # Try positions incrementally from current to intended
        while increment_distance < distance:
            # Move forward by increment
            increment_distance += self.collision_increment
            if increment_distance >= distance:
                # We've reached or exceeded the intended distance
                # No safe position found, return current position
                return current_pos
                
            # Calculate the new test position
            test_pos = np.array([
                current_pos[0] + direction[0] * increment_distance,
                current_pos[1] + direction[1] * increment_distance,
                0.0  # Always 0 for z component
            ])
            
            # Create robot point at test position
            test_point = Point(test_pos[0], test_pos[1]).buffer(robot_radius)
            
            # Check for collisions at test position
            test_collides = False
            for obstacle in MAP.obstacles:
                if test_point.intersects(obstacle):
                    test_collides = True
                    break
                    
            if test_point.intersects(MAP.boundary):
                test_collides = True
                
            # If no collision at this test position, return it
            if not test_collides:
                return test_pos
                
        # If we get here, we couldn't find a safe position
        return current_pos

    def lidar_publish_cb(self):
        lidar_points = MAP.calc_lidar_point_cloud(
            self.pos_true, self.lidar_delta_theta, self.lidar_r_max, self.lidar_r_min
        )
        lidar_points = self._apply_2d_noise(lidar_points, self.lidar_std_dev)
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"
        lidar_msg = point_cloud2.create_cloud_xyz32(header, lidar_points)
        self.lidar_pub.publish(lidar_msg)

        lidar_points_world = []
        for point in lidar_points:
            lidar_points_world.append(
                np.array(
                    [
                        point[0] + self.pos_true[0],
                        point[1] + self.pos_true[1],
                        0,
                    ]
                )
            )
        lidar_points_world_msg = point_cloud2.create_cloud_xyz32(
            header, lidar_points_world
        )
        self.lidar_viz_pub.publish(lidar_points_world_msg)

    def beacon_publish_cb(self):
        beacon_positions = MAP.calc_beacon_positions(self.pos_true)
        beacon_positions = self._apply_2d_noise(beacon_positions, self.beacon_std_dev)
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"
        beacon_msg = point_cloud2.create_cloud_xyz32(header, beacon_positions)
        self.beacon_pub.publish(beacon_msg)

        beacon_positions_world = []
        for point in beacon_positions:
            beacon_positions_world.append(
                np.array(
                    [
                        point[0] + self.pos_true[0],
                        point[1] + self.pos_true[1],
                        0,
                    ]
                )
            )
        beacon_positions_world_msg = point_cloud2.create_cloud_xyz32(
            header, beacon_positions_world
        )
        self.beacon_viz_pub.publish(beacon_positions_world_msg)

    def publish_true_pos(self):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "map"
        marker.ns = "true_position"
        marker.id = 0
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

        # For a sphere with radius 0.3, set the scale (diameter = 2 * radius)
        marker.scale.x = 0.6
        marker.scale.y = 0.6
        marker.scale.z = 0.6

        # Define the color (for example, green)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0  # Fully opaque

        self.pos_viz_pub.publish(marker)

    def publish_noisy_pos(self):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "map"
        marker.ns = "noisy_position"
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Set the position to the noisy position
        marker.pose.position.x = self.pos_noisy[0]
        marker.pose.position.y = self.pos_noisy[1]
        marker.pose.position.z = self.pos_noisy[2]

        # No rotation needed
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # For a sphere with radius 0.3, set the scale (diameter = 2 * radius)
        marker.scale.x = 0.6
        marker.scale.y = 0.6
        marker.scale.z = 0.6

        # Define the color (red for noisy pos)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0  # Fully opaque

        self.pos_noisy_viz_pub.publish(marker)

    def sim_update_cb(self):
        # First calculate the intended new position
        intended_pos = self.pos_true + self.vel_true * self.sim_dt
        
        # Check for collisions and get the safe position
        self.pos_true = self.check_collision(self.pos_true, intended_pos)
        
        # Calculate noisy position using 2D noise instead of 3D
        points = [self.pos_true]
        noisy_points = self._apply_2d_noise(points, self.pos_std_dev_dist)
        self.pos_noisy = noisy_points[0]
        
        self.publish_true_pos()
        self.publish_noisy_pos()


def main(args=None):
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