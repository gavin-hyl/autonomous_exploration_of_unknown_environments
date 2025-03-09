import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from multi_slam.Map import MAP
from std_msgs.msg import Header
import numpy as np
from visualization_msgs.msg import Marker


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

        self.declare_parameter("lidar_std_dev", 0.1)
        self.lidar_std_dev = self.get_parameter("lidar_std_dev").value

        self.declare_parameter("beacon_std_dev", 0.1)
        self.beacon_std_dev = self.get_parameter("beacon_std_dev").value

        self.declare_parameter("sim_dt", 0.1)
        self.sim_dt = self.get_parameter("sim_dt").value

        self.lidar_pub = self.create_publisher(PointCloud2, "lidar", 10)
        self.beacon_pub = self.create_publisher(PointCloud2, "beacon", 10)

        self.pos_viz_pub = self.create_publisher(Marker, "visualization_marker", 10)
        self.beacon_viz_pub = self.create_publisher(PointCloud2, "beacon_viz", 10)
        self.lidar_viz_pub = self.create_publisher(PointCloud2, "lidar_viz", 10)

        self.lidar_pub_timer = self.create_timer(0.1, self.lidar_publish_cb)
        self.beacon_pub_timer = self.create_timer(0.1, self.beacon_publish_cb)
        self.sim_update_timer = self.create_timer(self.sim_dt, self.sim_update_cb)

        self.pos_true = np.array([2, 2, 0], dtype=float)
        self.vel_true = np.array([0, 0, 0], dtype=float)
        self.pos_est = np.array([0, 0, 0], dtype=float)
        self.accel = np.array([0, 0, 0], dtype=float)

    def control_signal_cb(self, msg: Vector3):
        self.accel = np.array([msg.x, msg.y, msg.z])

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

    def sim_update_cb(self):
        self.pos_true += self.vel_true * self.sim_dt
        self.vel_true += self.accel * self.sim_dt
        self.publish_true_pos()
        # FIXME collision detection


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
