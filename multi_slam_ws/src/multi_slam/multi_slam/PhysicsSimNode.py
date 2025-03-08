import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import PointCloud
from sensor_msgs_py import point_cloud2
from multi_slam.Map import MAP
import numpy as np


class PhysicsSimNode(Node):
    def __init__(self):
        super().__init__('physics_sim')
        self.control_signal_sub = self.create_subscription(Vector3, 'control_signal', self.control_signal_cb, 10)

        self.declare_parameter('lidar_r_max', 5)
        self.lidar_r_max = self.get_parameter('lidar_r_max').value

        self.declare_parameter('lidar_r_min', 0.2)
        self.lidar_r_min = self.get_parameter('lidar_r_min').value

        self.declare_parameter('lidar_delta_theta', 3)
        self.lidar_delta_theta = self.get_parameter('lidar_delta_theta').value

        self.declare_parameter('lidar_std_dev', 0.1)
        self.lidar_std_dev = self.get_parameter('lidar_std_dev').value

        self.declare_parameter('beacon_std_dev', 0.1)
        self.beacon_std_dev = self.get_parameter('beacon_std_dev').value


        self.lidar_pub = self.create_publisher(PointCloud, 'lidar', 10)
        self.beacon_pub = self.create_publisher(PointCloud, 'beacon', 10)
        self.lidar_pub_timer = self.create_timer(0.1, self.lidar_publish_cb)
        self.beacon_pub_timer = self.create_timer(0.1, self.beacon_publish_cb)
        self.sim_update_timer = self.create_timer(0.1, self.sim_update_cb)
        self.pos_true = np.array([0, 0, 0])
        self.vel_true = np.array([0, 0, 0])
        self.pos_est = np.array([0, 0, 0])
        self.accel = np.array([0, 0, 0])
    
    def control_signal_cb(self, msg: Vector3):
        self.accel = np.array([msg.x, msg.y, msg.z])

    def _apply_2d_noise(self, points: np.array, std_dev: float) -> np.array:
        noise = np.random.normal(0, std_dev, points.shape)
        noise[:, 2] = 0
        return points + noise

    def lidar_publish_cb(self):
        lidar_points = MAP.calc_lidar_point_cloud(self.pos_true,
                                                    self.lidar_delta_theta,
                                                    self.lidar_r_max,
                                                    self.lidar_r_min)
        lidar_points = self._apply_2d_noise(lidar_points, self.lidar_std_dev)
        lidar_msg = point_cloud2.create_cloud_xyz32(None, lidar_points)
        self.lidar_pub.publish(lidar_msg)

    def beacon_publish_cb(self):
        beacon_positions = MAP.calc_beacon_positions(self.pos_true)
        beacon_positions = self._apply_2d_noise(beacon_positions, self.beacon_std_dev)
        beacon_msg = point_cloud2.create_cloud_xyz32(None, beacon_positions)
        self.beacon_pub.publish(beacon_msg)

    def sim_update_cb(self):
        pass



def main(args=None):
    rclpy.init(args=args)
    node = PhysicsSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('Exiting...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()