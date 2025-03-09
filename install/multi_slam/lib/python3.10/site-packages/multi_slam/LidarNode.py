import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from multi_slam import global_world_map

class LidarNode(Node):
    def __init__(self):
        super().__init__('lidar_node')
        
        # Use the global world map
        self.world_map = global_world_map
        
        # Subscribers and publishers
        self.pose_sub = self.create_subscription(
            Pose, '/robot_pose', self.update_point_cloud, 10)
        self.pointcloud_pub = self.create_publisher(PointCloud2, '/point_cloud', 10)
        
        # Lidar parameters
        self.scan_range = 5.0  # meters
        self.scan_resolution = 0.1  # meters
        
        self.get_logger().info('Lidar Node initialized')

    def update_point_cloud(self, pose):
        """Generate point cloud based on robot pose"""
        points = []
        
        # Simulate point cloud generation around robot
        for x in np.arange(-self.scan_range, self.scan_range, self.scan_resolution):
            for y in np.arange(-self.scan_range, self.scan_range, self.scan_resolution):
                point_x = pose.position.x + x
                point_y = pose.position.y + y
                
                if self.world_map.is_valid_position(int(point_x), int(point_y)):
                    points.append([point_x, point_y, 0.0])
        
        # Create PointCloud2 message
        cloud_msg = self.create_point_cloud_msg(points)
        self.pointcloud_pub.publish(cloud_msg)

    def create_point_cloud_msg(self, points):
        """Create PointCloud2 message from points"""
        cloud_msg = PointCloud2()
        cloud_msg.header = Header()
        cloud_msg.header.frame_id = 'map'
        
        cloud_msg.height = 1
        cloud_msg.width = len(points)
        cloud_msg.is_dense = False
        
        cloud_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        
        cloud_msg.point_step = 12
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
        
        # Convert points to byte array
        cloud_msg.data = np.array(points, dtype=np.float32).tobytes()
        
        return cloud_msg

def main(args=None):
    rclpy.init(args=args)
    node = LidarNode()
    
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()