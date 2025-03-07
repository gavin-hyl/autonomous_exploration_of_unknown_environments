#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
from shapely.geometry import Polygon
import math
from multi_slam_ws.src.multi_slam.multi_slam.Map import Map, MAP  # Import Map and MAP from map.py

class SimplePhysicsSimNode(Node):
    def __init__(self):
        super().__init__('physics_sim')
        
        # Parameters
        self.declare_parameter('robot_width', 1.0)
        self.declare_parameter('robot_length', 2.0)
        self.declare_parameter('delta_noise_std', 0.02)
        
        # Robot dimensions and noise parameters
        self.robot_width = self.get_parameter('robot_width').value
        self.robot_length = self.get_parameter('robot_length').value
        self.delta_noise_std = self.get_parameter('delta_noise_std').value
        
        # Use predefined MAP
        self.map = MAP
        
        # Current true position
        self.true_x = 5.0
        self.true_y = 5.0
        self.true_theta = 0.0
        
        # Publishers
        self.pos_publisher = self.create_publisher(
            Float32MultiArray, '/true_position', 10)
        
        # Subscribers - expects delta values
        self.create_subscription(
            Float32MultiArray, '/control_input', self.action_callback, 10)
        
        self.get_logger().info('Physics simulation node initialized')
    
    def create_robot_polygon(self, x, y, theta):
        """Create a polygon representing the robot"""
        half_length = self.robot_length / 2.0
        half_width = self.robot_width / 2.0
        
        corners = [
            (-half_length, -half_width),
            (half_length, -half_width),
            (half_length, half_width),
            (-half_length, half_width)
        ]
        
        rotated_corners = []
        for corner_x, corner_y in corners:
            rotated_x = corner_x * math.cos(theta) - corner_y * math.sin(theta) + x
            rotated_y = corner_x * math.sin(theta) + corner_y * math.cos(theta) + y
            rotated_corners.append((rotated_x, rotated_y))
        
        return Polygon(rotated_corners)
    
    def add_noise_to_delta(self, dx, dy, dtheta):
        """Add Gaussian noise to delta values"""
        dx_noisy = dx + np.random.normal(0, self.delta_noise_std)
        dy_noisy = dy + np.random.normal(0, self.delta_noise_std)
        dtheta_noisy = dtheta + np.random.normal(0, self.delta_noise_std/5)
        return dx_noisy, dy_noisy, dtheta_noisy
    
    def find_valid_position(self, start_x, start_y, start_theta, end_x, end_y, end_theta):
        """Find position right before collision"""
        # Check if end position is valid
        robot_poly = self.create_robot_polygon(end_x, end_y, end_theta)
        if not self.map.intersections(robot_poly):
            return end_x, end_y, end_theta
        
        # Step backwards from collision point
        step = 0.05  # Step size
        alpha = 1.0
        
        while alpha > 0:
            alpha -= step
            x = start_x + alpha * (end_x - start_x)
            y = start_y + alpha * (end_y - start_y)
            theta = start_theta + alpha * (end_theta - start_theta)
            
            robot_poly = self.create_robot_polygon(x, y, theta)
            if not self.map.intersections(robot_poly):
                return x, y, theta
        
        # Fallback to start position
        return start_x, start_y, start_theta
    
    def action_callback(self, msg):
        """Process incoming delta actions (dx, dy, dtheta)"""
        if len(msg.data) >= 3:
            # Get delta values
            dx, dy, dtheta = msg.data[0], msg.data[1], msg.data[2]
            
            # Add noise to delta values
            dx_noisy, dy_noisy, dtheta_noisy = self.add_noise_to_delta(dx, dy, dtheta)
            
            # Store original position
            start_x, start_y, start_theta = self.true_x, self.true_y, self.true_theta
            
            # Apply noisy deltas to get target position
            target_x = start_x + dx_noisy
            target_y = start_y + dy_noisy
            target_theta = start_theta + dtheta_noisy
            
            # Find valid position (handles collisions)
            valid_x, valid_y, valid_theta = self.find_valid_position(
                start_x, start_y, start_theta, target_x, target_y, target_theta)
            
            # Update true position
            self.true_x, self.true_y, self.true_theta = valid_x, valid_y, valid_theta
            
            # Publish true position directly
            pos_msg = Float32MultiArray()
            pos_msg.data = [float(self.true_x), float(self.true_y), float(self.true_theta)]
            self.pos_publisher.publish(pos_msg)
            
            # Log movement
            self.get_logger().info(f'Applied action: [{dx:.2f}, {dy:.2f}, {dtheta:.2f}]')
            self.get_logger().info(f'True position: [{self.true_x:.2f}, {self.true_y:.2f}, {self.true_theta:.2f}]')

def main(args=None):
    rclpy.init(args=args)
    node = SimplePhysicsSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()



    #!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
from shapely.geometry import Point, LineString
import math
from multi_slam_ws.src.multi_slam.multi_slam.Map import Map, MAP  # Import Map and MAP from map.py

class BeaconsNode(Node):
    def __init__(self):
        super().__init__('beacons_node')
        
        # Parameters
        self.declare_parameter('range_noise_std', 0.1)
        
        # Use predefined MAP
        self.map = MAP
        
        # Noise parameters for range measurements
        self.range_noise_std = self.get_parameter('range_noise_std').value
        
        # Define beacon positions (x, y)
        self.beacon_positions = [
            (2.0, 2.0),   # Beacon 1
            (8.0, 2.0),   # Beacon 2
            (8.0, 8.0),   # Beacon 3
            (2.0, 8.0),   # Beacon 4
        ]
        
        # Publisher for beacon measurements
        self.beacon_publisher = self.create_publisher(
            Float32MultiArray, '/beacon_measurements', 10)
        
        # Subscribe to robot's true position
        self.create_subscription(
            Float32MultiArray, '/true_position', self.position_callback, 10)
        
        self.get_logger().info(f'Beacons node initialized with {len(self.beacon_positions)} beacons')
    
    def has_line_of_sight(self, robot_x, robot_y, beacon_x, beacon_y):
        """Check if there's a clear line of sight between robot and beacon
        
        Args:
            robot_x, robot_y: Robot position
            beacon_x, beacon_y: Beacon position
            
        Returns:
            bool: True if there is a line of sight, False otherwise
        """
        # Create a line from robot to beacon
        line = LineString([(robot_x, robot_y), (beacon_x, beacon_y)])
        
        # Check if this line intersects with any obstacle in the map
        return not self.map.intersections(line)
    
    def add_noise_to_measurement(self, dx, dy):
        """Add Gaussian noise to delta measurements
        
        Args:
            dx, dy: True delta values from robot to beacon
            
        Returns:
            tuple: Noisy delta values (dx_noisy, dy_noisy)
        """
        # Calculate true range
        true_range = np.sqrt(dx**2 + dy**2)
        
        # Add noise to range
        noisy_range = true_range + np.random.normal(0, self.range_noise_std)
        
        # Calculate angle (no noise added to angle for simplicity)
        angle = np.arctan2(dy, dx)
        
        # Convert back to cartesian coordinates
        dx_noisy = noisy_range * np.cos(angle)
        dy_noisy = noisy_range * np.sin(angle)
        
        return dx_noisy, dy_noisy
    
    def position_callback(self, msg):
        """Callback for receiving robot's true position
        
        Args:
            msg: Message containing robot position [x, y, theta]
        """
        if len(msg.data) >= 3:
            robot_x, robot_y, _ = msg.data[0], msg.data[1], msg.data[2]
            self.update_and_publish(robot_x, robot_y)
    
    def update_and_publish(self, robot_x, robot_y):
        """Update visible beacons and publish measurements
        
        Args:
            robot_x, robot_y: Current robot position
        """
        visible_beacons = []
        
        for beacon_id, (beacon_x, beacon_y) in enumerate(self.beacon_positions):
            if self.has_line_of_sight(robot_x, robot_y, beacon_x, beacon_y):
                # Calculate true deltas
                dx = beacon_x - robot_x
                dy = beacon_y - robot_y
                
                # Add noise to measurements
                dx_noisy, dy_noisy = self.add_noise_to_measurement(dx, dy)
                
                # Store beacon info: [beacon_id, dx_noisy, dy_noisy]
                visible_beacons.extend([float(beacon_id), dx_noisy, dy_noisy])
        
        # Publish beacon measurements
        if visible_beacons:
            msg = Float32MultiArray()
            msg.data = visible_beacons
            self.beacon_publisher.publish(msg)
            
            num_visible = len(visible_beacons) // 3
            self.get_logger().info(f'Published {num_visible} visible beacon measurements')
        else:
            self.get_logger().info('No beacons currently visible')

def main(args=None):
    rclpy.init(args=args)
    node = BeaconsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from multi_slam import Map

class LidarNode(Node):
    def __init__(self):
        super().__init__('lidar_node')

        self.world_map = Map
        
        self.pose_sub = self.create_subscription(
            Pose, '/robot_pose', self.update_point_cloud, 10)
        self.pointcloud_pub = self.create_publisher(PointCloud2, '/point_cloud', 10)
        
        # Lidar parameters
        self.scan_range = 5.0  
        self.scan_resolution = 0.1  
        
        self.get_logger().info('Lidar Node initialized')

    def update_point_cloud(self, pose):
        points = []
        
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