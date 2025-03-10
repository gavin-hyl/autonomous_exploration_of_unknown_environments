import rclpy
from rclpy.node import Node
import math

from geometry_msgs.msg import Pose, Point
from multi_slam import global_world_map

class BeaconsNode(Node):
    def __init__(self):
        super().__init__('beacons_node')
        
        # Use the global world map
        self.world_map = global_world_map
        
        # Subscribers and publishers
        self.pose_sub = self.create_subscription(
            Pose, '/robot_pose', self.check_beacon_proximity, 10)
        self.beacon_pub = self.create_publisher(Point, '/beacon_detection', 10)
        
        # Detection range
        self.detection_range = 1.0
        
        self.get_logger().info('Beacons Node initialized')

    def check_beacon_proximity(self, pose):
        """Check if robot is near any beacons"""
        for beacon in self.world_map.beacons:
            dist = math.sqrt(
                (pose.position.x - beacon['x'])**2 + 
                (pose.position.y - beacon['y'])**2
            )
            
            if dist < self.detection_range:
                beacon_point = Point()
                beacon_point.x = beacon['x']
                beacon_point.y = beacon['y']
                self.beacon_pub.publish(beacon_point)
                self.get_logger().info(f'Beacon detected at {beacon_point.x}, {beacon_point.y}')

def main(args=None):
    rclpy.init(args=args)
    node = BeaconsNode()
    
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()