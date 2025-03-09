import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid
from multi_slam import WorldMap

class VisualizationNode(Node):
    def __init__(self):
        super().__init__('visualization_node')
        
        # Use the global world map
        self.world_map = WorldMap
        
        # Subscribers
        self.pose_sub = self.create_subscription(
            Pose, '/robot_pose', self.update_visualization, 10)
        
        # Publishers
        self.marker_pub = self.create_publisher(MarkerArray, '/visualization_markers', 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/world_map', 10)
        
        self.get_logger().info('Visualization Node initialized')

    def update_visualization(self, pose):
        """Update visualization markers"""
        marker_array = MarkerArray()
        
        # Robot marker
        robot_marker = Marker()
        robot_marker.type = Marker.CUBE
        robot_marker.pose = pose
        robot_marker.scale.x = 0.5
        robot_marker.scale.y = 0.5
        robot_marker.scale.z = 0.5
        robot_marker.color.a = 1.0
        robot_marker.color.r = 1.0
        marker_array.markers.append(robot_marker)
        
        # Beacon markers
        for i, beacon in enumerate(self.world_map.beacons):
            beacon_marker = Marker()
            beacon_marker.type = Marker.SPHERE
            beacon_marker.pose.position.x = beacon['x']
            beacon_marker.pose.position.y = beacon['y']
            beacon_marker.scale.x = 0.2
            beacon_marker.scale.y = 0.2
            beacon_marker.scale.z = 0.2
            beacon_marker.color.a = 1.0
            beacon_marker.color.g = 1.0
            marker_array.markers.append(beacon_marker)
        
        self.marker_pub.publish(marker_array)
        
        # Publish map
        self.publish_map()

    def publish_map(self):
        """Publish an OccupancyGrid representation of the world"""
        map_msg = OccupancyGrid()
        map_msg.header.frame_id = 'map'
        map_msg.info.resolution = self.world_map.resolution
        map_msg.info.width = self.world_map.width
        map_msg.info.height = self.world_map.height
        
        # Convert grid to occupancy grid
        map_msg.data = [int(cell * 100) for row in self.world_map.grid for cell in row]
        
        self.map_pub.publish(map_msg)

def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
    
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()