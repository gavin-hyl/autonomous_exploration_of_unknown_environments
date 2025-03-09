import rclpy
from rclpy.node import Node
import math

from geometry_msgs.msg import Twist, Point
from multi_slam import global_world_map

class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller_node')
        
        # Use the global world map
        self.world_map = global_world_map
        
        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/crazyflie/cmd_vel', 10)
        self.beacon_sub = self.create_subscription(
            Point, '/beacon_detection', self.beacon_callback, 10)
        
        # Control timer
        self.create_timer(0.1, self.control_loop)
        
        # Robot state
        self.current_pose = None
        self.target_beacon = None
        
        self.get_logger().info('Controller Node initialized')

    def beacon_callback(self, msg):
        """Handle beacon detection"""
        self.target_beacon = msg
        self.get_logger().info(f'Beacon detected at: {msg.x}, {msg.y}')

    def control_loop(self):
        """Simple control logic"""
        cmd_vel = Twist()
        
        if self.target_beacon and self.current_pose:
            # Simple proportional control towards beacon
            dx = self.target_beacon.x - self.current_pose.position.x
            dy = self.target_beacon.y - self.current_pose.position.y
            
            # Calculate distance and heading
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance > 0.5:  # Threshold for reaching beacon
                cmd_vel.linear.x = min(max(dx * 0.5, -1), 1)
                cmd_vel.linear.y = min(max(dy * 0.5, -1), 1)
        
        self.cmd_vel_pub.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()