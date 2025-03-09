import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import Twist, Pose
from multi_slam import WorldMap

class SimulationNode(Node):
    def __init__(self):
        super().__init__('simulation_node')
        
        # Use the global world map
        self.world_map = WorldMap
        
        # Subscribers and publishers
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/crazyflie/cmd_vel', self.update_robot_state, 10)
        self.pose_pub = self.create_publisher(Pose, '/robot_pose', 10)
        
        # Robot state
        self.current_pose = Pose()
        self.velocity = Twist()
        
        # Simulation timer
        self.create_timer(0.1, self.simulate_step)
        
        self.get_logger().info('Simulation Node initialized')

    def update_robot_state(self, msg):
        """Update robot velocity from control input"""
        self.velocity = msg

    def simulate_step(self):
        """Simulate robot movement based on velocity"""
        # Simple kinematic model
        self.current_pose.position.x += self.velocity.linear.x * 0.1
        self.current_pose.position.y += self.velocity.linear.y * 0.1
        
        # Basic collision detection with map boundaries
        x = self.current_pose.position.x
        y = self.current_pose.position.y
        
        if not self.world_map.is_valid_position(int(x), int(y)):
            self.get_logger().warn(f'Collision detected at {x}, {y}')
            # Simple bounce back
            self.current_pose.position.x -= self.velocity.linear.x * 0.1
            self.current_pose.position.y -= self.velocity.linear.y * 0.1
        
        # Publish updated pose
        self.pose_pub.publish(self.current_pose)

def main(args=None):
    rclpy.init(args=args)
    node = SimulationNode()
    
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()