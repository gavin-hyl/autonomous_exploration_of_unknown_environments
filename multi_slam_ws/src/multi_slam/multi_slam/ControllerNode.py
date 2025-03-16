import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3, PoseStamped
import numpy as np


class ControllerNode(Node):
    """
    Robot controller node that makes the robot move constantly upward.
    
    Subscribes to the estimated pose from SLAM and publishes control commands.
    """
    
    def __init__(self):
        """Initialize the controller node with publishers, subscribers, and parameters."""
        super().__init__("controller_node")
        
        # Publishers
        self.control_pub = self.create_publisher(Vector3, "control_signal", 10)
        
        # Subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped, "/estimated_pose", self.pose_callback, 10
        )
        
        # Parameters
        self.declare_parameter("max_speed", 1.0)
        self.max_speed = self.get_parameter("max_speed").value
        
        self.declare_parameter("control_frequency", 10.0)  # Hz
        control_freq = self.get_parameter("control_frequency").value
        self.control_timer = self.create_timer(1.0 / control_freq, self.control_loop)
        
        # State
        self.current_pose = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
        
        # Constant up movement control signal with higher speed
        self.control_input = np.array([0.0, 0.8 * self.max_speed, 0.0])  # Move upward (positive y) at 80% max speed
        
        # Counters for logging
        self.count_pose_callbacks = 0
        self.publish_count = 0
        
        self.get_logger().info("Controller node initialized with constant upward movement")
        self.get_logger().info(f"Using control input: [{self.control_input[0]}, {self.control_input[1]}, {self.control_input[2]}]")
        self.get_logger().info(f"Maximum speed: {self.max_speed}")
        self.get_logger().info(f"Control frequency: {control_freq} Hz")

    def pose_callback(self, msg: PoseStamped):
        """Store the latest estimated pose from SLAM Node."""
        self.current_pose = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            0.0
        ])

    def control_loop(self):
        """Output constant upward control command."""
        self.get_logger().debug("Publishing control command")
        self.publish_control()

    def publish_control(self):
        """Publish velocity command to move upward."""
        msg = Vector3()
        msg.x = float(self.control_input[0])
        msg.y = float(self.control_input[1])
        msg.z = 0.0
        self.control_pub.publish(msg)


def main(args=None):
    """Entry point for the controller node."""
    rclpy.init(args=args)
    node = ControllerNode()
    node.get_logger().info("Controller node started and ready to control robot")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()