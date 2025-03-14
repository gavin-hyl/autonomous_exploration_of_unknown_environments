import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3, PoseStamped
import numpy as np
import threading


class ControllerNode(Node):
    """
    Robot controller node that provides teleoperation and autonomous navigation capabilities.
    
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
        
        self.declare_parameter("teleop_enabled", True)
        self.teleop_enabled = self.get_parameter("teleop_enabled").value
        
        # State
        self.current_pose = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
        self.target = np.array([8.0, 8.0, 0.0])  # Example target
        
        # Teleop setup
        if self.teleop_enabled:
            self.key_mapping = {
                'w': np.array([1.0, 0.0, 0.0]),   # Forward
                's': np.array([-1.0, 0.0, 0.0]),  # Backward
                'a': np.array([0.0, 1.0, 0.0]),   # Left
                'd': np.array([0.0, -1.0, 0.0]),  # Right
                'x': np.array([0.0, 0.0, 0.0]),   # Stop
            }
            self.control_input = np.array([0.0, 0.0, 0.0])
            self.teleop_thread = threading.Thread(target=self.teleop_input_loop)
            self.teleop_thread.daemon = True
            self.teleop_thread.start()

    def pose_callback(self, msg: PoseStamped):
        """Store the latest estimated pose from SLAM Node."""
        self.current_pose = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            0.0  # Ignore orientation for simplicity
        ])

    def control_loop(self):
        """Calculate control commands (autonomous or teleop)."""
        if not self.teleop_enabled:
            # Autonomous navigation to target (simple proportional control)
            error = self.target - self.current_pose
            control = 0.5 * error[:2]  # Proportional gain
            control = np.clip(control, -self.max_speed, self.max_speed)
            self.control_input = np.array([control[0], control[1], 0.0])
        
        self.publish_control()

    def teleop_input_loop(self):
        """Read keyboard input for teleoperation."""
        while True:
            key = input("Enter WASD command (x to stop): ").lower()
            if key in self.key_mapping:
                self.control_input = self.key_mapping[key]
            else:
                self.get_logger().warn(f"Invalid key: {key}")

    def publish_control(self):
        """Publish velocity command."""
        msg = Vector3()
        msg.x = float(self.control_input[0])
        msg.y = float(self.control_input[1])
        msg.z = 0.0
        self.control_pub.publish(msg)


def main(args=None):
    """Entry point for the controller node."""
    rclpy.init(args=args)
    node = ControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()