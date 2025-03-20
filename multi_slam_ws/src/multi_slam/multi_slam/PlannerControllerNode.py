import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3, PoseStamped
from std_msgs.msg import Bool
import numpy as np


class PlannerControllerNode(Node):
    """
    Enhanced robot controller node that uses planned trajectory.
    
    Subscribes to the estimated pose from SLAM and the planned control commands.
    Can fallback to manual control when planning is not active.
    """
    
    def __init__(self):
        """Initialize the planner controller node with publishers, subscribers, and parameters."""
        super().__init__("planner_controller_node")
        
        # Publishers
        self.control_pub = self.create_publisher(Vector3, "control_signal", 10)
        
        # Subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped, "/estimated_pose", self.pose_callback, 10
        )
        self.planned_control_sub = self.create_subscription(
            Vector3, "/planned_control", self.planned_control_callback, 10
        )
        self.planning_status_sub = self.create_subscription(
            Bool, "/planning_status", self.planning_status_callback, 10
        )
        
        # Parameters
        self.declare_parameter("max_speed", 1.0)
        self.max_speed = self.get_parameter("max_speed").value
        
        self.declare_parameter("control_frequency", 10.0)  # Hz
        control_freq = self.get_parameter("control_frequency").value
        self.control_timer = self.create_timer(1.0 / control_freq, self.control_loop)
        
        self.declare_parameter("use_manual_fallback", True)
        self.use_manual_fallback = self.get_parameter("use_manual_fallback").value

        self.declare_parameter("fallback_control_x", 0.0)
        self.declare_parameter("fallback_control_y", 0.5)
        self.fallback_control = np.array([
            self.get_parameter("fallback_control_x").value,
            self.get_parameter("fallback_control_y").value,
            0.0
        ])
        
        # State
        self.current_pose = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
        self.planned_control_input = None
        self.last_planned_control_time = 0.0
        self.is_planning_active = False
        self.control_timeout = 1.0  # seconds to consider control stale
        
        # Counters for logging
        self.count_pose_callbacks = 0
        self.publish_count = 0
        
        self.get_logger().info("Planner controller node initialized")
        self.get_logger().info(f"Maximum speed: {self.max_speed}")
        self.get_logger().info(f"Control frequency: {control_freq} Hz")
        self.get_logger().info(f"Manual fallback: {'enabled' if self.use_manual_fallback else 'disabled'}")
        if self.use_manual_fallback:
            self.get_logger().info(f"Fallback control: [{self.fallback_control[0]}, {self.fallback_control[1]}]")

    def pose_callback(self, msg: PoseStamped):
        """Store the latest estimated pose from SLAM Node."""
        self.current_pose = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            0.0
        ])

    def planned_control_callback(self, msg: Vector3):
        """Store the latest planned control input from the planner."""
        self.planned_control_input = np.array([msg.x, msg.y, 0.0])
        self.last_planned_control_time = self.get_clock().now().seconds_nanoseconds()[0]
        self.get_logger().debug(f"Received planned control: [{msg.x:.2f}, {msg.y:.2f}]")

    def planning_status_callback(self, msg: Bool):
        """Update planning status from the planner."""
        self.is_planning_active = msg.data
        if self.is_planning_active:
            self.get_logger().debug("Planning is now active")
        else:
            self.get_logger().debug("Planning is now inactive")

    def is_control_input_stale(self):
        """Check if the last received control input is too old to use."""
        if self.planned_control_input is None:
            return True
            
        current_time = self.get_clock().now().seconds_nanoseconds()[0]
        time_diff = current_time - self.last_planned_control_time
        
        return time_diff > self.control_timeout

    def control_loop(self):
        """
        Generate and publish control command based on planning status.
        
        If planning is active and control input is fresh, use planned control.
        Otherwise, use fallback control if enabled.
        """
        # Check if we should use planned control
        if (self.is_planning_active and 
            self.planned_control_input is not None and 
            not self.is_control_input_stale()):
            
            # Use planned control
            control_input = self.planned_control_input
            self.get_logger().debug("Using planned control input")
            
        elif self.use_manual_fallback:
            # Use fallback control
            control_input = self.fallback_control
            self.get_logger().debug("Using fallback control input")
            
        else:
            # Stop if no valid control and no fallback
            control_input = np.zeros(3)
            self.get_logger().debug("No valid control input, stopping")
        
        # Scale control to respect max speed
        speed = np.linalg.norm(control_input[:2])
        if speed > self.max_speed:
            control_input[:2] = control_input[:2] * (self.max_speed / speed)
        
        # Publish control
        self.publish_control(control_input)

    def publish_control(self, control_input):
        """Publish velocity command."""
        msg = Vector3()
        msg.x = float(control_input[0])
        msg.y = float(control_input[1])
        msg.z = 0.0
        self.control_pub.publish(msg)
        self.publish_count += 1
        
        # Log every 50 publishes to avoid flooding
        if self.publish_count % 50 == 0:
            self.get_logger().info(f"Published control: [{control_input[0]:.2f}, {control_input[1]:.2f}]")


def main(args=None):
    """Entry point for the planner controller node."""
    rclpy.init(args=args)
    node = PlannerControllerNode()
    node.get_logger().info("Planner controller node started and ready to control robot")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main() 