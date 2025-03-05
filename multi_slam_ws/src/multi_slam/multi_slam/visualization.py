import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray

class Visualizer(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.control_input_sub = self.create_subscription(Float32MultiArray, '/control_input', self.control_intput_cb, 10) 

    def control_intput_cb(self, msg: Float32MultiArray):
        self.get_logger().info(f"Received control input: {msg.data}")