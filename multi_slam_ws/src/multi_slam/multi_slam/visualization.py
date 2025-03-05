import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)
        # Prevent unused variable warning.
        self.subscription

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: {msg.data}')