# # #!/usr/bin/env python3

# # import rclpy
# # from rclpy.node import Node
# # from geometry_msgs.msg import Vector3
# # import sys
# # import termios
# # import tty
# # import select
# # import threading
# # import numpy as np
# # import time
# # import os

# # msg = """
# # Robot Control Commands
# # ---------------------------
# # Movement directions:
# #         w    
# #    a    s    d
# #         x    

# # w/s : forward/backward
# # a/d : left/right
# # x : stop
# # q : quit

# # CTRL+C: Force quit
# # """

# # class KeyboardTeleop(Node):
# #     def __init__(self):
# #         super().__init__('keyboard_teleop')
        
# #         self.publisher = self.create_publisher(Vector3, '/control_signal', 10)
        
# #         self.key_mapping = {
# #             'w': np.array([0.0, 1.0, 0.0]),   # Forward (positive Y)
# #             's': np.array([0.0, -1.0, 0.0]),  # Backward (negative Y)
# #             'a': np.array([-1.0, 0.0, 0.0]),  # Left (negative X)
# #             'd': np.array([1.0, 0.0, 0.0]),   # Right (positive X)
# #             'x': np.array([0.0, 0.0, 0.0]),   # Stop
# #         }
        
# #         # Parameter settings
# #         self.declare_parameter('max_speed', 1.0)
# #         self.max_speed = self.get_parameter('max_speed').value
        
# #         self.declare_parameter('publish_rate', 20.0)  # Higher rate for better responsiveness
# #         self.publish_rate = self.get_parameter('publish_rate').value
        
# #         # Current state
# #         self.current_velocity = np.array([0.0, 0.0, 0.0])
# #         self.target_velocity = np.array([0.0, 0.0, 0.0])
# #         self.last_command_time = time.time()
# #         self.publishing = True
# #         self.last_key = 'x'
# #         self.status_msg = ""
        
# #         # Screen setup
# #         os.system('clear')
# #         print(msg)
        
# #         # Thread for publishing control commands
# #         self.publish_thread = threading.Thread(target=self.publish_loop)
# #         self.publish_thread.daemon = True
# #         self.publish_thread.start()
        
# #         # Thread for displaying status
# #         self.status_thread = threading.Thread(target=self.status_loop)
# #         self.status_thread.daemon = True
# #         self.status_thread.start()

# #     def get_key(self):
# #         """Get keyboard input in non-blocking mode"""
# #         settings = termios.tcgetattr(sys.stdin)
# #         tty.setraw(sys.stdin.fileno())
# #         rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
# #         if rlist:
# #             key = sys.stdin.read(1)
# #         else:
# #             key = ''
# #         termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
# #         return key

# #     def update_velocity(self):
# #         """Update velocity with first-order control"""
# #         # Direct first-order control - set current velocity to target velocity
# #         self.current_velocity = self.target_velocity.copy()

# #     def publish_control(self):
# #         """Publish control commands"""
# #         msg = Vector3()
# #         msg.x = float(self.current_velocity[0])
# #         msg.y = float(self.current_velocity[1])
# #         msg.z = float(self.current_velocity[2])
# #         self.publisher.publish(msg)

# #     def publish_loop(self):
# #         """Publish control commands at a fixed rate"""
# #         r = 1.0 / self.publish_rate
        
# #         while self.publishing:
# #             now = time.time()
            
# #             # Set current velocity to target velocity (first-order control)
# #             self.update_velocity()
            
# #             # Publish command
# #             self.publish_control()
            
# #             # Maintain fixed rate
# #             time.sleep(max(0.0, r - (time.time() - now)))
    
# #     def status_loop(self):
# #         """Display status information on screen"""
# #         while self.publishing:
# #             # Move cursor and display status
# #             os.system('clear')
# #             print(msg)
# #             print(f"Current Command: {self.last_key}")
# #             print(f"Velocity: X={self.current_velocity[0]:.2f}, Y={self.current_velocity[1]:.2f}")
# #             print(self.status_msg)
# #             time.sleep(0.2)

# # def main(args=None):
# #     rclpy.init(args=args)
# #     teleop = KeyboardTeleop()
    
# #     try:
# #         while True:
# #             key = teleop.get_key()
# #             if key in teleop.key_mapping.keys():
# #                 target_vel = teleop.key_mapping[key] * teleop.max_speed
# #                 teleop.target_velocity = target_vel
# #                 teleop.last_key = key
# #             elif key == 'q':
# #                 # Stop the robot before quitting
# #                 teleop.target_velocity = np.array([0.0, 0.0, 0.0])
# #                 teleop.current_velocity = np.array([0.0, 0.0, 0.0])
# #                 teleop.publish_control()
# #                 teleop.status_msg = "Shutting down program..."
# #                 time.sleep(0.5)
# #                 break
# #             elif key == '\x03':  # CTRL+C
# #                 teleop.status_msg = "Force quitting..."
# #                 time.sleep(0.5)
# #                 break
# #     except Exception as e:
# #         teleop.status_msg = f"Error occurred: {str(e)}"
# #         time.sleep(1.0)
# #     finally:
# #         # Cleanup
# #         teleop.publishing = False
# #         teleop.destroy_node()
# #         rclpy.shutdown()
# #         # Restore terminal settings
# #         os.system('reset')

# # if __name__ == '__main__':
# #     main() 

# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Vector3
# import sys
# import termios
# import tty
# import select
# import threading
# import numpy as np
# import time
# import os
# # Add this import at the top
# from std_msgs.msg import Bool

# msg = """
# Robot Control Commands
# ---------------------------
# Movement directions:
#         w    
#    a    s    d
#         x    

# w/s : forward/backward
# a/d : left/right
# x : stop
# q : quit

# CTRL+C: Force quit

# p : toggle planner (switch between manual and autonomous mode)
# """

# class KeyboardTeleop(Node):
#     def __init__(self):
#         super().__init__('keyboard_teleop')
        
#         # Changed to publish to teleop_control instead of directly to control_signal
#         self.publisher = self.create_publisher(Vector3, '/teleop_control', 10)
        
#         # Added publisher for planning status
#         self.planning_status_pub = self.create_publisher(Vector3, '/planning_status', 10)
        
#         self.key_mapping = {
#             'w': np.array([0.0, 1.0, 0.0]),   # Forward (positive Y)
#             's': np.array([0.0, -1.0, 0.0]),  # Backward (negative Y)
#             'a': np.array([-1.0, 0.0, 0.0]),  # Left (negative X)
#             'd': np.array([1.0, 0.0, 0.0]),   # Right (positive X)
#             'x': np.array([0.0, 0.0, 0.0]),   # Stop
#         }
        
#         # Parameter settings
#         self.declare_parameter('max_speed', 1.0)
#         self.max_speed = self.get_parameter('max_speed').value
        
#         self.declare_parameter('publish_rate', 20.0)  # Higher rate for better responsiveness
#         self.publish_rate = self.get_parameter('publish_rate').value
        
#         # Current state
#         self.current_velocity = np.array([0.0, 0.0, 0.0])
#         self.target_velocity = np.array([0.0, 0.0, 0.0])
#         self.last_command_time = time.time()
#         self.publishing = True
#         self.last_key = 'x'
#         self.status_msg = ""
#         self.planner_active = True  # Default to planner active
        
#         # Screen setup
#         os.system('clear')
#         print(msg)
        
#         # Thread for publishing control commands
#         self.publish_thread = threading.Thread(target=self.publish_loop)
#         self.publish_thread.daemon = True
#         self.publish_thread.start()
        
#         # Thread for displaying status
#         self.status_thread = threading.Thread(target=self.status_loop)
#         self.status_thread.daemon = True
#         self.status_thread.start()

#     def get_key(self):
#         """Get keyboard input in non-blocking mode"""
#         settings = termios.tcgetattr(sys.stdin)
#         tty.setraw(sys.stdin.fileno())
#         rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
#         if rlist:
#             key = sys.stdin.read(1)
#         else:
#             key = ''
#         termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
#         return key

#     def update_velocity(self):
#         """Update velocity with first-order control"""
#         # Direct first-order control - set current velocity to target velocity
#         self.current_velocity = self.target_velocity.copy()

#     def publish_control(self):
#         """Publish control commands"""
#         msg = Vector3()
#         msg.x = float(self.current_velocity[0])
#         msg.y = float(self.current_velocity[1])
#         msg.z = float(self.current_velocity[2])
#         self.publisher.publish(msg)

#     def publish_planner_status(self, active):
#         """Publish planning status"""
#         msg = Bool()
#         msg.data = active
#         self.planning_status_pub.publish(msg)

#     def publish_loop(self):
#         """Publish control commands at a fixed rate"""
#         r = 1.0 / self.publish_rate
        
#         while self.publishing:
#             now = time.time()
            
#             # Set current velocity to target velocity (first-order control)
#             self.update_velocity()
            
#             # Publish command
#             self.publish_control()
            
#             # Maintain fixed rate
#             time.sleep(max(0.0, r - (time.time() - now)))
    
#     def status_loop(self):
#         """Display status information on screen"""
#         while self.publishing:
#             # Move cursor and display status
#             os.system('clear')
#             print(msg)
#             print(f"Current Command: {self.last_key}")
#             print(f"Velocity: X={self.current_velocity[0]:.2f}, Y={self.current_velocity[1]:.2f}")
#             print(f"Mode: {'Autonomous planning' if self.planner_active else 'Manual control'}")
#             print(self.status_msg)
#             time.sleep(0.2)

# def main(args=None):
#     rclpy.init(args=args)
#     teleop = KeyboardTeleop()
    
#     try:
#         while True:
#             key = teleop.get_key()
#             if key in teleop.key_mapping.keys():
#                 target_vel = teleop.key_mapping[key] * teleop.max_speed
#                 teleop.target_velocity = target_vel
#                 teleop.last_key = key
#                 teleop.status_msg = "Manual control active" if not teleop.planner_active else "Command sent"
#             elif key == 'p':
#                 # Toggle planner active state
#                 teleop.planner_active = not teleop.planner_active
#                 teleop.publish_planner_status(teleop.planner_active)
#                 teleop.status_msg = f"{'Enabled' if teleop.planner_active else 'Disabled'} autonomous planning"
#             elif key == 'q':
#                 # Stop the robot before quitting
#                 teleop.target_velocity = np.array([0.0, 0.0, 0.0])
#                 teleop.current_velocity = np.array([0.0, 0.0, 0.0])
#                 teleop.publish_control()
#                 teleop.status_msg = "Shutting down program..."
#                 time.sleep(0.5)
#                 break
#             elif key == '\x03':  # CTRL+C
#                 teleop.status_msg = "Force quitting..."
#                 time.sleep(0.5)
#                 break
#     except Exception as e:
#         teleop.status_msg = f"Error occurred: {str(e)}"
#         time.sleep(1.0)
#     finally:
#         # Cleanup
#         teleop.publishing = False
#         teleop.destroy_node()
#         rclpy.shutdown()
#         # Restore terminal settings
#         os.system('reset')

# if __name__ == '__main__':
#     main()

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool
import sys
import termios
import tty
import select
import threading
import numpy as np
import time
import os

msg = """
Robot Control Commands
---------------------------
Movement directions:
        w    
   a    s    d
        x    

w/s : forward/backward
a/d : left/right
x : stop
q : quit

CTRL+C: Force quit

p : toggle planner (switch between manual and autonomous mode)
m : momentary manual control (press with direction keys)
"""

class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__('keyboard_teleop')
        
        # Publisher for control commands
        self.publisher = self.create_publisher(Vector3, '/teleop_control', 10)
        
        # Publisher for planner status
        self.planning_status_pub = self.create_publisher(Bool, '/use_planner', 10)
        
        self.key_mapping = {
            'w': np.array([0.0, 1.0, 0.0]),   # Forward (positive Y)
            's': np.array([0.0, -1.0, 0.0]),  # Backward (negative Y)
            'a': np.array([-1.0, 0.0, 0.0]),  # Left (negative X)
            'd': np.array([1.0, 0.0, 0.0]),   # Right (positive X)
            'x': np.array([0.0, 0.0, 0.0]),   # Stop
        }
        
        # Parameter settings
        self.declare_parameter('max_speed', 1.0)
        self.max_speed = self.get_parameter('max_speed').value
        
        self.declare_parameter('publish_rate', 20.0)  # Higher rate for better responsiveness
        self.publish_rate = self.get_parameter('publish_rate').value
        
        # Current state
        self.current_velocity = np.array([0.0, 0.0, 0.0])
        self.target_velocity = np.array([0.0, 0.0, 0.0])
        self.last_command_time = time.time()
        self.publishing = True
        self.last_key = 'x'
        self.status_msg = ""
        self.planner_active = True  # Default to planner active
        self.momentary_override = False  # For temporary manual control
        
        # Publish initial planner status
        self.publish_planner_status(self.planner_active)
        
        # Screen setup
        os.system('clear')
        print(msg)
        
        # Thread for publishing control commands
        self.publish_thread = threading.Thread(target=self.publish_loop)
        self.publish_thread.daemon = True
        self.publish_thread.start()
        
        # Thread for displaying status
        self.status_thread = threading.Thread(target=self.status_loop)
        self.status_thread.daemon = True
        self.status_thread.start()

    def get_key(self):
        """Get keyboard input in non-blocking mode"""
        settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key

    def update_velocity(self):
        """Update velocity with first-order control"""
        # Direct first-order control - set current velocity to target velocity
        self.current_velocity = self.target_velocity.copy()

    def publish_control(self):
        """Publish control commands"""
        msg = Vector3()
        msg.x = float(self.current_velocity[0])
        msg.y = float(self.current_velocity[1])
        msg.z = float(self.current_velocity[2])
        self.publisher.publish(msg)

    def publish_planner_status(self, active):
        """Publish planning status"""
        msg = Bool()
        msg.data = active
        self.planning_status_pub.publish(msg)

    def publish_loop(self):
        """Publish control commands at a fixed rate"""
        r = 1.0 / self.publish_rate
        
        while self.publishing:
            now = time.time()
            
            # Set current velocity to target velocity (first-order control)
            self.update_velocity()
            
            # Publish command
            self.publish_control()
            
            # Maintain fixed rate
            time.sleep(max(0.0, r - (time.time() - now)))
    
    def status_loop(self):
        """Display status information on screen"""
        while self.publishing:
            # Move cursor and display status
            os.system('clear')
            print(msg)
            print(f"Current Command: {self.last_key}")
            print(f"Velocity: X={self.current_velocity[0]:.2f}, Y={self.current_velocity[1]:.2f}")
            
            # Show the current mode with more detail
            if self.momentary_override:
                mode_str = "MOMENTARY MANUAL OVERRIDE"
            elif self.planner_active:
                mode_str = "Autonomous planning"
            else:
                mode_str = "Manual control"
            
            print(f"Mode: {mode_str}")
            print(self.status_msg)
            time.sleep(0.2)

def main(args=None):
    rclpy.init(args=args)
    teleop = KeyboardTeleop()
    
    # Variables for momentary control
    momentary_active = False
    
    try:
        while True:
            keys = []
            # Collect all available keys (for multi-key combinations)
            key = teleop.get_key()
            while key:
                keys.append(key)
                key = teleop.get_key()
            
            # Process collected keys
            if not keys:
                # No keys pressed, continue
                continue
                
            # Check for momentary override key
            if 'm' in keys:
                momentary_active = True
                teleop.momentary_override = True
                teleop.status_msg = "MOMENTARY OVERRIDE ACTIVE - Use direction keys"
            else:
                # Check if we should end momentary mode
                if momentary_active:
                    momentary_active = False
                    teleop.momentary_override = False
                    # Send stop command when releasing momentary override
                    teleop.target_velocity = np.array([0.0, 0.0, 0.0])
                    teleop.status_msg = "Returned to " + ("autonomous" if teleop.planner_active else "manual") + " mode"
            
            # Process direction keys
            for key in keys:
                if key in teleop.key_mapping.keys():
                    target_vel = teleop.key_mapping[key] * teleop.max_speed
                    teleop.target_velocity = target_vel
                    teleop.last_key = key
                    if momentary_active:
                        teleop.status_msg = "MOMENTARY OVERRIDE - Manual control active"
                    elif not teleop.planner_active:
                        teleop.status_msg = "Manual control active"
                    else:
                        teleop.status_msg = "Command sent (planner still active)"
                elif key == 'p' and not momentary_active:
                    # Toggle planner active state (only if not in momentary mode)
                    teleop.planner_active = not teleop.planner_active
                    teleop.publish_planner_status(teleop.planner_active)
                    teleop.status_msg = f"{'Enabled' if teleop.planner_active else 'Disabled'} autonomous planning"
                elif key == 'q':
                    # Stop the robot before quitting
                    teleop.target_velocity = np.array([0.0, 0.0, 0.0])
                    teleop.current_velocity = np.array([0.0, 0.0, 0.0])
                    teleop.publish_control()
                    teleop.status_msg = "Shutting down program..."
                    time.sleep(0.5)
                    teleop.publishing = False
                    break
                elif key == '\x03':  # CTRL+C
                    teleop.status_msg = "Force quitting..."
                    time.sleep(0.5)
                    teleop.publishing = False
                    break
            
            # Break outer loop if inner loop signaled exit
            if not teleop.publishing:
                break
                
    except Exception as e:
        teleop.status_msg = f"Error occurred: {str(e)}"
        time.sleep(1.0)
    finally:
        # Cleanup
        teleop.publishing = False
        teleop.destroy_node()
        rclpy.shutdown()
        # Restore terminal settings
        os.system('reset')

if __name__ == '__main__':
    main()