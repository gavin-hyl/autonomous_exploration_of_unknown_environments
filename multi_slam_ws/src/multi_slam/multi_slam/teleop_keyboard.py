#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
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
"""

class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__('keyboard_teleop')
        
        self.publisher = self.create_publisher(Vector3, 'control_signal', 10)
        
        self.key_mapping = {
            'w': np.array([0.0, 1.0, 0.0]),   # Forward (positive Y)
            's': np.array([0.0, -1.0, 0.0]),  # Backward (negative Y)
            'a': np.array([-1.0, 0.0, 0.0]),  # Left (negative X)
            'd': np.array([1.0, 0.0, 0.0]),   # Right (positive X)
            'x': np.array([0.0, 0.0, 0.0]),   # Stop
        }

        # original key mapping 
        # 'w': np.array([1.0, 0.0, 0.0]),   # Forward
        # 's': np.array([-1.0, 0.0, 0.0]),  # Backward
        # 'a': np.array([0.0, 1.0, 0.0]),   # Left
        # 'd': np.array([0.0, -1.0, 0.0]),  # Right

        
        # Parameter settings
        self.declare_parameter('max_speed', 1.0)
        self.max_speed = self.get_parameter('max_speed').value
        
        self.declare_parameter('acceleration', 2.0)
        self.acceleration = self.get_parameter('acceleration').value
        
        self.declare_parameter('deceleration', 4.0)
        self.deceleration = self.get_parameter('deceleration').value
        
        self.declare_parameter('publish_rate', 20.0)  # Higher rate for better responsiveness
        self.publish_rate = self.get_parameter('publish_rate').value
        
        # Current state
        self.current_velocity = np.array([0.0, 0.0, 0.0])
        self.target_velocity = np.array([0.0, 0.0, 0.0])
        self.last_command_time = time.time()
        self.publishing = True
        self.last_key = 'x'
        self.status_msg = ""
        
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

    def update_velocity(self, dt):
        """Update velocity with smooth acceleration/deceleration"""
        if np.array_equal(self.target_velocity, self.current_velocity):
            return
            
        # Process each axis
        for i in range(3):
            if self.current_velocity[i] < self.target_velocity[i]:
                # Acceleration
                self.current_velocity[i] = min(
                    self.target_velocity[i],
                    self.current_velocity[i] + self.acceleration * dt
                )
            elif self.current_velocity[i] > self.target_velocity[i]:
                # Deceleration
                self.current_velocity[i] = max(
                    self.target_velocity[i],
                    self.current_velocity[i] - self.deceleration * dt
                )
        
        # Set very small values to zero
        self.current_velocity[np.abs(self.current_velocity) < 0.05] = 0.0

    def publish_control(self):
        """Publish control commands"""
        msg = Vector3()
        msg.x = float(self.current_velocity[0])
        msg.y = float(self.current_velocity[1])
        msg.z = float(self.current_velocity[2])
        self.publisher.publish(msg)

    def publish_loop(self):
        """Publish control commands at a fixed rate"""
        r = 1.0 / self.publish_rate
        last_time = time.time()
        
        while self.publishing:
            now = time.time()
            dt = now - last_time
            last_time = now
            
            # Process smooth acceleration/deceleration
            self.update_velocity(dt)
            
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
            print(self.status_msg)
            time.sleep(0.2)

def main(args=None):
    rclpy.init(args=args)
    teleop = KeyboardTeleop()
    
    try:
        while True:
            key = teleop.get_key()
            if key in teleop.key_mapping.keys():
                target_vel = teleop.key_mapping[key] * teleop.max_speed
                teleop.target_velocity = target_vel
                teleop.last_key = key
            elif key == 'q':
                # Stop the robot before quitting
                teleop.target_velocity = np.array([0.0, 0.0, 0.0])
                teleop.current_velocity = np.array([0.0, 0.0, 0.0])
                teleop.publish_control()
                teleop.status_msg = "Shutting down program..."
                time.sleep(0.5)
                break
            elif key == '\x03':  # CTRL+C
                teleop.status_msg = "Force quitting..."
                time.sleep(0.5)
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