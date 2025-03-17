'''
Testing the ROS2 bag reading and plotting the data
'''

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message_type
from geometry_msgs.msg import Vector3, Marker
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Float32MultiArray
import rosbag2_py
from rclpy.time import Time

class PlotData:
    def __init__(self, bag_path):
        """Initialize the class with path to the rosbag"""
        self.bag_path = bag_path
        
        # Data storage
        self.timestamps = {}
        self.true_positions = []
        self.estimated_positions = []
        self.particles = []
        self.scores = []
        self.control_inputs = []
        
        # Topics to extract data from
        self.true_pos_topic = "visualization_marker_true"
        self.estimated_pos_topic = "/pos_hat"
        self.control_topic = "/control_signal"
        
        # Add particles topic
        # This is a custom topic we need to add to SlamNode.py to publish particle data
        self.particles_topic = "/particles"
        
        # Add scores topic
        # This is a custom topic we need to add to SlamNode.py to publish score data
        self.scores_topic = "/scores"
        
        # Initialize the data storage for each topic
        self.timestamps[self.true_pos_topic] = []
        self.timestamps[self.estimated_pos_topic] = []
        self.timestamps[self.control_topic] = []
        self.timestamps[self.particles_topic] = []
        self.timestamps[self.scores_topic] = []
        
    def read_bag(self):
        """Read the ROS2 bag file and extract messages from relevant topics"""
        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=self.bag_path, storage_id='sqlite3'),
            rosbag2_py.ConverterOptions(
                input_serialization_format='cdr',
                output_serialization_format='cdr')
        )
        
        topic_types = reader.get_all_topics_and_types()
        type_map = {topic_info.name: topic_info.type for topic_info in topic_types}
        
        print(f"Available topics in the bag: {list(type_map.keys())}")
        
        # Read all messages from the bag
        while reader.has_next():
            topic_name, data, timestamp = reader.read_next()
            
            if topic_name == self.true_pos_topic:
                # Extract true position from Marker message
                msg_type = get_message_type(type_map[topic_name])
                msg = deserialize_message(data, msg_type)
                
                self.timestamps[topic_name].append(timestamp)
                self.true_positions.append([
                    msg.pose.position.x,
                    msg.pose.position.y,
                    msg.pose.position.z
                ])
                
            elif topic_name == self.estimated_pos_topic:
                # Extract estimated position from Vector3 message
                msg_type = get_message_type(type_map[topic_name])
                msg = deserialize_message(data, msg_type)
                
                self.timestamps[topic_name].append(timestamp)
                self.estimated_positions.append([msg.x, msg.y, msg.z])
                
            elif topic_name == self.control_topic:
                # Extract control input from Vector3 message
                msg_type = get_message_type(type_map[topic_name])
                msg = deserialize_message(data, msg_type)
                
                self.timestamps[topic_name].append(timestamp)
                self.control_inputs.append([msg.x, msg.y, msg.z])
                
            elif topic_name == self.particles_topic:
                # Extract particle data from Float32MultiArray message
                msg_type = get_message_type(type_map[topic_name])
                msg = deserialize_message(data, msg_type)
                
                # Reshape data from flat array to N x 3 (x, y, z for each particle)
                num_particles = len(msg.data) // 3
                particle_data = np.array(msg.data).reshape(num_particles, 3)
                
                self.timestamps[topic_name].append(timestamp)
                self.particles.append(particle_data)
                
            elif topic_name == self.scores_topic:
                # Extract score data
                msg_type = get_message_type(type_map[topic_name])
                msg = deserialize_message(data, msg_type)
                
                self.timestamps[topic_name].append(timestamp)
                self.scores.append(msg.data)
        
        # Convert lists to numpy arrays for easier handling
        self.true_positions = np.array(self.true_positions)
        self.estimated_positions = np.array(self.estimated_positions)
        self.control_inputs = np.array(self.control_inputs)
        
        print(f"Extracted {len(self.true_positions)} true position messages")
        print(f"Extracted {len(self.estimated_positions)} estimated position messages")
        print(f"Extracted {len(self.control_inputs)} control input messages")
        print(f"Extracted {len(self.particles)} particle data messages")
        print(f"Extracted {len(self.scores)} score data messages")
    
    def plot_positions(self):
        """Plot true position and estimated position on the same 2D graph"""
        if len(self.true_positions) == 0 or len(self.estimated_positions) == 0:
            print("No position data to plot")
            return
            
        plt.figure(figsize=(10, 8))
        
        # Plot true positions
        plt.plot(self.true_positions[:, 0], self.true_positions[:, 1], 'g-', label='True Position')
        
        # Plot estimated positions
        plt.plot(self.estimated_positions[:, 0], self.estimated_positions[:, 1], 'r-', label='Estimated Position')
        
        # Mark start points
        plt.plot(self.true_positions[0, 0], self.true_positions[0, 1], 'go', markersize=10, label='True Start')
        plt.plot(self.estimated_positions[0, 0], self.estimated_positions[0, 1], 'ro', markersize=10, label='Estimated Start')
        
        # Mark end points
        plt.plot(self.true_positions[-1, 0], self.true_positions[-1, 1], 'g*', markersize=10, label='True End')
        plt.plot(self.estimated_positions[-1, 0], self.estimated_positions[-1, 1], 'r*', markersize=10, label='Estimated End')
        
        plt.grid(True)
        plt.axis('equal')
        plt.title('Robot Positions: True vs Estimated')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.legend()
        plt.savefig('positions.png')
        plt.show()
    
    def plot_position_error(self):
        """Plot position error over time"""
        if len(self.true_positions) == 0 or len(self.estimated_positions) == 0:
            print("No position data to plot error")
            return
            
        # Interpolate timestamps to align data for error calculation
        # For simplicity, we'll just use the shorter of the two arrays
        length = min(len(self.true_positions), len(self.estimated_positions))
        
        errors = []
        for i in range(length):
            error = np.linalg.norm(self.true_positions[i, :2] - self.estimated_positions[i, :2])
            errors.append(error)
            
        plt.figure(figsize=(10, 6))
        plt.plot(range(length), errors)
        plt.grid(True)
        plt.title('Position Estimation Error Over Time')
        plt.xlabel('Sample Index')
        plt.ylabel('Error (m)')
        plt.savefig('position_error.png')
        plt.show()
    
    def plot_particles(self, frame_indices=None):
        """Plot particle distributions for selected frames"""
        if len(self.particles) == 0:
            print("No particle data to plot")
            return
            
        if frame_indices is None:
            # Choose a few frames evenly spaced throughout the data
            total_frames = len(self.particles)
            if total_frames <= 4:
                frame_indices = range(total_frames)
            else:
                frame_indices = [0, total_frames//3, 2*total_frames//3, total_frames-1]
                
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, ax_idx in enumerate(frame_indices):
            if i >= 4:  # Only plot 4 frames
                break
                
            if ax_idx < len(self.particles):
                ax = axes[i]
                particles = self.particles[ax_idx]
                
                # Plot particles
                ax.scatter(particles[:, 0], particles[:, 1], s=2, alpha=0.3, c='b', label='Particles')
                
                # Plot mean position
                mean_pos = np.mean(particles, axis=0)
                ax.plot(mean_pos[0], mean_pos[1], 'ro', markersize=8, label='Mean Position')
                
                # Plot true position if available
                if ax_idx < len(self.true_positions):
                    ax.plot(self.true_positions[ax_idx, 0], self.true_positions[ax_idx, 1], 'g*', 
                            markersize=10, label='True Position')
                
                ax.grid(True)
                ax.set_aspect('equal')
                ax.set_title(f'Particle Distribution (Frame {ax_idx})')
                ax.set_xlabel('X Position (m)')
                ax.set_ylabel('Y Position (m)')
                ax.legend()
                
        plt.tight_layout()
        plt.savefig('particle_distributions.png')
        plt.show()
    
    def plot_scores(self):
        """Plot score distribution over time"""
        if len(self.scores) == 0:
            print("No score data to plot")
            return
            
        # For each score array, calculate mean and variance
        means = []
        variances = []
        
        for score_array in self.scores:
            means.append(np.mean(score_array))
            variances.append(np.var(score_array))
            
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(means)
        plt.grid(True)
        plt.title('Mean Score Over Time')
        plt.xlabel('Frame Index')
        plt.ylabel('Mean Score')
        
        plt.subplot(2, 1, 2)
        plt.plot(variances)
        plt.grid(True)
        plt.title('Score Variance Over Time')
        plt.xlabel('Frame Index')
        plt.ylabel('Score Variance')
        
        plt.tight_layout()
        plt.savefig('scores.png')
        plt.show()
    
    def plot_control_inputs(self):
        """Plot control inputs over time"""
        if len(self.control_inputs) == 0:
            print("No control input data to plot")
            return
            
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        plt.plot(self.control_inputs[:, 0])
        plt.grid(True)
        plt.title('X Control Input Over Time')
        plt.ylabel('X Velocity (m/s)')
        
        plt.subplot(3, 1, 2)
        plt.plot(self.control_inputs[:, 1])
        plt.grid(True)
        plt.title('Y Control Input Over Time')
        plt.ylabel('Y Velocity (m/s)')
        
        plt.subplot(3, 1, 3)
        plt.plot(self.control_inputs[:, 2])
        plt.grid(True)
        plt.title('Z Control Input Over Time')
        plt.xlabel('Sample Index')
        plt.ylabel('Z Velocity (m/s)')
        
        plt.tight_layout()
        plt.savefig('control_inputs.png')
        plt.show()

    def plot_all(self):
        """Plot all available data"""
        self.plot_positions()
        self.plot_position_error()
        self.plot_particles()
        self.plot_scores()
        self.plot_control_inputs()

def add_publishers_to_slamnode():
    """Code to add the necessary publishers to SlamNode.py
    
    This is just a reference function showing what code needs to be added to SlamNode.py
    to publish particle and score data.
    """
    # In SlamNode.__init__, add:
    '''
    self.particles_pub = self.create_publisher(Float32MultiArray, "/particles", 10)
    self.scores_pub = self.create_publisher(Float32MultiArray, "/scores", 10)
    '''
    
    # In SlamNode.slam_loop, add after localization.update_position:
    '''
    # Publish particle data
    particle_msg = Float32MultiArray()
    particles_flat = self.localization.particles.flatten().tolist()
    particle_msg.data = particles_flat
    self.particles_pub.publish(particle_msg)
    
    # Publish score data
    score_msg = Float32MultiArray()
    score_msg.data = self.localization.scores.tolist()
    self.scores_pub.publish(score_msg)
    '''
    
    # You'll also need to make particles and scores attributes in the Localization class

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path_to_rosbag>")
        return
        
    bag_path = sys.argv[1]
    if not os.path.exists(bag_path):
        print(f"Bag directory {bag_path} does not exist")
        return
        
    plot_data = PlotData(bag_path)
    plot_data.read_bag()
    plot_data.plot_all()

if __name__ == "__main__":
    main() 