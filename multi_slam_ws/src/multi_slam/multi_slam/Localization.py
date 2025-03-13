import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
import std_msgs.msg
import struct

class Localization:
    def __init__(self, initial_location, std_dev_noise, num_particles, dt):
        self.current_location = initial_location
        self.covariance_matrix = np.eye(3)
        self.num_particles = num_particles

        self.std_dev_noise = std_dev_noise
        self.occupancy_threshold = 0.5
        self.initial_location = initial_location
        
        # Vectorized particle generation
        # noise = np.random.normal(0, self.std_dev_noise, (self.num_particles, 2))
        # particles = pos_hat_new + np.pad(noise, ((0, 0), (0, 1)))  # pad with zeros for z

        self.particles = self.initial_location - np.zeros((self.num_particles, 3))
        self.particles = self.particles + np.pad(noise, ((0, 0), (0, 1)))  # pad with zeros for z
    
    def update_position(self, beacon_data, estimated_map):
        if self.particles is None:
            self.particles = self.initial_location - np.zeros((self.num_particles, 3))
            self.particles = self.particles + np.pad(noise, ((0, 0), (0, 1)))

        particles = self.particles

        # Calculate scores (could be parallelized if needed)
        scores = np.array([self.calculate_score(p, beacon_data, estimated_map) for p in particles])

        # Vectorized softmax
        scores = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
        scores /= scores.sum()

        # Efficient resampling
        particles_idx = np.random.choice(self.num_particles, size=self.num_particles, p=scores)
        particles = np.array(particles)[particles_idx]

        cov = np.cov(particles.T)

        return particles, cov

    def calculate_score(self, particle, beacon_data, estimated_map):
        score = 0 
        
        # Check beacons in batch rather than one at a time
        global_beacons = [self.current_location + beacon for beacon in beacon_data]
        
        # Use numpy operations for faster calculations
        for global_beacon in global_beacons:
            # Sample fewer points along the line
            points = self.create_2d_line_fast(particle[0:2], global_beacon[0:2])
            
            # Batch check occupancy
            occupancies = estimated_map.world_to_prob_batch(points)
            if np.any(occupancies > self.occupancy_threshold):
                return -1e9
            
            closest_beacon, _, _ = estimated_map.get_closest_beacon(global_beacon)
            if closest_beacon is None:
                return -1e9

            score += max(0, min(100, 1 / np.sqrt(sum((closest_beacon - global_beacon) ** 2))))
        
        return score

    def create_2d_line(self, start, end):
        x1, y1 = start
        x2, y2 = end
        points = []
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        step_x = 1 if x1 < x2 else -1
        step_y = 1 if y1 < y2 else -1
        
        error = dx - dy
        x, y = x1, y1

        while True:
            points.append((x, y))
            
            if x == x2 and y == y2:
                break
            
            error2 = error * 2
            
            if error2 > -dy:
                error -= dy
                x += step_x
                
            if error2 < dx:
                error += dx
                y += step_y
        
        return points

    def create_2d_line_fast(self, start, end, num_samples=10):
        """Faster line creation with fewer samples"""
        x1, y1 = start
        x2, y2 = end
        
        # Create evenly spaced points along the line
        t = np.linspace(0, 1, num_samples)
        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t
        
        return np.column_stack((x, y))
