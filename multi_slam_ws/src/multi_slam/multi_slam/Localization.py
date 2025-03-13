import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
import std_msgs.msg
import struct

def array_to_pointcloud2(array_2d, frame_id="map"):
    """Convert a 2D numpy array to ROS2 PointCloud2 message.
    
    Args:
        array_2d: 2D numpy array where:
                  - row index will be the Y coordinate
                  - column index will be the X coordinate
                  - array value will be the Z coordinate
        frame_id: Frame ID for the point cloud message
    
    Returns:
        sensor_msgs.msg.PointCloud2: The point cloud message
    """
    # Create a PointCloud2 message
    cloud_msg = PointCloud2()
    
    # Get dimensions
    height, width = array_2d.shape
    
    # Define fields for the point cloud
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    
    # Create the header
    cloud_msg.header = std_msgs.msg.Header()
    cloud_msg.header.stamp = Node.get_clock().now().to_msg()
    cloud_msg.header.frame_id = frame_id
    
    # Set point cloud properties
    cloud_msg.height = 1  # Unorganized point cloud
    cloud_msg.width = height * width  # Total number of points
    cloud_msg.fields = fields
    cloud_msg.is_bigendian = False
    cloud_msg.point_step = 12  # 3 * float32 (4 bytes each)
    cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
    cloud_msg.is_dense = True
    
    # Create the point data
    points_list = []
    for y in range(height):
        for x in range(width):
            z = array_2d[y, x]
            # Pack float32 into bytes
            points_list.append(struct.pack('fff', float(x), float(y), float(z)))
    
    # Convert list of byte strings to a single byte string
    cloud_msg.data = b''.join(points_list)
    
    return cloud_msg

def pointcloud2_to_array(cloud_msg, default_value=0.0):
    """Convert a ROS2 PointCloud2 message to a 2D numpy array.
    
    Args:
        cloud_msg: sensor_msgs.msg.PointCloud2 message
        default_value: Value to fill in empty cells of the array
    
    Returns:
        numpy.ndarray: 2D array representing the point cloud
    """
    # Extract points from the PointCloud2 message
    points = list(pc2.read_points(cloud_msg, field_names=('x', 'y', 'z'), skip_nans=True))
    
    if not points:
        return np.array([])
    
    # Find dimensions
    x_values = [p[0] for p in points]
    y_values = [p[1] for p in points]
    
    x_min, x_max = int(min(x_values)), int(max(x_values))
    y_min, y_max = int(min(y_values)), int(max(y_values))
    
    # Create empty array
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    array_2d = np.full((height, width), default_value)
    
    # Fill array with point values
    for x, y, z in points:
        array_2d[int(y) - y_min, int(x) - x_min] = z
    
    return array_2d

class Localization:
    def __init__(self, initial_location, std_dev_noise, num_particles, dt):
        self.current_location = initial_location
        self.covariance_matrix = np.eye(3)
        self.num_particles = num_particles

        self.std_dev_noise = std_dev_noise
        self.occupancy_threshold = 0.5
        
        # Vectorized particle generation
        # noise = np.random.normal(0, self.std_dev_noise, (self.num_particles, 2))
        # particles = pos_hat_new + np.pad(noise, ((0, 0), (0, 1)))  # pad with zeros for z

    def update_position(self, pos_hat_particles, beacon_data, estimated_map):
        particles = pointcloud2_to_array(pos_hat_particles)

        # Calculate scores (could be parallelized if needed)
        scores = np.array([self.calculate_score(p, beacon_data, estimated_map) for p in particles])

        # Vectorized softmax
        scores = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
        scores /= scores.sum()

        # Efficient resampling
        particles_idx = np.random.choice(self.num_particles, size=self.num_particles, p=scores)
        particles = np.array(particles)[particles_idx]

        return array_to_pointcloud2(particles)  

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
