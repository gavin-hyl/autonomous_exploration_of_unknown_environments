import numpy as np

class Localization:
    def __init__(self, initial_location, std_dev_noise, num_particles, dt):
        self.current_location = initial_location
        self.covariance_matrix = np.eye(3)
        self.num_particles = num_particles
        self.dt = dt

        self.std_dev_noise = std_dev_noise
        self.occupancy_threshold = 0.5


    def update_position(self, control_signal, beacon_data, estimated_map):
        # control signal: single array (delta_x, delta_y, delta_z)
        # beacon data: list of arrays (delta_x, delta_y, delta_z)
        # estimated map: object to get occupancy grid and beacon data

        predicted_location = self.current_location + control_signal * self.dt

        # Vectorized particle generation
        noise = np.random.normal(0, self.std_dev_noise, (self.num_particles, 2))
        particles = predicted_location + np.pad(noise, ((0, 0), (0, 1)))  # pad with zeros for z
        
        # Calculate scores (could be parallelized if needed)
        scores = np.array([self.calculate_score(p, beacon_data, estimated_map) for p in particles])
        
        # Vectorized softmax
        scores = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
        scores /= scores.sum()
        
        # Efficient resampling
        particles_idx = np.random.choice(self.num_particles, size=self.num_particles, p=scores)
        particles = np.array(particles)[particles_idx]
        
        # Update current location and covariance
        self.current_location = np.mean(particles, axis=0)
        self.covariance_matrix = np.cov(particles, rowvar=False)
        
        return self.current_location, self.covariance_matrix

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
                return -float('inf')
            
            closest_beacon, _, _ = estimated_map.get_closest_beacon(global_beacon)
            if closest_beacon is None:
                return -float('inf')

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
