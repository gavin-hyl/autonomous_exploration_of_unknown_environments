import numpy as np

class BeaconManager:
    """Class to handle beacon tracking and comparison"""
    
    def __init__(self, distance_threshold=2.0):
        # Standard beacons (from sensor observations)
        self.beacon_positions = []
        self.beacon_covariances = []
        
        # Particle-based beacons
        self.particle_beacons = []  # List of lists of particles for each beacon
        self.particle_averages = []  # Average position of each beacon's particles
        self.particle_covariances = []  # Covariance of each beacon's particles
        
        self.distance_threshold = distance_threshold
    
    def find_closest_beacon(self, point, use_particles=False):
        """Find closest beacon to a point
        
        Args:
            point: Point in world coordinates
            use_particles: Whether to use particle-based beacons
            
        Returns:
            (position, covariance, index) or (None, None, None) if not found
        """
        positions = self.particle_averages if use_particles else self.beacon_positions
        covariances = self.particle_covariances if use_particles else self.beacon_covariances
        
        if not positions:
            return None, None, None
            
        distances = np.linalg.norm(np.array(positions) - point, axis=1)
        index = np.argmin(distances)
        min_distance = distances[index]
        
        if min_distance > self.distance_threshold:
            return None, None, None
            
        return positions[index], covariances[index], index
    
    def get_beacon_match_votes(self, cluster):
        """Count votes for which beacon a cluster matches
        
        Args:
            cluster: List of particle positions
            
        Returns:
            Dictionary of {beacon_index: vote_count}
        """
        votes = {}
        for particle in cluster:
            _, _, idx = self.find_closest_beacon(particle, use_particles=True)
            idx = idx if idx is not None else -1
            votes[idx] = votes.get(idx, 0) + 1
        return votes
    
    def determine_beacon_match(self, cluster):
        """Determine which beacon a cluster matches based on voting
        
        Returns:
            (position, index) of matched beacon or (None, None) if no match
        """
        votes = self.get_beacon_match_votes(cluster)
        if not votes:
            return None, None
            
        winner = max(votes.keys(), key=lambda k: votes[k])
        
        if winner == -1:
            return None, None
            
        return self.particle_averages[winner], winner
    
    def update_standard_beacon(self, position, covariance):
        """Add or update a standard beacon
        
        Args:
            position: Beacon position
            covariance: Beacon position covariance
        
        Returns:
            Index of the beacon
        """
        closest_beacon, closest_cov, closest_idx = self.find_closest_beacon(position)
        
        if closest_beacon is not None:
            # Update existing beacon using Kalman filter
            new_cov = np.linalg.pinv(np.linalg.pinv(closest_cov) + np.linalg.pinv(covariance))
            new_pos = new_cov @ (np.linalg.pinv(closest_cov) @ closest_beacon + 
                                 np.linalg.pinv(covariance) @ position)
            self.beacon_positions[closest_idx] = new_pos
            self.beacon_covariances[closest_idx] = new_cov
            return closest_idx
        else:
            # Add new beacon
            self.beacon_positions.append(position)
            self.beacon_covariances.append(covariance)
            return len(self.beacon_positions) - 1
    
    def update_beacon_particles(self, cluster, covariance=None):
        """Add particles to a beacon or create a new beacon
        
        Args:
            cluster: List of particle positions
            covariance: Optional covariance matrix for new beacons
        
        Returns:
            Index of the updated/created beacon
        """
        position, index = self.determine_beacon_match(cluster)
        
        if index is not None:
            # Update existing beacon
            self.particle_beacons[index].extend(cluster)
            self.particle_averages[index] = np.mean(self.particle_beacons[index], axis=0)
            self.particle_covariances[index] = np.cov(np.array(self.particle_beacons[index]).T)
            return index
        else:
            # Create new beacon
            self.particle_beacons.append(list(cluster))
            self.particle_averages.append(np.mean(cluster, axis=0))
            
            # Calculate covariance from the cluster or use provided covariance
            cluster_cov = np.cov(np.array(cluster).T) if len(cluster) > 1 else np.eye(3)
            self.particle_covariances.append(covariance if covariance is not None else cluster_cov)
            return len(self.particle_beacons) - 1

class Mapping:
    def __init__(self,
                 map_size: tuple[float, float],
                 map_origin: tuple[float, float],
                 grid_size: float = 0.05,
                 ):
        self.grid_size = grid_size
        self.map_origin = map_origin
        self.grid_width = int(map_size[0] / grid_size)
        self.grid_height = int(map_size[1] / grid_size)
        self.lor_grid = np.zeros((self.grid_width, self.grid_height))
        self.lor_grid_guess = np.zeros((self.grid_width, self.grid_height))
        self.lor_known = np.zeros((self.grid_width, self.grid_height)) # 1 for known, 0 for unknown

        # Constants
        self.L_FREE = -0.1
        self.L_OCC = 0.3
        self.L_OCC_GUESS = self.L_OCC / 100
        self.BEACON_DIST_THRESH = 2
        self.LOR_SATURATION = 100

        # Create beacon manager
        self.beacon_manager = BeaconManager(distance_threshold=self.BEACON_DIST_THRESH)
        
        # For backward compatibility
        self.beacon_positions = self.beacon_manager.beacon_positions
        self.beacon_covariances = self.beacon_manager.beacon_covariances
        self.average_beacon_positions_by_particle = self.beacon_manager.particle_averages
        self.beacon_covariances_by_particle = self.beacon_manager.particle_covariances
        self.total_beacon_particles = self.beacon_manager.particle_beacons

    def update(self,
               robot_pos: np.ndarray,
               robot_cov: np.ndarray,
               lidar_data: list[np.ndarray],
               lidar_range: tuple[float, float],
               beacon_data: list[np.ndarray],
               beacon_particles: list[list[np.ndarray]]):
        """
        Update the map with new sensor data.
        
        Args:
            robot_pos: Robot position (x,y,z) in world coordinates
            robot_cov: 3x3 covariance matrix of robot position
            lidar_data: List of (x,y,z) points in robot frame
            lidar_range: (min_range, max_range) of the LiDAR in meters10
            beacon_data: List of (x,y,z) beacon positions in robot frame
        """
        if not lidar_data and not beacon_data:
            return
            
        if lidar_data:
            # discount points after beam breaks
            lidar_array = np.array(lidar_data)
            
            distances = np.linalg.norm(lidar_array[:, :2], axis=1)
            
            # Process each valid point
            for i, point in enumerate(lidar_array):
                point_world = robot_pos + point
                hit = (distances[i] < lidar_range[1] - 0.01)
                theta = np.arctan2(point[1], point[0])
                max_point = point_world + lidar_range[1] * np.array([np.cos(theta), np.sin(theta), 0])

                pos_grid = self._coord_to_grid(robot_pos[0], robot_pos[1])
                point_grid = self._coord_to_grid(point_world[0], point_world[1])
                max_point_grid = self._coord_to_grid(max_point[0], max_point[1])
                
                # Get grid cells along the ray
                ray_cells = self._bresenham_line(pos_grid, point_grid)
                extended_ray_cells = self._bresenham_line(point_grid, max_point_grid)

                # Update cells - mark all but last as free
                for (grid_x, grid_y) in ray_cells[:-1]:
                    self.lor_grid[grid_y, grid_x] += self.L_FREE
                    self.lor_known[grid_y, grid_x] = 1
                
                if not hit:
                    continue

                if len(ray_cells) > 0:
                    grid_x, grid_y = ray_cells[-1]
                    self.lor_grid[grid_y, grid_x] += self.L_OCC
                    self.lor_known[grid_y, grid_x] = 1
                if len(extended_ray_cells) > 0:
                    for (grid_x, grid_y) in extended_ray_cells:
                        self.lor_grid_guess[grid_y, grid_x] += self.L_OCC_GUESS

        # Process beacon data with beacon manager
        if beacon_data:
            for measurement in beacon_data:
                measurement_world = robot_pos + measurement
                self.beacon_manager.update_standard_beacon(measurement_world, robot_cov)

        # Process beacon particles with beacon manager
        if beacon_particles is not None:
            for cluster in beacon_particles:
                self.beacon_manager.update_beacon_particles(cluster)

        # Apply log-odds bounds to prevent saturation
        sat = self.LOR_SATURATION
        self.lor_grid = np.clip(self.lor_grid, -sat, sat)
        self.lor_grid_guess = np.clip(self.lor_grid_guess, -sat, sat)

    def get_closest_beacon(self, point_world, compare_with_beacon_particles=False):
        """
        Get the closest beacon to the given point.
        
        Args:
            point_world: (x,y,z) point in world coordinates
            compare_with_beacon_particles: Whether to use particle-based beacons
            
        Returns:
            (position, covariance, index) of closest beacon or (None, None, None) if not found
        """
        return self.beacon_manager.find_closest_beacon(point_world, use_particles=compare_with_beacon_particles)

    # Keep existing methods for backwards compatibility
    def get_closest_beacon_by_particle(self, cluster):
        """Legacy method - use BeaconManager instead"""
        position, index = self.beacon_manager.determine_beacon_match(cluster)
        return position, index

    def _bresenham_line(self, start, end):
        """
        Takes in two grid coordinates and returns the cells that lie along the line.
        """
        # Extract the coordinates
        (xs, ys) = start[:2]
        (xe, ye) = end[:2]

        if xs == xe or ys == ye:
            return []

        # Move along ray (excluding endpoint).
        if (np.abs(xe-xs) >= np.abs(ye-ys)):
            return[(u, int(ys + (ye-ys)/(xe-xs) * (u+0.5-xs)))
                   for u in range(int(xs), int(xe), int(np.sign(xe-xs)))]
        else:
            return[(int(xs + (xe-xs)/(ye-ys) * (v+0.5-ys)), v)
                   for v in range(int(ys), int(ye), int(np.sign(ye-ys)))]

    def _coord_to_grid(self, x, y):
        """Convert world coordinates to grid coordinates"""
        grid_x = int((x - self.map_origin[0]) / self.grid_size)
        grid_y = int((y - self.map_origin[1]) / self.grid_size)
        # 그리드 인덱스가 범위를 벗어나지 않도록 제한
        grid_x = np.clip(grid_x, 0, self.grid_width - 1)
        grid_y = np.clip(grid_y, 0, self.grid_height - 1)
        return grid_x, grid_y

    def _grid_to_coord(self, x, y):
        """
        Convert grid indices to world coordinates (center of the cell).

        Args:
            x (int): grid index in x.
            y (int): grid index in y.

        Returns:
            (float, float): Corresponding world coordinates.
        """
        world_x = self.map_origin[0] + (x + 0.5) * self.grid_size
        world_y = self.map_origin[1] + (y + 0.5) * self.grid_size
        return world_x, world_y

    def world_to_prob(self, world_x, world_y):
        grid_x, grid_y = self._coord_to_grid(world_x, world_y)
        lor = self.lor_grid[grid_x, grid_y]
        prob = np.exp(lor) / (1 + np.exp(lor))
        return prob
    
    def world_to_prob_batch(self, coords):
        """
        Convert world coordinates to probabilities in batch.

        Args:
            coords: Nx2 array of world coordinates (x,y)

        Returns:
            1D array of probabilities for each coordinate
        """
        # Convert to grid coordinates
        grid_x = ((coords[:, 0] - self.map_origin[0]) / self.grid_size).astype(int)
        grid_y = ((coords[:, 1] - self.map_origin[1]) / self.grid_size).astype(int)

        # Get log odds values
        log_odds = self.lor_grid[grid_x, grid_y]

        # Convert to probabilities using vectorized operations
        probs = np.exp(log_odds) / (1 + np.exp(log_odds))

        return probs
    

    def get_prob_grid(self):
        """
        Get the probability grid.
        """
        log_odds_grid = self.lor_grid * self.lor_known + self.lor_grid_guess * (1 - self.lor_known)
        return np.exp(log_odds_grid) / (1 + np.exp(log_odds_grid))
    