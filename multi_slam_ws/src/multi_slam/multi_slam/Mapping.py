import numpy as np

class Mapping:
    def __init__(self,
                 map_size: tuple[float, float],
                 map_origin: tuple[float, float],
                 grid_size: float = 0.05,
                 ):
        self.grid_size = grid_size
        self.map_origin = map_origin
        # Calculate grid dimensions
        self.grid_width = int(map_size[0] / grid_size)
        self.grid_height = int(map_size[1] / grid_size)
        self.log_odds_grid = np.zeros((self.grid_width, self.grid_height))
        self.beacon_positions = []
        self.beacon_covariances = []
        
        # Cache for Bresenham's algorithm
        self._bresenham_cache = {}
        self._cache_misses = 0
        
        # Constants
        self.L_FREE = -0.1
        self.L_OCC = 0.3
        self.BEACON_DIST_THRESH = 0.5

    def update(self,
               robot_pos: np.ndarray,
               robot_cov: np.ndarray,
               lidar_data: list[np.ndarray],
               lidar_range: tuple[float, float],
               beacon_data: list[np.ndarray]):
        """
        Update the map with new sensor data.
        
        Args:
            robot_pos: Robot position (x,y,z) in world coordinates
            robot_cov: 3x3 covariance matrix of robot position
            lidar_data: List of (x,y,z) points in robot frame
            lidar_range: (min_range, max_range) of the LiDAR in meters
            beacon_data: List of (x,y,z) beacon positions in robot frame
        """
        # Skip update if there's no data to process
        if not lidar_data and not beacon_data:
            return
            
        # Process LiDAR data (subsample for speed)
        if lidar_data:
            # Convert to numpy array for faster operations if not already
            lidar_array = np.array(lidar_data)
            
            # Calculate distances for each lidar point
            distances = np.linalg.norm(lidar_array[:, :2], axis=1)
            
            # Filter by range - vectorized operation
            valid_indices = (distances >= lidar_range[0]) & (distances <= lidar_range[1])
            filtered_lidar = lidar_array[valid_indices]
            
            # Process each valid point
            for point in filtered_lidar:
                # Convert point from robot frame to world frame
                point_world = robot_pos + point
                
                # Get grid cells along the ray
                ray_cells = self._bresenham_line(robot_pos, point_world)
                
                # Update cells - mark all but last as free
                for i, (grid_x, grid_y) in enumerate(ray_cells[:-1]):
                    # Skip border cells
                    if (grid_x <= 0 or grid_x >= self.grid_width-1 or 
                        grid_y <= 0 or grid_y >= self.grid_height-1):
                        continue
                        
                    # Apply log-odds update - mark as free space (negative update)
                    # Use distance-based decay for free space - less confident further away
                    distance_factor = min(1.0, 0.5 + 0.5 * i / max(1, len(ray_cells)))
                    self.log_odds_grid[grid_x, grid_y] -= 0.4 * distance_factor
                
                # Mark the endpoint as occupied if it's within the grid
                if ray_cells and len(ray_cells) > 0:
                    grid_x, grid_y = ray_cells[-1]
                    if (0 < grid_x < self.grid_width-1 and 0 < grid_y < self.grid_height-1):
                        self.log_odds_grid[grid_x, grid_y] += 0.9
        
        # Process beacon data
        if beacon_data:
            for beacon in beacon_data:
                # Convert beacon from robot frame to world frame
                beacon_world = robot_pos + beacon
                
                # Scale covariance by distance - higher uncertainty at greater distances
                distance = np.linalg.norm(beacon)
                scaled_cov = robot_cov * (1.0 + 0.1 * distance)
                
                # Find closest beacon in map, or add a new one
                closest_beacon = self.get_closest_beacon(beacon_world)
                if closest_beacon is not None:
                    # Update existing beacon
                    # Here you'd update position using Kalman filter or similar
                    pass
                else:
                    # Add new beacon to the list
                    self.beacon_positions.append(beacon_world)
                    self.beacon_covariances.append(scaled_cov)
        
        # Apply log-odds bounds to prevent saturation
        self.log_odds_grid = np.clip(self.log_odds_grid, -10.0, 10.0)

    def get_closest_beacon(self, point_world):
        if len(self.beacon_positions) == 0:
            return None
        dists = np.linalg.norm(np.array(self.beacon_positions) - point_world, axis=1)
        mindex = np.argmin(dists)
        min_dist = dists[mindex]
        if min_dist > self.BEACON_DIST_THRESH:
            return None
        return self.beacon_positions[mindex], self.beacon_covariances[mindex]

    def _bresenham_line(self, start, end):
        """
        Implements a highly optimized Bresenham's line algorithm.
        Returns the grid cells that a line from start to end passes through.
        
        Args:
            start: Start point in world coordinates (x, y, z)
            end: End point in world coordinates (x, y, z)
            
        Returns:
            List of (grid_x, grid_y) tuples representing grid cells the line passes through
        """
        # Convert world coordinates to grid coordinates
        x1, y1 = self._coord_to_grid(start[0], start[1])
        x2, y2 = self._coord_to_grid(end[0], end[1])
        
        # Early boundary checking - completely outside grid
        if (x1 < 0 and x2 < 0) or (y1 < 0 and y2 < 0) or \
           (x1 >= self.grid_width and x2 >= self.grid_width) or \
           (y1 >= self.grid_height and y2 >= self.grid_height):
            return []
        
        # Generate a cache key for this line
        cache_key = (x1, y1, x2, y2)
        # Check if we've already computed this line
        if hasattr(self, '_bresenham_cache') and cache_key in self._bresenham_cache:
            return self._bresenham_cache[cache_key]
            
        # Implementation of Bresenham's algorithm optimized for speed
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        # Determine dominant direction for optimization
        if dx > dy:
            # Process horizontally dominant lines
            if x1 > x2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
            
            # Pre-allocate array for x values (faster than appending)
            x_values = np.arange(x1, x2 + 1, dtype=np.int32)
            
            # Calculate corresponding y values
            if y1 == y2:  # Horizontal line
                y_values = np.full_like(x_values, y1)
            else:
                slope = (y2 - y1) / (x2 - x1)
                y_values = np.floor(y1 + slope * (x_values - x1) + 0.5).astype(np.int32)
                
            # Stack to create coordinate pairs
            coords = np.stack((x_values, y_values), axis=1)
            
            # Apply boundary filter (vectorized)
            mask = ((coords[:, 0] >= 0) & (coords[:, 0] < self.grid_width) & 
                    (coords[:, 1] >= 0) & (coords[:, 1] < self.grid_height))
            coords = coords[mask]
            
            # Convert to list of tuples
            points = list(map(tuple, coords))
        else:
            # Process vertically dominant lines
            if y1 > y2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
                
            # Pre-allocate array for y values
            y_values = np.arange(y1, y2 + 1, dtype=np.int32)
            
            # Calculate corresponding x values
            if x1 == x2:  # Vertical line
                x_values = np.full_like(y_values, x1)
            else:
                slope = (x2 - x1) / (y2 - y1)
                x_values = np.floor(x1 + slope * (y_values - y1) + 0.5).astype(np.int32)
            
            # Stack to create coordinate pairs
            coords = np.stack((x_values, y_values), axis=1)
            
            # Apply boundary filter (vectorized)
            mask = ((coords[:, 0] >= 0) & (coords[:, 0] < self.grid_width) & 
                    (coords[:, 1] >= 0) & (coords[:, 1] < self.grid_height))
            coords = coords[mask]
            
            # Convert to list of tuples
            points = list(map(tuple, coords))
        
        # Store in cache
        self._bresenham_cache[cache_key] = points
        self._cache_misses += 1
        
        return points

    def _coord_to_grid(self, x, y):
        """
        Convert world coordinates to grid indices.

        Args:
            x (float): world x coordinate.
            y (float): world y coordinate.

        Returns:
            (int, int): Corresponding grid indices.
        """
        grid_x = int((x - self.map_origin[0]) / self.grid_size)
        grid_y = int((y - self.map_origin[1]) / self.grid_size)
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
        lor = self.log_odds_grid[grid_x, grid_y]
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
        log_odds = self.log_odds_grid[grid_x, grid_y]

        # Convert to probabilities using vectorized operations
        probs = np.exp(log_odds) / (1 + np.exp(log_odds))

        return probs