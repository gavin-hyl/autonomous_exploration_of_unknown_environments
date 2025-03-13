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
        self.BEACON_DIST_THRESH = 2

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
        if not lidar_data and not beacon_data:
            return
            
        if lidar_data:
            # discount points after beam breaks
            lidar_array = np.array(lidar_data)
            
            distances = np.linalg.norm(lidar_array[:, :2], axis=1)
            
            # Process each valid point
            for i, point in enumerate(lidar_array):
                point_world = robot_pos + point

                pos_grid = self._coord_to_grid(robot_pos[0], robot_pos[1])
                point_grid = self._coord_to_grid(point_world[0], point_world[1])
                
                # Get grid cells along the ray
                ray_cells = self._bresenham_line(pos_grid, point_grid)
                
                # Update cells - mark all but last as free
                for (grid_x, grid_y) in ray_cells[:-1]:
                    self.log_odds_grid[grid_y, grid_x] += self.L_FREE
                
                if len(ray_cells) > 0 and distances[i] < lidar_range[1] - 0.01:
                    grid_x, grid_y = ray_cells[-1]
                    self.log_odds_grid[grid_y, grid_x] += self.L_OCC
        
        # Process beacon data
        if beacon_data:
            for measurement in beacon_data:
                measurement_world = robot_pos + measurement
                closest_beacon, closest_cov, closest_idx = self.get_closest_beacon(measurement_world)
                if closest_beacon is not None:
                    new_cov = np.linalg.pinv(np.linalg.pinv(closest_cov) + np.linalg.pinv(robot_cov))
                    new_pos = new_cov @ (np.linalg.pinv(closest_cov) @ closest_beacon + np.linalg.pinv(robot_cov) @ measurement_world)
                    self.beacon_positions[closest_idx] = new_pos
                    self.beacon_covariances[closest_idx] = new_cov
                else:
                    # Add new beacon to the list
                    self.beacon_positions.append(measurement_world)
                    self.beacon_covariances.append(robot_cov)

        # Apply log-odds bounds to prevent saturation
        self.log_odds_grid = np.clip(self.log_odds_grid, -10.0, 10.0)

    def get_closest_beacon(self, point_world):
        if len(self.beacon_positions) == 0:
            return None, None, None
        dists = np.linalg.norm(np.array(self.beacon_positions) - point_world, axis=1)
        mindex = np.argmin(dists)
        min_dist = dists[mindex]
        if min_dist > self.BEACON_DIST_THRESH:
            return None, None, None
        return self.beacon_positions[mindex], self.beacon_covariances[mindex], mindex

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