import numpy as np

class Mapping:
    def __init__(self,
                 map_size: tuple[float, float],
                 map_origin: tuple[float, float],
                 grid_size: float = 0.1,
                 ):
        self.grid_size = grid_size
        self.map_origin = map_origin
        # Note: ensure the grid shape is passed as a tuple
        self.log_odds_grid = np.zeros((int(map_size[0] / grid_size), int(map_size[1] / grid_size)))
        self.beacon_positions = []
        self.beacon_covariances = []
        
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
        Update the grid with new data.

        Args:
            robot_pos: x, y, z
            robot_cov: 3x3 covariance matrix
            lidar_data: list of 1d numpy arrays
            lidar_range: tuple of min and max range
            beacon_data: list of 1d numpy arrays

        Returns:
            None
        """
        lidar_data_world = []
        for point in lidar_data:
            point_world = point + robot_pos
            lidar_data_world.append(point_world)
        cov_scaling = 1/np.linalg.det(robot_cov)
        # Update the log-odds grid
        for point in lidar_data_world:
            dist = np.linalg.norm(point - robot_pos)
            if dist < lidar_range[0]:
                pass
            elif dist < lidar_range[1]:
                # Update the log-odds grid from rmin to point
                for cell in self.bresenham_line(robot_pos, point):
                    self.log_odds_grid[cell] += self.L_OCC * cov_scaling
            else:
                for cell in self.bresenham_line(robot_pos, point):
                    self.log_odds_grid[cell] += self.L_FREE * cov_scaling
        
        # Update the beacon positions
        beacon_data_world = []
        for point in beacon_data:
            point_world = point + robot_pos
            beacon_data_world.append(point_world)
            beacon_world, beacon_cov, mindex = self.get_closest_beacon(point_world)
            beacon_cov_inv = np.linalg.inv(beacon_cov)
            robot_cov_inv = np.linalg.inv(robot_cov)
            if beacon_world is None:
                self.beacon_positions.append(point_world)
                self.beacon_covariances.append(robot_cov)
            else:
                dist = np.linalg.norm(point_world - beacon_world)
                if dist > self.BEACON_DIST_THRESH:
                    continue
                cov_scaling = np.linalg.inv(beacon_cov_inv + robot_cov_inv)
                self.beacon_covariances[mindex] = cov_scaling
                self.beacon_positions[mindex] = cov_scaling @ (
                    beacon_cov_inv * beacon_world + robot_cov_inv * point_world
                )

    def get_closest_beacon(self, point_world):
        if len(self.beacon_positions) == 0:
            return None, None
        dists = np.linalg.norm(np.array(self.beacon_positions) - point_world, axis=1)
        mindex = np.argmin(dists)
        min_dist = dists[mindex]
        if min_dist > self.BEACON_DIST_THRESH:
            return None, None, None
        return self.beacon_positions[mindex], self.beacon_covariances[mindex], mindex


    def _bresenham_line(self, x1, y1, x2, y2):
        """
        Bresenham's line algorithm to determine which grid cells lie on the line between two points.

        Returns:
            List of (x, y) grid indices.
        """
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        while True:
            points.append((x1, y1))
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
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
        
    