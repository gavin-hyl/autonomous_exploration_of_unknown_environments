import numpy as np
import math
import cv2
from scipy.ndimage import sobel, gaussian_filter, binary_dilation
from scipy.spatial import KDTree
from typing import List, Tuple, Optional
import random
import time
import logging
from sklearn.cluster import DBSCAN


########## RRT Node Class ##########
class RRTNode:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.path_x = []  # x coordinates to this node
        self.path_y = []  # y coordinates to this node
        self.parent = None  # parent node

########## Planner Class ##########
class Planner:

    def __init__(self, 
                 map_resolution=0.1,
                 rrt_step_size=0.5,
                 rrt_max_iter=500,
                 rrt_goal_sample_rate=5,
                 rrt_connect_circle_dist=0.5,
                 pd_p_gain=2.0,
                 pd_d_gain=0.2,
                 entropy_weight=1.0,
                 beacon_weight=2.0,
                 beacon_attraction_radius=3.0,
                 coverage_threshold=0.95,
                 sensor_range=10.0,
                 min_frontier_size=3,
                 corridor_threshold=2.0):
        """        
        Args:
            map_resolution (float): Map resolution (meters/cell)
            rrt_step_size (float): RRT step size
            rrt_max_iter (int): Maximum number of RRT iterations
            rrt_goal_sample_rate (int): Goal point sampling rate (%)
            rrt_connect_circle_dist (float): Node connection allowed distance
            pd_p_gain (float): Proportional gain for PD controller
            pd_d_gain (float): Derivative gain for PD controller
            entropy_weight (float): Entropy map weight
            beacon_weight (float): Beacon position weight
            beacon_attraction_radius (float): Beacon attraction radius
            coverage_threshold (float): Threshold for exploration coverage to consider complete
            sensor_range (float): Maximum sensor range for planning
            min_frontier_size (int): Minimum size of frontier clusters to consider
            corridor_threshold (float): Threshold for corridor width detection
        """
        # RRT parameters
        self.rrt_step_size = rrt_step_size
        self.rrt_max_iter = rrt_max_iter
        self.rrt_goal_sample_rate = rrt_goal_sample_rate
        self.rrt_connect_circle_dist = rrt_connect_circle_dist
        
        # PD controller parameters
        self.pd_p_gain = pd_p_gain
        self.pd_d_gain = pd_d_gain
        
        # Map properties
        self.map_resolution = map_resolution
        self.occupancy_grid = None
        self.grid_width = 0
        self.grid_height = 0
        self.grid_origin = (0, 0)
        
        # Weights and radii - Beacon attraction range
        self.entropy_weight = entropy_weight
        self.beacon_weight = beacon_weight
        self.beacon_attraction_radius = beacon_attraction_radius
        
        # State variables
        self.current_pos = np.array([0.0, 0.0])
        self.prev_pos = np.array([0.0, 0.0])
        self.current_path = []
        self.current_path_index = 0
        self.beacons = []
        self.is_planning = False
        self.last_control = np.array([0.0, 0.0])
        self.prev_error = np.array([0.0, 0.0])
        self.planning_time = time.time()
        
        # RRT visualization data
        self.rrt_nodes = []     # Store nodes in the RRT tree
        self.rrt_edges = []     # Store edges between nodes
        self.rrt_samples = []   # Store random samples used for RRT
        self.entropy_map = None # Store latest entropy map 
        self.boundary_map = None # Store latest boundary map
        
        # Variables for oscillation prevention
        self.original_position = None  # Store initial position
        self.previous_goals = []  # Store previous goal points
        self.max_previous_goals = 5  # Maximum number of previous goals to store
        self.forbidden_angle_range = 30  # Forbidden angle range (degrees)
        self.visited_areas = []  # Store visited areas
        self.visited_area_radius = 2.0  # Radius to consider as visited area (meters)
        
        # Variables for goal point stabilization
        self.goal_dwell_time = 5.0  # Time to stay at goal point before replanning (seconds)
        self.goal_reached_time = None  # Time when goal was reached
        self.final_goal_reached = False  # Flag to indicate if final goal was reached
        self.goal_reached_dist = 0.3  # Distance threshold to consider goal reached (meters)
        self.waypoint_reached_dist = 0.2  # Distance threshold to consider waypoint reached (meters)
        self.min_replan_time = 15.0  # Minimum time between replanning (seconds)
        
        # 최적화된 계획 흐름을 위한 변수들
        self.coverage_threshold = coverage_threshold  # 탐험 완료 임계값
        self.sensor_range = sensor_range  # 센서 범위
        self.min_frontier_size = min_frontier_size  # 최소 프론티어 크기
        self.corridor_threshold = corridor_threshold  # 코리도 감지 임계값
        self.exploration_complete = False  # 탐험 완료 플래그

    def update_map(self, occupancy_grid, grid_origin, grid_resolution):
        """
        Update the occupancy grid map
        
        Args:
            occupancy_grid (numpy.ndarray): Occupancy grid 
            grid_origin (tuple): Grid origin (x, y)
            grid_resolution (float): Grid resolution (meters/cell)
        """
        self.occupancy_grid = occupancy_grid
        self.grid_height, self.grid_width = occupancy_grid.shape
        self.grid_origin = grid_origin
        self.map_resolution = grid_resolution

    def update_position(self, current_pos):
        """
        Update the robot position
        
        Args:
            current_pos (numpy.ndarray): Current robot position [x, y]
        """
        self.prev_pos = self.current_pos.copy()
        self.current_pos = np.array(current_pos[:2])  # Use only x, y
        
        # If initial position is not set, store current position as initial position
        if self.original_position is None:
            self.original_position = self.current_pos.copy()
            print(f"Original position set to: {self.original_position}")
            
        # Update visited areas
        self.update_visited_areas(self.current_pos)
            
    def update_visited_areas(self, position):
        """
        Add current position to visited areas list
        
        Args:
            position (numpy.ndarray): Current position [x, y]
        """
        # Check distance with already visited areas
        for i, area in enumerate(self.visited_areas):
            dist = np.linalg.norm(position - area['position'])
            if dist < self.visited_area_radius:
                # If area is already visited, increase visit count
                self.visited_areas[i]['visits'] += 1
                return
                
        # Add new visited area
        self.visited_areas.append({
            'position': position.copy(),
            'visits': 1,
            'time': time.time()
        })
        
        # Clean up old visited areas (limit to 20)
        if len(self.visited_areas) > 20:
            # Sort by time and remove oldest
            self.visited_areas.sort(key=lambda x: x['time'])
            self.visited_areas.pop(0)
    
    def calculate_angle(self, point1, point2):
        """
        Calculate angle between two points (radians)
        
        Args:
            point1 (numpy.ndarray): Start point [x, y]
            point2 (numpy.ndarray): End point [x, y]
            
        Returns:
            float: Angle (radians, -pi~pi)
        """
        return math.atan2(point2[1] - point1[1], point2[0] - point1[0])
        
    def is_forbidden_direction(self, goal_pos):
        """
        Check if goal position is in forbidden direction (±30 degrees from original position direction)
        
        Args:
            goal_pos (numpy.ndarray): Goal position [x, y]
            
        Returns:
            bool: True if direction is forbidden, False otherwise
        """
        if self.original_position is None:
            return False
            
        # Calculate angle to original position from current position
        angle_to_original = self.calculate_angle(self.current_pos, self.original_position)
        
        # Calculate angle to goal position from current position
        angle_to_goal = self.calculate_angle(self.current_pos, goal_pos)
        
        # Calculate angle difference (normalized to -pi~pi range)
        angle_diff = (angle_to_goal - angle_to_original)
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
            
        # If absolute angle difference is within forbidden range, direction is forbidden
        forbidden_rad = math.radians(self.forbidden_angle_range)
        return abs(angle_diff) < forbidden_rad
    
    def is_previously_visited(self, pos, radius=2.0):
        """
        Check if position is in previously visited area
        
        Args:
            pos (numpy.ndarray): Position to check [x, y]
            radius (float): Radius to consider as visited area
            
        Returns:
            bool: True if area was frequently visited, False otherwise
        """
        for area in self.visited_areas:
            dist = np.linalg.norm(np.array(pos) - area['position'])
            if dist < radius and area['visits'] > 2:  # Forbid revisiting areas visited more than 3 times
                return True
        return False
        
    def is_similar_to_previous_goals(self, pos, threshold=1.5):
        """
        Check if position is similar to previous goal points
        
        Args:
            pos (numpy.ndarray): Position to check [x, y]
            threshold (float): Distance threshold to consider as similar
            
        Returns:
            bool: True if similar to previous goal, False otherwise
        """
        for prev_goal in self.previous_goals:
            dist = np.linalg.norm(np.array(pos) - prev_goal)
            if dist < threshold:
                return True
        return False

    def update_beacons(self, beacons):
        """
        Update beacon positions
        
        Args:
            beacons (list): List of beacon positions [[x1, y1], [x2, y2], ...]
        """
        self.beacons = [np.array(beacon[:2]) for beacon in beacons]  # Use only x, y

    def world_to_grid(self, world_x, world_y):
        """
        Convert world coordinates to grid coordinates
        
        Args:
            world_x (float): World coordinate x
            world_y (float): World coordinate y
            
        Returns:
            tuple: Grid coordinates (grid_x, grid_y)
        """
        grid_x = int((world_x - self.grid_origin[0]) / self.map_resolution)
        grid_y = int((world_y - self.grid_origin[1]) / self.map_resolution)
        return grid_x, grid_y

    def grid_to_world(self, grid_x, grid_y):
        """
        Convert grid coordinates to world coordinates
        
        Args:
            grid_x (int): Grid coordinate x
            grid_y (int): Grid coordinate y
            
        Returns:
            tuple: World coordinates (world_x, world_y)
        """
        world_x = grid_x * self.map_resolution + self.grid_origin[0]
        world_y = grid_y * self.map_resolution + self.grid_origin[1]
        return world_x, world_y

    # TK part 
    def generate_entropy_map(self):  
        """
        creates entropy map from the occupancy grid 

        unknown areas (occ_grid = -1) has highest entropy (1.0)
        free areas (occ_grid = 0) has low entrophy (0.2)
        occupied areas (occ_grid > 50) has no entrophy (0.0)

        """
        
        if self.occupancy_grid is None:
            return None
        
        # Assume occ grid is 0-1
        o_grid = self.occupancy_grid
        entropy_map = -o_grid * np.log2(o_grid + 1e-10) - (1 - o_grid) * np.log2(1 - o_grid + 1e-10)
        entropy_map = np.nan_to_num(entropy_map)

        # entropy_map = np.ones_like(self.occupancy_grid, dtype=float)
        # unknown = (self.occupancy_grid == -1) # unknown
        # free = (self.occupancy_grid == 0)     # free
        # occupied = (self.occupancy_grid > 50)  # above 50 is occupied

        # entropy_map[unknown] = 1.0 # high entropy
        # entropy_map[free] = 0.2    # low entropy
        # entropy_map[occupied] = 0.0 # no entropy


        # gaussian filter for smoothnes (optional)
        entropy_map = gaussian_filter(entropy_map, sigma=2)

        # Store for visualization
        self.entropy_map = entropy_map.copy()

        return entropy_map


    def compute_entropy_gradient(self, entropy_map):
        """
        gradient of entrophy map using sobel filter

        """

        sobel_h = sobel(entropy_map, axis=0) # horizontal gradient
        sobel_v = sobel(entropy_map, axis=1) # vertical gradient

        gradient_y = sobel_h
        gradient_x = sobel_v

        return gradient_x, gradient_y
        
        
    def detect_exploration_boundary(self, entropy_map):
        """
        edge detection using sobel filter / gives you boundary map
        """
        
        # get gradients from entropy map
        gradient_x, gradient_y = self.compute_entropy_gradient(entropy_map)
        gradient_norm = np.sqrt(gradient_x**2 + gradient_y**2)

        # gradient threshold 
        thres =  np.max(gradient_norm) * 0.4 ###### adjust threshold
        boundary_map = (gradient_norm > thres).astype(float)

        # gaussian filter for smoothnes (optional)
        boundary_map = gaussian_filter(boundary_map, sigma=1)

        # Store for visualization
        self.boundary_map = boundary_map.copy()

        return boundary_map


    def select_goal_point(self):
        """
        selects next goal point for the robot - only from known areas
        """
        # robot pos to grid pos 
        robot_grid_x, robot_grid_y = self.world_to_grid(self.current_pos[0], self.current_pos[1])

        # check if grid pos is valid
        if not (0 <= robot_grid_x < self.grid_width and 0 <= robot_grid_y < self.grid_height):
            return None
        
        # find candidate points
        if self.occupancy_grid is None:
            return None
        
        entropy_map = self.generate_entropy_map()
        if entropy_map is None:
            return None
        
        # boundaries based on entrophy
        boundary_map = self.detect_exploration_boundary(entropy_map)

        # mean filter applied to boundary map
        filtered_boundary = cv2.boxFilter(boundary_map, -1, (15, 15)) # 15 by 15 kernel
        # replacing the filtered boundary w max value in 25 by 25 kernel
        max_filtered = cv2.dilate(filtered_boundary, np.ones((25, 25)))
        local_maximas = (filtered_boundary == max_filtered) & (filtered_boundary > 0.3)

        goal_pts = []
        y_idx, x_idx = np.where(local_maximas)

        # Calculate exploration direction (from origin to current position)
        if self.original_position is not None:
            exploration_direction = self.current_pos - self.original_position
            if np.linalg.norm(exploration_direction) > 0.001:
                exploration_direction = exploration_direction / np.linalg.norm(exploration_direction)
            else:
                exploration_direction = np.array([1.0, 0.0])  # Default direction
        else:
            exploration_direction = np.array([1.0, 0.0])

        # LiDAR 스캔 범위 설정 (미터 단위)
        lidar_range = 5.0  # LiDAR의 최대 스캔 범위
        search_radius = min(lidar_range, 1000.0 / self.map_resolution)  # LiDAR 범위와 기존 검색 범위 중 작은 값 사용

        for i in range(len(y_idx)):
            x = x_idx[i]
            y = y_idx[i]

            # Skip occupied cells and unknown cells
            if self.occupancy_grid[y, x] >= 50 or self.occupancy_grid[y, x] == -1:
                continue

            # Convert to world coordinates
            world_x, world_y = self.grid_to_world(x, y)
            world_pos = np.array([world_x, world_y])
            
            # Calculate distance from robot
            d = np.linalg.norm(world_pos - self.current_pos)
            
            # Skip if too close or too far from LiDAR range
            if d < 1.0 or d > lidar_range:
                continue

            # Skip if not in known area
            if not self.is_in_known_area(world_x, world_y):
                continue
            
            # Check if direction is forbidden
            if self.is_forbidden_direction(world_pos):
                continue
                
            # Check if area was previously visited
            if self.is_previously_visited(world_pos):
                continue
                
            # Check if similar to previous goals
            if self.is_similar_to_previous_goals(world_pos):
                continue

            gradient_x, gradient_y = self.compute_entropy_gradient(entropy_map)
            grad_norm = filtered_boundary[y, x]

            d_score = 1.0 - (d / search_radius)
            grad_score = grad_norm / (np.max(gradient_x)**2 + np.max(gradient_y)**2)**0.5

            # Calculate direction score based on exploration direction
            direction_to_point = world_pos - self.current_pos
            if np.linalg.norm(direction_to_point) > 0.001:
                direction_to_point = direction_to_point / np.linalg.norm(direction_to_point)
                direction_alignment = np.dot(direction_to_point, exploration_direction)
                direction_score = max(0, direction_alignment)  # Only reward forward movement
            else:
                direction_score = 0

            # Adjust weights to prefer forward exploration
            d_weight = 0.3  # Distance weight
            grad_weight = 0.3  # Gradient weight
            direction_weight = 0.4  # Direction weight (increased)
            score_total = (d_score * d_weight + 
                         grad_score * grad_weight + 
                         direction_score * direction_weight)

            goal_pts.append((world_x, world_y, score_total))
        
        # if there are no goal points, find any valid known free cell
        if not goal_pts:
            print("No high entropy goal points found. Searching for any known free cell...")
            attempts = 0
            while attempts < 300:  # Increase attempts for better chance of finding valid point
                # Generate random angle and distance for radial search
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(1.0, lidar_range)  # LiDAR 범위 내에서만 검색
                
                # Convert to grid coordinates
                rand_grid_x = int(robot_grid_x + distance * math.cos(angle))
                rand_grid_y = int(robot_grid_y + distance * math.sin(angle))
                
                # Check if within grid bounds
                if (0 <= rand_grid_x < self.grid_width and 
                    0 <= rand_grid_y < self.grid_height):
                    
                    # Filter for known and free cells only
                    if 0 <= self.occupancy_grid[rand_grid_y, rand_grid_x] < 50:
                        world_x, world_y = self.grid_to_world(rand_grid_x, rand_grid_y)
                        world_pos = np.array([world_x, world_y])
                        
                        # Additional checks
                        if (not self.is_forbidden_direction(world_pos) and 
                            not self.is_previously_visited(world_pos) and
                            not self.is_similar_to_previous_goals(world_pos)):
                            print(f"Found random goal in known area: ({world_x}, {world_y})")
                            return world_x, world_y
                    
                attempts += 1
            
            print("Could not find valid goal point. Staying at current position.")
            return self.current_pos[0], self.current_pos[1]
        
        best_goal_pt = max(goal_pts, key=lambda x: x[2])
        
        # Add goal point to previous goals list
        goal_pos = np.array([best_goal_pt[0], best_goal_pt[1]])
        self.previous_goals.append(goal_pos.copy())
        
        # Limit previous goals list size
        if len(self.previous_goals) > self.max_previous_goals:
            self.previous_goals.pop(0)
            
        return best_goal_pt[0], best_goal_pt[1]

    def check_collision(self, x, y):
        """
        Check if a given position collides with obstacles or is in unknown area
        
        Args:
            x (float): World coordinate x
            y (float): World coordinate y
            
        Returns:
            bool: True if collision or unknown area, False otherwise
        """
        if self.occupancy_grid is None:
            return True
        
        grid_x, grid_y = self.world_to_grid(x, y)
        
        # Consider out of grid as collision
        if grid_x < 0 or grid_x >= self.grid_width or grid_y < 0 or grid_y >= self.grid_height:
            return True
        
        # Consider occupied cells (≥50) or unknown cells (-1) as collision
        if self.occupancy_grid[grid_y, grid_x] >= 50 or self.occupancy_grid[grid_y, grid_x] == -1:
            return True
        
        return False
        
    def is_in_known_area(self, x, y):
        """
        Check if a given position is in known area (not unknown or out of bounds)
        
        Args:
            x (float): World coordinate x
            y (float): World coordinate y
            
        Returns:
            bool: True if in known area, False otherwise
        """
        if self.occupancy_grid is None:
            return False
        
        grid_x, grid_y = self.world_to_grid(x, y)
        
        # Out of grid is not a known area
        if grid_x < 0 or grid_x >= self.grid_width or grid_y < 0 or grid_y >= self.grid_height:
            return False
        
        # -1 indicates unknown area
        if self.occupancy_grid[grid_y, grid_x] == -1:
            return False
            
        # Check surrounding cells (ensure not on the edge of known space)
        radius = 2  # cells radius to check
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                ny, nx = grid_y + dy, grid_x + dx
                if (0 <= nx < self.grid_width and 0 <= ny < self.grid_height and 
                    self.occupancy_grid[ny, nx] == -1):
                    return False  # Near unknown area
        
        return True

    def check_path_collision(self, from_x, from_y, to_x, to_y):
        """
        Check if a path between two points collides with obstacles
        
        Args:
            from_x (float): Start point x
            from_y (float): Start point y
            to_x (float): End point x
            to_y (float): End point y
            
        Returns:
            bool: True if collision, False otherwise
        """
        # Calculate distance between the two points
        dist = math.sqrt((to_x - from_x)**2 + (to_y - from_y)**2)
        
        # Determine number of sampling points based on distance and resolution
        # 더 조밀한 샘플링을 위해 해상도의 1/4 크기로 설정
        step_size = self.map_resolution * 0.25
        steps = max(2, int(dist / step_size))
        
        # Inflation radius (in grid cells) - 로봇 크기에 맞게 조정
        inflation_radius = 5  # 로봇 크기에 맞게 조정
        
        # Sample along the path
        for i in range(steps + 1):
            t = i / steps
            x = from_x + t * (to_x - from_x)
            y = from_y + t * (to_y - from_y)
            
            grid_x, grid_y = self.world_to_grid(x, y)
            
            # Check inflated area around the point for obstacles
            for dy in range(-inflation_radius, inflation_radius+1):
                for dx in range(-inflation_radius, inflation_radius+1):
                    nx, ny = grid_x + dx, grid_y + dy
                    
                    # Skip if out of grid
                    if nx < 0 or nx >= self.grid_width or ny < 0 or ny >= self.grid_height:
                        continue
                    
                    # Check if obstacle or unknown
                    if self.occupancy_grid[ny, nx] >= 50 or self.occupancy_grid[ny, nx] == -1:
                        print(f"Collision detected at grid position ({nx}, {ny}) with value {self.occupancy_grid[ny, nx]}")
                        return True  # Collision detected
        
        return False

    def rrt_planning(self, start_pos, goal_pos, max_attempts=3):
        """
        Path planning using the RRT algorithm - constrained to known areas
        
        Args:
            start_pos (numpy.ndarray): Start position [x, y]
            goal_pos (numpy.ndarray): Goal position [x, y]
            max_attempts (int): Maximum number of attempts
            
        Returns:
            list: Path coordinate list [[x1, y1], [x2, y2], ...], None if failed
        """
        self.is_planning = True
        
        # Record start time
        start_time = time.time()
        
        # Ensure start and goal are in known areas
        if not self.is_in_known_area(start_pos[0], start_pos[1]):
            print("Start position is in unknown area.")
            self.is_planning = False
            return None
            
        # If goal point is occupied or in unknown area, find new goal point
        if self.check_collision(goal_pos[0], goal_pos[1]) or not self.is_in_known_area(goal_pos[0], goal_pos[1]):
            print("Goal point is occupied or in unknown area. Selecting another point...")
            goal_x, goal_y = self.select_goal_point()
            if goal_x is None or goal_y is None:
                self.is_planning = False
                return None
            goal_pos = np.array([goal_x, goal_y])
        
        # Create a list of known free grid cells for sampling
        known_free_cells = []
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if 0 < self.occupancy_grid[y, x] < 50:  # Known and free
                    known_free_cells.append((x, y))
        
        if not known_free_cells:
            print("No known free cells available for RRT sampling")
            self.is_planning = False
            return None
        
        # Try RRT for maximum number of attempts
        for attempt in range(max_attempts):
            # Initialize nodes
            start_node = RRTNode(start_pos[0], start_pos[1])
            goal_node = RRTNode(goal_pos[0], goal_pos[1])
            node_list = [start_node]
            
            # Clear visualization data for each attempt
            self.rrt_nodes = [(start_pos[0], start_pos[1])]
            self.rrt_edges = []
            self.rrt_samples = []
            
            for i in range(self.rrt_max_iter):
                # Sample random position
                if random.randint(0, 100) > self.rrt_goal_sample_rate:
                    # Generate random position from known free cells
                    if known_free_cells:
                        # Sample from known free cells
                        rand_grid_x, rand_grid_y = random.choice(known_free_cells)
                        rand_x, rand_y = self.grid_to_world(rand_grid_x, rand_grid_y)
                        
                        # Add small random offset within the cell
                        rand_x += random.uniform(-0.5, 0.5) * self.map_resolution
                        rand_y += random.uniform(-0.5, 0.5) * self.map_resolution
                    else:
                        # Fallback to area around start position
                        rand_x = random.uniform(start_pos[0] - 5.0, start_pos[0] + 5.0)
                        rand_y = random.uniform(start_pos[1] - 5.0, start_pos[1] + 5.0)
                    
                    rand_node = RRTNode(rand_x, rand_y)
                    
                    # Only store valid samples in known areas
                    if self.is_in_known_area(rand_x, rand_y):
                        self.rrt_samples.append((rand_x, rand_y))
                else:
                    # Sample goal position
                    rand_node = RRTNode(goal_pos[0], goal_pos[1])
                    
                    # Store sample for visualization
                    self.rrt_samples.append((goal_pos[0], goal_pos[1]))
                
                # Find nearest node
                nearest_ind = self.get_nearest_node_index(node_list, rand_node)
                nearest_node = node_list[nearest_ind]
                
                # Create new node
                new_node = self.steer(nearest_node, rand_node)
                
                # Skip if new node is in unknown area
                if not self.is_in_known_area(new_node.x, new_node.y):
                    continue
                
                # Check collision
                if self.check_path_collision(nearest_node.x, nearest_node.y, new_node.x, new_node.y):
                    continue
                
                # Add new node
                node_list.append(new_node)
                
                # Store node and edge for visualization
                self.rrt_nodes.append((new_node.x, new_node.y))
                self.rrt_edges.append(((nearest_node.x, nearest_node.y), (new_node.x, new_node.y)))
                
                # Check if reached near goal
                dist_to_goal = math.sqrt((new_node.x - goal_node.x)**2 + (new_node.y - goal_node.y)**2)
                if dist_to_goal <= self.rrt_step_size:
                    # Check if can connect directly to goal in known area
                    if not self.check_path_collision(new_node.x, new_node.y, goal_node.x, goal_node.y):
                        # Connect final node to goal
                        final_node = self.steer(new_node, goal_node)
                        node_list.append(final_node)
                        
                        # Store goal node and edge for visualization
                        self.rrt_nodes.append((goal_node.x, goal_node.y))
                        self.rrt_edges.append(((new_node.x, new_node.y), (goal_node.x, goal_node.y)))
                        
                        # Generate path
                        path = self.generate_path(node_list)
                        
                        # Optimize path
                        path = self.optimize_path(path)
                        
                        self.is_planning = False
                        print(f"RRT path planning successful! (Attempt {attempt+1}/{max_attempts}, iteration {i+1})")
                        return path
                
                # Check timeout
                if time.time() - start_time > 5.0:  # 5 second limit
                    print(f"RRT timeout. (Attempt {attempt+1}/{max_attempts})")
                    break
            
            print(f"RRT attempt {attempt+1}/{max_attempts} failed. Trying again...")
        
        # All attempts failed
        self.is_planning = False
        print("All RRT attempts failed.")
        return None

    def get_nearest_node_index(self, node_list, target_node):
        """
        Return the index of the node closest to the target node
        
        Args:
            node_list (list): List of nodes
            target_node (RRTNode): Target node
            
        Returns:
            int: Index of the closest node
        """
        distances = [(node.x - target_node.x)**2 + (node.y - target_node.y)**2 for node in node_list]
        return distances.index(min(distances))

    def steer(self, from_node, to_node):
        """
        Create a new node by moving from from_node toward to_node by steer_step distance
        
        Args:
            from_node (RRTNode): Start node
            to_node (RRTNode): Target node
            
        Returns:
            RRTNode: New node
        """
        dist = math.sqrt((to_node.x - from_node.x)**2 + (to_node.y - from_node.y)**2)
        
        # If distance is less than step_size, return as is
        if dist < self.rrt_step_size:
            new_node = RRTNode(to_node.x, to_node.y)
        else:
            # Calculate direction vector
            theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
            new_node = RRTNode(
                from_node.x + self.rrt_step_size * math.cos(theta),
                from_node.y + self.rrt_step_size * math.sin(theta)
            )
        
        # Set path and parent
        new_node.path_x = from_node.path_x.copy()
        new_node.path_y = from_node.path_y.copy()
        new_node.path_x.append(new_node.x)
        new_node.path_y.append(new_node.y)
        new_node.parent = from_node
        
        return new_node

    def generate_path(self, node_list):
        """
        Generate path from node list
        
        Args:
            node_list (list): List of nodes
            
        Returns:
            list: Path coordinate list [[x1, y1], [x2, y2], ...]
        """
        # Trace back from last node to start node
        path = []
        node = node_list[-1]  # Last node
        
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        
        # Add start node
        path.append([node.x, node.y])
        
        # Reverse path (start -> goal order)
        path.reverse()
        
        return path

    def optimize_path(self, path):
        """
        Optimize a path to prefer corridor centers and smoothness
        코리도 중심과 매끄러움을 선호하도록 경로 최적화
        
        Args:
            path (list): List of waypoints [[x1, y1], [x2, y2], ...]
            
        Returns:
            list: Optimized path
        """
        if not path or len(path) < 2:
            return path
            
        optimized_path = [path[0]]  # 시작점은 유지
        
        # 중간 웨이포인트 최적화
        for i in range(1, len(path) - 1):
            # 웨이포인트의 여유 공간 계산
            grid_x, grid_y = self.world_to_grid(path[i][0], path[i][1])
            clearance = self.calculate_clearance((grid_y, grid_x))
            
            # 여유 공간이 적으면 코리도 중심으로 조정
            if clearance < self.corridor_threshold:
                adjusted_point = self.adjust_toward_corridor_center(path[i])
                optimized_path.append(adjusted_point)
            else:
                optimized_path.append(path[i])
        
        optimized_path.append(path[-1])  # 목표점은 유지
        
        # 경로 평활화 (불필요한 웨이포인트 제거)
        i = 0
        while i < len(optimized_path) - 2:
            # 두 웨이포인트를 직접 연결했을 때 충돌이 없으면 중간 웨이포인트 제거
            if not self.check_path_collision(
                optimized_path[i][0], optimized_path[i][1],
                optimized_path[i+2][0], optimized_path[i+2][1]
            ):
                optimized_path.pop(i+1)
            else:
                i += 1
                
        return optimized_path
        
    def calculate_entropy_map(self):
        """
        Calculate entropy map from occupancy grid
        엔트로피 맵을 계산하여 탐색 경계를 찾는 데 사용
        
        Returns:
            numpy.ndarray: Entropy map
        """
        # Entropy = -p*log(p) - (1-p)*log(1-p)
        # 0이나 1에 가까운 값에 대해 로그가 정의되지 않으므로 값 클리핑
        p = np.clip(self.occupancy_grid / 100.0, 0.001, 0.999)  # Occupancy grid is 0-100
        entropy = -p * np.log(p) - (1-p) * np.log(1-p)
        
        # 장애물 영역(p > 0.65)은 엔트로피를 0으로 설정
        entropy[p > 0.65] = 0
        
        # 정규화 (0-1 범위로)
        if np.max(entropy) > 0:
            entropy = entropy / np.max(entropy)
            
        self.entropy_map = entropy
        return entropy
        
    def find_frontiers(self):
        """
        Find and cluster frontier cells between known free space and unknown space
        탐색 경계(프론티어)를 찾아 군집화
        
        Returns:
            list: List of frontier clusters, each cluster is a list of (y, x) coordinates
        """
        if self.occupancy_grid is None:
            return []
            
        # 자유 공간(< 50)과 미지 공간(= -1) 식별
        free_cells = (self.occupancy_grid < 50) & (self.occupancy_grid >= 0)
        unknown_cells = self.occupancy_grid == -1
        
        # 미지 공간 팽창
        kernel = np.ones((3, 3))
        dilated_unknown = binary_dilation(unknown_cells, kernel)
        
        # 프론티어 셀 = 자유 공간 + 팽창된 미지 공간 경계
        frontier_cells = free_cells & dilated_unknown
        
        # 프론티어 셀 좌표 추출
        frontier_coords = np.argwhere(frontier_cells)
        
        if len(frontier_coords) == 0:
            return []
        
        # DBSCAN으로 프론티어 셀 군집화
        try:
            clustering = DBSCAN(eps=3, min_samples=self.min_frontier_size).fit(frontier_coords)
            labels = clustering.labels_
            
            # 군집별로 프론티어 셀 분류
            frontier_clusters = []
            for label in np.unique(labels):
                if label != -1:  # 노이즈 포인트 제외
                    cluster = frontier_coords[labels == label]
                    if len(cluster) >= self.min_frontier_size:
                        frontier_clusters.append(cluster)
            
            return frontier_clusters
        except Exception as e:
            print(f"프론티어 군집화 오류: {e}")
            # 오류 발생 시 각 좌표를 개별 군집으로 처리
            if len(frontier_coords) > self.min_frontier_size:
                return [frontier_coords]
            return []
    
    def calculate_information_gain(self, frontier_cluster):
        """
        Calculate expected information gain for a frontier cluster
        프론티어 군집에 대한 예상 정보 획득량 계산
        
        Args:
            frontier_cluster (numpy.ndarray): Array of (y, x) coordinates of frontier points
            
        Returns:
            tuple: (information_gain, center_point)
        """
        # 프론티어 군집의 중심점 계산
        center = np.mean(frontier_cluster, axis=0).astype(int)
        
        # 로봇에서 프론티어까지의 거리 계산
        robot_grid_x, robot_grid_y = self.world_to_grid(self.current_pos[0], self.current_pos[1])
        distance = np.linalg.norm(np.array([robot_grid_y, robot_grid_x]) - center)
        
        # 이 프론티어에서 볼 수 있는 미지 셀 수 추정
        visible_unknown = self.estimate_visible_unknown(center)
        
        # 여유 공간(장애물까지의 거리) 계산
        clearance = self.calculate_clearance(center)
        
        # 여유 공간과 미지 셀 수를 고려한 정보 획득량
        # (중요도: 미지 셀 > 여유 공간 > 거리)
        visible_weight = 1.0
        clearance_weight = 0.5
        distance_weight = -0.3  # 거리가 멀수록 불리
        
        # 모든 값을 0-1 범위로 정규화
        norm_visible = min(1.0, visible_unknown / 100)  # 100개 미지 셀을 최대로 가정
        norm_clearance = min(1.0, clearance / 10)  # 10칸의 여유를 최대로 가정
        norm_distance = 1.0 - min(1.0, distance / (self.sensor_range / self.map_resolution))
        
        # 총 점수 계산
        gain = (norm_visible * visible_weight + 
                norm_clearance * clearance_weight + 
                norm_distance * distance_weight)
        
        # 중심점 그리드 좌표를 월드 좌표로 변환
        center_world_x, center_world_y = self.grid_to_world(center[1], center[0])
        
        return gain, np.array([center_world_x, center_world_y])
    
    def estimate_visible_unknown(self, point):
        """
        Estimate the number of unknown cells visible from a point
        주어진 점에서 볼 수 있는 미지 셀의 수 추정
        
        Args:
            point (numpy.ndarray): (y, x) grid coordinates
            
        Returns:
            int: Number of estimated visible unknown cells
        """
        y, x = point
        h, w = self.occupancy_grid.shape
        
        # 센서 범위를 그리드 단위로 변환
        sensor_radius = int(self.sensor_range / self.map_resolution)
        
        # 범위 내 좌표 생성
        y_grid, x_grid = np.ogrid[-sensor_radius:sensor_radius+1, 
                                -sensor_radius:sensor_radius+1]
        
        # 원형 마스크 생성
        mask = x_grid**2 + y_grid**2 <= sensor_radius**2
        
        # 바운드 적용
        y_min = max(0, y - sensor_radius)
        y_max = min(h, y + sensor_radius + 1)
        x_min = max(0, x - sensor_radius)
        x_max = min(w, x + sensor_radius + 1)
        
        # 마스크 크기 조정
        mask_y_min = max(0, sensor_radius - y)
        mask_y_max = mask_y_min + (y_max - y_min)
        mask_x_min = max(0, sensor_radius - x)
        mask_x_max = mask_x_min + (x_max - x_min)
        
        mask_bounded = mask[mask_y_min:mask_y_max, mask_x_min:mask_x_max]
        
        # 영역 내 점유도 값 추출
        region = self.occupancy_grid[y_min:y_max, x_min:x_max]
        
        # 마스크 적용 및 미지 셀 카운트
        if region.shape[0] > 0 and region.shape[1] > 0 and mask_bounded.shape[0] > 0 and mask_bounded.shape[1] > 0:
            # 미지 셀 (-1) 카운트
            unknown_count = np.sum((region == -1) & mask_bounded)
            return unknown_count
        
        return 0
    
    def calculate_clearance(self, point):
        """
        Calculate the clearance (distance to nearest obstacle) for a point
        주어진 점에서 가장 가까운 장애물까지의 거리 계산
        
        Args:
            point (numpy.ndarray): (y, x) grid coordinates
            
        Returns:
            float: Distance to nearest obstacle in grid cells
        """
        y, x = point
        h, w = self.occupancy_grid.shape
        
        # 탐색 반경 (그리드 셀 단위)
        search_radius = 10
        
        # 바운드 적용
        y_min = max(0, y - search_radius)
        y_max = min(h, y + search_radius + 1)
        x_min = max(0, x - search_radius)
        x_max = min(w, x + search_radius + 1)
        
        # 영역 추출
        region = self.occupancy_grid[y_min:y_max, x_min:x_max]
        
        # 장애물 셀 (점유도 >= 50) 식별
        obstacle_cells = region >= 50
        
        if not np.any(obstacle_cells):
            return search_radius
        
        # 장애물 셀 좌표 추출
        obstacle_coords = np.argwhere(obstacle_cells)
        
        # 중심점에 상대적인 좌표로 변환
        point_rel = np.array([y - y_min, x - x_min])
        
        # 거리 계산
        distances = np.sqrt(np.sum((obstacle_coords - point_rel)**2, axis=1))
        
        # 최소 거리 반환
        min_distance = np.min(distances) if len(distances) > 0 else search_radius
        return min_distance
        
    def select_frontier_goal(self):
        """
        Select the next goal point based on frontier information gain
        프론티어 정보 획득량에 기반한 다음 목표점 선택
        
        Returns:
            numpy.ndarray or None: Selected goal point [x, y] in world coordinates, or None if no valid goal
        """
        # 프론티어 찾기
        frontier_clusters = self.find_frontiers()
        
        if not frontier_clusters:
            print("더 이상 프론티어가 없습니다. 탐험 완료.")
            self.exploration_complete = True
            return None
        
        # 각 프론티어 군집에 대한 정보 획득량 계산
        gains_and_centers = []
        for cluster in frontier_clusters:
            gain, center = self.calculate_information_gain(cluster)
            # 이미 방문한 영역이나 금지된 방향은 제외
            if not self.is_previously_visited(center) and not self.is_forbidden_direction(center):
                gains_and_centers.append((gain, center))
        
        if not gains_and_centers:
            print("유효한 프론티어 목표가 없습니다.")
            return None
        
        # 정보 획득량 내림차순 정렬
        gains_and_centers.sort(key=lambda x: x[0], reverse=True)
        
        # 정보 획득량이 가장 높은 프론티어 중심 반환
        best_gain, best_center = gains_and_centers[0]
        print(f"선택된 프론티어 목표: {best_center}, 정보 획득량: {best_gain}")
        
        return best_center
    
    def adjust_toward_corridor_center(self, waypoint):
        """
        Adjust a waypoint toward the center of the corridor
        경로 상의 웨이포인트를 코리도 중심으로 조정
        
        Args:
            waypoint (numpy.ndarray): Waypoint [x, y] in world coordinates
            
        Returns:
            numpy.ndarray: Adjusted waypoint [x, y] in world coordinates
        """
        # 월드 좌표를 그리드 좌표로 변환
        grid_x, grid_y = self.world_to_grid(waypoint[0], waypoint[1])
        
        # 8방향으로 장애물까지의 거리 계산
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # 상하좌우
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # 대각선
        ]
        
        distances = []
        for dy, dx in directions:
            dist = 0
            while True:
                ny, nx = grid_y + dy * dist, grid_x + dx * dist
                
                # 바운드 체크
                if not (0 <= ny < self.grid_height and 0 <= nx < self.grid_width):
                    break
                
                # 장애물 체크 (점유도 >= 50)
                if self.occupancy_grid[ny, nx] >= 50:
                    break
                
                dist += 1
                
                # 탐색 거리 제한
                if dist > 15:
                    break
            
            distances.append(dist)
        
        # 코리도 중심 방향 벡터 계산
        direction_vector = np.zeros(2)
        for i, (dy, dx) in enumerate(directions):
            # 거리가 짧을수록 더 강한 힘을 적용
            if distances[i] > 0:
                force = 1.0 / distances[i]
                direction_vector[0] += dy * force
                direction_vector[1] += dx * force
        
        # 정규화 및 반전 (장애물로부터 멀어지는 방향)
        if np.linalg.norm(direction_vector) > 0:
            direction_vector = -direction_vector / np.linalg.norm(direction_vector)
            
            # 조정 적용 (최대 2칸)
            adjustment = np.clip(direction_vector * 2, -2, 2)
            adjusted_y = int(grid_y + adjustment[0])
            adjusted_x = int(grid_x + adjustment[1])
            
            # 조정된 좌표가 유효한지 확인
            if (0 <= adjusted_y < self.grid_height and 
                0 <= adjusted_x < self.grid_width and 
                self.occupancy_grid[adjusted_y, adjusted_x] < 50):
                # 그리드 좌표를 월드 좌표로 변환
                world_x, world_y = self.grid_to_world(adjusted_x, adjusted_y)
                return np.array([world_x, world_y])
        
        # 조정이 불가능하면 원래 웨이포인트 반환
        return waypoint
        
    def compute_exploration_coverage(self):
        """
        Calculate exploration coverage and determine if exploration is complete
        탐험 커버리지를 계산하고 탐험 완료 여부 결정
        
        Returns:
            tuple: (explored_percent, is_complete, reason)
        """
        if self.occupancy_grid is None:
            return 0.0, False, "맵이 없음"
            
        # 알려진 셀 (자유 공간 또는 장애물)
        known_cells = np.sum((self.occupancy_grid < 50) & (self.occupancy_grid >= 0) | (self.occupancy_grid >= 50))
        
        # 전체 셀 수 (패딩/외부 영역 제외)
        unknown_cells = np.sum(self.occupancy_grid == -1)
        total_cells = known_cells + unknown_cells
        
        # 패딩/외부 영역이 많으면 총 셀 수를 추정
        if total_cells < 100:
            total_cells = self.grid_width * self.grid_height
            
        # 탐험 퍼센트 계산
        explored_percent = (known_cells / total_cells) if total_cells > 0 else 1.0
        
        # 자유 공간과 장애물 비율 계산
        free_cells = np.sum((self.occupancy_grid < 50) & (self.occupancy_grid >= 0))
        occupied_cells = np.sum(self.occupancy_grid >= 50)
        free_space_ratio = free_cells / (free_cells + occupied_cells) if (free_cells + occupied_cells) > 0 else 0.5
        
        # 경계 맵 생성 및 경계 밀도 계산
        entropy_map = self.calculate_entropy_map()
        boundary_map = self.detect_exploration_boundary(entropy_map)
        boundary_cells = np.sum(boundary_map > 0.3)
        boundary_density = boundary_cells / total_cells if total_cells > 0 else 0.0
        
        # 탐험 완료 조건 검사
        is_complete = False
        reason = ""
        
        # 탐험 퍼센트가 임계값 이상
        if explored_percent >= self.coverage_threshold:
            is_complete = True
            reason = f"탐험 커버리지 {explored_percent:.1%}가 임계값 {self.coverage_threshold:.1%} 이상"
        
        # 경계 밀도가 매우 낮음 (더 이상 탐험할 경계가 거의 없음)
        elif boundary_density < 0.01:
            is_complete = True
            reason = f"경계 밀도 {boundary_density:.1%}가 매우 낮음"
        
        # 자유 공간 비율이 비정상적 (대부분 벽이거나 모두 자유 공간)
        elif free_space_ratio < 0.05 or free_space_ratio > 0.95:
            is_complete = True
            reason = f"자유 공간 비율 {free_space_ratio:.1%}가 비정상적"
        
        # 프론티어가 없음
        elif not self.find_frontiers():
            is_complete = True
            reason = "더 이상 탐험할 프론티어가 없음"
            
        # 상세 통계 출력
        print(f"탐험 통계 - 총 셀: {total_cells}, 알려진 셀: {known_cells}, 미지 셀: {unknown_cells}")
        print(f"탐험 커버리지: {explored_percent:.1%}, 자유 공간 비율: {free_space_ratio:.1%}")
        print(f"경계 밀도: {boundary_density:.1%}, 완료 여부: {is_complete}, 이유: {reason}")
        
        self.exploration_complete = is_complete
        return explored_percent, is_complete, reason
        
    def improved_select_goal_point(self):
        """
        Advanced goal point selection using frontier-based exploration
        프론티어 기반 탐험을 사용한 고급 목표점 선택
        
        Returns:
            tuple or None: (x, y) goal point in world coordinates, or None if exploration is complete
        """
        # 탐험 커버리지 계산 및 완료 여부 확인
        _, is_complete, reason = self.compute_exploration_coverage()
        
        if is_complete:
            print(f"탐험 완료: {reason}")
            self.exploration_complete = True
            return None
            
        # 프론티어 기반으로 목표점 선택
        goal = self.select_frontier_goal()
        
        # 프론티어가 없으면 엔트로피 기반 목표점 선택으로 폴백
        if goal is None:
            print("프론티어 기반 목표점 선택 실패. 엔트로피 기반 방법으로 전환.")
            return self.select_goal_point()
            
        return goal[0], goal[1]

    def plan_and_control(self, dt):
        """
        Plan path and compute control to follow the path
        
        Args:
            dt (float): Time interval for control computation
        
        Returns:
            tuple: (control, goal_point, current_path)
                control (numpy.ndarray): Control input [vx, vy]
                goal_point (numpy.ndarray): Goal point [x, y]
                current_path (list): Current path [[x1, y1], [x2, y2], ...]
        """
        current_time = time.time()
        
        # Check if we're already planning
        if self.is_planning:
            # Check if we've had a path for a while 
            if self.current_path is not None and len(self.current_path) > 0:
                # Use the existing path for control
                control = self.compute_control(self.current_pos, self.current_path, dt)
                goal_point = np.array([self.current_path[-1][0], self.current_path[-1][1]])
                return control, goal_point, self.current_path
            else:
                # No path yet, just return the previous control
                return self.last_control, None, None
        
        # If we've just reached the goal and need to wait
        if self.goal_reached_time is not None:
            dwell_time = current_time - self.goal_reached_time
            if dwell_time < self.goal_dwell_time:
                # Still in dwell time at goal, don't replan yet
                print(f"In goal dwell time. Remaining: {self.goal_dwell_time - dwell_time:.1f}s")
                return np.array([0.0, 0.0]), self.current_path[-1] if self.current_path and len(self.current_path) > 0 else None, self.current_path
        
        # Check if it's too soon to replan
        time_since_planning = current_time - self.planning_time
        if time_since_planning < self.min_replan_time and self.current_path is not None and len(self.current_path) > 0:
            print(f"Too soon to replan. Time since last plan: {time_since_planning:.1f}s")
            control = self.compute_control(self.current_pos, self.current_path, dt)
            goal_point = np.array([self.current_path[-1][0], self.current_path[-1][1]])
            return control, goal_point, self.current_path
            
        # Start planning
        self.is_planning = True
        self.planning_time = current_time
        
        try:
            # Compute exploration coverage
            coverage, is_complete, reason = self.compute_exploration_coverage()
            
            # Check if exploration is complete
            if is_complete:
                print(f"Exploration complete: {reason}")
                self.is_planning = False
                return np.array([0.0, 0.0]), None, None
            
            # Select new goal point using the improved method
            goal_x, goal_y = self.improved_select_goal_point()
            if goal_x is None or goal_y is None:
                print("No goal point available.")
                self.is_planning = False
                return np.array([0.0, 0.0]), None, None
                
            goal_point = np.array([goal_x, goal_y])
            
            # Monitor current goal approach
            if self.monitor_path_execution():
                # Path execution is stuck, clear current path
                self.current_path = None
                self.current_path_index = 0
                print("Path execution detected as stuck. Replanning...")
                
            # Check if existing path is still valid
            reuse_path = False
            if self.current_path is not None and len(self.current_path) > 0:
                current_goal = self.current_path[-1]
                if (np.linalg.norm(np.array(current_goal) - goal_point) < 0.5 and 
                    not self.check_path_collision(self.current_pos[0], self.current_pos[1], current_goal[0], current_goal[1])):
                    print("Current path is still valid. Reusing.")
                    reuse_path = True
            
            if not reuse_path:
                # Plan new path
                print(f"Planning path from {self.current_pos} to {goal_point}")
                path = self.plan_path(self.current_pos[0], self.current_pos[1], goal_point[0], goal_point[1])
                
                if path is None or len(path) == 0:
                    print("Path planning failed. Using direct path.")
                    path = [[self.current_pos[0], self.current_pos[1]], [goal_point[0], goal_point[1]]]
                
                # Apply path optimization with corridor center preference
                path = self.optimize_path(path)
                
                self.current_path = path
                self.current_path_index = 0
            
            # Compute control for the path
            control = self.compute_control(self.current_pos, self.current_path, dt)
            
            # Planning complete
            self.is_planning = False
            
            # Reset goal reached time if we're planning a new path
            self.goal_reached_time = None
            self.final_goal_reached = False
            
            # Return control, goal point, and path
            goal_point = np.array([self.current_path[-1][0], self.current_path[-1][1]])
            return control, goal_point, self.current_path
            
        except Exception as e:
            print(f"Error in planning: {str(e)}")
            self.is_planning = False
            # In case of error, return last control
            return self.last_control, None, None
            
