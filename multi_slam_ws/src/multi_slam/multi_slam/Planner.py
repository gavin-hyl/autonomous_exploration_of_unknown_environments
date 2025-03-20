import numpy as np
import math
import cv2
from scipy.ndimage import sobel, gaussian_filter
from scipy.spatial import KDTree
from typing import List, Tuple, Optional
import random
import time
import logging


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
                 beacon_attraction_radius=3.0):
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
        Optimize path (remove unnecessary intermediate nodes)
        
        Args:
            path (list): Path coordinate list [[x1, y1], [x2, y2], ...]
            
        Returns:
            list: Optimized path coordinate list
        """
        if path is None or len(path) <= 2:
            return path
        
        optimized_path = [path[0]]  # Add start point
        i = 0
        
        while i < len(path) - 1:
            # Check next points from current point
            for j in range(len(path) - 1, i, -1):
                # Check if direct connection is possible
                if not self.check_path_collision(path[i][0], path[i][1], path[j][0], path[j][1]):
                    # If direct connection is possible, add that point and skip intermediate points
                    optimized_path.append(path[j])
                    i = j
                    break
            else:
                # If direct connection is not possible, add next point
                i += 1
                if i < len(path):
                    optimized_path.append(path[i])
        
        return optimized_path

    ##################################################

    def compute_control(self, current_pos, path, dt):
        """
        return control input to follow a given path (using PD controller)
        
        """
        
        if path is None or len(path) < 2:
            print(f"Warning: Path is {path} - returning zero control")
            return np.array([0.0, 0.0])
        
        # Ensure current path index is valid
        if self.current_path_index >= len(path):
            print(f"Warning: Path index {self.current_path_index} >= path length {len(path)}")
            self.current_path_index = len(path) - 1

        target_pos = np.array(path[self.current_path_index])
        
        print(f"Path control - current: {current_pos}, target: {target_pos}, distance: {np.linalg.norm(target_pos - current_pos)}, index: {self.current_path_index}/{len(path)}")

        # Determine if this is the final waypoint in the path
        is_final_waypoint = (self.current_path_index == len(path) - 1)
        
        # Use different distance thresholds for final waypoint vs. intermediate waypoints
        dist_threshold = self.goal_reached_dist if is_final_waypoint else self.waypoint_reached_dist
        
        print(f"Is final waypoint: {is_final_waypoint}, threshold: {dist_threshold}")

        # reached target point
        if np.linalg.norm(target_pos - current_pos) < dist_threshold:
            # If this is the final waypoint, mark it
            if is_final_waypoint:
                if not self.final_goal_reached:
                    self.final_goal_reached = True
                    self.goal_reached_time = time.time()
                    print(f"Final goal reached at: {target_pos}, time: {self.goal_reached_time}")
                
                # If we're at the final goal, slow down more significantly
                slow_factor = 0.5
                error = target_pos - current_pos
                d_error = (error - self.prev_error) / dt if dt > 0 else np.array([0.0, 0.0])
                self.prev_error = error.copy()
                control_input = self.pd_p_gain * error * slow_factor + self.pd_d_gain * d_error * slow_factor
                
                # Ensure minimum control to maintain position
                control_norm = np.linalg.norm(control_input)
                if control_norm < 0.05:  # Very small control
                    print(f"Very small control at goal: {control_input}, setting to zero")
                    control_input = np.array([0.0, 0.0])  # Just stop
                
                self.last_control = control_input.copy()
                return control_input
            else:
                # Move to next waypoint in the path
                self.current_path_index += 1
                # Reset the goal reached flag when starting to move to a new waypoint
                self.final_goal_reached = False
                self.goal_reached_time = None
                
                # Update target to the new waypoint
                if self.current_path_index < len(path):
                    target_pos = np.array(path[self.current_path_index])
                    print(f"Moving to next waypoint: {target_pos}, index: {self.current_path_index}")
                else:
                    self.current_path_index = len(path) - 1
                    print("Reached end of path, stopping")
                    return np.array([0.0, 0.0])
        
        # Regular controller logic for following the path
        error = target_pos - current_pos

        d_error = (error - self.prev_error) / dt if dt > 0 else np.array([0.0, 0.0])
        self.prev_error = error.copy()

        # pd control 
        control_input = self.pd_p_gain * error + self.pd_d_gain * d_error
        
        print(f"PD control: p_gain={self.pd_p_gain}, d_gain={self.pd_d_gain}")
        print(f"Error: {error}, d_error: {d_error}")
        print(f"Raw control: {control_input}")

        # control input clipping 
        max_speed = 5.0  # 증가된 최대 속도
        control_norm = np.linalg.norm(control_input)

        if control_norm > max_speed:
            control_input = control_input / control_norm * max_speed
            print(f"Control clipped to: {control_input}")
        
        # 최소 제어 입력 설정 - 로봇이 너무 천천히 움직이지 않도록
        min_control_norm = 0.2  # 최소 제어 입력 크기
        if 0 < control_norm < min_control_norm:
            control_input = control_input / control_norm * min_control_norm
            print(f"Control boosted to minimum: {control_input}")

        self.last_control = control_input.copy()
        print(f"Final control output: {control_input}, norm: {np.linalg.norm(control_input)}")

        return control_input
       

    def plan_and_control(self, dt):
        """
        Path planning and control input calculation
        
        Args:
            dt (float): Time interval
            
        Returns:
            tuple: (control_input, goal_point, path)
                - control_input: Velocity command [vx, vy]
                - goal_point: Current goal point
                - path: Planned path
        """
        if self.occupancy_grid is None:
            return None, None, None
        
        # Define replanning conditions
        need_new_plan = (
            self.current_path is None or  # No path exists
            len(self.current_path) == 0 or  # Empty path
            (  # Regular replanning condition with timing
                time.time() - self.planning_time > self.min_replan_time and  # Minimum replanning time passed
                not self.final_goal_reached  # Not at the final goal
            ) or
            (  # Final goal reached and dwell time passed
                self.final_goal_reached and
                self.goal_reached_time is not None and
                time.time() - self.goal_reached_time > self.goal_dwell_time
            )
        )
        
        # If new plan is needed
        if need_new_plan:
            # If it is already planning, keep previous control
            if self.is_planning:
                goal_point = np.array([self.current_path[-1][0], self.current_path[-1][1]]) if self.current_path and len(self.current_path) > 0 else None
                return self.last_control, goal_point, self.current_path
            
            # Reset goal reached flags
            self.final_goal_reached = False
            self.goal_reached_time = None
            
            # Select new goal point
            goal_x, goal_y = self.select_goal_point()
            if goal_x is None or goal_y is None:
                return None, None, None
            
            goal_pos = np.array([goal_x, goal_y])
            print(f"New goal point: {goal_pos}, planning time: {time.time()}")

            # path planner using RRT
            path = self.rrt_planning(self.current_pos, goal_pos)

            if path is None:
                return self.last_control, goal_pos, []

            self.current_path = path
            self.current_path_index = 0
            self.planning_time = time.time()

            control = self.compute_control(self.current_pos, self.current_path, dt)
        else:
            # Continue with existing path
            control = self.compute_control(self.current_pos, self.current_path, dt)
        
        goal_point = np.array([self.current_path[-1][0], self.current_path[-1][1]]) if self.current_path and len(self.current_path) > 0 else None
        return control, goal_point, self.current_path
            
    def compute_exploration_coverage(self):
        """
        Compute the percentage of the map that has been explored
        
        Returns:
            float: Percentage of explored area (0.0 to 1.0)
            bool: True if exploration is considered complete
        """
        if self.occupancy_grid is None:
            return 0.0, False
        
        # Count cells
        total_cells = self.grid_width * self.grid_height
        unknown_cells = np.sum(self.occupancy_grid == -1)
        occupied_cells = np.sum(self.occupancy_grid >= 50)
        free_cells = np.sum((self.occupancy_grid >= 0) & (self.occupancy_grid < 50))
        
        # Calculate exploration percentage
        explored_percent = 1.0 - (unknown_cells / total_cells)
        
        # Check boundary conditions
        boundary_map = self.detect_exploration_boundary(self.generate_entropy_map())
        boundary_count = np.sum(boundary_map > 0.3)
        
        # Calculate coverage metrics
        free_space_ratio = free_cells / (free_cells + occupied_cells) if (free_cells + occupied_cells) > 0 else 0
        boundary_density = boundary_count / total_cells
        
        print(f"Exploration stats:")
        print(f"- Total cells: {total_cells}")
        print(f"- Unknown cells: {unknown_cells}")
        print(f"- Occupied cells: {occupied_cells}")
        print(f"- Free cells: {free_cells}")
        print(f"- Explored percentage: {explored_percent*100:.1f}%")
        print(f"- Free space ratio: {free_space_ratio*100:.1f}%")
        print(f"- Boundary density: {boundary_density*100:.1f}%")
        
        # Exploration is complete if:
        # 1. High exploration percentage (>95%)
        # 2. Low boundary density (<0.1%)
        # 3. Reasonable free space ratio (20-80%)
        is_complete = (
            explored_percent > 0.95 and  # Most of the map is explored
            boundary_density < 0.001 and  # Few exploration boundaries
            0.2 <= free_space_ratio <= 0.8  # Reasonable mix of free and occupied space
        )
        
        if is_complete:
            print("Exploration complete! All criteria met.")
        else:
            print("Exploration incomplete:")
            if explored_percent <= 0.95:
                print("- Not enough area explored")
            if boundary_density >= 0.001:
                print("- Too many exploration boundaries")
            if free_space_ratio < 0.2 or free_space_ratio > 0.8:
                print("- Unreasonable free space ratio")
        
        return explored_percent, is_complete
        
