# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.ndimage import gaussian_filter, sobel, convolve
# import math
# import random

# class Planner:
#     """
#     Entropy-based autonomous exploration planner.
    
#     Computes information-rich areas in the environment for exploration,
#     generates paths to selected goals using RRT, and provides control signals
#     to follow these paths.
#     """
    
#     def __init__(
#         self,
#         map_resolution=0.1,
#         rrt_step_size=0.5,
#         rrt_max_iter=1000,
#         rrt_goal_sample_rate=5,
#         rrt_connect_circle_dist=0.5,
#         pd_p_gain=1.0,
#         pd_d_gain=0.1,
#         entropy_weight=1.0,
#         beacon_weight=2.0,
#         beacon_attraction_radius=3.0,
#         goal_persistence_time=10.0,  # Time to keep trying a goal (seconds)
#         goal_reached_threshold=0.5   # Distance to consider a goal reached (meters)
#         ):
#         """
#         Initialize the planner with exploration and control parameters.
        
#         Args:
#             map_resolution: Resolution of the occupancy grid (meters per cell)
#             rrt_step_size: Step size for RRT exploration
#             rrt_max_iter: Maximum iterations for RRT search
#             rrt_goal_sample_rate: Percentage chance to sample the goal directly in RRT
#             rrt_connect_circle_dist: Maximum distance to connect nodes in RRT
#             pd_p_gain: Proportional gain for the PD controller
#             pd_d_gain: Derivative gain for the PD controller
#             entropy_weight: Weight for entropy gradient in goal selection
#             beacon_weight: Weight for beacon attraction in goal selection
#             beacon_attraction_radius: Radius within which beacons attract the robot
#             goal_persistence_time: Time to keep trying a goal (seconds)
#             goal_reached_threshold: Distance to consider a goal reached (meters)
#         """
#         # Map parameters
#         self.map_grid = None
#         self.map_resolution = map_resolution
#         self.map_origin = None
        
#         # RRT parameters
#         self.rrt_step_size = rrt_step_size
#         self.rrt_max_iter = rrt_max_iter
#         self.rrt_goal_sample_rate = rrt_goal_sample_rate
#         self.rrt_connect_circle_dist = rrt_connect_circle_dist
        
#         # Controller parameters
#         self.pd_p_gain = pd_p_gain
#         self.pd_d_gain = pd_d_gain
        
#         # Exploration parameters
#         self.entropy_weight = entropy_weight
#         self.beacon_weight = beacon_weight
#         self.beacon_attraction_radius = beacon_attraction_radius
        
#         # Robot state
#         self.robot_position = np.array([0, 0, 0])
#         self.prev_error = np.array([0, 0])
        
#         # Path following
#         self.current_path = []
#         self.current_path_index = 0
#         self.waypoint_threshold = 0.3  # Distance to consider a waypoint reached
        
#         # Goal management
#         self.current_goal = None
#         self.goal_start_time = None
#         self.goal_persistence_time = goal_persistence_time
#         self.goal_reached_threshold = goal_reached_threshold
#         self.goal_attempt_count = 0
#         self.max_goal_attempts = 3  # Maximum attempts to reach a goal before selecting a new one
        
#         # Beacons
#         self.beacon_positions = []
        
#         # Maps
#         self.entropy_map = None
#         self.boundary_map = None
#         self.gradient_magnitude = None
        
#         # For visualization
#         self.rrt_nodes = []
#         self.rrt_edges = []
#         self.rrt_samples = []
            
#     def update_map(self, occupancy_grid, map_origin, map_resolution):
#         """
#         Update the map used for planning.
        
#         Args:
#             occupancy_grid: 2D numpy array of occupancy probabilities (0-1)
#             map_origin: (x, y) tuple of map origin in world coordinates
#             map_resolution: Resolution of the map in meters per cell
#         """
#         self.map_grid = occupancy_grid
#         self.map_origin = map_origin
#         self.map_resolution = map_resolution
        
#         # Clear previous computed maps
#         self.entropy_map = None
#         self.boundary_map = None
#         self.gradient_magnitude = None
        
#     def update_position(self, position):
#         """
#         Update the robot's position.
        
#         Args:
#             position: [x, y, theta] numpy array of robot position
#         """
#         self.robot_position = position
        
#     def update_beacons(self, beacon_positions):
#         """
#         Update the known beacon positions.
        
#         Args:
#             beacon_positions: List of [x, y, z] positions of beacons
#         """
#         self.beacon_positions = beacon_positions
    
#     # def plan_and_control(self, dt, current_time=None):
#     #     """
#     #     Generate a plan and control signal for autonomous exploration.
        
#     #     Args:
#     #         dt: Time step for control
#     #         current_time: Current simulation time (for goal persistence)
            
#     #     Returns:
#     #         (control_input, goal_point, path): 
#     #             control_input: [vx, vy] numpy array of control velocities
#     #             goal_point: [x, y] numpy array of selected goal position
#     #             path: List of [x, y] points along the planned path
#     #     """
#     #     if current_time is None:
#     #         current_time = 0.0
        
#     #     # Check if we should select a new goal
#     #     new_goal_needed = False
        
#     #     # Case 1: No current goal
#     #     if self.current_goal is None:
#     #         new_goal_needed = True
        
#     #     # Case 2: Goal reached - simplified condition
#     #     elif np.linalg.norm(self.robot_position[:2] - self.current_goal) < self.goal_reached_threshold:
#     #         # Goal is reached - always get a new one
#     #         print(f"DEBUG: Goal reached with distance {np.linalg.norm(self.robot_position[:2] - self.current_goal):.2f}")
#     #         new_goal_needed = True
#     #         self.goal_attempt_count = 0
        
#     #     # Case 3: Goal persistence timeout
#     #     elif self.goal_start_time is not None and (current_time - self.goal_start_time) > self.goal_persistence_time:
#     #         self.goal_attempt_count += 1
            
#     #         # If we've tried too many times, find a new goal
#     #         if self.goal_attempt_count >= self.max_goal_attempts:
#     #             new_goal_needed = True
#     #             self.goal_attempt_count = 0
#     #         else:
#     #             # Try again with the same goal
#     #             # Reset the goal start time to extend the timeout
#     #             self.goal_start_time = current_time
        
#     #     # Select a new goal if needed
#     #     if new_goal_needed:
#     #         max_attempts = 10  # Maximum attempts to find a suitable goal
#     #         goal_point = None
            
#     #         for attempt in range(max_attempts):
#     #             goal_point = self.select_exploration_goal()
                
#     #             if goal_point is None:
#     #                 print(f"DEBUG: No goal point found on attempt {attempt+1}/{max_attempts}")
#     #                 continue
                    
#     #             # Don't select goals too close to the current position
#     #             if np.linalg.norm(self.robot_position[:2] - goal_point) < 2.0:
#     #                 print(f"DEBUG: Goal too close on attempt {attempt+1}/{max_attempts}, distance: {np.linalg.norm(self.robot_position[:2] - goal_point):.2f}")
#     #                 continue
                
#     #             # Goal is good, break the loop
#     #             print(f"DEBUG: Found suitable goal point on attempt {attempt+1}")
#     #             break
            
#     #         # If we still don't have a goal after all attempts, try a random one
#     #         if goal_point is None:
#     #             print("DEBUG: Using random goal as fallback")
#     #             # Choose a random direction and distance
#     #             angle = np.random.uniform(0, 2 * np.pi)
#     #             distance = np.random.uniform(3.0, 8.0)  # Between 3 and 8 meters
#     #             goal_point = self.robot_position[:2] + np.array([
#     #                 distance * np.cos(angle),
#     #                 distance * np.sin(angle)
#     #             ])
                
#     #             # Check if the random goal is in a free space
#     #             i, j = self.world_to_grid(goal_point[0], goal_point[1])
#     #             if hasattr(self, 'map_grid') and self.map_grid is not None:
#     #                 # Try up to 20 random angles to find a free space
#     #                 for _ in range(20):
#     #                     if i >= 0 and i < self.map_grid.shape[1] and j >= 0 and j < self.map_grid.shape[0]:
#     #                         if self.map_grid[j, i] < 0.5:  # Free space
#     #                             break
                        
#     #                     # Try another angle
#     #                     angle = np.random.uniform(0, 2 * np.pi)
#     #                     goal_point = self.robot_position[:2] + np.array([
#     #                         distance * np.cos(angle),
#     #                         distance * np.sin(angle)
#     #                     ])
#     #                     i, j = self.world_to_grid(goal_point[0], goal_point[1])
            
#     #         # Update goal state
#     #         self.current_goal = goal_point
#     #         self.goal_start_time = current_time
#     #         print(f"DEBUG: New goal set to {goal_point}")
#     #     else:
#     #         # Use existing goal
#     #         goal_point = self.current_goal
        
#     #     # Plan a path to the goal
#     #     path = self.plan_path(self.robot_position[:2], goal_point)

#     #     # Store the current path
#     #     if path and len(path) > 1:
#     #         self.current_path = path
#     #         self.current_path_index = 1  # Start with the second point (first after start)
            
#     #         # Generate control to follow the path - ONLY if we have a valid path
#     #         control_input = self.generate_control(dt)
#     #         return control_input, goal_point, path
#     #     else:
#     #         # No valid path found
#     #         self.goal_attempt_count += 1
            
#     #         # Clear the current path to prevent using old path data
#     #         self.current_path = []
            
#     #         # If we've tried too many times, find a new goal next time
#     #         if self.goal_attempt_count >= self.max_goal_attempts:
#     #             print("DEBUG: No valid path found after multiple attempts, will select new goal next time")
#     #             self.current_goal = None
            
#     #         # Return no control input when path planning fails
#     #         return None, self.current_goal, []
    
#     def plan_and_control(self, dt, current_time=None):
#         """
#         Generate a plan and control signal for autonomous exploration.
        
#         Args:
#             dt: Time step for control
#             current_time: Current simulation time (for goal persistence)
            
#         Returns:
#             (control_input, goal_point, path): 
#                 control_input: [vx, vy] numpy array of control velocities
#                 goal_point: [x, y] numpy array of selected goal position
#                 path: List of [x, y] points along the planned path
#         """
#         if current_time is None:
#             current_time = 0.0
        
#         # Check if we should select a new goal
#         new_goal_needed = False
        
#         # Case 1: No current goal
#         if self.current_goal is None:
#             new_goal_needed = True
#             print("DEBUG: No current goal, selecting new goal")
        
#         # Case 2: Goal reached - simplified condition
#         elif np.linalg.norm(self.robot_position[:2] - self.current_goal) < self.goal_reached_threshold:
#             # Goal is reached - always get a new one
#             print(f"DEBUG: Goal reached with distance {np.linalg.norm(self.robot_position[:2] - self.current_goal):.2f}")
#             new_goal_needed = True
#             self.goal_attempt_count = 0
        
#         # Case 3: Goal persistence timeout
#         elif self.goal_start_time is not None and (current_time - self.goal_start_time) > self.goal_persistence_time:
#             self.goal_attempt_count += 1
#             print(f"DEBUG: Goal timeout #{self.goal_attempt_count}, attempts limit: {self.max_goal_attempts}")
            
#             # If we've tried too many times, find a new goal
#             if self.goal_attempt_count >= self.max_goal_attempts:
#                 print("DEBUG: Max attempts reached, selecting new goal")
#                 new_goal_needed = True
#                 self.goal_attempt_count = 0
#             else:
#                 # Try again with the same goal but ensure it's actually reachable
#                 path_test = self.plan_path(self.robot_position[:2], self.current_goal)
#                 if not path_test or len(path_test) <= 1:
#                     print("DEBUG: Goal appears unreachable, selecting new goal")
#                     new_goal_needed = True
#                     self.goal_attempt_count = 0
#                 else:
#                     # Reset the goal start time to extend the timeout
#                     self.goal_start_time = current_time
#                     print(f"DEBUG: Goal still appears reachable, continuing with attempt #{self.goal_attempt_count}")
        
#         # Select a new goal if needed
#         if new_goal_needed:
#             max_attempts = 10  # Maximum attempts to find a suitable goal
#             goal_point = None
            
#             for attempt in range(max_attempts):
#                 goal_point = self.select_exploration_goal()
                
#                 if goal_point is None:
#                     print(f"DEBUG: No goal point found on attempt {attempt+1}/{max_attempts}")
#                     continue
                    
#                 # Don't select goals too close to the current position
#                 if np.linalg.norm(self.robot_position[:2] - goal_point) < 2.0:
#                     print(f"DEBUG: Goal too close on attempt {attempt+1}/{max_attempts}, distance: {np.linalg.norm(self.robot_position[:2] - goal_point):.2f}")
#                     continue
                
#                 # Before committing to a goal, check if it's reachable
#                 test_path = self.plan_path(self.robot_position[:2], goal_point)
#                 if not test_path or len(test_path) <= 1:
#                     print(f"DEBUG: Goal unreachable on attempt {attempt+1}/{max_attempts}")
#                     continue
                    
#                 # Goal is good, break the loop
#                 print(f"DEBUG: Found suitable goal point on attempt {attempt+1}")
#                 break
            
#             # If we still don't have a goal after all attempts, try a random one
#             if goal_point is None:
#                 print("DEBUG: Using random goal as fallback")
#                 # Choose a random direction and distance
#                 angle = np.random.uniform(0, 2 * np.pi)
#                 distance = np.random.uniform(3.0, 8.0)  # Between 3 and 8 meters
#                 goal_point = self.robot_position[:2] + np.array([
#                     distance * np.cos(angle),
#                     distance * np.sin(angle)
#                 ])
                
#                 # Check if the random goal is in a free space
#                 i, j = self.world_to_grid(goal_point[0], goal_point[1])
#                 if hasattr(self, 'map_grid') and self.map_grid is not None:
#                     # Try up to 20 random angles to find a free space
#                     for _ in range(20):
#                         if i >= 0 and i < self.map_grid.shape[1] and j >= 0 and j < self.map_grid.shape[0]:
#                             if self.map_grid[j, i] < 0.5:  # Free space
#                                 break
                        
#                         # Try another angle
#                         angle = np.random.uniform(0, 2 * np.pi)
#                         goal_point = self.robot_position[:2] + np.array([
#                             distance * np.cos(angle),
#                             distance * np.sin(angle)
#                         ])
#                         i, j = self.world_to_grid(goal_point[0], goal_point[1])
                    
#                     # Final check for reachability of random goal
#                     test_path = self.plan_path(self.robot_position[:2], goal_point)
#                     if not test_path or len(test_path) <= 1:
#                         print("DEBUG: Random goal is unreachable, will use direct path as fallback")
            
#             # Update goal state
#             self.current_goal = goal_point
#             self.goal_start_time = current_time
#             self.goal_attempt_count = 0
#             print(f"DEBUG: New goal set to ({goal_point[0]:.2f}, {goal_point[1]:.2f})")
#         else:
#             # Use existing goal
#             goal_point = self.current_goal
#             print(f"DEBUG: Continuing with existing goal ({goal_point[0]:.2f}, {goal_point[1]:.2f})")
        
#         # Plan a path to the goal
#         path = self.plan_path(self.robot_position[:2], goal_point)
        
#         # Debug path information
#         if path:
#             print(f"DEBUG: Path planned with {len(path)} waypoints")
#         else:
#             print("DEBUG: Failed to plan a path")

#         # Store the current path
#         if path and len(path) > 1:
#             self.current_path = path
#             self.current_path_index = 1  # Start with the second point (first after start)
            
#             # Generate control to follow the path - ONLY if we have a valid path
#             control_input = self.generate_control(dt)
#             print(f"DEBUG: Generated control: ({control_input[0]:.2f}, {control_input[1]:.2f})")
#             return control_input, goal_point, path
#         else:
#             # No valid path found
#             self.goal_attempt_count += 1
#             print(f"DEBUG: No valid path found, attempt #{self.goal_attempt_count}")
            
#             # Clear the current path to prevent using old path data
#             self.current_path = []
            
#             # If we've tried too many times, find a new goal next time
#             if self.goal_attempt_count >= self.max_goal_attempts:
#                 print("DEBUG: No valid path found after multiple attempts, will select new goal next time")
#                 self.current_goal = None
            
#             # Return no control input when path planning fails
#             return None, self.current_goal, []

#     def generate_entropy_map(self):
#         """
#         Generate an entropy map from the occupancy grid.
        
#         Returns:
#             entropy_map: 2D numpy array of information entropy
#         """
#         if self.map_grid is None:
#             return None
        
#         # Ensure probabilities are in [0.001, 0.999] range to avoid log(0) issues
#         p = np.clip(self.map_grid, 0.001, 0.999)
        
#         # Calculate entropy: -p*log(p) - (1-p)*log(1-p)
#         entropy = -p * np.log(p) - (1 - p) * np.log(1 - p)
        
#         # Apply mean filtering
#         kernel = np.ones((3, 3)) / 9.0
#         entropy_smooth = convolve(entropy, kernel)
        
#         self.entropy_map = entropy_smooth
#         return entropy_smooth
    
#     def compute_entropy_gradient(self, entropy_map=None):
#         """
#         Compute the gradient of the entropy map.
        
#         Args:
#             entropy_map: Optional entropy map to use (generated if None)
            
#         Returns:
#             (gradient_x, gradient_y, gradient_magnitude): Gradient components and magnitude
#         """
#         if entropy_map is None:
#             entropy_map = self.generate_entropy_map()
            
#         if entropy_map is None:
#             return None, None, None
        
#         # Apply Sobel filter
#         gradient_x = sobel(entropy_map, axis=1)
#         gradient_y = sobel(entropy_map, axis=0)
        
#         # Compute gradient magnitude
#         gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
#         # Normalize to [0, 1]
#         max_magnitude = np.max(gradient_magnitude)
#         if max_magnitude > 0:
#             gradient_magnitude = gradient_magnitude / max_magnitude
        
#         self.gradient_magnitude = gradient_magnitude
#         return gradient_x, gradient_y, gradient_magnitude
    
#     def generate_boundary_map(self):
#         """
#         Generate a boundary map that highlights transitions between known and unknown areas.
        
#         Returns:
#             boundary_map: 2D numpy array of boundary values
#         """
#         if self.map_grid is None:
#             return None
        
#         # Compute entropy gradient if not already done
#         if self.gradient_magnitude is None:
#             _, _, self.gradient_magnitude = self.compute_entropy_gradient()
        
#         # The gradient magnitude already represents boundaries
#         self.boundary_map = self.gradient_magnitude
        
#         return self.boundary_map
    
#     def select_exploration_goal(self):
#         """
#         Select the best exploration goal based on entropy gradient and beacon positions.
        
#         Returns:
#             goal_point: [x, y] numpy array of selected goal position
#         """
#         # Generate boundary map if not available
#         if self.boundary_map is None:
#             self.generate_boundary_map()
            
#         if self.boundary_map is None or self.map_grid is None:
#             print("ERROR: Boundary map or map grid is None, cannot select goal")
#             return None
        
#         # Log current state
#         print(f"DEBUG: Robot position: [{self.robot_position[0]:.2f}, {self.robot_position[1]:.2f}]")
#         print(f"DEBUG: Map shape: {self.map_grid.shape}, Boundary map shape: {self.boundary_map.shape}")
        
#         # Check if boundary map has any high values
#         high_boundary_cells = np.where(self.boundary_map > 0.3)  # Lower threshold to find more candidates
#         print(f"DEBUG: Number of high boundary cells: {len(high_boundary_cells[0])}")
        
#         # Create a copy to avoid modifying the original
#         score_map = self.boundary_map.copy()
        
#         # Get grid dimensions
#         height, width = score_map.shape
        
#         # Robot position in grid coordinates
#         robot_x, robot_y = self.world_to_grid(self.robot_position[0], self.robot_position[1])
        
#         # Compute scores for each cell based on distance and beacon proximity
#         best_score = -float('inf')
#         best_cell = None
        
#         # Sample a subset of cells to evaluate (for efficiency)
#         sample_rate = 0.1  # Increase from 0.05 to 0.1 to evaluate 10% of cells
#         num_samples = int(height * width * sample_rate)
        
#         # Skip if grid is very small
#         if num_samples < 10:
#             num_samples = min(height * width, 100)
        
#         # Random sampling for evaluation
#         candidate_cells = []
        
#         # Only consider cells with high boundary values
#         high_boundary_threshold = 0.3  # Lower threshold from 0.5 to 0.3
#         high_boundary_cells = np.where(self.boundary_map > high_boundary_threshold)
        
#         # If we have high boundary cells, prioritize those
#         if len(high_boundary_cells[0]) > 0:
#             high_boundary_indices = list(zip(high_boundary_cells[0], high_boundary_cells[1]))
#             if len(high_boundary_indices) > num_samples:
#                 candidate_cells = random.sample(high_boundary_indices, num_samples)
#             else:
#                 candidate_cells = high_boundary_indices
        
#         # If we need more candidates, add random cells
#         if len(candidate_cells) < num_samples:
#             additional_samples = num_samples - len(candidate_cells)
#             for _ in range(additional_samples):
#                 i = random.randint(0, height - 1)
#                 j = random.randint(0, width - 1)
#                 candidate_cells.append((i, j))
        
#         # If still no candidates, create some random positions around the robot
#         if not candidate_cells:
#             print("DEBUG: No candidate cells found, generating random positions")
#             for _ in range(20):
#                 # Random distance between 3 and 10 meters
#                 distance = random.uniform(3.0, 10.0)
#                 # Random angle
#                 angle = random.uniform(0, 2 * np.pi)
#                 # Calculate position
#                 x = self.robot_position[0] + distance * np.cos(angle)
#                 y = self.robot_position[1] + distance * np.sin(angle)
#                 # Convert to grid coordinates
#                 i, j = self.world_to_grid(x, y)
#                 # Ensure within grid bounds
#                 if 0 <= i < width and 0 <= j < height:
#                     candidate_cells.append((j, i))  # Note: grid indices are (j,i)
        
#         print(f"DEBUG: Evaluating {len(candidate_cells)} candidate cells")
        
#         for i, j in candidate_cells:
#             # Skip occupied cells
#             if i >= 0 and i < height and j >= 0 and j < width and self.map_grid[i, j] > 0.7:
#                 continue
                
#             # Get world coordinates
#             cell_x, cell_y = self.grid_to_world(j, i)
            
#             # Compute distance to robot
#             dx = cell_x - self.robot_position[0]
#             dy = cell_y - self.robot_position[1]
#             distance = math.sqrt(dx*dx + dy*dy)
            
#             # Skip cells that are too close to the robot
#             if distance < 0.5:
#                 continue
                
#             # Skip cells that are too far from the robot
#             if distance > 12.0:  # Increased from 10.0 to 12.0
#                 continue
            
#             # Computer score based on boundary value and distance
#             score = (self.boundary_map[i, j] * self.entropy_weight) / max(0.5, distance)
            
#             # Add beacon attraction/repulsion factor
#             beacon_factor = 0
#             for beacon in self.beacon_positions:
#                 beacon_dist = math.sqrt((cell_x - beacon[0])**2 + (cell_y - beacon[1])**2)
                
#                 # Attraction to beacons within radius
#                 if beacon_dist < self.beacon_attraction_radius:
#                     beacon_factor += 1.0 / max(0.5, beacon_dist)
            
#             # Add beacon factor to score
#             score += beacon_factor * self.beacon_weight
            
#             if score > best_score:
#                 best_score = score
#                 best_cell = (i, j)
        
#         # If no valid cell found, return None
#         if best_cell is None:
#             print("DEBUG: No valid goal cell found after scoring")
#             return None
            
#         # Convert best cell to world coordinates
#         i, j = best_cell
#         goal_x, goal_y = self.grid_to_world(j, i)
        
#         print(f"DEBUG: Selected goal at ({goal_x:.2f}, {goal_y:.2f}) with score {best_score:.4f}")
#         return np.array([goal_x, goal_y])
    
#     def plan_path(self, start, goal):
#         """
#         Plan a path from start to goal using RRT.
        
#         Args:
#             start: [x, y] start position
#             goal: [x, y] goal position
            
#         Returns:
#             path: List of [x, y] positions along the path
#         """
#         # Check if we're already very close to the goal
#         if np.linalg.norm(np.array(start) - np.array(goal)) < self.waypoint_threshold:
#             # If we're very close, just return a direct path
#             return [np.array(start), np.array(goal)]
        
#         # Clear previous RRT data
#         self.rrt_nodes = []
#         self.rrt_edges = []
#         self.rrt_samples = []
        
#         # Initialize RRT
#         start_node = np.array(start)
#         goal_node = np.array(goal)
        
#         # Check if goal is directly reachable
#         if self.is_collision_free(start_node, goal_node):
#             return [start_node, goal_node]
        
#         # RRT nodes and edges
#         nodes = [start_node]
#         parents = [0]  # Parent index (self for root)
        
#         # Store for visualization
#         self.rrt_nodes = [start_node]
        
#         # Main RRT loop
#         for i in range(self.rrt_max_iter):
#             # Sample a point
#             if random.randint(0, 100) < self.rrt_goal_sample_rate:
#                 # Sample the goal directly
#                 sample = goal_node
#             else:
#                 # Random sampling
#                 sample = self.sample_free()
            
#             # Store for visualization
#             self.rrt_samples.append(sample)
            
#             # Find nearest node
#             nearest_idx = self.find_nearest(nodes, sample)
#             nearest_node = nodes[nearest_idx]
            
#             # Steer towards sample
#             new_node = self.steer(nearest_node, sample, self.rrt_step_size)
            
#             # Check if the path is collision-free
#             if self.is_collision_free(nearest_node, new_node):
#                 # Add the new node
#                 nodes.append(new_node)
#                 parents.append(nearest_idx)
                
#                 # Store for visualization
#                 self.rrt_nodes.append(new_node)
#                 self.rrt_edges.append((nearest_node, new_node))
                
#                 # Check if we can connect to the goal
#                 if np.linalg.norm(new_node - goal_node) <= self.rrt_connect_circle_dist:
#                     if self.is_collision_free(new_node, goal_node):
#                         # Goal reached, construct the path
#                         nodes.append(goal_node)
#                         parents.append(len(nodes) - 2)  # Parent is the last node
                        
#                         # Store for visualization
#                         self.rrt_nodes.append(goal_node)
#                         self.rrt_edges.append((new_node, goal_node))
                        
#                         return self.construct_path(nodes, parents)
        
#         # If we couldn't find a path but we're close enough to the goal,
#         # just return a direct path
#         if np.linalg.norm(start_node - goal_node) < self.goal_reached_threshold * 2:
#             return [start_node, goal_node]
            
#         # RRT failed to find a path
#         return []
    
#     def sample_free(self):
#         """
#         Sample a random collision-free point.
        
#         Returns:
#             point: [x, y] random point
#         """
#         if self.map_grid is None:
#             # Default sampling range if no map
#             x = random.uniform(-10, 10)
#             y = random.uniform(-10, 10)
#             return np.array([x, y])
        
#         # Get map dimensions
#         height, width = self.map_grid.shape
        
#         # Maximum attempts to find a free space
#         max_attempts = 100
        
#         for _ in range(max_attempts):
#             # Sample random grid cell
#             i = random.randint(0, height - 1)
#             j = random.randint(0, width - 1)
            
#             # Check if cell is free (threshold < 0.5 means free space)
#             if self.map_grid[i, j] < 0.5:
#                 # Convert to world coordinates
#                 x, y = self.grid_to_world(j, i)
#                 return np.array([x, y])
        
#         # If no free space found after max attempts, sample randomly
#         x = random.uniform(
#             self.map_origin[0], 
#             self.map_origin[0] + width * self.map_resolution
#         )
#         y = random.uniform(
#             self.map_origin[1],
#             self.map_origin[1] + height * self.map_resolution
#         )
        
#         return np.array([x, y])
    
#     def find_nearest(self, nodes, point):
#         """
#         Find the nearest node to a given point.
        
#         Args:
#             nodes: List of nodes
#             point: Target point
            
#         Returns:
#             idx: Index of the nearest node
#         """
#         dists = [np.linalg.norm(node - point) for node in nodes]
#         return np.argmin(dists)
    
#     def steer(self, from_node, to_node, step_size):
#         """
#         Steer from one node towards another with a maximum step size.
        
#         Args:
#             from_node: Starting node
#             to_node: Target node
#             step_size: Maximum distance to move
            
#         Returns:
#             new_node: New node after steering
#         """
#         dist = np.linalg.norm(to_node - from_node)
        
#         if dist <= step_size:
#             return to_node
        
#         # Compute direction vector
#         direction = (to_node - from_node) / dist
        
#         # Compute new node
#         new_node = from_node + direction * step_size
        
#         return new_node
    
#     def is_collision_free(self, from_node, to_node):
#         """
#         Check if a path between two nodes is collision-free.
        
#         Args:
#             from_node: Starting node
#             to_node: Target node
            
#         Returns:
#             is_free: True if path is collision-free, False otherwise
#         """
#         if self.map_grid is None:
#             return True
        
#         # Number of checks along the path
#         resolution = max(1, int(np.linalg.norm(to_node - from_node) / (self.map_resolution * 0.5)))
        
#         # Check points along the path
#         for i in range(resolution + 1):
#             t = i / resolution
#             point = from_node * (1 - t) + to_node * t
            
#             # Convert to grid coordinates
#             grid_i, grid_j = self.world_to_grid(point[0], point[1])
            
#             # Check if in bounds
#             if 0 <= grid_i < self.map_grid.shape[1] and 0 <= grid_j < self.map_grid.shape[0]:
#                 # Check if occupied
#                 if self.map_grid[grid_j, grid_i] > 0.5:  # Threshold for occupancy
#                     return False
#             else:
#                 # Out of bounds is considered collision
#                 return False
        
#         return True
    
#     def construct_path(self, nodes, parents):
#         """
#         Construct a path from start to goal using the RRT tree.
        
#         Args:
#             nodes: List of nodes
#             parents: List of parent indices
            
#         Returns:
#             path: List of nodes along the path from start to goal
#         """
#         # Start from the goal (last node)
#         path = [nodes[-1]]
#         parent_idx = parents[-1]
        
#         # Follow parents until we reach the start (index 0)
#         while parent_idx != 0:
#             path.append(nodes[parent_idx])
#             parent_idx = parents[parent_idx]
        
#         # Add the start node
#         path.append(nodes[0])
        
#         # Reverse to get path from start to goal
#         path.reverse()
        
#         return path
    
#     def generate_control(self, dt):
#         """
#         Generate control signal to follow the current path.
        
#         Args:
#             dt: Time step
            
#         Returns:
#             control: [vx, vy] velocity control
#         """
#         if not self.current_path or self.current_path_index >= len(self.current_path):
#             return np.zeros(2)
        
#         # Current target waypoint
#         target = self.current_path[self.current_path_index]
        
#         # Error (vector from robot to target)
#         error_x = target[0] - self.robot_position[0]
#         error_y = target[1] - self.robot_position[1]
        
#         # Check if we've reached the waypoint
#         dist_to_target = math.sqrt(error_x**2 + error_y**2)
        
#         if dist_to_target < self.waypoint_threshold:
#             # Advance to next waypoint
#             self.current_path_index += 1
            
#             # If we've reached the end of the path
#             if self.current_path_index >= len(self.current_path):
#                 return np.zeros(2)
                
#             # Update target
#             target = self.current_path[self.current_path_index]
#             error_x = target[0] - self.robot_position[0]
#             error_y = target[1] - self.robot_position[1]
        
#         # Calculate error and error derivative
#         error = np.array([error_x, error_y])
#         error_derivative = (error - self.prev_error) / dt
        
#         # PD control
#         control = self.pd_p_gain * error + self.pd_d_gain * error_derivative
        
#         # Limit control magnitude
#         control_mag = np.linalg.norm(control)
#         if control_mag > 2.0:
#             control = control * (2.0 / control_mag)
            
#         # Update previous error
#         self.prev_error = error
        
#         return control
        
#     def world_to_grid(self, x, y):
#         """
#         Convert world coordinates to grid indices.
        
#         Args:
#             x, y: World coordinates
            
#         Returns:
#             i, j: Grid indices
#         """
#         if self.map_grid is None or self.map_origin is None:
#             return 0, 0
            
#         i = int((x - self.map_origin[0]) / self.map_resolution)
#         j = int((y - self.map_origin[1]) / self.map_resolution)
        
#         # Ensure within grid bounds
#         i = max(0, min(i, self.map_grid.shape[1] - 1))
#         j = max(0, min(j, self.map_grid.shape[0] - 1))
        
#         return i, j
        
#     def grid_to_world(self, i, j):
#         """
#         Convert grid indices to world coordinates.
        
#         Args:
#             i, j: Grid indices
            
#         Returns:
#             x, y: World coordinates
#         """
#         if self.map_origin is None:
#             return 0, 0
            
#         x = self.map_origin[0] + (i + 0.5) * self.map_resolution
#         y = self.map_origin[1] + (j + 0.5) * self.map_resolution
        
#         return x, y

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel, convolve
import math
import random
import heapq

class Planner:
    """
    Entropy-based autonomous exploration planner.
    
    Computes information-rich areas in the environment for exploration,
    generates paths to selected goals using A*, and provides control signals
    to follow these paths.
    """
    
    def __init__(
        self,
        map_resolution=0.1,
        astar_allow_diagonal=True,
        pd_p_gain=1.0,
        pd_d_gain=0.1,
        entropy_weight=1.0,
        beacon_weight=2.0,
        beacon_attraction_radius=3.0,
        goal_persistence_time=10.0,  # Time to keep trying a goal (seconds)
        goal_reached_threshold=0.5   # Distance to consider a goal reached (meters)
        ):
        """
        Initialize the planner with exploration and control parameters.
        
        Args:
            map_resolution: Resolution of the occupancy grid (meters per cell)
            astar_allow_diagonal: Whether to allow diagonal movements in A*
            pd_p_gain: Proportional gain for the PD controller
            pd_d_gain: Derivative gain for the PD controller
            entropy_weight: Weight for entropy gradient in goal selection
            beacon_weight: Weight for beacon attraction in goal selection
            beacon_attraction_radius: Radius within which beacons attract the robot
            goal_persistence_time: Time to keep trying a goal (seconds)
            goal_reached_threshold: Distance to consider a goal reached (meters)
        """
        # Map parameters
        self.map_grid = None
        self.map_resolution = map_resolution
        self.map_origin = None
        
        # A* parameters
        self.astar_allow_diagonal = astar_allow_diagonal
        # Cost for movements (straight and diagonal)
        self.straight_cost = 1.0
        self.diagonal_cost = 1.414  # sqrt(2)
        
        # Controller parameters
        self.pd_p_gain = pd_p_gain
        self.pd_d_gain = pd_d_gain
        
        # Exploration parameters
        self.entropy_weight = entropy_weight
        self.beacon_weight = beacon_weight
        self.beacon_attraction_radius = beacon_attraction_radius
        
        # Robot state
        self.robot_position = np.array([0, 0, 0])
        self.prev_error = np.array([0, 0])
        
        # Path following
        self.current_path = []
        self.current_path_index = 0
        self.waypoint_threshold = 0.3  # Distance to consider a waypoint reached
        
        # Goal management
        self.current_goal = None
        self.goal_start_time = None
        self.goal_persistence_time = goal_persistence_time
        self.goal_reached_threshold = goal_reached_threshold
        self.goal_attempt_count = 0
        self.max_goal_attempts = 3  # Maximum attempts to reach a goal before selecting a new one
        
        # Beacons
        self.beacon_positions = []
        
        # Maps
        self.entropy_map = None
        self.boundary_map = None
        self.gradient_magnitude = None
        
        # For visualization
        self.astar_open_set = []
        self.astar_closed_set = []
        self.astar_path = []

        self.visited_regions = None  # Will be initialized when map is first updated
        self.visit_memory_decay = 0.995  # Decay factor for visited areas (per update)
        self.visit_influence_radius = 50  # Radius in cells to mark as visited
        self.visit_penalty_weight = 0.8 

        # # Open loop control parameters
        # self.waypoint_timer = 0.0
        # self.current_waypoint_time = 0.0
        # self.fixed_speed = 1.0  # Fixed speed in m/s
        # self.turning_speed_factor = 0.7  # Reduce speed for turns
        # self.min_angle_for_turn = np.pi/6  # ~30 degrees
    
    # Add this method to the Planner class
    def update_visited_regions(self):
        """
        Update the visited regions grid by marking current position and decaying old visits.
        This creates a "memory" of where the robot has been to encourage exploring new areas.
        """
        # Initialize visited_regions if not already done
        if self.visited_regions is None and self.map_grid is not None:
            self.visited_regions = np.zeros_like(self.map_grid)
            print("DEBUG: Initialized visited regions grid")
        
        if self.visited_regions is None:
            return  # Map not initialized yet
        
        # Decay the visit memory slightly each update
        self.visited_regions *= self.visit_memory_decay
        
        # Get current position in grid coordinates
        i, j = self.world_to_grid(self.robot_position[0], self.robot_position[1])
        
        # Mark area around robot as visited (higher weight to current position)
        for di in range(-self.visit_influence_radius, self.visit_influence_radius+1):
            for dj in range(-self.visit_influence_radius, self.visit_influence_radius+1):
                ni, nj = i + di, j + dj
                if 0 <= ni < self.visited_regions.shape[1] and 0 <= nj < self.visited_regions.shape[0]:
                    distance = np.sqrt(di**2 + dj**2)
                    if distance <= self.visit_influence_radius:
                        # Weight decreases with distance from robot
                        visit_weight = 1.0 - (distance / self.visit_influence_radius)
                        self.visited_regions[nj, ni] = min(1.0, self.visited_regions[nj, ni] + visit_weight)


    def update_map(self, occupancy_grid, map_origin, map_resolution):
        """
        Update the map used for planning.
        
        Args:
            occupancy_grid: 2D numpy array of occupancy probabilities (0-1)
            map_origin: (x, y) tuple of map origin in world coordinates
            map_resolution: Resolution of the map in meters per cell
        """
        self.map_grid = occupancy_grid
        self.map_origin = map_origin
        self.map_resolution = map_resolution
        
        # Initialize or resize visited_regions to match map_grid
        if self.visited_regions is None or self.visited_regions.shape != self.map_grid.shape:
            self.visited_regions = np.zeros_like(self.map_grid)
            print("DEBUG: Initialized/resized visited regions grid")
        
        # Clear previous computed maps
        self.entropy_map = None
        self.boundary_map = None
        self.gradient_magnitude = None
        
    def update_position(self, position):
        """
        Update the robot's position.
        
        Args:
            position: [x, y, theta] numpy array of robot position
        """
        self.robot_position = position
        
    def update_beacons(self, beacon_positions):
        """
        Update the known beacon positions.
        
        Args:
            beacon_positions: List of [x, y, z] positions of beacons
        """
        self.beacon_positions = beacon_positions
    
    def plan_and_control(self, dt, current_time=None):
        """
        Generate a plan and control signal for autonomous exploration.
        
        Args:
            dt: Time step for control
            current_time: Current simulation time (for goal persistence)
            
        Returns:
            (control_input, goal_point, path): 
                control_input: [vx, vy] numpy array of control velocities
                goal_point: [x, y] numpy array of selected goal position
                path: List of [x, y] points along the planned path
        """
        # Update visited regions based on current position
        self.update_visited_regions()
        
        if current_time is None:
            current_time = 0.0
        
        # Check if we should select a new goal
        new_goal_needed = False
        
        # Case 1: No current goal
        if self.current_goal is None:
            new_goal_needed = True
            print("DEBUG: No current goal, selecting new goal")
        
        # Case 2: Goal reached - simplified condition
        elif np.linalg.norm(self.robot_position[:2] - self.current_goal) < self.goal_reached_threshold:
            # Goal is reached - always get a new one
            print(f"DEBUG: Goal reached with distance {np.linalg.norm(self.robot_position[:2] - self.current_goal):.2f}")
            new_goal_needed = True
            self.goal_attempt_count = 0
        
        # Case 3: Goal persistence timeout
        elif self.goal_start_time is not None and (current_time - self.goal_start_time) > self.goal_persistence_time:
            self.goal_attempt_count += 1
            print(f"DEBUG: Goal timeout #{self.goal_attempt_count}, attempts limit: {self.max_goal_attempts}")
            
            # If we've tried too many times, find a new goal
            if self.goal_attempt_count >= self.max_goal_attempts:
                print("DEBUG: Max attempts reached, selecting new goal")
                new_goal_needed = True
                self.goal_attempt_count = 0
            else:
                # Try again with the same goal but ensure it's actually reachable
                path_test = self.plan_path(self.robot_position[:2], self.current_goal)
                if not path_test or len(path_test) <= 1:
                    print("DEBUG: Goal appears unreachable, selecting new goal")
                    new_goal_needed = True
                    self.goal_attempt_count = 0
                else:
                    # Reset the goal start time to extend the timeout
                    self.goal_start_time = current_time
                    print(f"DEBUG: Goal still appears reachable, continuing with attempt #{self.goal_attempt_count}")
        
        # Select a new goal if needed
        if new_goal_needed:
            max_attempts = 10  # Maximum attempts to find a suitable goal
            goal_point = None
            
            for attempt in range(max_attempts):
                goal_point = self.select_exploration_goal()
                
                if goal_point is None:
                    print(f"DEBUG: No goal point found on attempt {attempt+1}/{max_attempts}")
                    continue
                    
                # Don't select goals too close to the current position
                if np.linalg.norm(self.robot_position[:2] - goal_point) < 2.0:
                    print(f"DEBUG: Goal too close on attempt {attempt+1}/{max_attempts}, distance: {np.linalg.norm(self.robot_position[:2] - goal_point):.2f}")
                    continue
                
                # Before committing to a goal, check if it's reachable
                test_path = self.plan_path(self.robot_position[:2], goal_point)
                if not test_path or len(test_path) <= 1:
                    print(f"DEBUG: Goal unreachable on attempt {attempt+1}/{max_attempts}")
                    continue
                    
                # Goal is good, break the loop
                print(f"DEBUG: Found suitable goal point on attempt {attempt+1}")
                break
            
            # If we still don't have a goal after all attempts, try a random one
            if goal_point is None:
                print("DEBUG: Using random goal as fallback")
                # Choose a random direction and distance
                angle = np.random.uniform(0, 2 * np.pi)
                distance = np.random.uniform(3.0, 8.0)  # Between 3 and 8 meters
                goal_point = self.robot_position[:2] + np.array([
                    distance * np.cos(angle),
                    distance * np.sin(angle)
                ])
                
                # Check if the random goal is in a free space
                i, j = self.world_to_grid(goal_point[0], goal_point[1])
                if hasattr(self, 'map_grid') and self.map_grid is not None:
                    # Try up to 20 random angles to find a free space
                    for _ in range(20):
                        if i >= 0 and i < self.map_grid.shape[1] and j >= 0 and j < self.map_grid.shape[0]:
                            if self.map_grid[j, i] < 0.5:  # Free space
                                break
                        
                        # Try another angle
                        angle = np.random.uniform(0, 2 * np.pi)
                        goal_point = self.robot_position[:2] + np.array([
                            distance * np.cos(angle),
                            distance * np.sin(angle)
                        ])
                        i, j = self.world_to_grid(goal_point[0], goal_point[1])
                    
                    # Final check for reachability of random goal
                    test_path = self.plan_path(self.robot_position[:2], goal_point)
                    if not test_path or len(test_path) <= 1:
                        print("DEBUG: Random goal is unreachable, will use direct path as fallback")
            
            # Update goal state
            self.current_goal = goal_point
            self.goal_start_time = current_time
            self.goal_attempt_count = 0
            print(f"DEBUG: New goal set to ({goal_point[0]:.2f}, {goal_point[1]:.2f})")
        else:
            # Use existing goal
            goal_point = self.current_goal
            print(f"DEBUG: Continuing with existing goal ({goal_point[0]:.2f}, {goal_point[1]:.2f})")
        
        # Plan a path to the goal
        path = self.plan_path(self.robot_position[:2], goal_point)
        
        # Debug path information
        if path:
            print(f"DEBUG: Path planned with {len(path)} waypoints")
        else:
            print("DEBUG: Failed to plan a path")

        # Store the current path
        if path and len(path) > 1:
            self.current_path = path
            self.current_path_index = 1  # Start with the second point (first after start)
            
            # Generate control to follow the path - ONLY if we have a valid path
            control_input = self.generate_control(dt)
            print(f"DEBUG: Generated control: ({control_input[0]:.2f}, {control_input[1]:.2f})")
            return control_input, goal_point, path
        else:
            # No valid path found
            self.goal_attempt_count += 1
            print(f"DEBUG: No valid path found, attempt #{self.goal_attempt_count}")
            
            # Clear the current path to prevent using old path data
            self.current_path = []
            
            # If we've tried too many times, find a new goal next time
            if self.goal_attempt_count >= self.max_goal_attempts:
                print("DEBUG: No valid path found after multiple attempts, will select new goal next time")
                self.current_goal = None
            
            # Return no control input when path planning fails
            return None, self.current_goal, []

        # # In the plan_and_control method
        # if path and len(path) > 1:
        #     self.current_path = path
        #     self.current_path_index = 0  # Start with the first point
            
        #     # Generate control using open loop control
        #     control_input = self.generate_control(self.current_path, self.current_path_index, dt)
            
        #     # Increment path index based on estimated progress (time-based)
        #     distance_to_next = np.linalg.norm(self.current_path[self.current_path_index + 1] - 
        #                                     self.current_path[self.current_path_index])
        #     time_to_next = distance_to_next / np.linalg.norm(control_input)
            
        #     # After the estimated time has passed, move to the next waypoint
        #     self.waypoint_timer += dt
        #     if self.waypoint_timer >= time_to_next:
        #         self.current_path_index += 1
        #         self.waypoint_timer = 0
            
        #     return control_input, goal_point, path

    def generate_entropy_map(self):
        """
        Generate an entropy map from the occupancy grid.
        
        Returns:
            entropy_map: 2D numpy array of information entropy
        """
        if self.map_grid is None:
            return None
        
        # Ensure probabilities are in [0.001, 0.999] range to avoid log(0) issues
        p = np.clip(self.map_grid, 0.001, 0.999)
        
        # Calculate entropy: -p*log(p) - (1-p)*log(1-p)
        entropy = -p * np.log(p) - (1 - p) * np.log(1 - p)
        
        # Apply mean filtering
        kernel = np.ones((3, 3)) / 9.0
        entropy_smooth = convolve(entropy, kernel)
        
        self.entropy_map = entropy_smooth
        return entropy_smooth
    
    def compute_entropy_gradient(self, entropy_map=None):
        """
        Compute the gradient of the entropy map.
        
        Args:
            entropy_map: Optional entropy map to use (generated if None)
            
        Returns:
            (gradient_x, gradient_y, gradient_magnitude): Gradient components and magnitude
        """
        if entropy_map is None:
            entropy_map = self.generate_entropy_map()
            
        if entropy_map is None:
            return None, None, None
        
        # Apply Sobel filter
        gradient_x = sobel(entropy_map, axis=1)
        gradient_y = sobel(entropy_map, axis=0)
        
        # Compute gradient magnitude
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Normalize to [0, 1]
        max_magnitude = np.max(gradient_magnitude)
        if max_magnitude > 0:
            gradient_magnitude = gradient_magnitude / max_magnitude
        
        self.gradient_magnitude = gradient_magnitude
        return gradient_x, gradient_y, gradient_magnitude
    
    def generate_boundary_map(self):
        """
        Generate a boundary map that highlights transitions between known and unknown areas.
        
        Returns:
            boundary_map: 2D numpy array of boundary values
        """
        if self.map_grid is None:
            return None
        
        # Compute entropy gradient if not already done
        if self.gradient_magnitude is None:
            _, _, self.gradient_magnitude = self.compute_entropy_gradient()
        
        # The gradient magnitude already represents boundaries
        self.boundary_map = self.gradient_magnitude
        
        return self.boundary_map
    
    def select_exploration_goal(self):
        """
        Select the best exploration goal based on entropy gradient and beacon positions.
        
        Returns:
            goal_point: [x, y] numpy array of selected goal position
        """
        # Generate boundary map if not available
        if self.boundary_map is None:
            self.generate_boundary_map()
            
        if self.boundary_map is None or self.map_grid is None:
            print("ERROR: Boundary map or map grid is None, cannot select goal")
            return None
        
        # Log current state
        print(f"DEBUG: Robot position: [{self.robot_position[0]:.2f}, {self.robot_position[1]:.2f}]")
        print(f"DEBUG: Map shape: {self.map_grid.shape}, Boundary map shape: {self.boundary_map.shape}")
        
        # Check if boundary map has any high values
        high_boundary_cells = np.where(self.boundary_map > 0.3)  # Lower threshold to find more candidates
        print(f"DEBUG: Number of high boundary cells: {len(high_boundary_cells[0])}")
        
        # Create a copy to avoid modifying the original
        score_map = self.boundary_map.copy()
        
        # Get grid dimensions
        height, width = score_map.shape
        
        # Robot position in grid coordinates
        robot_x, robot_y = self.world_to_grid(self.robot_position[0], self.robot_position[1])
        
        # Compute scores for each cell based on distance and beacon proximity
        best_score = -float('inf')
        best_cell = None
        
        # Sample a subset of cells to evaluate (for efficiency)
        sample_rate = 0.1  # Increase from 0.05 to 0.1 to evaluate 10% of cells
        num_samples = int(height * width * sample_rate)
        
        # Skip if grid is very small
        if num_samples < 10:
            num_samples = min(height * width, 100)
        
        # Random sampling for evaluation
        candidate_cells = []
        
        # Only consider cells with high boundary values
        high_boundary_threshold = 0.3  # Lower threshold from 0.5 to 0.3
        high_boundary_cells = np.where(self.boundary_map > high_boundary_threshold)
        
        # If we have high boundary cells, prioritize those
        if len(high_boundary_cells[0]) > 0:
            high_boundary_indices = list(zip(high_boundary_cells[0], high_boundary_cells[1]))
            if len(high_boundary_indices) > num_samples:
                candidate_cells = random.sample(high_boundary_indices, num_samples)
            else:
                candidate_cells = high_boundary_indices
        
        # If we need more candidates, add random cells
        if len(candidate_cells) < num_samples:
            additional_samples = num_samples - len(candidate_cells)
            for _ in range(additional_samples):
                i = random.randint(0, height - 1)
                j = random.randint(0, width - 1)
                candidate_cells.append((i, j))
        
        # If still no candidates, create some random positions around the robot
        if not candidate_cells:
            print("DEBUG: No candidate cells found, generating random positions")
            for _ in range(20):
                # Random distance between 3 and 10 meters
                distance = random.uniform(3.0, 10.0)
                # Random angle
                angle = random.uniform(0, 2 * np.pi)
                # Calculate position
                x = self.robot_position[0] + distance * np.cos(angle)
                y = self.robot_position[1] + distance * np.sin(angle)
                # Convert to grid coordinates
                i, j = self.world_to_grid(x, y)
                # Ensure within grid bounds
                if 0 <= i < width and 0 <= j < height:
                    candidate_cells.append((j, i))  # Note: grid indices are (j,i)
        
        print(f"DEBUG: Evaluating {len(candidate_cells)} candidate cells")
        
        # When evaluating candidate cells:
        for i, j in candidate_cells:
            # Skip occupied cells
            if i >= 0 and i < height and j >= 0 and j < width and self.map_grid[i, j] > 0.7:
                continue
                
            # Get world coordinates
            cell_x, cell_y = self.grid_to_world(j, i)
            
            # Compute distance to robot
            dx = cell_x - self.robot_position[0]
            dy = cell_y - self.robot_position[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Skip cells that are too close to the robot
            if distance < 0.5:
                continue
                
            # Skip cells that are too far from the robot
            if distance > 12.0:
                continue
            
            # Computer score based on boundary value and distance
            score = (self.boundary_map[i, j] * self.entropy_weight) / max(0.5, distance)
            
            # Add beacon attraction/repulsion factor
            beacon_factor = 0
            for beacon in self.beacon_positions:
                beacon_dist = math.sqrt((cell_x - beacon[0])**2 + (cell_y - beacon[1])**2)
                
                # Attraction to beacons within radius
                if beacon_dist < self.beacon_attraction_radius:
                    beacon_factor += 1.0 / max(0.5, beacon_dist)
            
            # Add beacon factor to score
            score += beacon_factor * self.beacon_weight
            
            # Penalize previously visited areas
            if self.visited_regions is not None:
                visit_penalty = self.visit_penalty_weight * self.visited_regions[i, j]
                score *= (1.0 - visit_penalty)
            
            if score > best_score:
                best_score = score
                best_cell = (i, j)
    

        # If no valid cell found, return None
        if best_cell is None:
            print("DEBUG: No valid goal cell found after scoring")
            return None
            
        # Convert best cell to world coordinates
        i, j = best_cell
        goal_x, goal_y = self.grid_to_world(j, i)
        
        print(f"DEBUG: Selected goal at ({goal_x:.2f}, {goal_y:.2f}) with score {best_score:.4f}")
        return np.array([goal_x, goal_y])
    
    def plan_path(self, start, goal):
        """
        Plan a path from start to goal using A*.
        
        Args:
            start: [x, y] start position
            goal: [x, y] goal position
            
        Returns:
            path: List of [x, y] positions along the path
        """
        # Check if we're already very close to the goal
        if np.linalg.norm(np.array(start) - np.array(goal)) < self.waypoint_threshold:
            # If we're very close, just return a direct path
            return [np.array(start), np.array(goal)]
        
        # If map is not available, return direct path
        if self.map_grid is None:
            return [np.array(start), np.array(goal)]
        
        # Clear previous A* data
        self.astar_open_set = []
        self.astar_closed_set = []
        self.astar_path = []
        
        # Convert start and goal to grid coordinates
        start_grid = self.world_to_grid(start[0], start[1])
        goal_grid = self.world_to_grid(goal[0], goal[1])
        
        # Make sure start and goal are valid
        height, width = self.map_grid.shape
        if (start_grid[0] < 0 or start_grid[0] >= width or 
            start_grid[1] < 0 or start_grid[1] >= height or
            goal_grid[0] < 0 or goal_grid[0] >= width or
            goal_grid[1] < 0 or goal_grid[1] >= height):
            print("DEBUG: Start or goal outside map boundaries")
            return []
        
        # Check if start or goal are in occupied cells
        if self.map_grid[start_grid[1], start_grid[0]] > 0.5 or self.map_grid[goal_grid[1], goal_grid[0]] > 0.5:
            print("DEBUG: Start or goal in occupied cell")
            # For goal, we can try to find the nearest free cell
            if self.map_grid[goal_grid[1], goal_grid[0]] > 0.5:
                nearest_free = self.find_nearest_free_cell(goal_grid)
                if nearest_free:
                    goal_grid = nearest_free
                    goal = np.array(self.grid_to_world(goal_grid[0], goal_grid[1]))
                    print(f"DEBUG: Adjusted goal to nearest free cell: {goal}")
                else:
                    print("DEBUG: Could not find nearest free cell for goal")
                    return []
            else:
                return []
        
        # Define movement directions (8-connected grid)
        if self.astar_allow_diagonal:
            # 8-connected grid (horizontal, vertical, and diagonal movements)
            directions = [
                (0, 1),   # down
                (1, 0),   # right
                (0, -1),  # up
                (-1, 0),  # left
                (1, 1),   # down-right
                (1, -1),  # up-right
                (-1, 1),  # down-left
                (-1, -1)  # up-left
            ]
            costs = [
                self.straight_cost,  # down
                self.straight_cost,  # right
                self.straight_cost,  # up
                self.straight_cost,  # left
                self.diagonal_cost,  # down-right
                self.diagonal_cost,  # up-right
                self.diagonal_cost,  # down-left
                self.diagonal_cost   # up-left
            ]
        else:
            # 4-connected grid (only horizontal and vertical movements)
            directions = [
                (0, 1),   # down
                (1, 0),   # right
                (0, -1),  # up
                (-1, 0)   # left
            ]
            costs = [
                self.straight_cost,  # down
                self.straight_cost,  # right
                self.straight_cost,  # up
                self.straight_cost   # left
            ]
        
        # Initialize A* data structures
        open_set = []  # Priority queue for open nodes
        closed_set = set()  # Set of closed nodes
        g_score = {}  # Cost from start to node
        f_score = {}  # Total estimated cost (g_score + heuristic)
        came_from = {}  # Parent node mapping
        
        # Initialize start node
        start_node = (start_grid[0], start_grid[1])
        goal_node = (goal_grid[0], goal_grid[1])
        
        g_score[start_node] = 0
        f_score[start_node] = self.heuristic(start_node, goal_node)
        
        # Push start node to open set (f_score, counter for tiebreaking, node)
        counter = 0
        heapq.heappush(open_set, (f_score[start_node], counter, start_node))
        
        # Store for visualization
        self.astar_open_set = [start_node]
        
        # Main A* loop
        while open_set:
            # Get node with lowest f_score
            _, _, current = heapq.heappop(open_set)
            
            # Store for visualization
            self.astar_closed_set.append(current)
            
            # Check if goal reached
            if current == goal_node:
                # Reconstruct path
                path = self.reconstruct_path(came_from, current)
                path = self.smooth_path(path)
                
                # Convert path to world coordinates
                world_path = [np.array(self.grid_to_world(x, y)) for x, y in path]
                
                # Store for visualization
                self.astar_path = path
                
                return world_path
            
            # Add current to closed set
            closed_set.add(current)
            
            # Explore neighbors
            for i, (dx, dy) in enumerate(directions):
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check if neighbor is valid
                if (neighbor[0] < 0 or neighbor[0] >= width or 
                    neighbor[1] < 0 or neighbor[1] >= height):
                    continue
                
                # Check if neighbor is in closed set
                if neighbor in closed_set:
                    continue
                
                # Check if neighbor is in an occupied cell
                if self.map_grid[neighbor[1], neighbor[0]] > 0.5:
                    continue
                
                # Compute tentative g_score
                tentative_g_score = g_score[current] + costs[i]
                
                # Initialize neighbor if not seen before
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # Update path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_node)
                    
                    # Add to open set if not already there
                    in_open_set = False
                    for _, _, node in open_set:
                        if node == neighbor:
                            in_open_set = True
                            break
                    
                    if not in_open_set:
                        counter += 1
                        heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
                        
                        # Store for visualization
                        self.astar_open_set.append(neighbor)
        
        # No path found
        print("DEBUG: A* failed to find a path")
        return []
    
    def heuristic(self, a, b):
        """
        Calculate heuristic (Euclidean distance) between two grid cells.
        
        Args:
            a: (x, y) coordinates of first cell
            b: (x, y) coordinates of second cell
            
        Returns:
            distance: Euclidean distance between cells
        """
        return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
    
    def reconstruct_path(self, came_from, current):
        """
        Reconstruct path from A* search results.
        
        Args:
            came_from: Dictionary mapping nodes to their parents
            current: Current (goal) node
            
        Returns:
            path: List of (x, y) grid coordinates along the path
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        
        # Reverse to get path from start to goal
        path.reverse()
        
        return path
    
    def smooth_path(self, path):
        """
        Apply path smoothing to remove unnecessary waypoints.
        
        Args:
            path: List of (x, y) grid coordinates
            
        Returns:
            smoothed_path: Smoothed path with fewer waypoints
        """
        if len(path) <= 2:
            return path
        
        smoothed_path = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            current = path[i]
            
            # Look ahead as far as possible with a clear line of sight
            for j in range(len(path) - 1, i, -1):
                if self.is_collision_free_grid(current, path[j]):
                    smoothed_path.append(path[j])
                    i = j
                    break
            
            # If no clear line of sight, just add the next point
            if i == len(path) - 1 or not self.is_collision_free_grid(current, path[i + 1]):
                i += 1
                if i < len(path):
                    smoothed_path.append(path[i])
        
        return smoothed_path
    
    def is_collision_free_grid(self, from_cell, to_cell):
        """
        Check if a path between two grid cells is collision-free.
        
        Args:
            from_cell: (x, y) starting grid cell
            to_cell: (x, y) target grid cell
            
        Returns:
            is_free: True if path is collision-free, False otherwise
        """
        # Bresenham's line algorithm for grid-based collision checking
        x0, y0 = from_cell
        x1, y1 = to_cell
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while x0 != x1 or y0 != y1:
            if 0 <= x0 < self.map_grid.shape[1] and 0 <= y0 < self.map_grid.shape[0]:
                if self.map_grid[y0, x0] > 0.5:  # Cell is occupied
                    return False
            else:
                return False  # Out of bounds
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        
        # Check final cell
        if 0 <= x1 < self.map_grid.shape[1] and 0 <= y1 < self.map_grid.shape[0]:
            return self.map_grid[y1, x1] <= 0.5
        else:
            return False
    
    def find_nearest_free_cell(self, cell):
        """
        Find the nearest unoccupied cell to the given cell.
        
        Args:
            cell: (x, y) grid coordinates
            
        Returns:
            nearest: (x, y) coordinates of nearest free cell, or None if not found
        """
        # Maximum search radius
        max_radius = 10
        
        # Get map dimensions
        height, width = self.map_grid.shape
        
        # BFS to find nearest free cell
        queue = [(cell[0], cell[1], 0)]
        visited = set([(cell[0], cell[1])])
        
        # Explore in all 8 directions
        directions = [
            (0, 1), (1, 0), (0, -1), (-1, 0),  # 4-connected
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # diagonals
        ]
        
        while queue:
            x, y, dist = queue.pop(0)
            
            # Check if this cell is free
            if 0 <= x < width and 0 <= y < height and self.map_grid[y, x] <= 0.5:
                return (x, y)
            
            # Stop if we've gone too far
            if dist >= max_radius:
                continue
            
            # Add neighbors to the queue
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in visited and 0 <= nx < width and 0 <= ny < height:
                    visited.add((nx, ny))
                    queue.append((nx, ny, dist + 1))
        
        # No free cell found within radius
        return None

    def is_collision_free(self, from_node, to_node):
        """
        Check if a path between two world coordinate nodes is collision-free.
        
        Args:
            from_node: Starting node in world coordinates
            to_node: Target node in world coordinates
            
        Returns:
            is_free: True if path is collision-free, False otherwise
        """
        if self.map_grid is None:
            return True
        
        # Number of checks along the path
        resolution = max(1, int(np.linalg.norm(to_node - from_node) / (self.map_resolution * 0.5)))
        
        # Check points along the path
        for i in range(resolution + 1):
            t = i / resolution
            point = from_node * (1 - t) + to_node * t
            
            # Convert to grid coordinates
            grid_i, grid_j = self.world_to_grid(point[0], point[1])
            
            # Check if in bounds
            if 0 <= grid_i < self.map_grid.shape[1] and 0 <= grid_j < self.map_grid.shape[0]:
                # Check if occupied
                if self.map_grid[grid_j, grid_i] > 0.5:  # Threshold for occupancy
                    return False
            else:
                # Out of bounds is considered collision
                return False
        
        return True

    def generate_control(self, dt):
        """
        Generate control signal to follow the current path.
        
        Args:
            dt: Time step
            
        Returns:
            control: [vx, vy] velocity control
        """
        if not self.current_path or self.current_path_index >= len(self.current_path):
            return np.zeros(2)
        
        # Current target waypoint
        target = self.current_path[self.current_path_index]
        
        # Error (vector from robot to target)
        error_x = target[0] - self.robot_position[0]
        error_y = target[1] - self.robot_position[1]
        
        # Check if we've reached the waypoint
        dist_to_target = math.sqrt(error_x**2 + error_y**2)
        
        if dist_to_target < self.waypoint_threshold:
            # Advance to next waypoint
            self.current_path_index += 1
            
            # If we've reached the end of the path
            if self.current_path_index >= len(self.current_path):
                return np.zeros(2)
                
            # Update target
            target = self.current_path[self.current_path_index]
            error_x = target[0] - self.robot_position[0]
            error_y = target[1] - self.robot_position[1]
        
        # Calculate error and error derivative
        error = np.array([error_x, error_y])
        error_derivative = (error - self.prev_error) / dt
        
        # PD control
        control = self.pd_p_gain * error + self.pd_d_gain * error_derivative
        
        # Limit control magnitude
        control_mag = np.linalg.norm(control)
        if control_mag > 2.0:
            control = control * (2.0 / control_mag)
            
        # Update previous error
        self.prev_error = error
        
        return control
        
    # def generate_control(self, path, current_index, dt):
    #     """
    #     Generate an open loop control signal based on the planned path.
        
    #     Args:
    #         path: List of waypoints
    #         current_index: Current path index
    #         dt: Time step
                
    #     Returns:
    #         control: [vx, vy] velocity control
    #     """
    #     if not path or current_index >= len(path) - 1:
    #         return np.zeros(2)
        
    #     # Get current and next waypoint
    #     current_point = path[current_index]
    #     next_point = path[current_index + 1]
        
    #     # Compute direction vector
    #     direction = next_point - current_point
    #     distance = np.linalg.norm(direction)
        
    #     # Normalize direction vector
    #     if distance > 0:
    #         direction = direction / distance
        
    #     # Compute velocity (fixed speed in the direction of the next waypoint)
    #     speed = 1.0  # Fixed speed in m/s
    #     control = direction * speed
        
    #     return control
    
    def world_to_grid(self, x, y):
        """
        Convert world coordinates to grid indices.
        
        Args:
            x, y: World coordinates
            
        Returns:
            i, j: Grid indices
        """
        if self.map_grid is None or self.map_origin is None:
            return 0, 0
            
        i = int((x - self.map_origin[0]) / self.map_resolution)
        j = int((y - self.map_origin[1]) / self.map_resolution)
        
        # Ensure within grid bounds
        i = max(0, min(i, self.map_grid.shape[1] - 1))
        j = max(0, min(j, self.map_grid.shape[0] - 1))
        
        return i, j
        
    def grid_to_world(self, i, j):
        """
        Convert grid indices to world coordinates.
        
        Args:
            i, j: Grid indices
            
        Returns:
            x, y: World coordinates
        """
        if self.map_origin is None:
            return 0, 0
            
        x = self.map_origin[0] + (i + 0.5) * self.map_resolution
        y = self.map_origin[1] + (j + 0.5) * self.map_resolution
        
        return x, y