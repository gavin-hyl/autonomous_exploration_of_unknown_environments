#!/usr/bin/env python3

import numpy as np
import math
import cv2
from scipy.ndimage import sobel, gaussian_filter
from scipy.spatial import KDTree
from typing import List, Tuple, Optional
import random
import time
import logging

class RRTNode:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.path_x = []  # x coordinates to this node
        self.path_y = []  # y coordinates to this node
        self.parent = None  # parent node

class Planner:

    def __init__(self, 
                 map_resolution=0.05,
                 rrt_step_size=0.5,
                 rrt_max_iter=500,
                 rrt_goal_sample_rate=5,
                 rrt_connect_circle_dist=0.5,
                 pd_p_gain=1.0,
                 pd_d_gain=0.1,
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

    def generate_entropy_map(self):    # < ------ code from here 
        """
        Generate an entropy map from the occupancy grid
        
        Higher entropy means higher uncertainty in that area.
        Unknown areas (-1) have the highest entropy.
        
        Returns:
            numpy.ndarray: Entropy map
        """
        if self.occupancy_grid is None:
            return None
        
        # Assign high entropy to unknown areas (-1)
        entropy_map = np.ones_like(self.occupancy_grid, dtype=float)
        unknown_mask = (self.occupancy_grid == -1)
        free_mask = (self.occupancy_grid == 0)
        occupied_mask = (self.occupancy_grid > 50)  # Consider values above 50 as occupied
        
        entropy_map[unknown_mask] = 1.0  # Unknown areas: high entropy
        entropy_map[free_mask] = 0.2     # Free areas: low entropy
        entropy_map[occupied_mask] = 0.0  # Occupied areas: no entropy
        
        # Apply Gaussian filter to create a smooth entropy map
        entropy_map = gaussian_filter(entropy_map, sigma=3)
        
        return entropy_map

    def compute_entropy_gradient(self, entropy_map):
        """
        Calculate the gradient of the entropy map
        
        Args:
            entropy_map (numpy.ndarray): Entropy map
            
        Returns:
            tuple: x and y direction gradients (grad_x, grad_y)
        """
        grad_y = sobel(entropy_map, axis=0)
        grad_x = sobel(entropy_map, axis=1)
        
        return grad_x, grad_y

    def detect_exploration_boundary(self, entropy_map):
        """
        Detect exploration boundaries
        
        Uses Sobel filter to detect edges in the entropy map
        
        Args:
            entropy_map (numpy.ndarray): Entropy map
            
        Returns:
            numpy.ndarray: Boundary map (high values are boundaries)
        """
        grad_x, grad_y = self.compute_entropy_gradient(entropy_map)
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Apply threshold to emphasize boundaries
        threshold = np.max(gradient_magnitude) * 0.3  # Use 30% of max value as threshold
        boundary_map = (gradient_magnitude > threshold).astype(float)
        
        # Apply Gaussian filter to create a smooth boundary map
        boundary_map = gaussian_filter(boundary_map, sigma=1)
        
        return boundary_map

    def select_goal_point(self):
        """
        Select the next goal point for the robot
        
        Considers the entropy map and beacon positions to select the optimal goal point
        
        Returns:
            tuple: Goal point world coordinates (goal_x, goal_y)
        """
        if self.occupancy_grid is None:
            return None
        
        # Generate entropy map
        entropy_map = self.generate_entropy_map()
        if entropy_map is None:
            return None
        
        # Detect exploration boundaries
        boundary_map = self.detect_exploration_boundary(entropy_map)
        
        # Convert current robot position to grid coordinates
        robot_grid_x, robot_grid_y = self.world_to_grid(self.current_pos[0], self.current_pos[1])
        
        # Check if grid coordinates are valid
        if not (0 <= robot_grid_x < self.grid_width and 0 <= robot_grid_y < self.grid_height):
            return None
        
        # Find boundary point candidates
        candidate_points = []
        search_radius = int(5.0 / self.map_resolution)  # 5 meter radius
        
        for y in range(max(0, robot_grid_y - search_radius), min(self.grid_height, robot_grid_y + search_radius)):
            for x in range(max(0, robot_grid_x - search_radius), min(self.grid_width, robot_grid_x + search_radius)):
                # Consider only boundary points and unoccupied cells
                if boundary_map[y, x] > 0.5 and self.occupancy_grid[y, x] < 50:
                    # Calculate distance to robot
                    dist = math.sqrt((x - robot_grid_x)**2 + (y - robot_grid_y)**2)
                    
                    # Exclude points that are too close or too far
                    if 10 < dist < search_radius:
                        # Calculate score considering entropy gradient
                        grad_x, grad_y = self.compute_entropy_gradient(entropy_map)
                        grad_magnitude = math.sqrt(grad_x[y, x]**2 + grad_y[y, x]**2)
                        
                        # Distance-based score (closer is better)
                        distance_score = 1.0 - (dist / search_radius)
                        
                        # Entropy gradient-based score
                        gradient_score = grad_magnitude / (np.max(grad_x)**2 + np.max(grad_y)**2)**0.5
                        
                        # Total score calculation
                        score = distance_score * 0.3 + gradient_score * 0.7
                        
                        world_x, world_y = self.grid_to_world(x, y)
                        candidate_points.append((world_x, world_y, score))
        
        # Consider beacon positions
        for beacon in self.beacons:
            beacon_grid_x, beacon_grid_y = self.world_to_grid(beacon[0], beacon[1])
            
            # Check if grid coordinates are valid
            if not (0 <= beacon_grid_x < self.grid_width and 0 <= beacon_grid_y < self.grid_height):
                continue
            
            # Calculate distance between robot and beacon
            dist_to_beacon = np.linalg.norm(self.current_pos - beacon[:2])
            
            # If beacon is within attraction radius, add to candidates
            if dist_to_beacon < self.beacon_attraction_radius:
                # Find explorable points near the beacon
                beacon_radius = int(1.0 / self.map_resolution)  # 1 meter radius
                
                for y in range(max(0, beacon_grid_y - beacon_radius), min(self.grid_height, beacon_grid_y + beacon_radius)):
                    for x in range(max(0, beacon_grid_x - beacon_radius), min(self.grid_width, beacon_grid_x + beacon_radius)):
                        # Consider only unoccupied cells
                        if self.occupancy_grid[y, x] < 50:
                            # Calculate distance to beacon
                            dist = math.sqrt((x - beacon_grid_x)**2 + (y - beacon_grid_y)**2)
                            
                            # Consider only points at appropriate distance
                            if dist < beacon_radius:
                                world_x, world_y = self.grid_to_world(x, y)
                                
                                # Beacon-based score (closer is better)
                                beacon_score = 1.0 - (dist / beacon_radius)
                                
                                # Consider distance to robot
                                robot_dist = math.sqrt((x - robot_grid_x)**2 + (y - robot_grid_y)**2)
                                if 5 < robot_dist < search_radius:
                                    candidate_points.append((world_x, world_y, beacon_score * self.beacon_weight))
        
        # If no candidate points, select random point
        if not candidate_points:
            attempts = 0
            while attempts < 100:
                # Generate random coordinates
                rand_grid_x = random.randint(0, self.grid_width - 1)
                rand_grid_y = random.randint(0, self.grid_height - 1)
                
                # Consider only unoccupied cells
                if self.occupancy_grid[rand_grid_y, rand_grid_x] < 50:
                    # Calculate distance to robot
                    dist = math.sqrt((rand_grid_x - robot_grid_x)**2 + (rand_grid_y - robot_grid_y)**2)
                    
                    # Consider only points at appropriate distance
                    if 10 < dist < search_radius:
                        world_x, world_y = self.grid_to_world(rand_grid_x, rand_grid_y)
                        return world_x, world_y
                
                attempts += 1
            
            # If random point selection fails, return current position
            return self.current_pos[0], self.current_pos[1]
        
        # Select point with highest score
        best_candidate = max(candidate_points, key=lambda x: x[2])
        return best_candidate[0], best_candidate[1]

    def check_collision(self, x, y):
        """
        Check if a given position collides with obstacles
        
        Args:
            x (float): World coordinate x
            y (float): World coordinate y
            
        Returns:
            bool: True if collision, False otherwise
        """
        if self.occupancy_grid is None:
            return True
        
        grid_x, grid_y = self.world_to_grid(x, y)
        
        # Consider out of grid as collision
        if grid_x < 0 or grid_x >= self.grid_width or grid_y < 0 or grid_y >= self.grid_height:
            return True
        
        # Consider occupied cells (≥50) as collision
        if self.occupancy_grid[grid_y, grid_x] >= 50:
            return True
        
        return False

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
        
        # Determine number of sampling points based on distance
        steps = max(2, int(dist / (self.map_resolution * 0.5)))
        
        # Sample along the path
        for i in range(steps + 1):
            t = i / steps
            x = from_x + t * (to_x - from_x)
            y = from_y + t * (to_y - from_y)
            
            if self.check_collision(x, y):
                return True
        
        return False

    def rrt_planning(self, start_pos, goal_pos, max_attempts=3):
        """
        Path planning using the RRT algorithm
        
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
        
        # If goal point is occupied, find new goal point
        if self.check_collision(goal_pos[0], goal_pos[1]):
            print("Goal point is occupied. Selecting another point...")
            goal_x, goal_y = self.select_goal_point()
            if goal_x is None or goal_y is None:
                self.is_planning = False
                return None
            goal_pos = np.array([goal_x, goal_y])
        
        # Try RRT for maximum number of attempts
        for attempt in range(max_attempts):
            # Initialize nodes
            start_node = RRTNode(start_pos[0], start_pos[1])
            goal_node = RRTNode(goal_pos[0], goal_pos[1])
            node_list = [start_node]
            
            for i in range(self.rrt_max_iter):
                # Sample random position
                if random.randint(0, 100) > self.rrt_goal_sample_rate:
                    # Generate random position
                    rand_x = random.uniform(start_pos[0] - 10.0, start_pos[0] + 10.0)
                    rand_y = random.uniform(start_pos[1] - 10.0, start_pos[1] + 10.0)
                    rand_node = RRTNode(rand_x, rand_y)
                else:
                    # Sample goal position
                    rand_node = RRTNode(goal_pos[0], goal_pos[1])
                
                # Find nearest node
                nearest_ind = self.get_nearest_node_index(node_list, rand_node)
                nearest_node = node_list[nearest_ind]
                
                # Create new node
                new_node = self.steer(nearest_node, rand_node)
                
                # Check collision
                if self.check_path_collision(nearest_node.x, nearest_node.y, new_node.x, new_node.y):
                    continue
                
                # Add new node
                node_list.append(new_node)
                
                # Check if reached near goal
                dist_to_goal = math.sqrt((new_node.x - goal_node.x)**2 + (new_node.y - goal_node.y)**2)
                if dist_to_goal <= self.rrt_step_size:
                    # Check if can connect directly to goal
                    if not self.check_path_collision(new_node.x, new_node.y, goal_node.x, goal_node.y):
                        # Connect final node to goal
                        final_node = self.steer(new_node, goal_node)
                        node_list.append(final_node)
                        
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

    def compute_control(self, current_pos, path, dt):
        """
        Calculate control input to follow a given path (using PD controller)
        
        Args:
            current_pos (numpy.ndarray): Current position [x, y]
            path (list): Path coordinate list [[x1, y1], [x2, y2], ...]
            dt (float): Time interval
            
        Returns:
            numpy.ndarray: Control input [vx, vy]
        """
        if path is None or len(path) < 2:
            return np.array([0.0, 0.0])
        
        # Ensure current path index is valid
        if self.current_path_index >= len(path):
            self.current_path_index = len(path) - 1
        
        # Target point
        target_pos = np.array(path[self.current_path_index])
        
        # Distance between current point and target point
        distance = np.linalg.norm(target_pos - current_pos)
        
        # Check if reached target point
        if distance < 0.2:  # Within 20cm
            # Move to next point
            self.current_path_index += 1
            
            # If last point
            if self.current_path_index >= len(path):
                self.current_path_index = len(path) - 1
                # Path completed
                return np.array([0.0, 0.0])
            
            # New target point
            target_pos = np.array(path[self.current_path_index])
        
        # PD control
        error = target_pos - current_pos
        
        # Calculate derivative term
        d_error = (error - self.prev_error) / dt if dt > 0 else np.array([0.0, 0.0])
        self.prev_error = error.copy()
        
        # PD control law
        control = self.pd_p_gain * error + self.pd_d_gain * d_error
        
        # Limit control input
        max_control = 1.0  # Maximum control input
        control_norm = np.linalg.norm(control)
        if control_norm > max_control:
            control = control / control_norm * max_control
        
        # Save control input
        self.last_control = control.copy()
        
        return control

    def plan_and_control(self, dt):
        """
        Path planning and control input calculation
        
        Args:
            dt (float): Time interval
            
        Returns:
            numpy.ndarray: Control input [vx, vy]
        """
        # If no map data, control not possible
        if self.occupancy_grid is None:
            return np.array([0.0, 0.0])
        
        # If no current path or replanning needed
        if self.current_path is None or len(self.current_path) == 0 or time.time() - self.planning_time > 10.0:
            # If already planning, maintain previous control
            if self.is_planning:
                return self.last_control
            
            # Select new goal point
            goal_x, goal_y = self.select_goal_point()
            if goal_x is None or goal_y is None:
                return np.array([0.0, 0.0])
            
            goal_pos = np.array([goal_x, goal_y])
            print(f"New goal point: {goal_pos}")
            
            # Plan path using RRT
            path = self.rrt_planning(self.current_pos, goal_pos)
            
            # If path planning failed, maintain previous control
            if path is None:
                return self.last_control
            
            # Update new path and planning time
            self.current_path = path
            self.current_path_index = 0
            self.planning_time = time.time()
        
        # Calculate control input using PD controller
        control = self.compute_control(self.current_pos, self.current_path, dt)
        
        return control
