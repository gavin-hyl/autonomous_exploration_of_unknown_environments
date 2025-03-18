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
        thres =  np.max(gradient_norm) * 0.3 ###### adjust threshold
        boundary_map = (gradient_norm > thres).astype(float)

        # gaussian filter for smoothnes (optional)
        boundary_map = gaussian_filter(boundary_map, sigma=1)

        return boundary_map


    def select_goal_point(self):
        """
        selects next goal point for the robot
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

        search_radius = int(5.0 / self.map_resolution)

        for i in range(len(y_idx)):
            x = x_idx[i]
            y = y_idx[i]

            if self.occupancy_grid[y, x] > 50:
                continue

            d = math.sqrt((x - robot_grid_x)**2 + (y - robot_grid_y)**2)
            

            if 10 < d < search_radius:
                gradient_x, gradient_y = self.compute_entropy_gradient(entropy_map)
                grad_norm = filtered_boundary[y, x]

                d_score = 1.0 - (d / search_radius)
                grad_score = grad_norm / (np.max(gradient_x)**2 + np.max(gradient_y)**2)**0.5

                score_total = d_score * 0.3 + grad_score * 0.7

                world_x, world_y = self.grid_to_world(x, y)
                goal_pts.append((world_x, world_y, score_total))
        

        # beacon positions
        for beacon in self.beacons:
            beacon_grid_x, beacon_grid_y = self.world_to_grid(beacon[0], beacon[1])

            if not (0 <= beacon_grid_x < self.grid_width and 0 <= beacon_grid_y < self.grid_height):
                continue
            
            for world_x, world_y, score in goal_pts.copy():
                d_g_to_b = math.sqrt((world_x - beacon[0])**2 + (world_y - beacon[1])**2)

                if d_g_to_b < self.beacon_attraction_radius:
                    proximity_factor = 1.0 - (d_g_to_b / self.beacon_attraction_radius)

                    new_score = score + proximity_factor * self.beacon_weight

                    idx = goal_pts.index((world_x, world_y, score))
                    goal_pts[idx] = (world_x, world_y, new_score)

        # if there are no goal points, select random point
        if not goal_pts:
            attempts = 0
            while attempts < 100:
                rand_grid_x = random.randint(0, self.grid_width - 1)
                rand_grid_y = random.randint(0, self.grid_height - 1)

                # unoccupied cells
                if self.occupancy_grid[rand_grid_y, rand_grid_x] < 50:
                    dist = math.sqrt((rand_grid_x - robot_grid_x)**2 + (rand_grid_y - robot_grid_y)**2)
                    
                    if 10 < dist < search_radius:
                        world_x, world_y = self.grid_to_world(rand_grid_x, rand_grid_y)
                        return world_x, world_y
                    
                attempts += 1
            
            return self.current_pos[0], self.current_pos[1]
        
        best_goal_pt = max(goal_pts, key=lambda x: x[2])
        return best_goal_pt[0], best_goal_pt[1]
                       

    def check_collision(self, x, y):
        """
        Check if a given position collides with obstacles
        
        Args:
            x (float): World coordinate x
            y (float): World coordinate y
            
        Returns:
            bool: True if collision, False otherwise
        """
        pass

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
        pass

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
        pass

    def get_nearest_node_index(self, node_list, target_node):
        """
        Return the index of the node closest to the target node
        
        Args:
            node_list (list): List of nodes
            target_node (RRTNode): Target node
            
        Returns:
            int: Index of the closest node
        """
        pass

    def steer(self, from_node, to_node):
        """
        Create a new node by moving from from_node toward to_node by steer_step distance
        
        Args:
            from_node (RRTNode): Start node
            to_node (RRTNode): Target node
            
        Returns:
            RRTNode: New node
        """
        pass

    def generate_path(self, node_list):
        """
        Generate path from node list
        
        Args:
            node_list (list): List of nodes
            
        Returns:
            list: Path coordinate list [[x1, y1], [x2, y2], ...]
        """
        pass

    def optimize_path(self, path):
        """
        Optimize path (remove unnecessary intermediate nodes)
        
        Args:
            path (list): Path coordinate list [[x1, y1], [x2, y2], ...]
            
        Returns:
            list: Optimized path coordinate list
        """
        pass

    def compute_control(self, current_pos, path, dt):
        """
        return control input to follow a given path (using PD controller)
        
        """
        
        if path is None or len(path) < 2:
            return np.array([0.0, 0.0])
        
        # Ensure current path index is valid
        if self.current_path_index >= len(path):
            self.current_path_index = len(path) - 1

        target_pos = np.array(path[self.current_path_index])
        distance = np.linalg.norm(target_pos - current_pos)

        # reached target point
        if distance < 0.1:
            self.current_path_index += 1

            # at last point of path
            if self.current_path_index >= len(path):
                self.current_path_index = len(path) - 1
                return np.array([0.0, 0.0])
            
            target_pos = np.array(path[self.current_path_index])
        
        # controller 
        error = target_pos - current_pos

        d_error = (error - self.prev_error) / dt if dt > 0 else np.array([0.0, 0.0])
        self.prev_error = error.copy()

        # pd control 
        control_input = self.pd_p_gain * error + self.pd_d_gain * d_error

        # control input clipping 
        max_speed = 1.0
        control_norm = np.linalg.norm(control_input)

        if control_norm > max_speed:
            control_input = control_input / control_norm * max_speed

        self.last_control = control_input.copy()

        return control_input
       

    def plan_and_control(self, dt):
        """
        Path planning and control input calculation
        
        Args:
            dt (float): Time interval
            
        Returns:
            numpy.ndarray: Control input [vx, vy]
        """
        if self.occupancy_grid is None:
            return np.array([0.0, 0.0])
        
        # no current path or replanning is needed
        if self.current_path is None or len(self.current_path) == 0 or time.time() - self.planning_time > 10.0:
            # if it is already planning, keep previous control
            if self.is_planning:
                return self.last_control
            
            goal_x, goal_y = self.select_goal_point()
            if goal_x is None or goal_y is None:
                return np.array([0.0, 0.0])
            
            goal_pos = np.array([goal_x, goal_y])
            print(f"New goal point: {goal_pos}")

            # path planner using RRT
            path = self.rrt_planning(self.current_pos, goal_pos)

            if path is None:
                return self.last_control

            self.current_path = path
            self.current_path_index = 0
            self.planning_time = time.time()

            control = self.compute_control(self.current_pos, self.current_path, dt)
        
        return control
            
