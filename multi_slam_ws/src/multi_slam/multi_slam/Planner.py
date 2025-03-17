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

    def generate_entropy_map(self):  
        
        pass

    def compute_entropy_gradient(self, entropy_map):

        pass

    def detect_exploration_boundary(self, entropy_map):
        pass

    def select_goal_point(self):
        pass

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
        Calculate control input to follow a given path (using PD controller)
        
        Args:
            current_pos (numpy.ndarray): Current position [x, y]
            path (list): Path coordinate list [[x1, y1], [x2, y2], ...]
            dt (float): Time interval
            
        Returns:
            numpy.ndarray: Control input [vx, vy]
        """
        pass

    def plan_and_control(self, dt):
        """
        Path planning and control input calculation
        
        Args:
            dt (float): Time interval
            
        Returns:
            numpy.ndarray: Control input [vx, vy]
        """
        pass
