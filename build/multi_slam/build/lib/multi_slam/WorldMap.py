import numpy as np
import random

class WorldMap:
    def __init__(self, width=100, height=100, resolution=0.1):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.grid = np.zeros((height, width), dtype=np.int8)
        self.beacons = []
        self.obstacles = []
        
        # Initialize some random beacons
        self._generate_random_beacons()

    def _generate_random_beacons(self, num_beacons=5):
        """Generate random beacons in the world"""
        for _ in range(num_beacons):
            x = random.uniform(0, self.width * self.resolution)
            y = random.uniform(0, self.height * self.resolution)
            self.add_beacon(x, y)

    def add_beacon(self, x, y):
        """Add a beacon to the map"""
        beacon = {'x': x, 'y': y}
        self.beacons.append(beacon)

    def add_obstacle(self, x, y):
        """Add an obstacle to the map"""
        obstacle = {'x': x, 'y': y}
        self.obstacles.append(obstacle)

    def is_valid_position(self, x, y):
        """Check if a position is valid (not an obstacle)"""
        return (0 <= x < self.width and 
                0 <= y < self.height and 
                self.grid[int(y), int(x)] != 1)