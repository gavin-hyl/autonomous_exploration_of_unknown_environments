from typing import List, Optional
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely.geometry.base import BaseGeometry

class Map:
    """
    Represents a 2D map with a rectangular boundary, obstacles, and beacons.
    
    This class supports checking intersections between a given geometry and
    the map's boundary/obstacles, as well as adding obstacles and beacons.
    
    Attributes:
        boundary (LineString): The boundary of the map as a rectangular polygon's edge.
        obstacles (List[BaseGeometry]): A list of geometric obstacles on the map.
        beacons (List[Point]): A list of beacon positions.
    """
    def __init__(
        self, 
        x_min: float, 
        y_min: float, 
        x_max: float, 
        y_max: float, 
        obstacles: Optional[List[BaseGeometry]] = None,
        beacons: Optional[List[Point]] = None
    ) -> None:
        """
        Initialize the map with a rectangular boundary.
        
        Args:
            x_min (float): Minimum x-coordinate (left boundary).
            y_min (float): Minimum y-coordinate (bottom boundary).
            x_max (float): Maximum x-coordinate (right boundary).
            y_max (float): Maximum y-coordinate (top boundary).
            obstacles (Optional[List[BaseGeometry]]): An optional list of obstacles.
            beacons (Optional[List[Point]]): An optional list of beacons.
        """
        # Create a rectangular polygon and then use its boundary (LineString)
        self.boundary: LineString = Polygon([
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max)
        ]).boundary
        self.obstacles: List[BaseGeometry] = [] if obstacles is None else obstacles
        self.beacons: List[Point] = [] if beacons is None else beacons
        
        # Store map dimensions
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
    
    def add_obstacle(self, obstacle: BaseGeometry) -> None:
        """
        Add an obstacle to the map.
        
        Args:
            obstacle (BaseGeometry): The obstacle to add.
        """
        self.obstacles.append(obstacle)
    
    def add_beacon(self, x: float, y: float) -> None:
        """
        Add a beacon to the map.
        
        Args:
            x (float): The x-coordinate of the beacon.
            y (float): The y-coordinate of the beacon.
        """
        self.beacons.append(Point(x, y))
    
    def check_collision(self, geometry: BaseGeometry) -> bool:
        """
        Check if the given geometry collides with any obstacle or crosses the boundary.
        
        Args:
            geometry (BaseGeometry): The geometry to check.
            
        Returns:
            bool: True if there is a collision, False otherwise.
        """
        # Check collision with boundary
        if not Polygon(self.boundary).contains(geometry):
            return True
        
        # Check collision with obstacles
        for obstacle in self.obstacles:
            if geometry.intersects(obstacle):
                return True
        
        return False
    
    def get_dimensions(self):
        """
        Get the dimensions of the map.
        
        Returns:
            tuple: (x_min, y_min, x_max, y_max)
        """
        return (self.x_min, self.y_min, self.x_max, self.y_max)


# Create default map instance
DEFAULT_MAP = Map(
    x_min=-10.0,
    y_min=-10.0,
    x_max=10.0,
    y_max=10.0
)

# Add some obstacles
DEFAULT_MAP.add_obstacle(Polygon([(-5, -5), (-3, -5), (-3, -3), (-5, -3)]))
DEFAULT_MAP.add_obstacle(Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]))
DEFAULT_MAP.add_obstacle(Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)]))

# Add some beacons for localization
DEFAULT_MAP.add_beacon(-8.0, -8.0)
DEFAULT_MAP.add_beacon(8.0, -8.0)
DEFAULT_MAP.add_beacon(8.0, 8.0)
DEFAULT_MAP.add_beacon(-8.0, 8.0) 