from typing import List, Optional
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

        The rectangle is defined by the lower-left corner (x_min, y_min) and
        the upper-right corner (x_max, y_max).

        Args:
            x_min (float): Minimum x-coordinate (left boundary).
            y_min (float): Minimum y-coordinate (bottom boundary).
            x_max (float): Maximum x-coordinate (right boundary).
            y_max (float): Maximum y-coordinate (top boundary).
            obstacles (Optional[List[BaseGeometry]]): An optional list of obstacles.
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

    def _add_obstacle(self, obstacle: BaseGeometry) -> None:
        """
        Add an obstacle to the map.

        Args:
            obstacle (BaseGeometry): A geometric object representing an obstacle.
        """
        self.obstacles.append(obstacle)

    def _add_beacon(self, beacon: Point) -> None:
        """
        Add a beacon to the map.

        Args:
            beacon (Point): A point representing the beacon's location.
        """
        self.beacons.append(beacon)

    def _extract_points(self, geom: BaseGeometry) -> List[Point]:
        """
        Extract representative points from a geometry.

        - For a Point, returns [geom].
        - For a MultiPoint, returns all its points.
        - For LineString or LinearRing, returns the midpoint.
        - For MultiLineString, returns the midpoint of each line.
        - For GeometryCollection, recursively processes each geometry.
        - For other types (e.g., Polygon), returns the centroid as a fallback.

        Args:
            geom (BaseGeometry): The geometry to extract points from.

        Returns:
            List[Point]: A list of representative points extracted from the geometry.
        """
        points: List[Point] = []
        if geom.is_empty:
            return points

        if geom.geom_type == 'Point':
            points.append(geom)
        elif geom.geom_type == 'MultiPoint':
            points.extend(list(geom.geoms))
        elif geom.geom_type in ['LineString', 'LinearRing']:
            # Compute the midpoint of the line
            mid: Point = geom.interpolate(geom.length / 2)
            points.append(mid)
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                mid = line.interpolate(line.length / 2)
                points.append(mid)
        elif geom.geom_type == 'GeometryCollection':
            for g in geom.geoms:
                points.extend(self._extract_points(g))
        else:
            # For geometries like Polygon, use the centroid as a fallback.
            points.append(geom.centroid)
        return points

    def intersects(self, geom: BaseGeometry) -> List[Point]:
        """
        Find intersections between a given geometry and the map's features.

        Checks for intersections with the map's boundary and all obstacles,
        and returns representative intersection points.

        Args:
            geom (BaseGeometry): The geometry to check for intersections.

        Returns:
            List[Point]: A list of points where intersections occur.
        """
        intersections: List[Point] = []
        
        # Intersection with the map boundary
        boundary_intersection = self.boundary.intersection(geom)
        if not boundary_intersection.is_empty:
            intersections.extend(self._extract_points(boundary_intersection))
        
        # Intersection with each obstacle
        for obstacle in self.obstacles:
            obs_intersection = obstacle.intersection(geom)
            if not obs_intersection.is_empty:
                intersections.extend(self._extract_points(obs_intersection))
        
        return intersections


# Example usage:

# Create a map with a rectangular boundary from (0,0) to (10,10)
MAP = Map(0, 0, 10, 10)

# Create and add a polygon obstacle
obstacle1 = Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])
MAP._add_obstacle(obstacle1)

# Create and add a line obstacle
obstacle2 = LineString([(1, 9), (9, 1)])
MAP._add_obstacle(obstacle2)
