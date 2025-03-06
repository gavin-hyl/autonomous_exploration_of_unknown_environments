import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString

class Map:
    """
    A class to represent a map with a rectangular boundary and obstacles.
    The map can check for intersections with geometric objects and plot the map and geometric objects.

    Attributes:
        - boundary (Shapely LineString): The boundary of the map.
        - obstacles (list of Shapely geometries): A list of obstacles
    """
    def __init__(self, x_min, ymin, x_max, y_max, obstacles=None):
        """
        Initialize a map with a rectangular boundary.
        The rectangle is defined by the lower left (minx, miny)
        and upper right (maxx, maxy) coordinates.
        """
        self.boundary = Polygon([
            (x_min, ymin),
            (x_max, ymin),
            (x_max, y_max),
            (x_min, y_max)
        ]).boundary
        self.obstacles = [] if obstacles is None else obstacles


    def _add_obstacle(self, obstacle):
        """
        Add an obstacle to the map.

        Parameters:
            - obstacle (Shapely geometry): A geometric object representing an obstacle.
        
        Returns:
            None
        """
        self.obstacles.append(obstacle)


    def _extract_points(self, geom):
        """
        Helper method to extract points from a geometry:
         - For Point, return [geom].
         - For MultiPoint, return a list of its points.
         - For LineString or LinearRing, return the midpoint.
         - For MultiLineString, return the midpoint of each line.
         - For GeometryCollection, recursively process each geometry.
        
        Parameters:
            - geom (Shapely geometry): The geometry to extract points from.
        
        Returns:
            A list of Shapely Point objects.
        """
        points = []
        if geom.is_empty:
            return points
        if geom.geom_type == 'Point':
            points.append(geom)
        elif geom.geom_type == 'MultiPoint':
            points.extend(list(geom.geoms))
        elif geom.geom_type in ['LineString', 'LinearRing']:
            # Calculate the midpoint of the line (arbitrary choice here)
            mid = geom.interpolate(geom.length / 2)
            points.append(mid)
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                mid = line.interpolate(line.length / 2)
                points.append(mid)
        elif geom.geom_type == 'GeometryCollection':
            for g in geom.geoms:
                points.extend(self._extract_points(g))
        else:
            # For any other geometry type (e.g., Polygon), use the centroid as a fallback.
            points.append(geom.centroid)
        return points


    def intersects(self, geom):
        """
        Returns all the intersections of the given geometric object with the map boundary and the obstacles.

        Parameters:
            - geom (Shapely geometry): A geometric object to check for intersection.
        
        Returns:
            a list of intersections.
        """
        intersections = []
        # Compute intersection with the map boundary
        boundary_intersection = self.boundary.intersection(geom)
        if not boundary_intersection.is_empty:
            intersections.extend(self._extract_points(boundary_intersection))
        # Compute intersections with each obstacle
        for obstacle in self.obstacles:
            obs_intersection = obstacle.intersection(geom)
            if not obs_intersection.is_empty:
                intersections.extend(self._extract_points(obs_intersection))
        return intersections


    def _plot_geom(self, geom, label):
        """
        Add a geometric object to the plot.

        Parameters:
            - geom (Shapely geometry): A geometric object to plot.
            - label (str): A label for the geometric object.
        
        Returns:
            None
        """
        if geom.geom_type == 'Polygon':
            x, y = geom.exterior.xy
            plt.plot(x, y, label=label)
        elif geom.geom_type in ['LineString', 'LinearRing']:
            x, y = geom.xy
            plt.plot(x, y, label=label)
        elif geom.geom_type == 'Point':
            plt.plot(geom.x, geom.y, 'o', label=label)
        else:
            raise ValueError("Unsupported geometry type for plotting: " + geom.geom_type)
        

    def plot_map(self, geometries=None):
        """
        Plot the map's rectangular boundary, any added obstacles, and additional geometric objects.

        Parameters:
            - geometries (list of Shapely geometries): Additional geometric objects to plot.
        
        Returns:
            None (Run plt.show() to display the plot)
        """
        self._plot_geom(self.boundary, 'Map Boundary')

        for i, obstacle in enumerate(self.obstacles):
            self._plot_geom(obstacle, f'Obstacle {i+1}')

        if geometries:
            for i, geom in enumerate(geometries):
                self._plot_geom(geom, f'Geom {i+1}')

        plt.legend()


MAP = Map(0, 0, 10, 10)

obstacle1 = Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])
MAP._add_obstacle(obstacle1)

obstacle2 = LineString([(1, 9), (9, 1)])
MAP._add_obstacle(obstacle2)


# Tests
if __name__ == '__main__':
    my_map = Map(0, 0, 10, 10)

    obstacle1 = Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])
    my_map._add_obstacle(obstacle1)

    obstacle2 = LineString([(1, 9), (9, 1)])
    my_map._add_obstacle(obstacle2)

    point_inside = Point(5, 5)
    point_on_boundary = Point(0, 5)
    line_intersecting = LineString([(-5, 5), (15, 5)])
    line_outside = LineString([(15, 15), (20, 20)])

    print("Intersections for point_inside:", my_map.intersects(point_inside))
    print("Intersections for point_on_boundary:", my_map.intersects(point_on_boundary))
    print("Intersections for line_intersecting:", my_map.intersects(line_intersecting))
    print("Intersections for line_outside:", my_map.intersects(line_outside))

    my_map.plot_map([point_inside, point_on_boundary, line_intersecting, line_outside])
    plt.show()
