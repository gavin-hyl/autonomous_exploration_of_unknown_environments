from .Map import Map

# Create a global WorldMap instance that can be imported and shared
# Setting reasonable default map bounds (-10 to 10 in both x and y)
global_world_map = Map(x_min=-10.0, y_min=-10.0, x_max=10.0, y_max=10.0)