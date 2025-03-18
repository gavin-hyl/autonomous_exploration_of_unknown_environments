# slam_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Vector3, PoseStamped, Point
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from multi_slam.Localization import Localization
from multi_slam.Mapping import Mapping
import numpy as np
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, Bool, Float32MultiArray
from sensor_msgs_py.point_cloud2 import read_points
from multi_slam.Planner import Planner


class PlannerSLAMNode(Node):
    """
    Enhanced Simultaneous Localization and Mapping (SLAM) node with integrated Planner.
    
    Processes sensor data from LiDAR and beacons to perform SLAM,
    maintaining a probabilistic map and robot pose estimate.
    Includes planning capabilities for autonomous navigation.
    """
    
    def __init__(self):
        """Initialize the enhanced SLAM node with publishers, subscribers, and parameters."""
        super().__init__("planner_slam_node")
        
        # Parameters
        self.declare_parameter('map_size_x', 50.0)
        self.declare_parameter('map_size_y', 50.0)
        self.declare_parameter('map_origin_x', -25.0)
        self.declare_parameter('map_origin_y', -25.0)
        self.declare_parameter('grid_size', 0.1)
        self.declare_parameter('num_particles', 1000)
        self.declare_parameter('position_std_dev', 0.1)
        self.declare_parameter('initial_noise', 0.5)
        
        # Planner parameters
        self.declare_parameter('use_planner', True)
        self.declare_parameter('rrt_step_size', 0.5)
        self.declare_parameter('rrt_max_iter', 500)
        self.declare_parameter('rrt_goal_sample_rate', 5)
        self.declare_parameter('rrt_connect_circle_dist', 0.5)
        self.declare_parameter('pd_p_gain', 1.0)
        self.declare_parameter('pd_d_gain', 0.1)
        self.declare_parameter('entropy_weight', 1.0)
        self.declare_parameter('beacon_weight', 2.0)
        self.declare_parameter('beacon_attraction_radius', 3.0)

        # Get standard parameters
        map_size_x = self.get_parameter('map_size_x').value
        map_size_y = self.get_parameter('map_size_y').value
        map_origin_x = self.get_parameter('map_origin_x').value
        map_origin_y = self.get_parameter('map_origin_y').value
        grid_size = self.get_parameter('grid_size').value
        num_particles = self.get_parameter('num_particles').value
        position_std_dev = self.get_parameter('position_std_dev').value
        initial_noise = self.get_parameter('initial_noise').value
        
        # Get planner parameters
        self.use_planner = self.get_parameter('use_planner').value
        rrt_step_size = self.get_parameter('rrt_step_size').value
        rrt_max_iter = self.get_parameter('rrt_max_iter').value
        rrt_goal_sample_rate = self.get_parameter('rrt_goal_sample_rate').value
        rrt_connect_circle_dist = self.get_parameter('rrt_connect_circle_dist').value
        pd_p_gain = self.get_parameter('pd_p_gain').value
        pd_d_gain = self.get_parameter('pd_d_gain').value
        entropy_weight = self.get_parameter('entropy_weight').value
        beacon_weight = self.get_parameter('beacon_weight').value
        beacon_attraction_radius = self.get_parameter('beacon_attraction_radius').value
    
        # Initialize Map and Localization
        self.map = Mapping(
            map_size=(map_size_x, map_size_y),
            map_origin=(map_origin_x, map_origin_y),
            grid_size=grid_size
        )
        
        # Initial position (x,y,0) - 2D position with z=0
        initial_position = np.array([0.0, 0.0, 0.0])
        self.position = initial_position
        self.position_cov = np.eye(3) * 0.1  # Initial covariance
        
        # Update rate
        self.dt = 0.02
        
        # Initialize Localization
        self.localization = Localization(
            initial_location=initial_position, 
            initial_noise=initial_noise,
            std_dev_noise=position_std_dev,
            num_particles=num_particles,
            dt=self.dt
        )

        # Initialize Planner
        if self.use_planner:
            self.planner = Planner(
                map_resolution=grid_size,
                rrt_step_size=rrt_step_size,
                rrt_max_iter=rrt_max_iter,
                rrt_goal_sample_rate=rrt_goal_sample_rate,
                rrt_connect_circle_dist=rrt_connect_circle_dist,
                pd_p_gain=pd_p_gain,
                pd_d_gain=pd_d_gain,
                entropy_weight=entropy_weight,
                beacon_weight=beacon_weight,
                beacon_attraction_radius=beacon_attraction_radius
            )
            self.planner.update_map(self.map.get_prob_grid(), self.map.map_origin, self.map.grid_size)
        else:
            self.planner = None
        
        # Lidar range
        self.lidar_range = (0.1, 10.0)  # min and max range in meters
        
        # Current data
        self.lidar_data = []
        self.beacon_data = []
        self.control_input = np.zeros(3)  # vx, vy, 0
        self.pos_hat_new = np.array([0.0, 0.0, 0.0])
        self.particles = None
        self.sim_done = True
        self.planning_active = False
        self.goal_point = None
        self.current_path = []
        
        # 목표 지점 궤적 추적을 위한 변수
        self.goal_history = []  # 과거 목표 지점들을 저장할 리스트
        self.max_goal_history = 10  # 최대 기록할 과거 목표 지점 수

        # Subscribers
        self.create_subscription(
            PointCloud2, "/lidar", self.lidar_callback, 10
        )
        self.create_subscription(
            PointCloud2, "/beacon", self.beacon_callback, 10
        )
        self.create_subscription(
            Vector3, "/control_signal", self.control_callback, 10
        )
        self.create_subscription(
            PointCloud2, "/particles_pred", self.particles_pred_callback, 10
        )
        self.sim_done_sub = self.create_subscription(
            Bool, "/sim_done", self.sim_done_cb, 10
        )
        self.create_subscription(
            Bool, "/use_planner", self.use_planner_callback, 10
        )

        # Publishers
        self.particles_pub = self.create_publisher(PointCloud2, "particles", 10)
        self.slam_done_pub = self.create_publisher(Bool, "/slam_done", 10)
        self.pose_pub = self.create_publisher(Marker, "/pos_hat_viz", 10)
        self.map_pub = self.create_publisher(OccupancyGrid, "/occupancy_grid", 10)
        self.beacon_pub = self.create_publisher(MarkerArray, "/estimated_beacons", 10)
        self.planning_status_pub = self.create_publisher(Bool, "/planning_status", 10)
        self.pose_estimate_pub = self.create_publisher(PoseStamped, "/estimated_pose", 10)
        self.planned_path_pub = self.create_publisher(Marker, "/planned_path", 10)
        self.goal_point_pub = self.create_publisher(Marker, "/goal_point", 10)
        self.goal_history_pub = self.create_publisher(Marker, "/goal_history", 10)
        self.control_pub = self.create_publisher(Vector3, "/planned_control", 10)
        
        # RRT visualization publishers
        self.rrt_tree_pub = self.create_publisher(Marker, "/rrt_tree", 10)
        self.rrt_nodes_pub = self.create_publisher(Marker, "/rrt_nodes", 10)
        self.rrt_samples_pub = self.create_publisher(Marker, "/rrt_samples", 10)
        self.entropy_map_pub = self.create_publisher(OccupancyGrid, "/entropy_map", 10)
        self.boundary_map_pub = self.create_publisher(OccupancyGrid, "/boundary_map", 10)
        
        # Timer for visualization
        self.create_timer(1, self.publish_viz)
        self.create_timer(0.1, self.publish_planning_status)
        self.sim_done_cb(Bool(data=True))  # Call once to initialize


    def sim_done_cb(self, msg: Bool):
        """
        Process simulation step completion and update SLAM state.
        
        This is the main SLAM update loop that runs when the physics simulation
        has completed a step.
        """
        # Localization
        particles, cov = self.localization.update_position(
            self.beacon_data,
            self.map
        )

        pos_hat_new = np.average(particles, axis=0)
        pos_hat_new = pos_hat_new.astype(float)
        pos_hat_new[2] = 0.0

        self.pos_hat_new = pos_hat_new
        self.position = self.pos_hat_new
        self.position_cov = cov
        
        self.map.update(
            robot_pos=self.position,
            robot_cov=self.position_cov,
            lidar_data=self.lidar_data,
            lidar_range=self.lidar_range,
            beacon_data=self.beacon_data
        )
        
        self.get_logger().info("Updated map.")

        # Update planner and generate control if enabled
        if self.use_planner and self.planner is not None:
            self.planner.update_map(self.map.get_prob_grid(), self.map.map_origin, self.map.grid_size)
            self.planner.update_beacons(self.map.beacon_positions)
            self.planner.update_position(self.position)
            
            # Generate control input using planner
            try:
                self.planning_active = True
                control_input, new_goal_point, self.current_path = self.planner.plan_and_control(self.dt)
                
                # 새로운 목표 지점이 이전과 다르면 기록 업데이트
                if new_goal_point is not None and (self.goal_point is None or not np.array_equal(new_goal_point, self.goal_point)):
                    # 이전 목표 지점이 있으면 기록에 추가
                    if self.goal_point is not None:
                        self.goal_history.append(self.goal_point.copy())
                        # 최대 기록 개수 제한
                        if len(self.goal_history) > self.max_goal_history:
                            self.goal_history.pop(0)  # 가장 오래된 기록 제거
                    
                    # 새 목표 지점 업데이트
                    self.goal_point = new_goal_point
                
                # Publish RRT visualization
                self.publish_rrt_visualization()
                self.publish_goal_history()
                
                if control_input is not None:
                    # Publish the control input
                    control_msg = Vector3()
                    control_msg.x = float(control_input[0])
                    control_msg.y = float(control_input[1])
                    control_msg.z = 0.0
                    self.control_pub.publish(control_msg)
                    self.get_logger().info(f"Published control: [{control_input[0]:.2f}, {control_input[1]:.2f}]")
                else:
                    self.get_logger().info("No control input generated.")
                    self.planning_active = False
            except Exception as e:
                self.get_logger().error(f"Error in planner: {str(e)}")
                self.planning_active = False
        else:
            self.planning_active = False

        # Publish position estimate
        self.publish_pose_estimate()
        
        # Publish SLAM results
        self.publish_particles()
        self.publish_planned_path()
        self.publish_goal_point()
        self.slam_done_pub.publish(Bool(data=True))

    def particles_callback(self, msg: PointCloud2):
        """Process particles from visualization."""
        points = list(read_points(msg, field_names=("x", "y", "z")))
        self.particles = np.array([np.array([p[0], p[1], p[2]]) for p in points])

    def publish_viz(self):
        """Publish all visualization messages."""
        self.publish_pos_viz()
        self.publish_map_viz()
        self.publish_beacons_viz()
        self.publish_planned_path()
        self.publish_goal_point()
        self.publish_goal_history()
        self.publish_rrt_visualization()

    def lidar_callback(self, msg: PointCloud2):
        """Process LiDAR data."""
        points = list(read_points(msg, field_names=("x", "y", "z")))
        self.lidar_data = [np.array([p[0], p[1], p[2]]) for p in points]

    def beacon_callback(self, msg: PointCloud2):
        """Process beacon data."""
        points = list(read_points(msg, field_names=("x", "y", "z")))
        self.beacon_data = [np.array([p[0], p[1], p[2]]) for p in points]

    def particles_pred_callback(self, msg: PointCloud2):
        """Process predicted particles from motion model."""
        points = list(read_points(msg, field_names=("x", "y", "z")))
        self.localization.particles = [np.array([p[0], p[1], p[2]]) for p in points]

    def control_callback(self, msg: Vector3):
        """Process control input."""
        self.control_input = np.array([msg.x, msg.y, msg.z])

    def use_planner_callback(self, msg: Bool):
        """Enable or disable planner based on incoming message."""
        self.use_planner = msg.data
        self.get_logger().info(f"Planner {'enabled' if self.use_planner else 'disabled'}")

    def publish_planning_status(self):
        """Publish the current planning status."""
        status_msg = Bool(data=self.planning_active)
        self.planning_status_pub.publish(status_msg)

    def publish_pose_estimate(self):
        """Publish the current position estimate as a PoseStamped message."""
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "map"
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        
        pose_msg.pose.position.x = float(self.position[0])
        pose_msg.pose.position.y = float(self.position[1])
        pose_msg.pose.position.z = 0.0
        
        # No orientation information in our 2D model, so use identity quaternion
        pose_msg.pose.orientation.w = 1.0
        
        self.pose_estimate_pub.publish(pose_msg)

    def publish_particles(self):
        """Publish particle filter state."""
        particles = self.localization.particles

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"
        particles_msg = point_cloud2.create_cloud_xyz32(header, particles)
        self.particles_pub.publish(particles_msg)

    def publish_pos_viz(self):
        """Publish the estimated pose visualization marker."""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "pos_hat_viz"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Position
        marker.pose.position.x = self.position[0]
        marker.pose.position.y = self.position[1]
        marker.pose.position.z = 0.0
        
        # Scale
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        
        # Color
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        self.pose_pub.publish(marker)

    def publish_map_viz(self):
        """Publish occupancy grid for visualization."""
        # Create occupancy grid message
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.info.map_load_time = self.get_clock().now().to_msg()
        msg.info.resolution = self.map.grid_size
        msg.info.width = self.map.grid_width
        msg.info.height = self.map.grid_height
        
        # Set origin (position and orientation)
        msg.info.origin.position.x = self.map.map_origin[0]
        msg.info.origin.position.y = self.map.map_origin[1]
        
        # Convert log-odds to probabilities (0-100)
        probs = self.map.get_prob_grid()
        msg.data = (probs * 100).astype(int).flatten().tolist()
        
        self.map_pub.publish(msg)

    def publish_beacons_viz(self):
        """Publish estimated beacon positions for visualization."""
        marker_array = MarkerArray()
        
        for i, (beacon_pos, beacon_cov) in enumerate(zip(self.map.beacon_positions, self.map.beacon_covariances)):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "estimated_beacons"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            # Position
            marker.pose.position.x = beacon_pos[0]
            marker.pose.position.y = beacon_pos[1]
            marker.pose.position.z = 0.0  # 2D
            
            # Scale - represent uncertainty with size
            uncertainty = np.sqrt(np.trace(beacon_cov[:2, :2]))
            marker.scale.x = max(0.2, min(1.0, uncertainty))
            marker.scale.y = max(0.2, min(1.0, uncertainty))
            marker.scale.z = 0.2  # Fixed height
            
            # Color - blue for beacons
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 0.7
            
            marker_array.markers.append(marker)
            
        self.beacon_pub.publish(marker_array)

    def publish_planned_path(self):
        """Publish the planned path for visualization."""
        if not self.current_path or len(self.current_path) < 2:
            return
            
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "planned_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # Add points to the line strip
        for point in self.current_path:
            p = Point()
            p.x = float(point[0])
            p.y = float(point[1])
            p.z = 0.0
            marker.points.append(p)
        
        # Set scale
        marker.scale.x = 0.05  # Line width
        
        # Set color - green for path
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        
        self.planned_path_pub.publish(marker)

    def publish_goal_point(self):
        """Publish the goal point for visualization."""
        if self.goal_point is None:
            return
        
        # 현재 시간을 텍스트로 표시하기 위해 가져옵니다
        current_time = self.get_clock().now()
        time_str = f"{current_time.seconds_nanoseconds()[0]}"
        
        # 1. 목표 지점 구체 마커 (더 크고 밝은 색상)
        sphere_marker = Marker()
        sphere_marker.header.frame_id = "map"
        sphere_marker.header.stamp = current_time.to_msg()
        sphere_marker.ns = "goal_point"
        sphere_marker.id = 0
        sphere_marker.type = Marker.SPHERE
        sphere_marker.action = Marker.ADD
        
        # 위치
        sphere_marker.pose.position.x = float(self.goal_point[0])
        sphere_marker.pose.position.y = float(self.goal_point[1])
        sphere_marker.pose.position.z = 0.0
        
        # 크기 (더 크게)
        sphere_marker.scale.x = 0.5
        sphere_marker.scale.y = 0.5
        sphere_marker.scale.z = 0.5
        
        # 색상 (밝은 노란색)
        sphere_marker.color.r = 1.0
        sphere_marker.color.g = 1.0
        sphere_marker.color.b = 0.0
        sphere_marker.color.a = 0.8
        
        self.goal_point_pub.publish(sphere_marker)
        
        # 2. 목표 지점 화살표 마커 (위를 향하는 큰 화살표)
        arrow_marker = Marker()
        arrow_marker.header.frame_id = "map"
        arrow_marker.header.stamp = current_time.to_msg()
        arrow_marker.ns = "goal_point_arrow"
        arrow_marker.id = 1
        arrow_marker.type = Marker.ARROW
        arrow_marker.action = Marker.ADD
        
        # 위치
        arrow_marker.pose.position.x = float(self.goal_point[0])
        arrow_marker.pose.position.y = float(self.goal_point[1])
        arrow_marker.pose.position.z = 0.0
        
        # 방향 (위쪽을 가리키도록)
        arrow_marker.pose.orientation.x = 0.0
        arrow_marker.pose.orientation.y = 0.0
        arrow_marker.pose.orientation.z = 0.0
        arrow_marker.pose.orientation.w = 1.0
        
        # 크기
        arrow_marker.scale.x = 0.8  # 화살표 길이
        arrow_marker.scale.y = 0.2  # 화살표 넓이
        arrow_marker.scale.z = 0.2  # 화살표 높이
        
        # 색상 (빨간색)
        arrow_marker.color.r = 1.0
        arrow_marker.color.g = 0.0
        arrow_marker.color.b = 0.0
        arrow_marker.color.a = 0.8
        
        self.goal_point_pub.publish(arrow_marker)
        
        # 3. 목표 지점 텍스트 마커 (좌표와 시간 표시)
        text_marker = Marker()
        text_marker.header.frame_id = "map"
        text_marker.header.stamp = current_time.to_msg()
        text_marker.ns = "goal_point_text"
        text_marker.id = 2
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        
        # 위치 (구체 위에 표시)
        text_marker.pose.position.x = float(self.goal_point[0])
        text_marker.pose.position.y = float(self.goal_point[1])
        text_marker.pose.position.z = 0.7  # 구체 위에 표시
        
        # 텍스트 내용 (좌표와 시간)
        text_marker.text = f"Goal: ({self.goal_point[0]:.2f}, {self.goal_point[1]:.2f}) | t={time_str}"
        
        # 크기
        text_marker.scale.z = 0.3  # 텍스트 높이
        
        # 색상 (흰색)
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 1.0
        
        self.goal_point_pub.publish(text_marker)

    def publish_rrt_visualization(self):
        """Publish visualizations of the RRT planning process."""
        if not self.use_planner or self.planner is None:
            return
            
        # RRT Tree - edge connections
        self.publish_rrt_tree()
        
        # RRT Nodes - vertices
        self.publish_rrt_nodes()
        
        # RRT Samples - random samples that were considered during planning
        self.publish_rrt_samples()
        
        # Entropy and boundary maps
        self.publish_entropy_map()
        self.publish_boundary_map()
    
    def publish_rrt_tree(self):
        """Publish RRT tree edges as line list."""
        if not hasattr(self.planner, 'rrt_edges') or not self.planner.rrt_edges:
            return
            
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "rrt_tree"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        
        # Add points to the line list
        for edge in self.planner.rrt_edges:
            start_point = Point()
            start_point.x = float(edge[0][0])
            start_point.y = float(edge[0][1])
            start_point.z = 0.0
            
            end_point = Point()
            end_point.x = float(edge[1][0])
            end_point.y = float(edge[1][1])
            end_point.z = 0.0
            
            marker.points.append(start_point)
            marker.points.append(end_point)
        
        # Set scale
        marker.scale.x = 0.02  # Line width
        
        # Set color - light blue for tree
        marker.color.r = 0.3
        marker.color.g = 0.7
        marker.color.b = 1.0
        marker.color.a = 0.6
        
        self.rrt_tree_pub.publish(marker)
    
    def publish_rrt_nodes(self):
        """Publish RRT nodes as points."""
        if not hasattr(self.planner, 'rrt_nodes') or not self.planner.rrt_nodes:
            return
            
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "rrt_nodes"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        
        # Add all nodes
        for node in self.planner.rrt_nodes:
            p = Point()
            p.x = float(node[0])
            p.y = float(node[1])
            p.z = 0.0
            marker.points.append(p)
        
        # Set scale
        marker.scale.x = 0.1  # Point width
        marker.scale.y = 0.1  # Point height
        
        # Set color - blue for nodes
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 0.8
        
        self.rrt_nodes_pub.publish(marker)
    
    def publish_rrt_samples(self):
        """Publish random samples used during RRT planning."""
        if not hasattr(self.planner, 'rrt_samples') or not self.planner.rrt_samples:
            return
            
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "rrt_samples"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        
        # Add all samples
        for sample in self.planner.rrt_samples:
            p = Point()
            p.x = float(sample[0])
            p.y = float(sample[1])
            p.z = 0.0
            marker.points.append(p)
        
        # Set scale
        marker.scale.x = 0.05  # Point width
        marker.scale.y = 0.05  # Point height
        
        # Set color - orange for samples
        marker.color.r = 1.0
        marker.color.g = 0.6
        marker.color.b = 0.0
        marker.color.a = 0.5
        
        self.rrt_samples_pub.publish(marker)
    
    def publish_entropy_map(self):
        """Publish entropy map visualization."""
        if not hasattr(self.planner, 'entropy_map') or self.planner.entropy_map is None:
            return
            
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.info.map_load_time = self.get_clock().now().to_msg()
        msg.info.resolution = self.map.grid_size
        msg.info.width = self.map.grid_width
        msg.info.height = self.map.grid_height
        
        # Set origin
        msg.info.origin.position.x = self.map.map_origin[0]
        msg.info.origin.position.y = self.map.map_origin[1]
        
        # Scale entropy values to 0-100 range for visualization
        entropy_scaled = (self.planner.entropy_map * 100).astype(int)
        msg.data = entropy_scaled.flatten().tolist()
        
        self.entropy_map_pub.publish(msg)
    
    def publish_boundary_map(self):
        """Publish boundary map visualization."""
        if not hasattr(self.planner, 'boundary_map') or self.planner.boundary_map is None:
            return
            
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.info.map_load_time = self.get_clock().now().to_msg()
        msg.info.resolution = self.map.grid_size
        msg.info.width = self.map.grid_width
        msg.info.height = self.map.grid_height
        
        # Set origin
        msg.info.origin.position.x = self.map.map_origin[0]
        msg.info.origin.position.y = self.map.map_origin[1]
        
        # Scale boundary values to 0-100 range for visualization
        boundary_scaled = (self.planner.boundary_map * 100).astype(int)
        msg.data = boundary_scaled.flatten().tolist()
        
        self.boundary_map_pub.publish(msg)

    # 목표 지점 궤적 시각화 메서드 추가
    def publish_goal_history(self):
        """Publish history of goal points as a line strip visualization."""
        if not self.goal_history or len(self.goal_history) < 1:
            return
            
        # 현재 목표 지점을 포함한 전체 궤적 생성
        full_history = self.goal_history.copy()
        if self.goal_point is not None:
            full_history.append(self.goal_point)
            
        # 목표 궤적을 LINE_STRIP으로 표시
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "goal_history"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # 각 궤적 지점 추가
        for point in full_history:
            p = Point()
            p.x = float(point[0])
            p.y = float(point[1])
            p.z = 0.1  # 바닥에서 약간 위에 표시
            marker.points.append(p)
        
        # 선 스타일 설정
        marker.scale.x = 0.1  # 선 두께
        
        # 파란색 계열의 궤적
        marker.color.r = 0.0
        marker.color.g = 0.5
        marker.color.b = 1.0
        marker.color.a = 0.8
        
        self.goal_history_pub.publish(marker)
        
        # 각 궤적 지점을 작은 구체로 표시
        points_marker = Marker()
        points_marker.header.frame_id = "map"
        points_marker.header.stamp = self.get_clock().now().to_msg()
        points_marker.ns = "goal_history_points"
        points_marker.id = 1
        points_marker.type = Marker.SPHERE_LIST
        points_marker.action = Marker.ADD
        
        # 각 궤적 지점 추가
        for point in full_history:
            p = Point()
            p.x = float(point[0])
            p.y = float(point[1])
            p.z = 0.1  # 바닥에서 약간 위에 표시
            points_marker.points.append(p)
        
        # 구체 스타일 설정
        points_marker.scale.x = 0.15
        points_marker.scale.y = 0.15
        points_marker.scale.z = 0.15
        
        # 하늘색 계열의 구체
        points_marker.color.r = 0.0
        points_marker.color.g = 0.7
        points_marker.color.b = 1.0
        points_marker.color.a = 0.5
        
        self.goal_history_pub.publish(points_marker)


def main(args=None):
    """Entry point for the planner SLAM node."""
    rclpy.init(args=args)
    node = PlannerSLAMNode()
    node.get_logger().info("Planner SLAM node started")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main() 