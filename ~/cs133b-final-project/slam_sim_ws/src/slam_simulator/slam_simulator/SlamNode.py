#!/usr/bin/env python3
# SlamNode.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, Twist, Vector3
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import time
import math
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

# 로컬 모듈 임포트
from slam_simulator.localization import Localizer
from slam_simulator.mapping import Mapper

class SlamNode(Node):
    def __init__(self):
        super().__init__('slam_node')
        
        # 파라미터 선언
        self.declare_parameter('map_resolution', 0.05)  # 미터당 그리드 셀 수
        self.declare_parameter('map_width', 20.0)  # 미터 단위 맵 너비
        self.declare_parameter('map_height', 20.0)  # 미터 단위 맵 높이
        self.declare_parameter('update_frequency', 10.0)  # Hz
        self.declare_parameter('exploration_weight', 0.7)  # 탐색 가중치
        
        # 파라미터 가져오기
        self.map_resolution = self.get_parameter('map_resolution').value
        self.map_width = self.get_parameter('map_width').value
        self.map_height = self.get_parameter('map_height').value
        self.update_frequency = self.get_parameter('update_frequency').value
        self.exploration_weight = self.get_parameter('exploration_weight').value
        
        # 맵 초기화 (빈 맵)
        self.initialize_empty_map()
        
        # 로봇 위치 초기화 (맵 중앙)
        self.robot_pose = {
            'x': 0.0,
            'y': 0.0,
            'theta': 0.0
        }
        
        # 찾은 비콘 목록
        self.discovered_beacons = []
        
        # 로컬라이저 및 매퍼 초기화
        self.localizer = Localizer()
        self.mapper = Mapper(
            resolution=self.map_resolution,
            width=int(self.map_width / self.map_resolution),
            height=int(self.map_height / self.map_resolution)
        )
        
        # Publishers
        self.map_pub = self.create_publisher(OccupancyGrid, '/slam/map', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/slam/robot_pose', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.beacon_markers_pub = self.create_publisher(MarkerArray, '/slam/beacons', 10)
        
        # TF 브로드캐스터
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Subscribers
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        self.beacon_sub = self.create_subscription(
            PointCloud2,
            '/beacon',
            self.beacon_callback,
            10
        )
        
        # SLAM 메인 루프를 위한 타이머
        self.timer = self.create_timer(1.0 / self.update_frequency, self.slam_loop)
        
        # 디버깅 정보
        self.get_logger().info("SLAM 노드가 초기화되었습니다.")
    
    def initialize_empty_map(self):
        """빈 맵을 초기화합니다."""
        width_cells = int(self.map_width / self.map_resolution)
        height_cells = int(self.map_height / self.map_resolution)
        
        # -1: 미탐색, 0: 비어 있음, 100: 점유됨
        self.occupancy_grid = np.full((height_cells, width_cells), -1, dtype=np.int8)
        
        # 맵 원점 (맵 왼쪽 하단 모서리)
        self.map_origin = {
            'x': -self.map_width / 2.0,
            'y': -self.map_height / 2.0
        }
        
        self.get_logger().info(f"빈 맵이 초기화되었습니다. 크기: {width_cells}x{height_cells} 셀")
    
    def lidar_callback(self, msg):
        """라이다 데이터를 처리합니다."""
        # 로봇 현재 위치에서 라이다 스캔을 사용하여 장애물 검출
        self.latest_scan = msg
    
    def odom_callback(self, msg):
        """오도메트리 데이터를 처리합니다."""
        # 오도메트리 데이터를 로봇의 위치 추정에 사용
        self.latest_odom = msg
    
    def beacon_callback(self, msg):
        """비콘 감지 데이터를 처리합니다."""
        # 비콘 측정값을 저장
        self.latest_beacon = msg
    
    def slam_loop(self):
        """메인 SLAM 루프: 로컬라이제이션, 매핑 및 제어를 수행합니다."""
        # 1. 로컬라이제이션: 로봇의 현재 위치 추정
        if hasattr(self, 'latest_scan') and hasattr(self, 'latest_odom'):
            self.robot_pose = self.localizer.update(
                self.robot_pose, 
                self.latest_scan, 
                self.latest_odom, 
                self.occupancy_grid,
                self.map_resolution,
                self.map_origin
            )
            self.publish_pose()
            self.broadcast_tf()
        
        # 2. 매핑: 현재 위치 및 센서 데이터를 기반으로 맵 업데이트
        if hasattr(self, 'latest_scan'):
            self.occupancy_grid = self.mapper.update(
                self.occupancy_grid,
                self.robot_pose,
                self.latest_scan,
                self.map_resolution,
                self.map_origin
            )
            self.publish_map()
        
        # 3. 비콘 검출 및 관리
        if hasattr(self, 'latest_beacon'):
            self.process_beacons()
        
        # 4. 제어 업데이트: 비콘을 효율적으로 탐색하기 위한 전략
        cmd_vel = self.compute_control()
        self.cmd_vel_pub.publish(cmd_vel)
    
    def process_beacons(self):
        """비콘 데이터를 처리하고 시각화합니다."""
        # 여기서 비콘 정보를 추출하고 관리
        # 비콘 마커 발행
        marker_array = MarkerArray()
        # 비콘 처리 및 마커 생성 로직
        self.beacon_markers_pub.publish(marker_array)
    
    def compute_control(self):
        """비콘을 효과적으로 찾기 위한 제어 명령을 계산합니다."""
        cmd = Twist()
        
        # 미탐색 영역을 향한 이동과 비콘 검색 사이의 균형을 조정
        # 예: 프론티어 기반 탐색 또는 정보 이득 최대화
        
        # 간단한 예시: 랜덤 이동
        if np.random.random() < 0.1:  # 10% 확률로 방향 변경
            cmd.angular.z = (np.random.random() - 0.5) * 2.0  # -1.0 ~ 1.0 라디안/초
        else:
            cmd.linear.x = 0.2  # 0.2 m/s 전진
        
        return cmd
    
    def publish_map(self):
        """현재 맵을 발행합니다."""
        map_msg = OccupancyGrid()
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = "map"
        
        map_msg.info.resolution = self.map_resolution
        map_msg.info.width = self.occupancy_grid.shape[1]
        map_msg.info.height = self.occupancy_grid.shape[0]
        map_msg.info.origin.position.x = self.map_origin['x']
        map_msg.info.origin.position.y = self.map_origin['y']
        
        # NumPy 배열을 1D로 변환 (행 우선)
        map_msg.data = self.occupancy_grid.flatten().tolist()
        
        self.map_pub.publish(map_msg)
    
    def publish_pose(self):
        """로봇의 추정 위치를 발행합니다."""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"
        
        pose_msg.pose.position.x = self.robot_pose['x']
        pose_msg.pose.position.y = self.robot_pose['y']
        
        # 오일러 각도(theta)를 쿼터니언으로 변환
        theta = self.robot_pose['theta']
        pose_msg.pose.orientation.z = math.sin(theta / 2.0)
        pose_msg.pose.orientation.w = math.cos(theta / 2.0)
        
        self.pose_pub.publish(pose_msg)
    
    def broadcast_tf(self):
        """로봇의 TF를 브로드캐스트합니다."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "base_link"
        
        t.transform.translation.x = self.robot_pose['x']
        t.transform.translation.y = self.robot_pose['y']
        t.transform.translation.z = 0.0
        
        # 오일러 각도를 쿼터니언으로 변환
        theta = self.robot_pose['theta']
        t.transform.rotation.z = math.sin(theta / 2.0)
        t.transform.rotation.w = math.cos(theta / 2.0)
        
        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = SlamNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('SLAM 노드가 사용자에 의해 종료되었습니다.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 