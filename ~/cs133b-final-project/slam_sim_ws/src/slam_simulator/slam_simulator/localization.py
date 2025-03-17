#!/usr/bin/env python3
# localization.py
import numpy as np
from scipy.stats import norm
import math

class Localizer:
    """
    로봇의 위치 추정을 담당하는 클래스입니다.
    파티클 필터 또는 EKF(Extended Kalman Filter)를 사용해 로봇의 위치를 추정합니다.
    """
    
    def __init__(self, num_particles=200, motion_noise=(0.05, 0.05, 0.02)):
        """
        로컬라이저를 초기화합니다.
        
        Args:
            num_particles (int): 파티클 필터에 사용할 파티클 수
            motion_noise (tuple): 이동 모델 노이즈 (x, y, theta)
        """
        self.num_particles = num_particles
        self.motion_noise = motion_noise
        
        # 파티클 필터 관련 변수
        self.particles = None  # 파티클 배열 [x, y, theta, weight]
        self.initialized = False
        
        # 스캔 매칭 매개변수
        self.max_scan_range = 10.0  # 미터
        self.min_scan_match_score = 0.3
    
    def initialize_particles(self, initial_pose):
        """
        파티클을 초기 위치 주변에 분포시켜 초기화합니다.
        
        Args:
            initial_pose (dict): 초기 로봇 포즈 {x, y, theta}
        """
        self.particles = np.zeros((self.num_particles, 4))  # x, y, theta, weight
        
        # 초기 위치 주변에 파티클 분포
        for i in range(self.num_particles):
            self.particles[i, 0] = initial_pose['x'] + np.random.normal(0, self.motion_noise[0])
            self.particles[i, 1] = initial_pose['y'] + np.random.normal(0, self.motion_noise[1])
            self.particles[i, 2] = initial_pose['theta'] + np.random.normal(0, self.motion_noise[2])
            self.particles[i, 3] = 1.0 / self.num_particles  # 초기 가중치 균등 분포
        
        self.initialized = True
    
    def update(self, robot_pose, scan, odom, occupancy_grid, resolution, map_origin):
        """
        로봇의 위치를 업데이트합니다.
        
        Args:
            robot_pose (dict): 현재 로봇 포즈 추정치 {x, y, theta}
            scan (LaserScan): 라이다 스캔 데이터
            odom (Odometry): 오도메트리 데이터
            occupancy_grid (np.ndarray): 현재 점유 그리드 맵
            resolution (float): 맵 해상도 (미터/셀)
            map_origin (dict): 맵 원점 {x, y}
            
        Returns:
            dict: 업데이트된 로봇 포즈 {x, y, theta}
        """
        # 파티클 필터 초기화 확인
        if not self.initialized:
            self.initialize_particles(robot_pose)
            return robot_pose
        
        # 1. 이동 모델: 오도메트리 기반 파티클 이동 예측
        self.predict_particles(odom)
        
        # 2. 측정 모델: 라이다 스캔 데이터를 사용해 파티클 가중치 업데이트
        self.update_weights(scan, occupancy_grid, resolution, map_origin)
        
        # 3. 리샘플링: 가중치 기반으로 파티클 리샘플링
        self.resample_particles()
        
        # 4. 현재 추정 위치 계산: 파티클의 가중 평균
        estimated_pose = self.calculate_pose_estimate()
        
        # 결과 반환
        return estimated_pose
    
    def predict_particles(self, odom):
        """
        오도메트리 데이터를 기반으로 파티클의 위치를 예측합니다.
        
        Args:
            odom (Odometry): 오도메트리 데이터
        """
        # 오도메트리 데이터에서 선형 및 각속도 추출
        linear_vel = np.sqrt(odom.twist.twist.linear.x**2 + odom.twist.twist.linear.y**2)
        angular_vel = odom.twist.twist.angular.z
        
        # 델타 타임 계산 (여기서는 간단히 0.1초로 가정)
        dt = 0.1
        
        # 각 파티클 이동 예측
        for i in range(self.num_particles):
            # 현재 파티클 상태
            x, y, theta = self.particles[i, :3]
            
            # 노이즈 추가
            noisy_linear = linear_vel + np.random.normal(0, self.motion_noise[0])
            noisy_angular = angular_vel + np.random.normal(0, self.motion_noise[2])
            
            # 이동 모델 적용
            if abs(noisy_angular) < 1e-3:  # 직선 이동
                x += noisy_linear * dt * np.cos(theta)
                y += noisy_linear * dt * np.sin(theta)
            else:  # 곡선 이동
                x += (noisy_linear / noisy_angular) * (np.sin(theta + noisy_angular * dt) - np.sin(theta))
                y += (noisy_linear / noisy_angular) * (-np.cos(theta + noisy_angular * dt) + np.cos(theta))
            
            theta += noisy_angular * dt
            
            # 각도 정규화 (-pi, pi)
            theta = self.normalize_angle(theta)
            
            # 파티클 업데이트
            self.particles[i, 0] = x
            self.particles[i, 1] = y
            self.particles[i, 2] = theta
    
    def update_weights(self, scan, occupancy_grid, resolution, map_origin):
        """
        라이다 스캔과 맵을 비교하여 파티클 가중치를 업데이트합니다.
        
        Args:
            scan (LaserScan): 라이다 스캔 데이터
            occupancy_grid (np.ndarray): 점유 그리드 맵
            resolution (float): 맵 해상도 (미터/셀)
            map_origin (dict): 맵 원점 {x, y}
        """
        # 맵 정보
        height, width = occupancy_grid.shape
        
        # 스캔 데이터 추출
        angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
        ranges = np.array(scan.ranges)
        
        # 유효한 범위만 사용
        valid_indices = np.logical_and(
            ranges >= scan.range_min,
            ranges <= scan.range_max
        )
        
        if not np.any(valid_indices):
            return  # 유효한 스캔 데이터가 없음
        
        valid_angles = angles[valid_indices]
        valid_ranges = ranges[valid_indices]
        
        # 각 파티클에 대해 가중치 업데이트
        for i in range(self.num_particles):
            x, y, theta = self.particles[i, :3]
            
            # 파티클 위치에서 예상되는 거리 계산
            score = 0
            valid_matches = 0
            
            for j in range(len(valid_ranges)):
                angle = valid_angles[j]
                measured_range = valid_ranges[j]
                
                # 글로벌 각도 계산
                global_angle = self.normalize_angle(theta + angle)
                
                # 측정된 거리 위치 계산
                end_x = x + measured_range * np.cos(global_angle)
                end_y = y + measured_range * np.sin(global_angle)
                
                # 맵 좌표로 변환
                cell_x = int((end_x - map_origin['x']) / resolution)
                cell_y = int((end_y - map_origin['y']) / resolution)
                
                # 맵 범위 내 확인
                if 0 <= cell_x < width and 0 <= cell_y < height:
                    # 장애물이 있는 경우 점수 증가
                    if occupancy_grid[cell_y, cell_x] > 50:  # 50%보다 높은 점유 확률
                        score += 1
                    valid_matches += 1
            
            # 최종 점수 계산
            if valid_matches > 0:
                match_score = score / valid_matches
                # 정규 분포를 사용한 가중치 계산
                self.particles[i, 3] = np.exp(-10 * (1 - match_score)**2)
            else:
                self.particles[i, 3] = 0.1  # 최소 가중치
        
        # 가중치 정규화
        weight_sum = np.sum(self.particles[:, 3])
        if weight_sum > 0:
            self.particles[:, 3] /= weight_sum
    
    def resample_particles(self):
        """
        가중치에 따라 파티클을 리샘플링합니다.
        """
        # 가중치 배열
        weights = self.particles[:, 3]
        
        # 누적 합 계산
        cumsum = np.cumsum(weights)
        
        # 시작점 선택
        r = np.random.random() / self.num_particles
        
        # 새 파티클 인덱스 및 파티클
        new_particles = np.zeros_like(self.particles)
        
        # 리샘플링 휠 알고리즘
        i, j = 0, 0
        while i < self.num_particles:
            while cumsum[j] < r:
                j = (j + 1) % self.num_particles
            
            # 파티클 복사
            new_particles[i] = self.particles[j].copy()
            
            # 균등 가중치 할당
            new_particles[i, 3] = 1.0 / self.num_particles
            
            # 포인터 이동
            r += 1.0 / self.num_particles
            i += 1
        
        # 파티클 교체
        self.particles = new_particles
    
    def calculate_pose_estimate(self):
        """
        파티클의 가중 평균으로 로봇의 현재 위치를 추정합니다.
        
        Returns:
            dict: 추정된 로봇 포즈 {x, y, theta}
        """
        # 가중치
        weights = self.particles[:, 3]
        
        # 가중 평균 계산
        x_mean = np.average(self.particles[:, 0], weights=weights)
        y_mean = np.average(self.particles[:, 1], weights=weights)
        
        # 각도는 단순 평균이 아닌 특별한 처리가 필요
        sin_sum = np.sum(np.sin(self.particles[:, 2]) * weights)
        cos_sum = np.sum(np.cos(self.particles[:, 2]) * weights)
        theta_mean = np.arctan2(sin_sum, cos_sum)
        
        return {
            'x': x_mean,
            'y': y_mean,
            'theta': theta_mean
        }
    
    def normalize_angle(self, angle):
        """
        각도를 -pi에서 pi 사이로 정규화합니다.
        
        Args:
            angle (float): 정규화할 각도
            
        Returns:
            float: 정규화된 각도
        """
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle 