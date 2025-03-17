#!/usr/bin/env python3
# mapping.py
import numpy as np
import math
from scipy.ndimage import rotate

# 자체 Bresenham 알고리즘 구현
def bresenham_line(x0, y0, x1, y1):
    """
    Bresenham의 선 알고리즘을 구현합니다. 두 점 사이의 모든 셀을 반환합니다.
    
    Args:
        x0, y0: 시작점 좌표
        x1, y1: 종단점 좌표
        
    Returns:
        list: (x, y) 좌표 쌍의 리스트
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    
    return points

class Mapper:
    """
    SLAM 시스템의 맵 업데이트를 담당하는 클래스입니다.
    LiDAR 스캔 데이터를 사용하여 점유 그리드 맵을 생성하고 업데이트합니다.
    """
    
    def __init__(self, resolution=0.05, width=400, height=400):
        """
        매퍼를 초기화합니다.
        
        Args:
            resolution (float): 맵 해상도 (미터/셀)
            width (int): 맵 너비 (셀 단위)
            height (int): 맵 높이 (셀 단위)
        """
        self.resolution = resolution
        self.width = width
        self.height = height
        
        # 로그-오즈 맵 매개변수
        self.log_odds_hit = 0.7   # 맵 업데이트 로그 오즈 증가값 (장애물 확인)
        self.log_odds_miss = 0.4  # 맵 업데이트 로그 오즈 감소값 (빈 공간 확인)
        self.log_odds_max = 3.0   # 최대 로그 오즈 값
        self.log_odds_min = -2.0  # 최소 로그 오즈 값
        
        # 로그-오즈 맵 초기화 (0은 확률 0.5를 의미)
        self.log_odds_map = np.zeros((height, width))
    
    def update(self, occupancy_grid, robot_pose, scan, resolution, map_origin):
        """
        라이다 스캔을 사용하여 점유 그리드 맵을 업데이트합니다.
        
        Args:
            occupancy_grid (np.ndarray): 현재 점유 그리드 맵
            robot_pose (dict): 로봇 포즈 {x, y, theta}
            scan (LaserScan): 라이다 스캔 데이터
            resolution (float): 맵 해상도 (미터/셀)
            map_origin (dict): 맵 원점 {x, y}
            
        Returns:
            np.ndarray: 업데이트된 점유 그리드 맵
        """
        # 맵 크기
        height, width = occupancy_grid.shape
        
        # 로그-오즈 맵 크기가 다른 경우 초기화
        if self.log_odds_map.shape != occupancy_grid.shape:
            self.log_odds_map = np.zeros_like(occupancy_grid, dtype=float)
            # 기존 맵에서 알려진 영역 복사
            for y in range(height):
                for x in range(width):
                    if occupancy_grid[y, x] >= 0:  # 이미 탐색된 영역
                        p = occupancy_grid[y, x] / 100.0  # 확률로 변환
                        if p > 0.5:
                            self.log_odds_map[y, x] = np.log(p / (1 - p))
        
        # 로봇 위치를 맵 좌표로 변환
        robot_x = robot_pose['x']
        robot_y = robot_pose['y']
        robot_theta = robot_pose['theta']
        
        # 로봇 셀 좌표
        robot_cell_x = int((robot_x - map_origin['x']) / resolution)
        robot_cell_y = int((robot_y - map_origin['y']) / resolution)
        
        # 맵 범위 확인
        if not (0 <= robot_cell_x < width and 0 <= robot_cell_y < height):
            return occupancy_grid  # 로봇이 맵 밖에 있으면 업데이트 중단
        
        # 스캔 데이터 처리
        angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
        ranges = np.array(scan.ranges)
        
        # 유효한 범위만 사용
        valid_indices = np.logical_and(
            ranges >= scan.range_min,
            ranges <= scan.range_max
        )
        
        if not np.any(valid_indices):
            return occupancy_grid  # 유효한 스캔 데이터가 없음
        
        valid_angles = angles[valid_indices]
        valid_ranges = ranges[valid_indices]
        
        # 각 유효한 레이저 스캔에 대해 처리
        for i in range(len(valid_ranges)):
            range_val = valid_ranges[i]
            angle = valid_angles[i]
            
            # 글로벌 좌표계에서의 각도
            global_angle = robot_theta + angle
            
            # 빔 종단점 (레이저가 맞은 지점) 계산
            end_x = robot_x + range_val * np.cos(global_angle)
            end_y = robot_y + range_val * np.sin(global_angle)
            
            # 종단점 셀 좌표 계산
            end_cell_x = int((end_x - map_origin['x']) / resolution)
            end_cell_y = int((end_y - map_origin['y']) / resolution)
            
            # 맵 범위 확인
            if not (0 <= end_cell_x < width and 0 <= end_cell_y < height):
                continue  # 종단점이 맵 밖에 있으면 건너뛰기
            
            # 브레젠험 알고리즘으로 레이 트레이싱 (로봇에서 장애물까지)
            cells = bresenham_line(robot_cell_x, robot_cell_y, end_cell_x, end_cell_y)
            
            # 경로상의 모든 셀을 프리 스페이스(빈 공간)로 표시
            for j, (cell_x, cell_y) in enumerate(cells):
                # 맵 범위 확인
                if not (0 <= cell_x < width and 0 <= cell_y < height):
                    continue
                
                # 마지막 셀(장애물)과 경로 셀(빈 공간) 구분
                if j == len(cells) - 1:  # 마지막 셀 (장애물)
                    # 로그 오즈 증가
                    self.log_odds_map[cell_y, cell_x] += self.log_odds_hit
                    if self.log_odds_map[cell_y, cell_x] > self.log_odds_max:
                        self.log_odds_map[cell_y, cell_x] = self.log_odds_max
                else:  # 경로 셀 (빈 공간)
                    # 로그 오즈 감소
                    self.log_odds_map[cell_y, cell_x] -= self.log_odds_miss
                    if self.log_odds_map[cell_y, cell_x] < self.log_odds_min:
                        self.log_odds_map[cell_y, cell_x] = self.log_odds_min
        
        # 로그 오즈에서 점유 그리드로 변환
        updated_grid = np.full_like(occupancy_grid, -1)  # 미탐색 영역은 -1
        
        for y in range(height):
            for x in range(width):
                log_odds = self.log_odds_map[y, x]
                if log_odds != 0:  # 탐색된 영역만 변환
                    p = 1 - (1 / (1 + np.exp(log_odds)))
                    updated_grid[y, x] = int(p * 100)  # 확률을 0-100 범위로 변환
        
        return updated_grid
    
    def reset(self):
        """
        매퍼 상태를 초기화합니다.
        """
        self.log_odds_map = np.zeros((self.height, self.width))
    
    def to_occupancy_grid(self):
        """
        현재 로그-오즈 맵을 점유 그리드로 변환합니다.
        
        Returns:
            np.ndarray: 점유 그리드 맵 (0-100)
        """
        occupancy_grid = np.full_like(self.log_odds_map, -1, dtype=np.int8)
        
        for y in range(self.height):
            for x in range(self.width):
                log_odds = self.log_odds_map[y, x]
                if log_odds != 0:  # 탐색된 영역만 변환
                    p = 1 - (1 / (1 + np.exp(log_odds)))
                    occupancy_grid[y, x] = int(p * 100)
        
        return occupancy_grid 