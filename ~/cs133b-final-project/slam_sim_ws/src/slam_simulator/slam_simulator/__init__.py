"""
SLAM Simulator 패키지는 로봇의 SLAM(Simultaneous Localization and Mapping) 알고리즘을 시뮬레이션하기 위한 ROS 2 패키지입니다.

주요 구성 요소:
- SlamNode: 맵을 생성하고 로봇 위치를 업데이트하는 메인 노드
- Localizer: 로봇의 위치 추정을 담당하는 모듈
- Mapper: 맵 업데이트를 담당하는 모듈
"""

from slam_simulator.SlamNode import SlamNode
from slam_simulator.localization import Localizer
from slam_simulator.mapping import Mapper

__all__ = ['SlamNode', 'Localizer', 'Mapper'] 