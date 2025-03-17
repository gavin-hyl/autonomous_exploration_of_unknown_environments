from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'slam_simulator'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 시작 스크립트 추가
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # RVIZ 설정 파일 추가 (필요 시)
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tkleeneuron',
    maintainer_email='tlee2@caltech.edu',
    description='A ROS 2 package for implementing SLAM (Simultaneous Localization and Mapping)',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'slam_node = slam_simulator.SlamNode:main',
        ],
    },
) 