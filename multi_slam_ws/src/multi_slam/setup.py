from glob import glob
from setuptools import find_packages, setup

package_name = 'multi_slam'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gavin',
    maintainer_email='gavin.hyl@outlook.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'controller_node = multi_slam.ControllerNode:main',
            'simulation_node = multi_slam.SimulationNode:main',
            'beacons_node = multi_slam.BeaconsNode:main',
            'lidar_node = multi_slam.LidarNode:main',
            'visualization_node = multi_slam.VisualizationNode:main',
        ],
    },
)

