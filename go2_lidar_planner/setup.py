from glob import glob
from setuptools import find_packages, setup

package_name = 'go2_lidar_planner'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='CWNU Go2 Team',
    maintainer_email='oblivionwine@gmail.com',
    description='Go2 LiDAR potential-field A* planner and path follower.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'lidar_transformer_node = go2_lidar_planner.lidar_transformer_node:main',
            'lidar_accumulator_node = go2_lidar_planner.lidar_accumulator_node:main',
            'potential_astar_planner_node = go2_lidar_planner.potential_astar_planner_node:main',
            'go2_path_follower_node = go2_lidar_planner.go2_path_follower_node:main',
        ],
    },
)
