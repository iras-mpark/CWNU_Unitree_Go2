from glob import glob
from setuptools import find_packages, setup

package_name = 'oak_yolo_target'

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
    description='OAK RGB-D person target publisher using Ultralytics YOLO tracking.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'yolo_target_node = oak_yolo_target.yolo_target_node:main',
        ],
    },
)
