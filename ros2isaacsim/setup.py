from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ros2isaacsim'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'isaac_utils', 'config'), glob('isaac_utils/config/*.yaml')),
        (os.path.join('share', package_name, 'isaac_utils', 'config', 'robot'), glob('isaac_utils/config/robot/*.yaml')),
        (os.path.join('share', package_name, 'assets', 'scenes'), glob('ros2isaacsim/assets/scenes/*.usd')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shibuina',
    maintainer_email='anhddhe180559@fpt.edu.vn',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            "run_isaacsim=ros2isaacsim.run_isaacsim:main",
            "convert_urdf_usd=ros2isaacsim.convert_urdf_usd:main",
            'navigation_controller = ros2isaacsim.navigation_controller:main',
            'sdf_to_urdf=ros2isaacsim.SdftoUrdf:main',
            'agent_rl=ros2isaacsim.agent_RL:main',
            "client_pub_ped=ros2isaacsim.client_publisher:main",
            "spawn_mecanum_teleop=ros2isaacsim.spawn_mecanum_teleop:main",
            "wasd_combo_teleop=ros2isaacsim.wasd_combo_teleop:main",
            "cmd_vel_pulse=ros2isaacsim.cmd_vel_pulse:main",
            "spawn_v10_scene_lidar_validation=ros2isaacsim.spawn_v10_scene_lidar_validation:main",
            "export_collision_proxies=ros2isaacsim.export_collision_proxies:main",
            "export_voxel_map=ros2isaacsim.export_voxel_map:main",
        ],
    },
)
