import argparse
import os
from pathlib import Path

import yaml

from isaacsim_msgs.msg import Person
from isaacsim_msgs.srv import Pedestrian

import rclpy
from rclpy.node import Node

from ros2isaacsim.pedestrian_auto_sampler import (
    AutoPedestrianConfig,
    VoxelPedestrianSampler,
    dump_agents_yaml,
)


class SpawnPedestrians(Node):

    def __init__(self):
        super().__init__('Spawn_peds')
        self.cli = self.create_client(Pedestrian, '/isaac/spawn_pedestrian')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = Pedestrian.Request()

    def send_request(self, people):
        self.req.people = people
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


def _default_agents_yaml() -> str:
    candidates = []
    arena_ws = os.environ.get('ARENA_WS') or os.environ.get('ARENA_WS_DIR')
    if arena_ws:
        candidates.append(Path(arena_ws) / 'src/arena/arena-isaac/ros2isaacsim/isaac_utils/config/agent_data_gen.yaml')
        candidates.append(Path(arena_ws) / 'src/arena/arena-isaac/ros2isaacsim/isaac_utils/config/agents.yaml')
    candidates.append(Path.cwd() / 'src/arena/arena-isaac/ros2isaacsim/isaac_utils/config/agent_data_gen.yaml')
    candidates.append(Path(__file__).resolve().parents[1] / 'isaac_utils/config/agent_data_gen.yaml')
    for p in candidates:
        if p.exists():
            return str(p)
    return str(candidates[0]) if candidates else 'agent_data_gen.yaml'


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--agents-yaml', default=_default_agents_yaml(), help='Path to agent_data_gen.yaml')
    parser.add_argument('--auto-map', default='', help='Optional voxel map path for automatic pedestrian sampling')
    parser.add_argument('--auto-count', type=int, default=2, help='Number of pedestrians to auto sample')
    parser.add_argument('--auto-seed', type=int, default=12345, help='Random seed for auto sampling')
    parser.add_argument('--auto-clearance', type=float, default=0.35, help='Clearance from walls/obstacles in meters')
    parser.add_argument('--auto-min-path-length', type=float, default=1.2, help='Minimum route length in meters')
    parser.add_argument('--auto-max-path-length', type=float, default=5.5, help='Maximum route length in meters')
    parser.add_argument('--auto-boundary-margin', type=float, default=0.20, help='Margin from voxel bounds in meters')
    parser.add_argument('--auto-velocity-min', type=float, default=0.35, help='Minimum pedestrian speed in m/s')
    parser.add_argument('--auto-velocity-max', type=float, default=0.50, help='Maximum pedestrian speed in m/s')
    parser.add_argument('--auto-stage-prefix', default='social_ped_', help='Stage prefix for auto sampled pedestrians')
    parser.add_argument('--auto-character-names', default='', help='Comma separated Isaac People character names')
    parser.add_argument('--dump-generated-yaml', default='', help='Optional path to write generated pedestrian YAML for debugging')
    parsed, ros_args = parser.parse_known_args(args)

    rclpy.init(args=ros_args)

    pedestrian_client = SpawnPedestrians()

    if parsed.auto_map:
        character_names = [
            name.strip() for name in str(parsed.auto_character_names).split(",")
            if name.strip()
        ]
        sampler = VoxelPedestrianSampler(
            AutoPedestrianConfig(
                map_path=str(Path(parsed.auto_map).expanduser()),
                count=max(0, parsed.auto_count),
                seed=parsed.auto_seed,
                clearance=max(0.0, parsed.auto_clearance),
                min_path_length=max(0.2, parsed.auto_min_path_length),
                max_path_length=max(parsed.auto_min_path_length, parsed.auto_max_path_length),
                boundary_margin=max(0.0, parsed.auto_boundary_margin),
                velocity_min=min(parsed.auto_velocity_min, parsed.auto_velocity_max),
                velocity_max=max(parsed.auto_velocity_min, parsed.auto_velocity_max),
                stage_prefix=str(parsed.auto_stage_prefix),
                character_names=character_names or None,
            )
        )
        agent_data = sampler.sample_agents()
        if parsed.dump_generated_yaml:
            dump_agents_yaml(str(Path(parsed.dump_generated_yaml).expanduser()), agent_data)
    else:
        agents_gen_data_path = Path(parsed.agents_yaml).expanduser()
        if not agents_gen_data_path.exists():
            raise FileNotFoundError(f'Pedestrian YAML not found: {agents_gen_data_path}')

        with agents_gen_data_path.open() as f:
            agent_data = yaml.load(f, Loader=yaml.SafeLoader)

    people = []
    for agent in agent_data.values():
        person = Person()
        person.stage_prefix = agent.get('stage_prefix')
        person.character_name = agent.get('character_name')
        person.initial_pose = agent.get('initial_pose')
        person.goal_pose = agent.get('goal_pose')
        path_points = agent.get('path_points') or []
        if path_points:
            flat = []
            for p in path_points:
                flat.extend([float(p[0]), float(p[1]), float(p[2] if len(p) > 2 else 0.0)])
            person.path_points_flat = flat
        person.loop_path = bool(agent.get('loop_path', True))
        person.orientation = agent.get('orientation')
        person.controller_stats = bool(agent.get('controller_stats', False))
        person.velocity = float(agent.get('velocity', 1.0))
        people.append(person)

    response = pedestrian_client.send_request(people)
    pedestrian_client.get_logger().info(f'Spawn success: {response.ret}')

    pedestrian_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
