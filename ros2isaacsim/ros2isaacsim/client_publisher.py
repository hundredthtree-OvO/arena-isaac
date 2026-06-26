import argparse
import os
from pathlib import Path

import yaml

from isaacsim_msgs.msg import Person
from isaacsim_msgs.srv import Pedestrian

import rclpy
from rclpy.node import Node


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
    parsed, ros_args = parser.parse_known_args(args)

    rclpy.init(args=ros_args)

    pedestrian_client = SpawnPedestrians()

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
