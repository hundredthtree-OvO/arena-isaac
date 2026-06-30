import yaml
import random

def load_yaml_file(agents: str, zones: str):
    
    with open(agents) as f1:
        agent_data = yaml.load(f1, Loader=yaml.SafeLoader)

    with open(zones) as f2:
        zone_data = yaml.load(f2, Loader=yaml.SafeLoader)

    room_list = []

    num_agents = agent_data.get("number_of_agents")
    min_vel = agent_data.get("min_vel")
    max_vel = agent_data.get("max_vel")

    for zone in zone_data:
        room = {}
        coordinate = zone.get("polygon")
        origin = coordinate[0]
        horizon = coordinate[2][0] - coordinate[0][0]
        vertical = coordinate[2][1] - coordinate[0][1]
        room["origin"] = origin
        room["horizon"] = horizon
        room["vertical"] = vertical
        room_list.append(room)
    
    return num_agents, min_vel, max_vel, room_list

def load_config_files(agents: str, zones: str):

    num_agents, min_vel, max_vel, room_list = load_yaml_file(agents, zones)
    
    agent_data = {}

    for agent_id in range(num_agents):

        room_start = random.choice(room_list)
        room_goal = random.choice(room_list)

        while (room_start == room_goal):
            room_goal = random.choice(room_list)

        start_point = [random.uniform(room_start["origin"][0],
                                         room_start["origin"][0] + room_start["horizon"]), 
                       random.uniform(room_start["origin"][1],
                                         room_start["origin"][1] + room_start["vertical"]),
                       0.0]
        
        goal_point = [random.uniform(room_goal["origin"][0],
                                         room_goal["origin"][0] + room_goal["horizon"]), 
                      random.uniform(room_goal["origin"][1],
                                         room_goal["origin"][1] + room_goal["vertical"]),
                      0.0]
        
        orientation = random.uniform(0, 180)

        vel = random.uniform(min_vel, max_vel)

        agent_data[f'agent_{agent_id}'] = {
            'stage_prefix': f'character_{agent_id}',
            'character_name': 'original_male_adult_construction_05',
            'initial_pose': start_point,
            'goal_pose': goal_point,
            'orientation': orientation,
            'controller_stats': False,
            'velocity': vel
        }

    # Writing agent data to a YAML file
    with open('/home/kuro/isaacsim4.2_ws/src/ros2isaacsim/isaac_utils/config/agent_data_gen.yaml', 'w') as file:
        yaml.dump(agent_data, file, default_flow_style=False)

    print("Agent data has been generated")



def main():
    agent_path = "/home/kuro/isaacsim4.2_ws/src/ros2isaacsim/isaac_utils/config/agents.yaml"
    zone_path = "/home/kuro/isaacsim4.2_ws/src/ros2isaacsim/isaac_utils/config/zones.yaml"
    load_config_files(agent_path, zone_path)

if __name__ == "__main__":
    main()

