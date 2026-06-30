import yaml

def load_yaml_file(agents_gen_data: str):
    
    with open(agents_gen_data) as f1:
        agent_data = yaml.load(f1, Loader=yaml.SafeLoader)

    agent_list = ()

    for agent in agent_data.values():
        stage_prefix = agent.get("stage_prefix")
        character_name = agent.get("character_name")
        initial_pose = [round(agent.get("initial_pose")[0], 2), round(agent.get("initial_pose")[1], 2), round(agent.get("initial_pose")[2], 2)]
        goal_pose = agent.get("goal_pose")
        orientation = agent.get("orientation")
        controller_stats = agent.get("controller_stats")
        velocity = agent.get("velocity")

        print(type(velocity))

def main():
    load_yaml_file("/home/kuro/isaacsim4.2_ws/src/ros2isaacsim/isaac_utils/config/agent_data_gen.yaml")


if __name__ == "__main__":
    main()