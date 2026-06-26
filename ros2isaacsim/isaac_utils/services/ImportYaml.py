from ..yaml_utils import read_yaml_config
from .ImportUsds import usd_importer
from isaacsim_msgs.srv import ImportUsd, ImportYaml
from rclpy.qos import QoSProfile

profile = QoSProfile(depth=2000)


def yaml_importer(request, response):
    config = read_yaml_config(request.yaml_path)

    name = config['robot']['name']
    model = config['robot'].get('model', name)
    usd_path = config['robot']['usd_path']
    control = config['robot'].get('control', False)
    position = config['robot'].get('position', [0.0, 0.0, 0.0])
    orientation = config['robot'].get('orientation', [0.0, 0.0, 0.0, 1.0])

    yaml_request = ImportUsd.Request()
    yaml_request.name = name
    yaml_request.model = model
    yaml_request.usd_path = usd_path
    yaml_request.control = bool(control)
    yaml_request.pose.position.x = float(position[0])
    yaml_request.pose.position.y = float(position[1])
    yaml_request.pose.position.z = float(position[2])
    yaml_request.pose.orientation.x = float(orientation[0])
    yaml_request.pose.orientation.y = float(orientation[1])
    yaml_request.pose.orientation.z = float(orientation[2])
    yaml_request.pose.orientation.w = float(orientation[3])

    usd_response = usd_importer(yaml_request, response)
    response.ret = usd_response.ret
    return response


def import_yaml(controller):
    return controller.create_service(
        srv_type=ImportYaml,
        qos_profile=profile,
        srv_name='isaac/import_yaml',
        callback=yaml_importer,
    )
