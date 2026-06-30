import isaac_utils.utils.paths as Paths
import numpy as np
from isaacsim.core.utils.prims import get_prim_at_path
from rclpy.qos import QoSProfile

from isaacsim_msgs.srv import GetPrimAttributes

from .utils import safe

profile = QoSProfile(depth=2000)


@safe
def get_prim_attributes(request, response):
    prim = get_prim_at_path(Paths.scene.path(request.prim_path))

    translate = np.array(prim.GetAttribute("xformOp:translate").Get(), dtype=np.float32)
    quat = prim.GetAttribute("xformOp:orient").Get()

    response.pose.position.x = float(translate[0])
    response.pose.position.y = float(translate[1])
    response.pose.position.z = float(translate[2])

    response.pose.orientation.w = float(quat.real)
    response.pose.orientation.x = float(quat.imaginary[0])
    response.pose.orientation.y = float(quat.imaginary[1])
    response.pose.orientation.z = float(quat.imaginary[2])

    scale = prim.GetAttribute("xformOp:scale").Get()
    if scale is not None:
        response.scale = np.array(scale, dtype=np.float32).tolist()

    return response


def get_prim_attr(controller):
    service = controller.create_service(
        srv_type=GetPrimAttributes,
        qos_profile=profile,
        srv_name='isaac/get_prim_attributes',
        callback=get_prim_attributes
    )
    return service
