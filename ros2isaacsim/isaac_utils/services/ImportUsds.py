import os

import isaac_utils.utils.paths as Paths
import isaacsim.core.utils.prims as prim_utils
import numpy as np
from isaac_utils.utils import geom
from rclpy.qos import QoSProfile

from isaacsim_msgs.srv import ImportUsd

from .utils import safe

profile = QoSProfile(depth=2000)


@safe
def usd_importer(request, response):
    name = request.name
    usd_path = request.usd_path
    prim_path = Paths.scene.path(name)

    prim_utils.create_prim(
        prim_path=prim_path,
        prim_type="Xform",
        position=np.array(geom.Translation.parse(request.pose.position).tuple()),
        orientation=np.array(geom.Rotation.parse(request.pose.orientation).quat()),
        usd_path=os.path.abspath(usd_path) if not usd_path.startswith(("http://", "https://", "omniverse://")) else usd_path,
    )
    try:
        from isaac_utils.scene_collision_repair import inspect_and_repair_scene, config_from_env
        inspect_and_repair_scene(
            scene_root_path=prim_path,
            scene_name=name,
            scene_usd_path=usd_path,
            config=config_from_env(),
        )
    except Exception as e:
        print(f"[scene_collision_repair] failed for {prim_path}: {e}")

    response.ret = True
    return response


def import_usd(controller):
    service = controller.create_service(
        srv_type=ImportUsd,
        qos_profile=profile,
        srv_name='isaac/import_usd',
        callback=usd_importer
    )
    return service
