# fmt: off

from .polyfill import *

# Optional Arena-Rosnav imports. Some installations do not ship the nested
# arena_simulation_setup.utils.cattrs module; Isaac bridge should still start.
try:
    import arena_simulation_setup  # noqa: F401
except ModuleNotFoundError:
    arena_simulation_setup = None
try:
    import arena_simulation_setup.utils.cattrs  # noqa: F401
except ModuleNotFoundError:
    pass

import os
import time
import gc


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}


# Default to a lightweight bridge. People/AnimGraph/Replicator/NavMesh/material
# extensions are expensive and can grow RSS even while the bridge is idle. Enable
# them explicitly only for pedestrian/people tests.
ENABLE_PEOPLE_STACK = _env_bool('ARENA_ISAAC_ENABLE_PEOPLE_STACK', False)
ENABLE_MATERIAL_STACK = _env_bool('ARENA_ISAAC_ENABLE_MATERIAL_STACK', False)
ENABLE_NAVMESH = _env_bool('ARENA_ISAAC_ENABLE_NAVMESH', False)

ENABLE_CHARACTER_SERVICES = _env_bool('ARENA_ISAAC_ENABLE_CHARACTER_SERVICES', False)
ADD_GROUND_PLANE = _env_bool('ARENA_ISAAC_ADD_GROUND_PLANE', True)

from isaac_utils.animgraph_people import ensure_animgraph_experience

# Use Isaac Sim to import SimulationApp.  For Isaac Sim 4.5, preload the
# AnimGraph/People/IRA extensions through a custom .kit experience.
from isaacsim import SimulationApp


CONFIG = {"renderer": "Wireframe", "headless": False}
_EXPERIENCE = ensure_animgraph_experience(os.environ.get("ISAAC_PATH")) if ENABLE_PEOPLE_STACK else None
if _EXPERIENCE:
    simulation_app = SimulationApp(CONFIG, experience=_EXPERIENCE)
else:
    simulation_app = SimulationApp(CONFIG)
simulation_app.update()


# Import Isaac Sim dependencies
import carb
import omni.usd
import omni.timeline
from isaacsim.core.api.world import World
from isaacsim.core.utils import prims
from isaacsim.core.api import SimulationContext
from pxr import Sdf, Gf, UsdLux
import yaml
from isaacsim.core.utils import extensions, stage
from isaac_utils.utils.assets import get_assets_root_path_safe
from isaac_utils.animgraph_people import enable_animgraph_people_extensions
from isaac_utils.rtx_sensor_settings import ensure_rtx_sensor_coord_frame

from isaacsim.core.utils.prims import set_prim_attribute_value
from isaacsim.asset.importer.urdf import _urdf
import omni.kit.commands as commands
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import get_current_stage, open_stage
import random

# People extension loading is intentionally split into modes.
# In Isaac Sim 4.5 the full People/Nav/Replicator stack can perturb RTX LiDAR
# render products.  Keep rtx_scan/debug runs minimal unless explicitly asked.
PEOPLE_EXTENSION_MODE = os.environ.get(
    "ARENA_ISAAC_PEOPLE_EXTENSION_MODE", "minimal"
).strip().lower()

_PEOPLE_EXTENSION_SETS = {
    # Only schema registration.  Useful when the custom .kit is also minimal.
    "schema": [
        "omni.anim.graph.schema",
    ],
    # Safe default for rtx_scan debugging: no navigation, no recast, no replicator.
    "minimal": [
        "omni.anim.graph.schema",
        "omni.anim.graph.core",
    ],
    # People runtime only.  Still avoids nav/recast/replicator by default.
    "people": [
        "omni.anim.graph.schema",
        "omni.anim.graph.core",
        "omni.anim.people",
    ],
    # Add navigation only for explicit navmesh/people tests.
    "nav": [
        "omni.anim.graph.schema",
        "omni.anim.graph.core",
        "omni.anim.people",
        "omni.anim.navigation.schema",
        "omni.anim.navigation.core",
        "omni.anim.navigation.recast",
    ],
    # Previous behavior.  Use only when you intentionally need the full stack.
    # Full stack except isaacsim.replicator.agent.*.
    # If nav is normal but full_no_replicator is normal and full is broken,
    # the culprit is almost certainly Replicator Agent.
    "full_no_replicator": [
        "omni.anim.timeline",
        "omni.anim.graph.bundle",
        "omni.anim.graph.schema",
        "omni.anim.graph.core",
        "omni.anim.graph.ui",
        "omni.anim.retarget.bundle",
        "omni.anim.retarget.core",
        "omni.anim.retarget.ui",
        "omni.kit.scripting",
        "omni.graph.nodes",
        "omni.anim.curve.core",
        "omni.anim.navigation.schema",
        "omni.anim.navigation.core",
        "omni.anim.navigation.recast",
        "omni.anim.people",
    ],
    # Add Replicator Agent on top of the nav/people set, without other full-stack UI/retarget extras.
    "replicator_agent": [
        "omni.anim.graph.schema",
        "omni.anim.graph.core",
        "omni.anim.people",
        "omni.anim.navigation.schema",
        "omni.anim.navigation.core",
        "omni.anim.navigation.recast",
        "isaacsim.replicator.agent.core",
        "isaacsim.replicator.agent.ui",
    ],
    # Replicator Agent core only. This helps distinguish core vs UI.
    "replicator_agent_core": [
        "omni.anim.graph.schema",
        "omni.anim.graph.core",
        "omni.anim.people",
        "omni.anim.navigation.schema",
        "omni.anim.navigation.core",
        "omni.anim.navigation.recast",
        "isaacsim.replicator.agent.core",
    ],
    # Full stack except Replicator Agent and user-facing UI pieces.
    "full_no_replicator_no_ui": [
        "omni.anim.timeline",
        "omni.anim.graph.bundle",
        "omni.anim.graph.schema",
        "omni.anim.graph.core",
        "omni.anim.retarget.bundle",
        "omni.anim.retarget.core",
        "omni.kit.scripting",
        "omni.graph.nodes",
        "omni.anim.curve.core",
        "omni.anim.navigation.schema",
        "omni.anim.navigation.core",
        "omni.anim.navigation.recast",
        "omni.anim.people",
    ],
    "full": [
        "omni.anim.timeline",
        "omni.anim.graph.bundle",
        "omni.anim.graph.schema",
        "omni.anim.graph.core",
        "omni.anim.graph.ui",
        "omni.anim.retarget.bundle",
        "omni.anim.retarget.core",
        "omni.anim.retarget.ui",
        "omni.kit.scripting",
        "omni.graph.nodes",
        "omni.anim.curve.core",
        "omni.anim.navigation.schema",
        "omni.anim.navigation.core",
        "omni.anim.navigation.recast",
        "omni.anim.people",
        "isaacsim.replicator.agent.core",
        "isaacsim.replicator.agent.ui",
    ],
}

if PEOPLE_EXTENSION_MODE not in _PEOPLE_EXTENSION_SETS:
    print(
        f"[people_stack] unknown ARENA_ISAAC_PEOPLE_EXTENSION_MODE="
        f"{PEOPLE_EXTENSION_MODE!r}; falling back to 'minimal'",
        flush=True,
    )
    PEOPLE_EXTENSION_MODE = "minimal"

EXTENSIONS_PEOPLE = list(_PEOPLE_EXTENSION_SETS[PEOPLE_EXTENSION_MODE])

EXTENSIONS_MATERIAL = [
    'omni.kit.material.library',
    'omni.kit.browser.material',
    'omni.kit.browser.asset',
    'omni.kit.window.material'
]
if ENABLE_PEOPLE_STACK:
    for ext_people in EXTENSIONS_PEOPLE:
        extensions.enable_extension(ext_people)
    # Re-enable the Isaac Sim 4.5 people stack explicitly; harmless if preloaded by .kit.
    if _env_bool("ARENA_ISAAC_PEOPLE_ENABLE_HELPER_EXTENSIONS", False):
        enable_animgraph_people_extensions(extensions)

if ENABLE_MATERIAL_STACK:
    for ext_material in EXTENSIONS_MATERIAL:
        extensions.enable_extension(ext_material)

# Update once after optional extension changes.
simulation_app.update()

ensure_rtx_sensor_coord_frame(
    logger=carb,
    reason=f"bridge startup people_extension_mode={PEOPLE_EXTENSION_MODE}",
)

# -------------------------------------------------------------------------------------------------
# These lines are needed to restart the USD stage and make sure that the people extension is loaded
# -------------------------------------------------------------------------------------------------
omni.usd.get_context().new_stage()

extensions.disable_extension("omni.isaac.ros_bridge")
extensions.enable_extension("isaacsim.ros2.bridge")

import rclpy
import numpy as np
from std_srvs.srv import Trigger
from isaac_utils.origin_collision_probe import origin_collision_probe_service
from isaac_utils.scene_collision_probe import scene_collision_probe_service
from isaac_utils.lidar_scan_relay import register_lidar_scan_relays

# Optional people/navmesh/replicator imports.  Keep navigation and replicator out
# of the default people path; both can affect RTX/SDG/render-product state.
nav = None
ag = None
rep = None
sd = None

if ENABLE_NAVMESH:
    try:
        for _nav_ext in (
            "omni.anim.navigation.schema",
            "omni.anim.navigation.core",
            "omni.anim.navigation.recast",
        ):
            extensions.enable_extension(_nav_ext)
        simulation_app.update()
        import omni.anim.navigation.core as nav
    except Exception as exc:  # noqa: BLE001 - keep bridge alive for rtx_scan.
        try:
            carb.log_warn(f"omni.anim.navigation unavailable, navmesh disabled: {exc}")
        except Exception:
            print(f"omni.anim.navigation unavailable, navmesh disabled: {exc}", flush=True)
        nav = None

if ENABLE_PEOPLE_STACK:
    try:
        import omni.anim.graph.core as ag
    except Exception as exc:  # noqa: BLE001
        try:
            carb.log_warn(f"omni.anim.graph.core unavailable: {exc}")
        except Exception:
            print(f"omni.anim.graph.core unavailable: {exc}", flush=True)
        ag = None

    # Do not import Replicator/SyntheticData unless explicitly requested.
    if _env_bool("ARENA_ISAAC_PEOPLE_IMPORT_REPLICATOR", False):
        try:
            import omni.replicator.core as rep
            import omni.syntheticdata._syntheticdata as sd
        except Exception as exc:  # noqa: BLE001
            try:
                carb.log_warn(f"Replicator/SyntheticData unavailable: {exc}")
            except Exception:
                print(f"Replicator/SyntheticData unavailable: {exc}", flush=True)
            rep = None
            sd = None


from isaacsim_msgs.srv import ImportUsd, ImportYaml
from isaacsim_msgs.srv import Pedestrian
# PeopleManager also touches people/navigation settings; keep it out of the RTX
# LiDAR debug path unless actual character services are explicitly enabled.
if ENABLE_PEOPLE_STACK and ENABLE_CHARACTER_SERVICES:
    try:
        from pedestrian.simulator.logic.people_manager import PeopleManager
    except Exception as exc:  # noqa: BLE001
        try:
            carb.log_warn(f"PeopleManager unavailable; character services disabled: {exc}")
        except Exception:
            print(f"PeopleManager unavailable; character services disabled: {exc}", flush=True)
        PeopleManager = None
else:
    PeopleManager = None

#Import robot models
from isaac_utils.robot_graphs import assign_robot_model

#Import services
from isaac_utils.services import spawn_wall
from isaac_utils.services import move_prim
from isaac_utils.services import get_prim_attr
from isaac_utils.services import delete_prim
from isaac_utils.services import convert_urdf_to_usd
from isaac_utils.services import import_obstacle
from isaac_utils.services import spawn_ped
from isaac_utils.services import move_ped
from isaac_utils.services import delete_all_characters
from isaac_utils.services import start_pedestrian_state_publisher
from isaac_utils.services import start_pedestrian_visual_envelope_monitor
from isaac_utils.services import spawn_floor
from isaac_utils.services import spawn_door
from isaac_utils.managers.door_manager import door_manager
from isaac_utils.mecanum_teleop import mecanum_teleop_manager
#Import sensors
from isaac_utils.sensors import imu_setup,publish_imu, contact_sensor_setup, publish_contact_sensor_info, camera_set_up,publish_camera_tf,publish_depth,publish_camera_info,publish_pointcloud_from_depth,publish_rgb, lidar_setup,publish_lidar 

# other imports
from isaac_utils.utils import geom

from isaac_utils.graphs.time import PublishTime

import random

# fmt: on
# ======================================Base======================================
# Setting up world and enable ros2_bridge extentions.
# BACKGROUND_STAGE_PATH = "/background"
# BACKGROUND_USD_PATH = "/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd"
plane_material_paths = [
    'https://omniverse-content-production.s3.us-west-2.amazonaws.com/Materials/2023_1/Base/Wood/Walnut_Planks.mdl',
    # 'https://omniverse-content-production.s3.us-west-2.amazonaws.com/Materials/2023_1/vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Diamond.mdl',
    # 'https://omniverse-content-production.s3.us-west-2.amazonaws.com/Materials/2023_1/vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Diamond.mdl'
]
world = World()
if ADD_GROUND_PLANE:
    world.scene.add_ground_plane(size=100, z_position=0.0)
_stage = omni.usd.get_context().get_stage()
if ENABLE_MATERIAL_STACK and ADD_GROUND_PLANE:
    plane_mdl_path = random.choice(plane_material_paths)
    plane_mtl_name = plane_mdl_path.split('/')[-1][:-4]
    plane_mtl_path = "/World/Looks/PlaneMaterial"
    plane_mtl = _stage.GetPrimAtPath(plane_mtl_path)
    if not (plane_mtl and plane_mtl.IsValid()):
        create_res = omni.kit.commands.execute('CreateMdlMaterialPrimCommand',
                                               mtl_url=plane_mdl_path,
                                               mtl_name=plane_mtl_name,
                                               mtl_path=plane_mtl_path)

        bind_res = omni.kit.commands.execute('BindMaterialCommand',
                                             prim_path="/World/groundPlane",
                                             material_path=plane_mtl_path)
simulation_app.update()  # update the simulation once for update ros2_bridge.
simulation_context = SimulationContext(stage_units_in_meters=1.0)  # currently we use 1m for simulation.
light_1 = prims.create_prim(
    "/World/Light_1",
    "DomeLight",
    position=np.array([1.0, 1.0, 1.0]),
    attributes={
        "inputs:texture:format": "latlong",
        "inputs:intensity": 1000.0,
        "inputs:color": (1.0, 1.0, 1.0)
    }
)
assets_root_path = get_assets_root_path_safe()

# Navmesh config and baking. Disabled by default for the lightweight bridge;
# enable only for pedestrian tests.
simulation_app.update()
stage = omni.usd.get_context().get_stage()

if ENABLE_NAVMESH and nav is not None:
    omni.kit.commands.execute("CreateNavMeshVolumeCommand",
                              parent_prim_path=Sdf.Path("/World"),
                              layer=stage.GetRootLayer()
                              )
    simulation_app.update()

    omni.kit.commands.execute(
        'ChangeSetting',
        path='/exts/omni.anim.navigation.core/navMesh/config/agentRadius',
        value=0.35)

    if ENABLE_PEOPLE_STACK:
        omni.kit.commands.execute(
            'ChangeSetting',
            path='/exts/omni.anim.people/navigation_settings/dynamic_avoidance_enabled',
            value=True)
        omni.kit.commands.execute(
            'ChangeSetting',
            path='/exts/omni.anim.people/navigation_settings/navmesh_enabled',
            value=True)

    inav = nav.acquire_interface()
    x = inav.start_navmesh_baking()
    simulation_app.update()

# stage.add_reference_to_stage(assets_root_path, BACKGROUND_USD_PATH)

# Setting up URDF importer.
status, import_config = commands.execute("URDFCreateImportConfig")
import_config.merge_fixed_joints = False
import_config.convex_decomp = False
import_config.import_inertia_tensor = False
import_config.self_collision = False
import_config.fix_base = False
import_config.distance_scale = 1
import_config.make_default_prim = True
import_config.default_drive_type = (_urdf.UrdfJointTargetType.JOINT_DRIVE_VELOCITY)
extension_path = _urdf.ImportConfig()

# list devices.
robots = []
environments = []
robot_positions = []
robot_orientation = []
environment_positions = []
environment_orientation = []

# ================================================================================
# ============================read yaml file===============================


def read_yaml_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def yaml_importer(request, response):
    # Read configuration from YAML file
    yaml_path = request.yaml_path
    config = read_yaml_config(yaml_path)

    # Extract parameters
    name = config['robot']['name']
    model = config['robot']['model']
    usd_path = config['robot']['usd_path']
    control = config['robot']['control']
    position = config['robot']['position']
    orientation = config['robot']['orientation']

    # Prepare the request for ImportUsd service.  ImportUsd.srv in this branch
    # has name/model/usd_path/control/pose fields, not prim_path/position/orientation.
    yaml_request = ImportUsd.Request()
    yaml_request.name = name
    yaml_request.model = model
    yaml_request.usd_path = usd_path
    yaml_request.control = control
    yaml_request.pose.position.x = float(position[0])
    yaml_request.pose.position.y = float(position[1])
    yaml_request.pose.position.z = float(position[2])
    yaml_request.pose.orientation.x = float(orientation[0])
    yaml_request.pose.orientation.y = float(orientation[1])
    yaml_request.pose.orientation.z = float(orientation[2])
    yaml_request.pose.orientation.w = float(orientation[3])

    usd_response = usd_importer(yaml_request, response)

    # Pass the response back (optional, depending on how you want to structure your service)
    response.ret = usd_response.ret
    return response

# ============================usd importer service================================
# Usd importer (service) -> bool.


def usd_importer(request, response):
    name = request.name
    model = request.model
    usd_path = request.usd_path
    prim_path = f"/World/{name}"

    model_prim = prims.create_prim(
        prim_path=prim_path,
        position=np.array(geom.Translation.parse(request.pose.position).tuple()),
        orientation=np.array(geom.Rotation.parse(request.pose.orientation).quat()),
        usd_path=usd_path,
        semantic_label=model,
    )
    door_manager.add_robot(f"/World/{name}")

    response.ret = True
    if not request.control:
        # V12: inspect and conservatively repair imported scene collisions.
        # This keeps the visual USD unchanged and adds/updates static collision
        # proxy geometry for later kinematic collision guarding.
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
        environments.append(prim_path)
        return response
    camera_prim_path = prim_path + "/camera_link"
    camera = camera_set_up(camera_prim_path, "Camera")
    camera.initialize()
    publish_camera_info(name, camera, 20)
    publish_depth(name, camera, 20)
    publish_rgb(name, camera, 20)
    publish_pointcloud_from_depth(name, camera, 20)
    publish_camera_tf(name, prim_path, camera)

    lidar_prim_path = prim_path + "/base_scan"
    lidar = lidar_setup(lidar_prim_path, "Lidar")
    publish_lidar(name, prim_path, lidar)

    links = ["wheel_left_link", "wheel_right_link"]
    for link in links:
        imu_prim_path = prim_path + "/" + link + "/" + "IMU"
        contact_prim_path = prim_path + "/" + link + "/" + "ContactSensor"
        imu = imu_setup(imu_prim_path)
        contact_sensor = contact_sensor_setup(contact_prim_path)
        publish_contact_sensor_info(name, prim_path, link, contact_sensor)
        publish_imu(name, prim_path, link, imu)

    robots.append(prim_path)

    model = assign_robot_model(name, prim_path, model)

    # publish joint_states and control
    model.control_and_publish_joint_states()
    model.publish_odom_and_tf()

    world.reset()
    return response

# Usd importer service callback.


def import_yaml(controller):
    service = controller.create_service(
        srv_type=ImportYaml,
        srv_name='isaac/import_yaml',
        callback=yaml_importer
    )
    return service


def import_usd(controller):
    service = controller.create_service(
        srv_type=ImportUsd,
        srv_name='isaac/import_usd',
        callback=usd_importer
    )
    return service


def time_publisher(controller):
    PublishTime('/World/publish_time')



# ============================collision proxy export service=========================

def _default_proxy_export_path():
    return os.environ.get(
        "ARENA_ISAAC_COLLISION_PROXY_EXPORT_PATH",
        "/home/stardust/resources/arena_ws/arena_assets/collision_configs/shenxinfu_841837.proxies.yaml",
    )


def _default_proxy_export_scene_root():
    env = os.environ.get("ARENA_ISAAC_COLLISION_PROXY_EXPORT_SCENE_ROOT", "").strip()
    if env:
        return env
    try:
        if environments:
            return environments[-1]
    except Exception:
        pass
    return "/World/shenxinfu_841837"


def export_collision_proxies_service(controller):
    def _callback(request, response):
        try:
            from isaac_utils.scene_collision_repair import export_collision_proxies_to_yaml, config_from_env
            cfg = config_from_env()
            try:
                scene_root = controller.get_parameter("collision_proxy_export_scene_root").value
            except Exception:
                scene_root = _default_proxy_export_scene_root()
            try:
                output_path = controller.get_parameter("collision_proxy_export_path").value
            except Exception:
                output_path = _default_proxy_export_path()
            if not scene_root:
                scene_root = _default_proxy_export_scene_root()
            if not output_path:
                output_path = _default_proxy_export_path()
            data = export_collision_proxies_to_yaml(
                str(scene_root),
                str(output_path),
                scene_name=str(scene_root).rstrip("/").split("/")[-1],
                proxy_root_name=cfg.proxy_root_name,
            )
            response.success = True
            response.message = f"exported {len(data.get('proxies', []))} proxies to {output_path}"
        except Exception as exc:
            response.success = False
            response.message = f"failed to export collision proxies: {exc}"
        return response

    return controller.create_service(Trigger, "/isaac/export_collision_proxies", _callback)


# ============================v15 voxel map build service==========================

def _default_voxel_scene_root():
    env = os.environ.get("ARENA_ISAAC_VOXEL_SCENE_ROOT", "").strip()
    if env:
        return env
    try:
        if environments:
            return environments[-1]
    except Exception:
        pass
    return "/World/shenxinfu_841837"


def _default_voxel_map_path():
    return os.environ.get(
        "ARENA_ISAAC_VOXEL_MAP_PATH",
        "/home/stardust/resources/arena_ws/arena_assets/collision_configs/shenxinfu_841837.voxel.json.gz",
    )


def build_voxel_map_service(controller):
    def _get_param(name, default):
        try:
            v = controller.get_parameter(name).value
            return default if v is None or v == "" else v
        except Exception:
            return default

    def _callback(request, response):
        try:
            from isaac_utils.voxel_stage_builder import VoxelBuildConfig, build_voxel_map_from_stage, DEFAULT_SKIP_KEYWORDS
            scene_root = str(_get_param("voxel_map_scene_root", _default_voxel_scene_root()))
            output_path = str(_get_param("voxel_map_output_path", _default_voxel_map_path()))
            debug_svg = str(_get_param("voxel_map_debug_svg_path", output_path.replace(".json.gz", ".svg").replace(".json", ".svg")))
            debug_pcd = str(_get_param("voxel_map_debug_pcd_path", output_path.replace(".json.gz", ".pcd").replace(".json", ".pcd")))
            resolution = float(_get_param("voxel_map_resolution", float(os.environ.get("ARENA_ISAAC_VOXEL_RESOLUTION", "0.05"))))
            sample_step = float(_get_param("voxel_map_sample_step", float(os.environ.get("ARENA_ISAAC_VOXEL_SAMPLE_STEP", str(resolution)))))
            z_min = float(_get_param("voxel_map_z_min", float(os.environ.get("ARENA_ISAAC_VOXEL_Z_MIN", "0.05"))))
            z_max = float(_get_param("voxel_map_z_max", float(os.environ.get("ARENA_ISAAC_VOXEL_Z_MAX", "1.20"))))
            skip_raw = str(_get_param("voxel_map_skip_keywords", os.environ.get("ARENA_ISAAC_VOXEL_SKIP_KEYWORDS", ",".join(DEFAULT_SKIP_KEYWORDS))))
            include_raw = str(_get_param("voxel_map_include_keywords", os.environ.get("ARENA_ISAAC_VOXEL_INCLUDE_KEYWORDS", "")))
            skip_keywords = [x.strip() for x in skip_raw.split(",") if x.strip()]
            include_keywords = [x.strip() for x in include_raw.split(",") if x.strip()]
            cfg = VoxelBuildConfig(
                scene_root=scene_root,
                output_path=output_path,
                debug_svg_path=debug_svg,
                debug_pcd_path=debug_pcd,
                resolution=resolution,
                sample_step=sample_step,
                z_min=z_min,
                z_max=z_max,
                skip_keywords=skip_keywords,
                include_keywords=include_keywords,
                max_faces_per_mesh=int(_get_param("voxel_map_max_faces_per_mesh", int(os.environ.get("ARENA_ISAAC_VOXEL_MAX_FACES_PER_MESH", "100000")))),
                max_samples_per_mesh=int(_get_param("voxel_map_max_samples_per_mesh", int(os.environ.get("ARENA_ISAAC_VOXEL_MAX_SAMPLES_PER_MESH", "250000")))),
                max_stored_voxels=int(_get_param("voxel_map_max_stored_voxels", int(os.environ.get("ARENA_ISAAC_VOXEL_MAX_STORED_VOXELS", "300000")))),
                max_debug_stage_points=int(_get_param("voxel_map_max_debug_stage_points", int(os.environ.get("ARENA_ISAAC_VOXEL_MAX_DEBUG_STAGE_POINTS", "30000")))),
                create_stage_debug_points=str(_get_param("voxel_map_create_stage_debug_points", os.environ.get("ARENA_ISAAC_VOXEL_CREATE_STAGE_DEBUG_POINTS", "true"))).lower() in {"1", "true", "yes", "on"},
                stage_debug_path=str(_get_param("voxel_map_stage_debug_path", os.environ.get("ARENA_ISAAC_VOXEL_STAGE_DEBUG_PATH", ""))),
            )
            data = build_voxel_map_from_stage(cfg, logger=controller.get_logger())
            response.success = True
            response.message = (
                f"voxel map built: columns={data.get('column_count')} stored_voxels={data.get('stored_voxel_count')} "
                f"meshes={data.get('mesh_count_used')}/{data.get('mesh_count_seen')} output={output_path} svg={debug_svg} pcd={debug_pcd}"
            )
        except Exception as exc:
            response.success = False
            response.message = f"failed to build voxel map: {exc}"
        return response

    return controller.create_service(Trigger, "/isaac/build_voxel_map", _callback)


def export_walkable_map_service(controller):
    def _get_param(name, default):
        try:
            value = controller.get_parameter(name).value
            return default if value is None or value == "" else value
        except Exception:
            return default

    def _csv_floats(value, count, name):
        result = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
        if len(result) != count:
            raise ValueError(f"{name} requires {count} comma-separated values")
        return result

    def _callback(_request, response):
        try:
            from isaac_utils.walkable_map_builder import (
                WalkableMapConfig,
                build_walkable_map,
            )

            output_path = str(
                _get_param(
                    "walkable_map_output_path",
                    "/home/stardust/resources/arena_ws/arena_assets/navigation/"
                    "shenxinfu_841837.walkable.json",
                )
            )
            config = WalkableMapConfig(
                scene_root=str(
                    _get_param("walkable_map_scene_root", _default_voxel_scene_root())
                ),
                output_path=output_path,
                resolution=float(_get_param("walkable_map_resolution", 0.05)),
                origin=_csv_floats(
                    _get_param("walkable_map_origin", "-2.24,-0.90,0.75"),
                    3,
                    "walkable_map_origin",
                ),
                world_bounds=_csv_floats(
                    _get_param(
                        "walkable_map_world_bounds",
                        "-4.60,3.60,-2.10,1.80",
                    ),
                    4,
                    "walkable_map_world_bounds",
                ),
                exclude_path_keywords=tuple(
                    item.strip()
                    for item in str(
                        _get_param(
                            "walkable_map_exclude_path_keywords",
                            "/World/Characters,/World/xms_mecanum,_debug",
                        )
                    ).split(",")
                    if item.strip()
                ),
            )
            data = build_walkable_map(config, logger=controller.get_logger())
            response.success = True
            response.message = (
                f"walkable map exported: output={output_path}; "
                f"dimensions={data['width']}x{data['height']}; "
                f"counts={data['counts']}; fingerprint={data['scene_fingerprint'][:12]}"
            )
        except Exception as exc:
            response.success = False
            response.message = f"failed to export walkable map: {exc}"
            controller.get_logger().error(response.message)
        return response

    return controller.create_service(
        Trigger,
        "/isaac/export_walkable_map",
        _callback,
    )

# =================================================================================

# ===================================controller====================================
# create controller node for isaacsim.


def create_controller(time=120):
    rclpy.init()
    controller = rclpy.create_node("isaac_controller")
    try:
        controller.declare_parameter("collision_proxy_export_path", _default_proxy_export_path())
        controller.declare_parameter("collision_proxy_export_scene_root", os.environ.get("ARENA_ISAAC_COLLISION_PROXY_EXPORT_SCENE_ROOT", ""))
        # V15 voxel-map build parameters. These are set by scripts/arena_scene_profile.py voxel build.
        controller.declare_parameter("voxel_map_scene_root", _default_voxel_scene_root())
        controller.declare_parameter("voxel_map_output_path", _default_voxel_map_path())
        controller.declare_parameter("voxel_map_debug_svg_path", _default_voxel_map_path().replace(".json.gz", ".svg").replace(".json", ".svg"))
        controller.declare_parameter("voxel_map_debug_pcd_path", _default_voxel_map_path().replace(".json.gz", ".pcd").replace(".json", ".pcd"))
        controller.declare_parameter("voxel_map_resolution", float(os.environ.get("ARENA_ISAAC_VOXEL_RESOLUTION", "0.05")))
        controller.declare_parameter("voxel_map_sample_step", float(os.environ.get("ARENA_ISAAC_VOXEL_SAMPLE_STEP", os.environ.get("ARENA_ISAAC_VOXEL_RESOLUTION", "0.05"))))
        controller.declare_parameter("voxel_map_z_min", float(os.environ.get("ARENA_ISAAC_VOXEL_Z_MIN", "0.05")))
        controller.declare_parameter("voxel_map_z_max", float(os.environ.get("ARENA_ISAAC_VOXEL_Z_MAX", "1.20")))
        controller.declare_parameter("voxel_map_skip_keywords", os.environ.get("ARENA_ISAAC_VOXEL_SKIP_KEYWORDS", ""))
        controller.declare_parameter("voxel_map_include_keywords", os.environ.get("ARENA_ISAAC_VOXEL_INCLUDE_KEYWORDS", ""))
        controller.declare_parameter("voxel_map_max_faces_per_mesh", int(os.environ.get("ARENA_ISAAC_VOXEL_MAX_FACES_PER_MESH", "100000")))
        controller.declare_parameter("voxel_map_max_samples_per_mesh", int(os.environ.get("ARENA_ISAAC_VOXEL_MAX_SAMPLES_PER_MESH", "250000")))
        controller.declare_parameter("voxel_map_max_stored_voxels", int(os.environ.get("ARENA_ISAAC_VOXEL_MAX_STORED_VOXELS", "300000")))
        controller.declare_parameter("voxel_map_max_debug_stage_points", int(os.environ.get("ARENA_ISAAC_VOXEL_MAX_DEBUG_STAGE_POINTS", "30000")))
        controller.declare_parameter("voxel_map_create_stage_debug_points", os.environ.get("ARENA_ISAAC_VOXEL_CREATE_STAGE_DEBUG_POINTS", "true"))
        controller.declare_parameter("voxel_map_stage_debug_path", os.environ.get("ARENA_ISAAC_VOXEL_STAGE_DEBUG_PATH", ""))
        controller.declare_parameter("walkable_map_scene_root", _default_voxel_scene_root())
        controller.declare_parameter(
            "walkable_map_output_path",
            "/home/stardust/resources/arena_ws/arena_assets/navigation/"
            "shenxinfu_841837.walkable.json",
        )
        controller.declare_parameter("walkable_map_resolution", 0.05)
        controller.declare_parameter("walkable_map_origin", "-2.24,-0.90,0.75")
        controller.declare_parameter(
            "walkable_map_world_bounds",
            "-4.60,3.60,-2.10,1.80",
        )
        controller.declare_parameter(
            "walkable_map_exclude_path_keywords",
            "/World/Characters,/World/xms_mecanum,_debug",
        )
    except Exception:
        pass
    import_usd(controller)
    import_yaml(controller)
    spawn_wall(controller)
    move_prim(controller)
    get_prim_attr(controller)
    delete_prim(controller)
    convert_urdf_to_usd(controller)
    import_obstacle(controller)
    spawn_ped(controller)
    move_ped(controller)
    delete_all_characters(controller)
    start_pedestrian_state_publisher(controller)
    start_pedestrian_visual_envelope_monitor(controller)
    spawn_floor(controller)
    spawn_door(controller)
    export_collision_proxies_service(controller)
    build_voxel_map_service(controller)
    export_walkable_map_service(controller)
    origin_collision_probe_service(controller)
    scene_collision_probe_service(controller)
    # Let the DoorManager subscribe to ROS topics on this controller node
    try:
        door_manager.register_node(controller)
    except Exception as e:
        controller.get_logger().warning(f'Failed to register DoorManager with controller: {e}')
    try:
        mecanum_teleop_manager.register_node(controller)
    except Exception as e:
        controller.get_logger().warning(f'Failed to register MecanumTeleopManager with controller: {e}')
    controller._lidar_scan_relays = register_lidar_scan_relays()
    # Enable per-entity logging and filter to show only jackal-related outputs
    try:
        door_manager._log_every_tick = False
        door_manager._log_entity_filter = ['jackal']
        controller.get_logger().info('DoorManager per-tick logging enabled (filter=jackal)')
    except Exception as e:
        controller.get_logger().warning(f'Failed to set DoorManager logging flags: {e}')
    time_publisher(controller)
    return controller



def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _rss_mb() -> float:
    try:
        with open('/proc/self/statm', 'r') as f:
            rss_pages = int(f.read().split()[1])
        return rss_pages * os.sysconf('SC_PAGE_SIZE') / (1024.0 * 1024.0)
    except Exception:
        return -1.0


def _controller_active() -> bool:
    # Idle bridge should not update Isaac as fast as possible.  Once a robot is
    # registered, use active FPS for kinematic control.  This intentionally does
    # not inspect every possible sensor graph; for lidar validation the robot is
    # registered first.
    try:
        if getattr(mecanum_teleop_manager, 'robots', None):
            return len(mecanum_teleop_manager.robots) > 0
    except Exception:
        pass
    return False

# =================================================================================

# ======================================main=======================================

def main(args=None):
    """
    Main function to initialize the simulation, create the ROS 2 node,
    and run the simulation loop.
    """
    # Create the ROS 2 controller node. This also calls rclpy.init().
    controller = create_controller()

    # AnimGraph characters become queryable only after the timeline is playing.
    try:
        omni.timeline.get_timeline_interface().play()
    except Exception as e:
        controller.get_logger().warning(f"Failed to start Isaac timeline: {e}")

    try:
        # Main simulation loop.  Previous versions ran this loop as fast as the
        # CPU allowed (spin_once timeout=0 and no sleep).  On Isaac Sim 4.5 this
        # can drive 150-250% CPU while idle and also amplifies Kit/RTX allocation
        # growth.  Use a low idle FPS before any robot is registered, and an
        # active FPS after kinematic control starts.
        idle_fps = max(1.0, _env_float("ARENA_ISAAC_IDLE_FPS", 5.0))
        active_fps = max(1.0, _env_float("ARENA_ISAAC_ACTIVE_FPS", 30.0))
        mem_log_sec = max(0.0, _env_float("ARENA_ISAAC_MEM_LOG_SEC", 30.0))
        gc_interval_sec = max(0.0, _env_float("ARENA_ISAAC_GC_INTERVAL_SEC", 60.0))
        last_mem_log = time.monotonic()
        last_gc = time.monotonic()

        controller.get_logger().info(
            f"Isaac bridge loop limiter active: idle_fps={idle_fps}, "
            f"active_fps={active_fps}, mem_log_sec={mem_log_sec}, people_stack={ENABLE_PEOPLE_STACK}, material_stack={ENABLE_MATERIAL_STACK}, navmesh={ENABLE_NAVMESH}"
        )

        while simulation_app.is_running():
            frame_start = time.monotonic()

            # Step the simulation/Kit once.
            simulation_app.update()

            # Update door logic.
            door_manager.update()

            # Update custom mecanum teleop robots registered by /isaac/urdf_to_usd.
            mecanum_teleop_manager.update()

            # Tick the ROS 2 node.  A tiny timeout prevents a pure busy-poll loop
            # while still keeping service calls responsive.
            rclpy.spin_once(controller, timeout_sec=0.001)

            now = time.monotonic()
            if mem_log_sec > 0.0 and (now - last_mem_log) >= mem_log_sec:
                rss = _rss_mb()
                if rss >= 0:
                    controller.get_logger().info(
                        f"Isaac bridge memory: rss={rss:.1f} MB, "
                        f"mode={'active' if _controller_active() else 'idle'}"
                    )
                last_mem_log = now

            # Periodic GC only affects Python-side cycles; it will not fix native
            # Kit leaks, but it prevents small ROS/Python allocations from piling up.
            if gc_interval_sec > 0.0 and (now - last_gc) >= gc_interval_sec:
                gc.collect()
                last_gc = now

            target_fps = active_fps if _controller_active() else idle_fps
            target_dt = 1.0 / target_fps
            elapsed = time.monotonic() - frame_start
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    except KeyboardInterrupt:
        controller.get_logger().info('Received KeyboardInterrupt, shutting down.')
    finally:
        # Cleanly shut down the simulation and ROS 2
        controller.get_logger().info('Shutting down ROS 2 node and simulation.')
        lidar_scan_relays = getattr(controller, '_lidar_scan_relays', None)
        if lidar_scan_relays is not None:
            try:
                lidar_scan_relays.shutdown()
            except Exception as e:
                controller.get_logger().warning(f'Failed to stop RTX lidar scan relays: {e}')
        controller.destroy_node()
        rclpy.shutdown()
        simulation_app.close()

# =================================================================================
if __name__ == "__main__":
    main()
