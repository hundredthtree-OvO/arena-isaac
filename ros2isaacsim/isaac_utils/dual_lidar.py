"""Dual front/rear RTX lidar helpers for stable kinematic robot validation."""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import omni.graph.core as og
import omni.kit.commands
import omni.usd
from pxr import Gf, Usd, UsdGeom

try:
    from isaac_utils.robot_geometry import resolve_lidar_mounts_from_env
except Exception:  # pragma: no cover
    resolve_lidar_mounts_from_env = None  # type: ignore

try:
    from isaacsim.core.utils.extensions import enable_extension, get_extension_path_from_name
except Exception:  # pragma: no cover
    from omni.isaac.core.utils.extensions import enable_extension, get_extension_path_from_name  # type: ignore


def _log(logger, level: str, text: str):
    if logger is None:
        print(text)
        return
    fn = getattr(logger, level, None) or getattr(logger, "info", None)
    if fn:
        fn(text)
    else:
        print(text)


def _quat_from_yaw(yaw: float) -> Gf.Quatd:
    return Gf.Quatd(math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


def _quatf_from_yaw(yaw: float) -> Gf.Quatf:
    # UsdGeom.Xformable.AddOrientOp() creates a float-orient op by default on
    # Isaac Sim 4.5.  Setting a Gf.Quatd into that op raises:
    #   expected GfQuatf, got GfQuatd
    return Gf.Quatf(float(math.cos(0.5 * yaw)), 0.0, 0.0, float(math.sin(0.5 * yaw)))


def _ensure_xform(path: str):
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None
    prim = stage.GetPrimAtPath(path)
    if prim and prim.IsValid():
        return prim
    parent = os.path.dirname(path)
    if parent and parent != path:
        _ensure_xform(parent)
    return UsdGeom.Xform.Define(stage, path).GetPrim()


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _set_display_color(prim, rgb):
    try:
        UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2]))])
    except Exception:
        pass


def _set_local_xform(prim, *, translate=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), yaw=0.0):
    x = UsdGeom.Xformable(prim)
    try:
        x.ClearXformOpOrder()
    except Exception:
        pass
    x.AddTranslateOp().Set(Gf.Vec3d(float(translate[0]), float(translate[1]), float(translate[2])))
    x.AddOrientOp().Set(_quatf_from_yaw(float(yaw)))
    x.AddScaleOp().Set(Gf.Vec3f(float(scale[0]), float(scale[1]), float(scale[2])))


def _delete_prim_if_exists(path: str):
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    prim = stage.GetPrimAtPath(path)
    if prim and prim.IsValid():
        try:
            omni.kit.commands.execute("DeletePrims", paths=[path])
        except Exception:
            pass


def _create_lidar_debug_marker(sensor_prim_path: str, *, name: str, fov_deg: float, color=(0.0, 0.5, 1.0), logger=None):
    """Create visible local markers under an RTX lidar prim.

    Marker convention:
      sphere     = lidar origin
      green bar  = LaserScan angle=0 / frame +X
      orange bars = +/- FOV edges
    """
    if not _env_bool("ARENA_ISAAC_LIDAR_VISUALIZE", True):
        return
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    sensor = stage.GetPrimAtPath(sensor_prim_path)
    if sensor is None or not sensor.IsValid():
        return
    marker_root = f"{sensor_prim_path}/_debug_lidar_marker"
    _delete_prim_if_exists(marker_root)
    UsdGeom.Xform.Define(stage, marker_root)

    sphere = UsdGeom.Sphere.Define(stage, f"{marker_root}/origin_sphere")
    sphere.CreateRadiusAttr(1.0)
    _set_local_xform(sphere.GetPrim(), scale=(0.045, 0.045, 0.045))
    _set_display_color(sphere.GetPrim(), color)

    xbar = UsdGeom.Cube.Define(stage, f"{marker_root}/plus_x_angle_0_bar")
    _set_local_xform(xbar.GetPrim(), translate=(0.25, 0.0, 0.0), scale=(0.25, 0.015, 0.015), yaw=0.0)
    _set_display_color(xbar.GetPrim(), (0.0, 1.0, 0.0))

    half = 0.5 * math.radians(float(fov_deg))
    length = 0.60
    for a, suffix in [(-half, "fov_minus"), (half, "fov_plus")]:
        bar = UsdGeom.Cube.Define(stage, f"{marker_root}/{suffix}")
        _set_local_xform(
            bar.GetPrim(),
            translate=(0.5 * length * math.cos(a), 0.5 * length * math.sin(a), 0.0),
            scale=(0.5 * length, 0.010, 0.010),
            yaw=a,
        )
        _set_display_color(bar.GetPrim(), (1.0, 0.6, 0.0))
    _log(logger, "info", f"[dual_lidar] debug marker created for {name}: {marker_root}")


def _make_custom_2d_lidar_config(
    *,
    horizontal_samples: int,
    min_angle: float,
    max_angle: float,
    range_min: float,
    range_max: float,
    update_rate: float,
) -> str:
    """Create an Isaac RTX lidar profile name for a single-layer 2D scan.

    The command IsaacSensorCreateRtxLidar expects a profile name relative to the
    omni.isaac.sensor lidar config directory, not an arbitrary absolute JSON path.
    """
    # Isaac Sim 4.5 searches RTX lidar profiles under the isaacsim.sensors.rtx
    # extension's data/lidar_configs directory.  Previous v10 wrote the JSON
    # under omni.isaac.sensor, which is not in the profile search path on many
    # 4.5 installs, producing: getProfileJsonAtPaths could not find config file.
    base_dir = None
    for ext_name in ("isaacsim.sensors.rtx", "omni.isaac.sensor", "omni.sensors.nv.common"):
        try:
            ext_path = Path(get_extension_path_from_name(ext_name))
        except Exception:
            continue
        candidate = ext_path / "data" / "lidar_configs" / "CustomLidar_tmp_v10"
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            base_dir = candidate
            break
        except Exception:
            continue
    if base_dir is None:
        base_dir = Path(tempfile.gettempdir()) / "arena_isaac_lidar_configs"
        base_dir.mkdir(parents=True, exist_ok=True)

    az = np.linspace(math.degrees(min_angle), math.degrees(max_angle), int(horizontal_samples), endpoint=True)
    data = {
        "class": "sensor",
        "type": "lidar",
        "name": "ArenaV10Dual2DLidar",
        "driveWorksId": "GENERIC",
        "profile": {
            "scanType": "solidState",
            "intensityProcessing": "normalization",
            "rayType": "IDEALIZED",
            "nearRangeM": float(range_min),
            "farRangeM": float(range_max),
            "rangeResolutionM": 0.01,
            "rangeAccuracyM": 0.01,
            "rotationDirection": "CW",
            "wavelengthNm": 1550.0,
            "maxReturns": 1,
            "reportRateBaseHz": float(update_rate),
            "scanRateBaseHz": float(update_rate),
            "numberOfEmitters": int(horizontal_samples),
            "numberOfChannels": int(horizontal_samples),
            "numLines": 1,
            "numRaysPerLine": [int(horizontal_samples)],
            "rangeCount": 1,
            "ranges": [{"min": float(range_min), "max": float(range_max)}],
            "emitterStateCount": 1,
            "emitterStates": [
                {
                    "azimuthDeg": az.tolist(),
                    "elevationDeg": np.zeros(int(horizontal_samples), dtype=float).tolist(),
                    "fireTimeNs": np.linspace(0, 1e9 / float(update_rate), int(horizontal_samples)).astype(int).tolist(),
                }
            ],
            "intensityMappingType": "LINEAR",
        },
    }

    out = base_dir / "arena_v10_dual_2d_lidar.json"
    out.write_text(json.dumps(data), encoding="utf-8")
    print(f"[dual_lidar] wrote custom RTX lidar profile: {out}")
    # The command expects a config name relative to data/lidar_configs.
    if "lidar_configs" in str(base_dir):
        return os.path.join(base_dir.name, out.stem)
    return "Example_Rotary"



def _prime_auto_mount_env(logger=None):
    """Resolve legacy/profile lidar mount source for v16.3.

    Important correction from v16.2:
      ``auto_from_body_bbox`` may still produce +/-0.42 m on this robot because
      the raw USD body bbox includes outer wheel/roller extents.  That is a
      mathematically valid visual bbox but it is not the desired navigation
      sensor origin: the lidar origin should stay inside the collision guard
      footprint that the user visually tuned to the mobile base.

    Therefore, unless the user explicitly provides FRONT_X/REAR_X, we default
    to ``auto_from_guard_footprint``.  This computes:
        front_x = +guard_length/2 - inset_x
        rear_x  = -guard_length/2 + inset_x
    using ARENA_ISAAC_COLLISION_GUARD_LENGTH, which comes from the profile
    guard.length launch argument.
    """
    source = (os.environ.get("ARENA_ISAAC_LIDAR_MOUNT_SOURCE") or "").strip().lower()
    has_explicit_x = (
        os.environ.get("ARENA_ISAAC_LIDAR_FRONT_X") is not None
        or os.environ.get("ARENA_ISAAC_LIDAR_REAR_X") is not None
    )
    if not has_explicit_x:
        if source in {"", "manual", "auto", "auto_from_body", "auto_from_body_bbox", "body_bbox"}:
            os.environ["ARENA_ISAAC_LIDAR_MOUNT_SOURCE"] = "auto_from_guard_footprint"
            _log(
                logger,
                "info",
                f"[lidar_mount] source={source or '<unset>'} without explicit front_x/rear_x; "
                "using auto_from_guard_footprint so lidar origins stay inside the guard footprint",
            )
    if os.environ.get("ARENA_ISAAC_LIDAR_INSET_X") is None:
        os.environ["ARENA_ISAAC_LIDAR_INSET_X"] = "0.03"
    if os.environ.get("ARENA_ISAAC_LIDAR_Y") is None:
        os.environ["ARENA_ISAAC_LIDAR_Y"] = "0.0"
    if os.environ.get("ARENA_ISAAC_LIDAR_Z") is None:
        os.environ["ARENA_ISAAC_LIDAR_Z"] = "0.20"


def _env_float_local(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _resolve_lidar_mounts_v16_3(
    robot_root_path: str,
    *,
    default_front_xyz: Tuple[float, float, float],
    default_rear_xyz: Tuple[float, float, float],
    logger=None,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Resolve lidar mounts with guard-footprint semantics.

    This is intentionally implemented inside dual_lidar.py so every caller that
    creates RTX lidar sensors gets the same behavior, even if an older profile
    or service path bypasses robot_geometry.py.
    """
    source = (os.environ.get("ARENA_ISAAC_LIDAR_MOUNT_SOURCE") or "manual").strip().lower()
    y = _env_float_local("ARENA_ISAAC_LIDAR_Y", default_front_xyz[1])
    z = _env_float_local("ARENA_ISAAC_LIDAR_Z", default_front_xyz[2])
    inset = max(0.0, _env_float_local("ARENA_ISAAC_LIDAR_INSET_X", 0.03))

    has_explicit_x = (
        os.environ.get("ARENA_ISAAC_LIDAR_FRONT_X") is not None
        or os.environ.get("ARENA_ISAAC_LIDAR_REAR_X") is not None
    )
    if source == "manual" and has_explicit_x:
        front = (
            _env_float_local("ARENA_ISAAC_LIDAR_FRONT_X", default_front_xyz[0]),
            _env_float_local("ARENA_ISAAC_LIDAR_FRONT_Y", y),
            _env_float_local("ARENA_ISAAC_LIDAR_FRONT_Z", z),
        )
        rear = (
            _env_float_local("ARENA_ISAAC_LIDAR_REAR_X", default_rear_xyz[0]),
            _env_float_local("ARENA_ISAAC_LIDAR_REAR_Y", y),
            _env_float_local("ARENA_ISAAC_LIDAR_REAR_Z", z),
        )
        _log(
            logger,
            "info",
            f"[lidar_mount] manual explicit: front=({front[0]:+.3f},{front[1]:+.3f},{front[2]:+.3f}) "
            f"rear=({rear[0]:+.3f},{rear[1]:+.3f},{rear[2]:+.3f})",
        )
        return front, rear

    if source in {"auto_from_guard", "auto_from_guard_footprint", "guard", "guard_footprint"}:
        # Prefer a dedicated lidar footprint length if present; otherwise use
        # the collision guard length passed through the launch/profile.
        guard_length = _env_float_local(
            "ARENA_ISAAC_LIDAR_FOOTPRINT_LENGTH",
            _env_float_local("ARENA_ISAAC_COLLISION_GUARD_LENGTH", 0.70),
        )
        half_len = max(0.0, 0.5 * guard_length)
        front_x = max(0.0, half_len - inset)
        rear_x = -front_x
        front = (float(front_x), y, z)
        rear = (float(rear_x), y, z)
        _log(
            logger,
            "info",
            f"[lidar_mount] auto_from_guard_footprint: guard_length={guard_length:.3f} inset={inset:.3f} "
            f"front=({front[0]:+.3f},{front[1]:+.3f},{front[2]:+.3f}) "
            f"rear=({rear[0]:+.3f},{rear[1]:+.3f},{rear[2]:+.3f})",
        )
        return front, rear

    # Optional legacy/debug mode: still allow bbox mode if explicitly requested
    # after this v16.3 fix.  This can reproduce the old +/-0.42 result on the
    # current robot because the raw USD bbox includes wheel/roller extents.
    if source in {"auto_from_body", "auto_from_body_bbox", "body_bbox"} and resolve_lidar_mounts_from_env is not None:
        front, rear = resolve_lidar_mounts_from_env(
            robot_root_path,
            default_front_xyz=default_front_xyz,
            default_rear_xyz=default_rear_xyz,
            logger=logger,
        )
        _log(logger, "warning", "[lidar_mount] using raw body bbox mode; this may place lidar at +/-0.42 on xms_mecanum")
        return front, rear

    # Safe fallback: use guard footprint rather than historical +/-0.42.
    os.environ["ARENA_ISAAC_LIDAR_MOUNT_SOURCE"] = "auto_from_guard_footprint"
    return _resolve_lidar_mounts_v16_3(
        robot_root_path,
        default_front_xyz=default_front_xyz,
        default_rear_xyz=default_rear_xyz,
        logger=logger,
    )

def _force_local_pose(prim_path: str, translation: Tuple[float, float, float], yaw: float, logger=None):
    """Force the local pose of an existing lidar prim after creation.

    IsaacSensorCreateRtxLidar can leave stale xform ops if a prim with the same
    name existed in the stage.  Clearing and rewriting the local xform makes the
    stage pose, ROS frame convention, and debug marker agree.
    """
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    prim = stage.GetPrimAtPath(prim_path)
    if prim is None or not prim.IsValid():
        return
    try:
        x = UsdGeom.Xformable(prim)
        try:
            x.ClearXformOpOrder()
        except Exception:
            pass
        x.AddTranslateOp().Set(Gf.Vec3d(float(translation[0]), float(translation[1]), float(translation[2])))
        x.AddOrientOp().Set(_quatf_from_yaw(float(yaw)))
        _log(logger, "info", f"[dual_lidar] forced local pose for {prim_path}: xyz=({translation[0]:+.3f},{translation[1]:+.3f},{translation[2]:+.3f}) yaw={yaw:.3f}")
    except Exception as exc:
        _log(logger, "warning", f"[dual_lidar] failed to force local pose for {prim_path}: {exc}")

def _create_lidar_sensor(
    *,
    robot_root_path: str,
    sensor_name: str,
    translation: Tuple[float, float, float],
    yaw: float,
    config_name: str,
    logger=None,
):
    # Remove stale lidar prims from previous validation spawns; otherwise Isaac
    # can keep the old local transform even after the profile changes.
    expected_path = f"{robot_root_path}/{sensor_name}"
    _delete_prim_if_exists(expected_path)

    # Use parent+relative path so the lidar remains a child of the kinematic robot root.
    ok, lidar = omni.kit.commands.execute(
        "IsaacSensorCreateRtxLidar",
        path=sensor_name,
        parent=robot_root_path,
        config=config_name,
        translation=translation,
        orientation=_quat_from_yaw(yaw),
    )
    if not ok or lidar is None:
        _log(logger, "warning", f"[dual_lidar] Failed custom lidar config {config_name}; retrying Example_Rotary")
        ok, lidar = omni.kit.commands.execute(
            "IsaacSensorCreateRtxLidar",
            path=sensor_name,
            parent=robot_root_path,
            config="Example_Rotary",
            translation=translation,
            orientation=_quat_from_yaw(yaw),
        )
    if not ok or lidar is None:
        raise RuntimeError(f"Failed to create RTX lidar {sensor_name} under {robot_root_path}")
    lidar_path = str(lidar.GetPath())
    _force_local_pose(lidar_path, translation, yaw, logger=logger)
    return lidar_path


def _create_lidar_publish_graph(
    *,
    graph_path: str,
    lidar_prim_path: str,
    topic_name: str,
    frame_id: str,
):
    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": graph_path},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("RunOnce", "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"),
                ("RenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("LaserScanPublisher", "isaacsim.ros2.bridge.ROS2RtxLidarHelper"),
                ("PointCloudPublisher", "isaacsim.ros2.bridge.ROS2RtxLidarHelper"),
            ],
            keys.SET_VALUES: [
                ("RenderProduct.inputs:cameraPrim", lidar_prim_path),
                ("LaserScanPublisher.inputs:topicName", topic_name.lstrip("/")),
                ("LaserScanPublisher.inputs:frameId", frame_id),
                ("LaserScanPublisher.inputs:type", "laser_scan"),
                ("PointCloudPublisher.inputs:topicName", f"{topic_name.strip('/')}/points"),
                ("PointCloudPublisher.inputs:frameId", frame_id),
                ("PointCloudPublisher.inputs:type", "point_cloud"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "RunOnce.inputs:execIn"),
                ("RunOnce.outputs:step", "RenderProduct.inputs:execIn"),
                ("RenderProduct.outputs:execOut", "LaserScanPublisher.inputs:execIn"),
                ("RenderProduct.outputs:renderProductPath", "LaserScanPublisher.inputs:renderProductPath"),
                ("Context.outputs:context", "LaserScanPublisher.inputs:context"),
                ("RenderProduct.outputs:execOut", "PointCloudPublisher.inputs:execIn"),
                ("RenderProduct.outputs:renderProductPath", "PointCloudPublisher.inputs:renderProductPath"),
                ("Context.outputs:context", "PointCloudPublisher.inputs:context"),
            ],
        },
    )


def create_dual_lidar(
    *,
    robot_root_path: str,
    robot_name: str,
    logger=None,
    front_topic: str = "/front_scan",
    rear_topic: str = "/rear_scan",
    front_frame: str = "front_laser_link",
    rear_frame: str = "rear_laser_link",
    front_xyz: Tuple[float, float, float] = (0.42, 0.0, 0.20),
    rear_xyz: Tuple[float, float, float] = (-0.42, 0.0, 0.20),
    samples: int = 721,
    fov_deg: float = 270.0,
    range_min: float = 0.05,
    range_max: float = 12.0,
    update_rate: float = 10.0,
):
    """Attach front/rear 2D RTX lidar sensors to a kinematic robot root."""
    enable_extension("isaacsim.ros2.bridge")
    enable_extension("isaacsim.core.nodes")
    try:
        enable_extension("omni.isaac.sensor")
    except Exception:
        pass

    _ensure_xform(robot_root_path)
    half = math.radians(float(fov_deg)) * 0.5
    config_name = _make_custom_2d_lidar_config(
        horizontal_samples=int(samples),
        min_angle=-half,
        max_angle=half,
        range_min=float(range_min),
        range_max=float(range_max),
        update_rate=float(update_rate),
    )

    # V16.2: resolve lidar mounts from robot body geometry/profile instead of
    # relying on the historical fixed +/-0.42 m values.  If a legacy launch path
    # did not inject the profile env, default to auto_from_body_bbox unless the
    # user explicitly supplied manual FRONT_X/REAR_X.
    _prime_auto_mount_env(logger=logger)
    try:
        front_xyz, rear_xyz = _resolve_lidar_mounts_v16_3(
            robot_root_path,
            default_front_xyz=front_xyz,
            default_rear_xyz=rear_xyz,
            logger=logger,
        )
    except Exception as exc:
        _log(logger, "warning", f"[dual_lidar] failed to resolve lidar mounts with v16.3 policy; using defaults: {exc}")

    front_yaw = float(os.environ.get("ARENA_ISAAC_LIDAR_FRONT_YAW", "0.0"))
    rear_yaw = float(os.environ.get("ARENA_ISAAC_LIDAR_REAR_YAW", str(math.pi)))
    # Publish the resolved values back to process env so the actual odom/tf
    # publisher uses the exact same sensor transforms.
    os.environ["ARENA_ISAAC_LIDAR_FRONT_X"] = str(float(front_xyz[0]))
    os.environ["ARENA_ISAAC_LIDAR_FRONT_Y"] = str(float(front_xyz[1]))
    os.environ["ARENA_ISAAC_LIDAR_FRONT_Z"] = str(float(front_xyz[2]))
    os.environ["ARENA_ISAAC_LIDAR_REAR_X"] = str(float(rear_xyz[0]))
    os.environ["ARENA_ISAAC_LIDAR_REAR_Y"] = str(float(rear_xyz[1]))
    os.environ["ARENA_ISAAC_LIDAR_REAR_Z"] = str(float(rear_xyz[2]))
    os.environ["ARENA_ISAAC_LIDAR_FRONT_YAW"] = str(front_yaw)
    os.environ["ARENA_ISAAC_LIDAR_REAR_YAW"] = str(rear_yaw)

    front_path = _create_lidar_sensor(
        robot_root_path=robot_root_path,
        sensor_name="front_laser",
        translation=front_xyz,
        yaw=front_yaw,
        config_name=config_name,
        logger=logger,
    )
    rear_path = _create_lidar_sensor(
        robot_root_path=robot_root_path,
        sensor_name="rear_laser",
        translation=rear_xyz,
        yaw=rear_yaw,
        config_name=config_name,
        logger=logger,
    )

    # Debug markers are diagnostic only.  They must never prevent the actual
    # RTX lidar sensors and ROS publish graphs from being created.
    try:
        _create_lidar_debug_marker(front_path, name="front_laser", fov_deg=fov_deg, color=(0.0, 0.4, 1.0), logger=logger)
        _create_lidar_debug_marker(rear_path, name="rear_laser", fov_deg=fov_deg, color=(1.0, 0.2, 0.0), logger=logger)
    except Exception as exc:
        _log(logger, "warning", f"[dual_lidar] debug marker creation failed; lidar itself remains active: {exc}")

    _delete_prim_if_exists(f"{robot_root_path}/front_lidar_publish_graph")
    _delete_prim_if_exists(f"{robot_root_path}/rear_lidar_publish_graph")

    _create_lidar_publish_graph(
        graph_path=f"{robot_root_path}/front_lidar_publish_graph",
        lidar_prim_path=front_path,
        topic_name=front_topic,
        frame_id=front_frame,
    )
    _create_lidar_publish_graph(
        graph_path=f"{robot_root_path}/rear_lidar_publish_graph",
        lidar_prim_path=rear_path,
        topic_name=rear_topic,
        frame_id=rear_frame,
    )
    _log(
        logger,
        "info",
        f"[dual_lidar] Created front/rear RTX lidar for {robot_name}: "
        f"{front_topic} ({front_frame}) xyz=({front_xyz[0]:+.3f},{front_xyz[1]:+.3f},{front_xyz[2]:+.3f}) yaw={front_yaw:.3f} at {front_path}; "
        f"{rear_topic} ({rear_frame}) xyz=({rear_xyz[0]:+.3f},{rear_xyz[1]:+.3f},{rear_xyz[2]:+.3f}) yaw={rear_yaw:.3f} at {rear_path}",
    )
    return front_path, rear_path
