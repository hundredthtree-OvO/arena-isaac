"""Dual front/rear RTX lidar helpers for stable kinematic robot validation."""
from __future__ import annotations

import json
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import omni.graph.core as og
import omni.kit.commands
import omni.usd
from pxr import Gf, Usd, UsdGeom

try:
    from isaacsim.core.utils.extensions import enable_extension, get_extension_path_from_name
except Exception:  # pragma: no cover
    from omni.isaac.core.utils.extensions import enable_extension, get_extension_path_from_name  # type: ignore

from isaac_utils.rtx_sensor_settings import ensure_rtx_sensor_coord_frame


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


def _quatf_from_rpy(roll: float, pitch: float, yaw: float) -> Gf.Quatf:
    cr = math.cos(0.5 * roll)
    sr = math.sin(0.5 * roll)
    cp = math.cos(0.5 * pitch)
    sp = math.sin(0.5 * pitch)
    cy = math.cos(0.5 * yaw)
    sy = math.sin(0.5 * yaw)
    return Gf.Quatf(
        float(cr * cp * cy + sr * sp * sy),
        float(sr * cp * cy - cr * sp * sy),
        float(cr * sp * cy + sr * cp * sy),
        float(cr * cp * sy - sr * sp * cy),
    )


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


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


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


def _set_local_xform_rpy(prim, *, translate=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)):
    x = UsdGeom.Xformable(prim)
    try:
        x.ClearXformOpOrder()
    except Exception:
        pass
    x.AddTranslateOp().Set(Gf.Vec3d(float(translate[0]), float(translate[1]), float(translate[2])))
    x.AddOrientOp().Set(_quatf_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2])))


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


def _find_descendant_named(root_path: str, name: str):
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None
    root = stage.GetPrimAtPath(root_path)
    if root is None or not root.IsValid():
        return None
    target = str(name).strip().lstrip("/")
    direct = stage.GetPrimAtPath(f"{root_path}/{target}")
    if direct and direct.IsValid():
        return direct
    for prim in stage.Traverse():
        if not prim.IsValid() or not prim.IsActive():
            continue
        path = str(prim.GetPath())
        if path.startswith(root_path + "/") and path.endswith("/" + target):
            return prim
    return None


def _parse_calibration_lidar_urdf(path: Path):
    tree = ET.parse(path)
    root = tree.getroot()
    link = root.find("link")
    joint = root.find("joint")
    origin = joint.find("origin") if joint is not None else None
    parent = joint.find("parent") if joint is not None else None
    if link is None or origin is None:
        raise ValueError(f"invalid calibration lidar URDF: {path}")
    link_name = str(link.attrib.get("name", "")).strip()
    parent_name = str(parent.attrib.get("link", "base_footprint")).strip() if parent is not None else "base_footprint"
    xyz = tuple(float(x) for x in str(origin.attrib.get("xyz", "0 0 0")).split())
    rpy = tuple(float(x) for x in str(origin.attrib.get("rpy", "0 0 0")).split())
    if len(xyz) != 3 or len(rpy) != 3 or not link_name:
        raise ValueError(f"incomplete calibration lidar URDF origin/link: {path}")
    return link_name, parent_name, xyz, rpy


def _default_calibration_dir() -> Path:
    return Path(os.environ.get(
        "ARENA_ISAAC_LIDAR_CALIBRATION_DIR",
        "/home/stardust/resources/calibration/src/stardust_calibration/urdf/v25730",
    ))


def _ensure_calibration_lidar_link(robot_root_path: str, *, urdf_path: Path, fallback_parent: str = "base_footprint", logger=None):
    link_name, parent_name, xyz, rpy = _parse_calibration_lidar_urdf(urdf_path)
    existing_prim = _find_descendant_named(robot_root_path, link_name)
    if existing_prim is not None and existing_prim.IsValid():
        _log(
            logger,
            "info",
            f"[dual_lidar] using existing in-asset calibration mount {link_name}: {existing_prim.GetPath()}",
        )
        return str(existing_prim.GetPath()), link_name
    parent_prim = _find_descendant_named(robot_root_path, parent_name) or _find_descendant_named(robot_root_path, fallback_parent)
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage is not available")
    if parent_prim is None or not parent_prim.IsValid():
        parent_prim = stage.GetPrimAtPath(robot_root_path)
    if parent_prim is None or not parent_prim.IsValid():
        raise RuntimeError(f"cannot find lidar parent for {link_name}: {parent_name}")
    link_path = f"{str(parent_prim.GetPath())}/{link_name}"
    prim = _ensure_xform(link_path)
    if prim is None or not prim.IsValid():
        raise RuntimeError(f"failed to create calibration lidar link {link_path}")
    _set_local_xform_rpy(prim, translate=xyz, rpy=rpy)
    _log(
        logger,
        "info",
        f"[dual_lidar] calibration mount {link_name}: parent={parent_prim.GetPath()} xyz={xyz} rpy={rpy}",
    )
    return str(prim.GetPath()), link_name


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
    range_offset_m: float = 0.0,
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
            "rangeOffset": float(range_offset_m),
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
    publish_points = _env_bool("ARENA_ISAAC_LIDAR_PUBLISH_POINTS", False)
    keys = og.Controller.Keys
    create_nodes = [
        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
        ("RunOnce", "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"),
        ("RenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
        ("Context", "isaacsim.ros2.bridge.ROS2Context"),
        ("LaserScanPublisher", "isaacsim.ros2.bridge.ROS2RtxLidarHelper"),
    ]
    set_values = [
        ("RenderProduct.inputs:cameraPrim", lidar_prim_path),
        ("LaserScanPublisher.inputs:topicName", topic_name.lstrip("/")),
        ("LaserScanPublisher.inputs:frameId", frame_id),
        ("LaserScanPublisher.inputs:type", "laser_scan"),
    ]
    connect = [
        ("OnPlaybackTick.outputs:tick", "RunOnce.inputs:execIn"),
        ("RunOnce.outputs:step", "RenderProduct.inputs:execIn"),
        ("RenderProduct.outputs:execOut", "LaserScanPublisher.inputs:execIn"),
        ("RenderProduct.outputs:renderProductPath", "LaserScanPublisher.inputs:renderProductPath"),
        ("Context.outputs:context", "LaserScanPublisher.inputs:context"),
    ]
    if publish_points:
        create_nodes.append(("PointCloudPublisher", "isaacsim.ros2.bridge.ROS2RtxLidarHelper"))
        set_values.extend([
            ("PointCloudPublisher.inputs:topicName", f"{topic_name.strip('/')}/points"),
            ("PointCloudPublisher.inputs:frameId", frame_id),
            ("PointCloudPublisher.inputs:type", "point_cloud"),
        ])
        connect.extend([
            ("RenderProduct.outputs:execOut", "PointCloudPublisher.inputs:execIn"),
            ("RenderProduct.outputs:renderProductPath", "PointCloudPublisher.inputs:renderProductPath"),
            ("Context.outputs:context", "PointCloudPublisher.inputs:context"),
        ])
    og.Controller.edit(
        {"graph_path": graph_path},
        {
            keys.CREATE_NODES: create_nodes,
            keys.SET_VALUES: set_values,
            keys.CONNECT: connect,
        },
    )


def create_dual_lidar_mount_frames(
    *,
    robot_root_path: str,
    robot_name: str,
    logger=None,
    front_frame: str = "base_scan_01",
    rear_frame: str = "base_scan_02",
):
    """Create real-calibration lidar frame Xforms without RTX raycast sensors.

    Synthetic 2D navigation scans should keep the real base_scan_01/02 TFs but
    must not raycast from inside the full robot mesh.  This helper creates the
    calibration links plus zero-offset front_laser/rear_laser Xforms so
    mecanum_teleop can publish static sensor TFs from USD without creating any
    RTX sensor or ROS RTX publish graph.
    """
    _ensure_xform(robot_root_path)
    calibration_dir = _default_calibration_dir()
    front_mount_path, front_mount_frame = _ensure_calibration_lidar_link(
        robot_root_path,
        urdf_path=calibration_dir / "base_scan_01.urdf.xml",
        logger=logger,
    )
    rear_mount_path, rear_mount_frame = _ensure_calibration_lidar_link(
        robot_root_path,
        urdf_path=calibration_dir / "base_scan_02.urdf.xml",
        logger=logger,
    )
    front_frame = os.environ.get("ARENA_ISAAC_FRONT_LASER_FRAME", front_frame or front_mount_frame)
    rear_frame = os.environ.get("ARENA_ISAAC_REAR_LASER_FRAME", rear_frame or rear_mount_frame)

    front_marker_path = f"{front_mount_path}/front_laser"
    rear_marker_path = f"{rear_mount_path}/rear_laser"
    _delete_prim_if_exists(front_marker_path)
    _delete_prim_if_exists(rear_marker_path)
    front_marker = _ensure_xform(front_marker_path)
    rear_marker = _ensure_xform(rear_marker_path)
    if front_marker is not None and front_marker.IsValid():
        _set_local_xform(front_marker, translate=(0.0, 0.0, 0.0), yaw=0.0)
    if rear_marker is not None and rear_marker.IsValid():
        _set_local_xform(rear_marker, translate=(0.0, 0.0, 0.0), yaw=0.0)

    _log(
        logger,
        "info",
        f"[dual_lidar] synthetic mount frames for {robot_name}: "
        f"front={front_marker_path} frame={front_frame}, rear={rear_marker_path} frame={rear_frame}",
    )
    return front_marker_path, rear_marker_path


def create_dual_lidar(
    *,
    robot_root_path: str,
    robot_name: str,
    logger=None,
    front_topic: str = "/front_scan",
    rear_topic: str = "/rear_scan",
    front_frame: str = "base_scan_01",
    rear_frame: str = "base_scan_02",
    samples: int = 721,
    fov_deg: float = 270.0,
    range_min: float = 0.05,
    range_max: float = 12.0,
    update_rate: float = 10.0,
):
    """Attach front/rear 2D RTX lidar sensors to a mobile robot root."""
    publish_points = _env_bool("ARENA_ISAAC_LIDAR_PUBLISH_POINTS", False)
    range_offset_m = _env_float("ARENA_ISAAC_LIDAR_RANGE_OFFSET_M", 0.0)
    ensure_rtx_sensor_coord_frame(
        logger=logger,
        reason=f"create_dual_lidar robot={robot_name}",
    )
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
        range_offset_m=float(range_offset_m),
    )

    calibration_dir = _default_calibration_dir()
    front_mount_path, front_mount_frame = _ensure_calibration_lidar_link(
        robot_root_path,
        urdf_path=calibration_dir / "base_scan_01.urdf.xml",
        logger=logger,
    )
    rear_mount_path, rear_mount_frame = _ensure_calibration_lidar_link(
        robot_root_path,
        urdf_path=calibration_dir / "base_scan_02.urdf.xml",
        logger=logger,
    )
    front_frame = os.environ.get("ARENA_ISAAC_FRONT_LASER_FRAME", front_frame or front_mount_frame)
    rear_frame = os.environ.get("ARENA_ISAAC_REAR_LASER_FRAME", rear_frame or rear_mount_frame)
    front_frame = front_frame or front_mount_frame
    rear_frame = rear_frame or rear_mount_frame

    front_path = _create_lidar_sensor(
        robot_root_path=front_mount_path,
        sensor_name="front_laser",
        translation=(0.0, 0.0, 0.0),
        yaw=0.0,
        config_name=config_name,
        logger=logger,
    )
    rear_path = _create_lidar_sensor(
        robot_root_path=rear_mount_path,
        sensor_name="rear_laser",
        translation=(0.0, 0.0, 0.0),
        yaw=0.0,
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
        f"{front_topic} ({front_frame}) calibration_link={front_mount_path} at {front_path}; "
        f"{rear_topic} ({rear_frame}) calibration_link={rear_mount_path} at {rear_path}; "
        f"publish_points={publish_points}, range_offset_m={range_offset_m}",
    )
    return front_path, rear_path
