"""ROS /cmd_vel driven mecanum controller for Isaac Sim 4.5.

This module is intentionally independent from Arena's nav2 controller configs.  It is
used for Milestone B: import a custom URDF robot and drive its mecanum wheel joints
from geometry_msgs/Twist.

Modes:
- joint:     send wheel joint velocity targets only. This is the most physically
             faithful path, but it requires valid wheel/roller collision geometry.
- physx_diff_contact:
             send differential wheel velocity targets only. Chassis motion must
             come exclusively from PhysX wheel-ground contact; lateral commands,
             root writes, collision-guard filtering, and kinematic fallback are
             intentionally excluded.
- hybrid:    send wheel joint velocity targets and kinematically advance the root
             prim. Useful when visual wheel animation is desired but the URDF lacks
             roller collision/contact geometry.
- kinematic: kinematically advance the root prim only.
- physx_root_velocity:
             command planar root velocity and let PhysX integrate/contact-resolve
             the chassis motion. Wheels are treated as visual rolling only.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from isaacsim_msgs.srv import ResetRobot
from std_msgs.msg import String
from std_srvs.srv import SetBool
try:
    from rosgraph_msgs.msg import Clock
except Exception:  # pragma: no cover
    Clock = None  # type: ignore
try:
    from rclpy.qos import QoSProfile, ReliabilityPolicy
except Exception:  # pragma: no cover
    QoSProfile = None  # type: ignore
    ReliabilityPolicy = None  # type: ignore
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from ros2isaacsim.drive_kinematics import (
    DIFFERENTIAL_DRIVE,
    MECANUM_DRIVE,
    wheel_angular_speeds,
)
from ros2isaacsim.actual_motion_state import ActualMotionStateEstimator
from ros2isaacsim.motion_backends import motion_backend_for_mode
from ros2isaacsim.transform_math import child_relative_to_parent
from ros2isaacsim.physx_diff_contact import (
    PhysxDiffContactDrive,
    articulation_link_wrench_arrays,
    config_from_values,
    separated_tire_wrench,
    single_articulation_targets,
)
try:
    from isaac_utils.dynamic_actor_guard import (
        RobotGuardState,
        current_pedestrian_contacts,
        footprint_extents,
        predict_robot_pose,
        scale_robot_command_for_pedestrians,
    )
except Exception:  # pragma: no cover
    from ros2isaacsim.isaac_utils.dynamic_actor_guard import (
        RobotGuardState,
        current_pedestrian_contacts,
        footprint_extents,
        predict_robot_pose,
        scale_robot_command_for_pedestrians,
    )
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, Sdf, Usd, UsdGeom
try:
    from pxr import UsdPhysics
except Exception:  # pragma: no cover
    UsdPhysics = None  # type: ignore

try:
    from isaacsim.core.prims import Articulation, XFormPrim
except Exception:  # pragma: no cover - compatibility fallback for older Isaac paths
    from omni.isaac.core.articulations import Articulation  # type: ignore
    from omni.isaac.core.prims import XFormPrim  # type: ignore

try:
    from isaacsim.core.utils.types import ArticulationAction
except Exception:  # pragma: no cover
    from omni.isaac.core.utils.types import ArticulationAction  # type: ignore


MECANUM_MODEL_PREFIX = "mecanum730_xms5"

_MECANUM_HOLD_LINK_NAMES = {
    "XMS5_R800_W4G3B4C_base",
    "XMS5_R800_W4G3B4C_link1",
    "XMS5_R800_W4G3B4C_link2",
    "XMS5_R800_W4G3B4C_link3",
    "XMS5_R800_W4G3B4C_link4",
    "XMS5_R800_W4G3B4C_link5",
    "link6",
    "tool_link",
    "gripper_link1",
    "gripper_link2",
    "tool_tip_link",
}


@dataclass
class MecanumConfig:
    wheel_joints: List[str] = field(default_factory=lambda: [
        "wheel_fl_joint",
        "wheel_fr_joint",
        "wheel_rl_joint",
        "wheel_rr_joint",
    ])
    arm_hold_joints: List[str] = field(default_factory=lambda: [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    ])
    gripper_hold_joints: List[str] = field(default_factory=lambda: [
        "gripper_joint1",
        "gripper_joint2",
    ])
    # Navigation-safe folded arm pose.
    # For stage-1 social navigation we prefer a compact "arm tucked down" pose
    # over the forward-ready manipulation posture, to reduce wall/door contact.
    arm_hold_positions: List[float] = field(default_factory=lambda: [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ])
    gripper_hold_positions: List[float] = field(default_factory=lambda: [
        0.0,
        0.0,
    ])
    hold_drive_stiffness: float = 1.0e5
    hold_drive_damping: float = 1.0e4
    hold_drive_max_force: float = 1.0e5
    arm_hold_disable_gravity: bool = True
    # URDF wheel joint origins are approximately +/-0.1575, +/-0.1725 m.
    half_length: float = 0.1575
    half_width: float = 0.1725
    differential_track_width: float = 0.345
    # Estimated wheel radius. If the robot is too slow/fast, tune this first.
    wheel_radius: float = 0.060
    max_wheel_speed: float = 20.0
    wheel_drive_damping: float = 8.0
    wheel_drive_max_force: float = 35.0
    wheel_static_friction: float = 0.25
    wheel_dynamic_friction: float = 0.20
    differential_linear_gain: float = 1.0
    differential_angular_gain: float = 1.0
    separated_tire_force_enabled: bool = False
    tire_longitudinal_stiffness: float = 60.0
    tire_lateral_stiffness: float = 20.0
    tire_max_longitudinal_force: float = 12.0
    tire_max_lateral_force: float = 4.0
    tire_contact_refresh_sec: float = 0.10
    max_linear_speed: float = 0.8
    max_lateral_speed: float = 0.5
    max_angular_speed: float = 1.2
    # Joint sign convention may need tuning after first physical test.
    # Default is chosen for a common X mecanum layout.
    wheel_signs: List[float] = field(default_factory=lambda: [1.0, -1.0, 1.0, -1.0])
    mode: str = "joint"  # joint | physx_diff_contact | hybrid | kinematic | physx_root_velocity
    timeout_sec: float = 0.5
    differential_timeout_sec: float = 0.30
    # v18: optional velocity slew-rate limiter for more realistic mobile-base
    # commands.  Values can be overridden with ARENA_ISAAC_MOTION_* env vars.
    max_linear_accel: float = 0.40
    max_lateral_accel: float = 0.40
    max_angular_accel: float = 0.80
    max_linear_decel: float = 0.80
    max_lateral_decel: float = 0.80
    max_angular_decel: float = 1.20
    smoothing_enabled: bool = True
    # v18: PhysX-root-velocity stabilization.  The x/y/yaw command is still
    # user-controlled; z and roll/pitch are gently corrected to avoid floating
    # articulation drift/tip-over.
    physx_height_kp: float = 8.0
    physx_height_max_vel: float = 0.35
    physx_upright_kp: float = 10.0
    physx_upright_max_ang_vel: float = 1.5
    physx_fallback_kinematic: bool = True
    arm_hold_hard_sync: bool = True
    roller_freeze_keywords: List[str] = field(default_factory=lambda: ["roller"])
    spawn_settling_sec: float = 0.8
    spawn_settling_min_stable_sec: float = 0.3
    spawn_settling_timeout_sec: float = 3.0
    spawn_settling_max_roll_pitch_deg: float = 2.0
    spawn_settling_max_lin_speed: float = 0.03
    spawn_settling_max_ang_speed: float = 0.15
    spawn_settling_max_z_speed: float = 0.02


def is_mecanum_model(robot_model: str) -> bool:
    return (robot_model or "").startswith(MECANUM_MODEL_PREFIX)


def parse_mecanum_config(robot_model: str) -> MecanumConfig:
    """Parse a compact mode suffix from UrdfToUsd.robot_model.

    Supported values:
      - mecanum730_xms5
      - mecanum730_xms5_joint
      - mecanum730_xms5_hybrid
      - mecanum730_xms5_kinematic
      - mecanum730_xms5_lidar_physx_root_velocity
      - mecanum730_xms5_lidar_physx_diff_contact
      - mecanum730_xms5_physx
    """
    config = MecanumConfig()
    model = (robot_model or "").lower()
    if "physx_diff_contact" in model:
        config.mode = "physx_diff_contact"
    elif "physx_root_velocity" in model or model.endswith("_physx"):
        config.mode = "physx_root_velocity"
    elif model.endswith("_hybrid"):
        config.mode = "hybrid"
    elif model.endswith("_kinematic"):
        config.mode = "kinematic"
    else:
        config.mode = "joint"

    config.max_linear_speed = _env_float("ARENA_ISAAC_MAX_LINEAR_SPEED", config.max_linear_speed)
    config.max_lateral_speed = _env_float("ARENA_ISAAC_MAX_LATERAL_SPEED", config.max_lateral_speed)
    config.max_angular_speed = _env_float("ARENA_ISAAC_MAX_ANGULAR_SPEED", config.max_angular_speed)
    config.wheel_radius = max(
        1.0e-6,
        _env_float("ARENA_ISAAC_DIFF_WHEEL_RADIUS", config.wheel_radius),
    )
    config.max_wheel_speed = max(
        0.0,
        _env_float("ARENA_ISAAC_DIFF_MAX_WHEEL_SPEED", config.max_wheel_speed),
    )
    wheel_signs = _env_float_list("ARENA_ISAAC_DIFF_WHEEL_SIGNS", len(config.wheel_signs))
    if wheel_signs is not None:
        config.wheel_signs = wheel_signs
    config.wheel_drive_damping = max(
        0.0,
        _env_float("ARENA_ISAAC_DIFF_WHEEL_DRIVE_DAMPING", config.wheel_drive_damping),
    )
    config.wheel_drive_max_force = max(
        0.0,
        _env_float("ARENA_ISAAC_DIFF_WHEEL_DRIVE_MAX_FORCE", config.wheel_drive_max_force),
    )
    config.wheel_static_friction = max(
        0.0,
        _env_float("ARENA_ISAAC_DIFF_WHEEL_STATIC_FRICTION", config.wheel_static_friction),
    )
    config.wheel_dynamic_friction = max(
        0.0,
        min(
            config.wheel_static_friction,
            _env_float("ARENA_ISAAC_DIFF_WHEEL_DYNAMIC_FRICTION", config.wheel_dynamic_friction),
        ),
    )
    config.differential_linear_gain = max(
        1.0e-6,
        _env_float("ARENA_ISAAC_DIFF_LINEAR_GAIN", config.differential_linear_gain),
    )
    config.differential_angular_gain = max(
        1.0e-6,
        _env_float("ARENA_ISAAC_DIFF_ANGULAR_GAIN", config.differential_angular_gain),
    )
    config.separated_tire_force_enabled = _env_bool(
        "ARENA_ISAAC_DIFF_TIRE_FORCE_ENABLED",
        config.separated_tire_force_enabled,
    )
    config.tire_longitudinal_stiffness = max(
        0.0,
        _env_float(
            "ARENA_ISAAC_DIFF_TIRE_LONGITUDINAL_STIFFNESS",
            config.tire_longitudinal_stiffness,
        ),
    )
    config.tire_lateral_stiffness = max(
        0.0,
        _env_float(
            "ARENA_ISAAC_DIFF_TIRE_LATERAL_STIFFNESS",
            config.tire_lateral_stiffness,
        ),
    )
    config.tire_max_longitudinal_force = max(
        0.0,
        _env_float(
            "ARENA_ISAAC_DIFF_TIRE_MAX_LONGITUDINAL_FORCE",
            config.tire_max_longitudinal_force,
        ),
    )
    config.tire_max_lateral_force = max(
        0.0,
        _env_float(
            "ARENA_ISAAC_DIFF_TIRE_MAX_LATERAL_FORCE",
            config.tire_max_lateral_force,
        ),
    )
    config.tire_contact_refresh_sec = max(
        0.02,
        _env_float(
            "ARENA_ISAAC_DIFF_TIRE_CONTACT_REFRESH_SEC",
            config.tire_contact_refresh_sec,
        ),
    )
    config.differential_track_width = max(
        0.0,
        _env_float("ARENA_ISAAC_DIFF_TRACK_WIDTH", config.differential_track_width),
    )
    config.differential_timeout_sec = max(
        0.05,
        _env_float("ARENA_ISAAC_DIFF_CMD_TIMEOUT_SEC", config.differential_timeout_sec),
    )
    config.max_linear_accel = _env_float("ARENA_ISAAC_MOTION_MAX_LINEAR_ACCEL", config.max_linear_accel)
    config.max_lateral_accel = _env_float("ARENA_ISAAC_MOTION_MAX_LATERAL_ACCEL", config.max_lateral_accel)
    config.max_angular_accel = _env_float("ARENA_ISAAC_MOTION_MAX_ANGULAR_ACCEL", config.max_angular_accel)
    config.max_linear_decel = _env_float("ARENA_ISAAC_MOTION_MAX_LINEAR_DECEL", config.max_linear_decel)
    config.max_lateral_decel = _env_float("ARENA_ISAAC_MOTION_MAX_LATERAL_DECEL", config.max_lateral_decel)
    config.max_angular_decel = _env_float("ARENA_ISAAC_MOTION_MAX_ANGULAR_DECEL", config.max_angular_decel)
    config.smoothing_enabled = _env_bool("ARENA_ISAAC_MOTION_SMOOTHING_ENABLED", config.smoothing_enabled)
    config.physx_height_kp = _env_float("ARENA_ISAAC_PHYSX_ROOT_HEIGHT_KP", config.physx_height_kp)
    config.physx_height_max_vel = _env_float("ARENA_ISAAC_PHYSX_ROOT_HEIGHT_MAX_VEL", config.physx_height_max_vel)
    config.physx_upright_kp = _env_float("ARENA_ISAAC_PHYSX_ROOT_UPRIGHT_KP", config.physx_upright_kp)
    config.physx_upright_max_ang_vel = _env_float("ARENA_ISAAC_PHYSX_ROOT_UPRIGHT_MAX_ANG_VEL", config.physx_upright_max_ang_vel)
    config.physx_fallback_kinematic = _env_bool("ARENA_ISAAC_PHYSX_ROOT_FALLBACK_KINEMATIC", config.physx_fallback_kinematic)
    config.arm_hold_hard_sync = _env_bool("ARENA_ISAAC_ARM_HOLD_HARD_SYNC", config.arm_hold_hard_sync)
    config.hold_drive_stiffness = _env_float("ARENA_ISAAC_HOLD_DRIVE_STIFFNESS", config.hold_drive_stiffness)
    config.hold_drive_damping = _env_float("ARENA_ISAAC_HOLD_DRIVE_DAMPING", config.hold_drive_damping)
    config.hold_drive_max_force = _env_float("ARENA_ISAAC_HOLD_DRIVE_MAX_FORCE", config.hold_drive_max_force)
    config.arm_hold_disable_gravity = _env_bool("ARENA_ISAAC_ARM_HOLD_DISABLE_GRAVITY", config.arm_hold_disable_gravity)
    config.spawn_settling_sec = max(0.0, _env_float("ARENA_ISAAC_SPAWN_SETTLING_SEC", config.spawn_settling_sec))
    config.spawn_settling_min_stable_sec = max(0.0, _env_float("ARENA_ISAAC_SPAWN_SETTLING_MIN_STABLE_SEC", config.spawn_settling_min_stable_sec))
    config.spawn_settling_timeout_sec = max(config.spawn_settling_sec, _env_float("ARENA_ISAAC_SPAWN_SETTLING_TIMEOUT_SEC", config.spawn_settling_timeout_sec))
    config.spawn_settling_max_roll_pitch_deg = max(0.0, _env_float("ARENA_ISAAC_SPAWN_SETTLING_MAX_ROLL_PITCH_DEG", config.spawn_settling_max_roll_pitch_deg))
    config.spawn_settling_max_lin_speed = max(0.0, _env_float("ARENA_ISAAC_SPAWN_SETTLING_MAX_LIN_SPEED", config.spawn_settling_max_lin_speed))
    config.spawn_settling_max_ang_speed = max(0.0, _env_float("ARENA_ISAAC_SPAWN_SETTLING_MAX_ANG_SPEED", config.spawn_settling_max_ang_speed))
    config.spawn_settling_max_z_speed = max(0.0, _env_float("ARENA_ISAAC_SPAWN_SETTLING_MAX_Z_SPEED", config.spawn_settling_max_z_speed))
    hold_positions = _env_float_list("ARENA_ISAAC_ARM_HOLD_POSITIONS", len(config.arm_hold_joints))
    if hold_positions is not None:
        config.arm_hold_positions = hold_positions
    gripper_hold_positions = _env_float_list("ARENA_ISAAC_GRIPPER_HOLD_POSITIONS", len(config.gripper_hold_joints))
    if gripper_hold_positions is not None:
        config.gripper_hold_positions = gripper_hold_positions
    return config


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


def _env_float_list(name: str, expected_len: int) -> Optional[List[float]]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        values = [float(x.strip()) for x in str(raw).split(",")]
    except Exception:
        return None
    if len(values) != expected_len:
        return None
    return values


def _stamp_from_node(node):
    try:
        return node.get_clock().now().to_msg()
    except Exception:
        return None


def _clock_qos():
    """QoS compatible with most ROS 2 /clock publishers."""
    if QoSProfile is None or ReliabilityPolicy is None:
        return 10
    try:
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        return qos
    except Exception:
        return 10


def _time_msg_to_float(stamp) -> float:
    try:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9
    except Exception:
        return -1.0


def _make_transform(parent: str, child: str, stamp, xyz, quat_wxyz) -> TransformStamped:
    t = TransformStamped()
    if stamp is not None:
        t.header.stamp = stamp
    t.header.frame_id = str(parent)
    t.child_frame_id = str(child)
    t.transform.translation.x = float(xyz[0])
    t.transform.translation.y = float(xyz[1])
    t.transform.translation.z = float(xyz[2])
    t.transform.rotation.w = float(quat_wxyz[0])
    t.transform.rotation.x = float(quat_wxyz[1])
    t.transform.rotation.y = float(quat_wxyz[2])
    t.transform.rotation.z = float(quat_wxyz[3])
    return t


def _quat_wxyz_to_xyzw_fields(qwxyz):
    return float(qwxyz[1]), float(qwxyz[2]), float(qwxyz[3]), float(qwxyz[0])


def _quat_from_matrix_wxyz(matrix) -> np.ndarray:
    try:
        quat = Gf.Transform(matrix).GetRotation().GetQuat()
        im = quat.GetImaginary()
        return np.array([float(quat.GetReal()), float(im[0]), float(im[1]), float(im[2])], dtype=np.float32)
    except Exception:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _sensor_tf_from_env(prefix: str, default_x: float, default_y: float, default_z: float, default_yaw: float, base_frame: str, frame: str, stamp):
    prefix = prefix.upper()
    x = _env_float(f"ARENA_ISAAC_LIDAR_{prefix}_X", default_x)
    y = _env_float(f"ARENA_ISAAC_LIDAR_{prefix}_Y", default_y)
    z = _env_float(f"ARENA_ISAAC_LIDAR_{prefix}_Z", default_z)
    yaw = _env_float(f"ARENA_ISAAC_LIDAR_{prefix}_YAW", default_yaw)
    return _make_transform(base_frame, frame, stamp, (x, y, z), _yaw_to_quat_wxyz(yaw))


def _find_descendant_named(root_path: str, name: str):
    try:
        stage = get_current_stage()
    except Exception:
        stage = None
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


def _yaw_from_local_x_axis(base_inv, sensor_world) -> float:
    try:
        origin_world = sensor_world.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
        x_world = sensor_world.Transform(Gf.Vec3d(1.0, 0.0, 0.0))
        origin_local = base_inv.Transform(origin_world)
        x_local = base_inv.Transform(x_world)
        dx = float(x_local[0] - origin_local[0])
        dy = float(x_local[1] - origin_local[1])
        if abs(dx) + abs(dy) < 1e-9:
            return 0.0
        return math.atan2(dy, dx)
    except Exception:
        return 0.0



def _clamp(v: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(v)))


def _slew_axis(current: float, target: float, dt: float, accel: float, decel: float) -> float:
    if dt <= 0.0:
        return float(current)
    current = float(current)
    target = float(target)
    # Use decel when target is closer to zero in the same direction, otherwise accel.
    same_sign = (current == 0.0 or target == 0.0 or (current > 0.0) == (target > 0.0))
    slowing_down = same_sign and abs(target) < abs(current)
    limit = float(decel if slowing_down else accel) * float(dt)
    if limit <= 0.0:
        return target
    return current + _clamp(target - current, -limit, limit)


def _quat_apply_wxyz(q, v):
    # Rotate vector v by quaternion q in wxyz convention.
    w, x, y, z = [float(a) for a in q]
    vx, vy, vz = [float(a) for a in v]
    # q * [0,v] * q^-1, expanded.
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return np.array([
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ], dtype=np.float32)

def _sensor_tf_from_stage_or_env(
    *,
    robot_root_path: str,
    sensor_prim_name: str,
    prefix: str,
    default_x: float,
    default_y: float,
    default_z: float,
    default_yaw: float,
    base_frame: str,
    frame: str,
    stamp,
    logger=None,
):
    """Prefer the actual USD lidar prim pose for static sensor TF.

    The old implementation published base_link->laser TF from hard-coded env
    defaults (+/-0.42).  That can diverge from the RTX lidar prim after auto
    mounting.  This function computes the transform from the stage when the
    sensor prim exists, and only falls back to env/default values if it cannot.
    """
    try:
        stage = get_current_stage()
        sensor = stage.GetPrimAtPath(f"{robot_root_path}/{sensor_prim_name}") if stage is not None else None
        if sensor is None or not sensor.IsValid():
            sensor = _find_descendant_named(robot_root_path, sensor_prim_name)
        base = _find_descendant_named(robot_root_path, base_frame)
        if base is None or not base.IsValid():
            base = stage.GetPrimAtPath(robot_root_path) if stage is not None else None
        if sensor is not None and sensor.IsValid() and base is not None and base.IsValid():
            cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            base_world = cache.GetLocalToWorldTransform(base)
            sensor_world = cache.GetLocalToWorldTransform(sensor)
            sensor_local = child_relative_to_parent(base_world, sensor_world)
            p = sensor_local.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
            quat_wxyz = _quat_from_matrix_wxyz(sensor_local)
            if logger is not None:
                try:
                    logger.info(
                        f"[sensor_tf] {base_frame}->{frame} from USD prim {sensor.GetPath()}: "
                        f"xyz=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f}) "
                        f"quat=({float(quat_wxyz[0]):+.3f},{float(quat_wxyz[1]):+.3f},{float(quat_wxyz[2]):+.3f},{float(quat_wxyz[3]):+.3f})"
                    )
                except Exception:
                    pass
            return _make_transform(base_frame, frame, stamp, (float(p[0]), float(p[1]), float(p[2])), quat_wxyz)
    except Exception as exc:
        if logger is not None:
            try:
                logger.warning(f"[sensor_tf] failed to read {sensor_prim_name} USD pose; falling back to env: {exc}")
            except Exception:
                pass
    return _sensor_tf_from_env(prefix, default_x, default_y, default_z, default_yaw, base_frame, frame, stamp)


class MecanumRobot:
    def __init__(
        self,
        *,
        name: str,
        prim_path: str,
        articulation_path: Optional[str] = None,
        nav_base_path: Optional[str] = None,
        cmd_vel_topic: str,
        config: MecanumConfig,
        logger=None,
        node=None,
        odom_frame: str = "odom",
        base_frame: str = "base_link",
        asset_root_path: Optional[str] = None,
    ):

        self.name = name
        self.prim_path = prim_path
        self.articulation_path = articulation_path or prim_path
        self.nav_base_path = nav_base_path or prim_path
        self.asset_root_path = asset_root_path or prim_path
        self.cmd_vel_topic = cmd_vel_topic
        self.config = config
        self._motion_backend = motion_backend_for_mode(config.mode)
        self._actual_state_estimator = ActualMotionStateEstimator()
        self._physx_diff_drive = PhysxDiffContactDrive(
            config_from_values(
                wheel_radius=config.wheel_radius,
                track_width=config.differential_track_width,
                wheel_signs=config.wheel_signs,
                max_wheel_speed=config.max_wheel_speed,
                linear_gain=config.differential_linear_gain,
                angular_gain=config.differential_angular_gain,
            ),
            self._apply_joint_velocity_targets,
        )
        self.logger = logger
        self.node = node
        self.odom_frame = str(os.environ.get("ARENA_ISAAC_ODOM_FRAME", odom_frame or "odom"))
        self.base_frame = str(os.environ.get("ARENA_ISAAC_BASE_FRAME", base_frame or "base_link"))
        self.odom_topic = str(os.environ.get("ARENA_ISAAC_ODOM_TOPIC", "/odom"))
        self.applied_cmd_vel_topic = str(os.environ.get("ARENA_ISAAC_CMD_VEL_APPLIED_TOPIC", "/cmd_vel_applied"))
        self.front_laser_frame = str(os.environ.get("ARENA_ISAAC_FRONT_LASER_FRAME", "front_laser_link"))
        self.rear_laser_frame = str(os.environ.get("ARENA_ISAAC_REAR_LASER_FRAME", "rear_laser_link"))
        self.publish_actual_odom_tf = _env_bool("ARENA_ISAAC_PUBLISH_ACTUAL_ODOM_TF", True)
        self.publish_static_sensor_tf = _env_bool("ARENA_ISAAC_PUBLISH_SENSOR_STATIC_TF", True)
        self._odom_pub = None
        self._applied_pub = None
        self._hard_guard_event_pub = None
        self._pedestrian_contact_pub = None
        self._reset_status_pub = None
        self._tf_pub = None
        self._static_tf_pub = None
        self._static_tf_sent = False
        self._clock_sub = None
        self._last_clock_stamp = None
        self._last_clock_time_float = -1.0
        self._warned_waiting_for_clock = False
        self._use_clock_topic_for_tf_stamps = _env_bool("ARENA_ISAAC_USE_CLOCK_TOPIC_FOR_TF_STAMPS", True)
        if self.node is not None:
            try:
                if self.publish_actual_odom_tf:
                    self._odom_pub = self.node.create_publisher(Odometry, self.odom_topic, 10)
                    self._tf_pub = TransformBroadcaster(self.node)
                    if self.publish_static_sensor_tf:
                        self._static_tf_pub = StaticTransformBroadcaster(self.node)
                self._applied_pub = self.node.create_publisher(Twist, self.applied_cmd_vel_topic, 10)
                self._hard_guard_event_pub = self.node.create_publisher(
                    String,
                    "/isaac/pedestrian_hard_guard_events",
                    10,
                )
                self._pedestrian_contact_pub = self.node.create_publisher(
                    String,
                    "/isaac/pedestrian_contact_events",
                    10,
                )
                self._reset_status_pub = self.node.create_publisher(
                    String,
                    "/isaac/mecanum_reset_status",
                    10,
                )
                if self._use_clock_topic_for_tf_stamps and Clock is not None:
                    self._clock_sub = self.node.create_subscription(Clock, "/clock", self._clock_cb, _clock_qos())
            except Exception as exc:
                self._log_warn(f"[{self.name}] failed to create actual odom/tf publishers: {exc}")
        self._lock = threading.Lock()
        self._last_cmd_time = 0.0
        self._last_update_time: Optional[float] = None
        self._vx = 0.0
        self._vy = 0.0
        self._wz = 0.0
        self._diff_vx = 0.0
        self._diff_wz = 0.0
        self._last_diff_cmd_time = 0.0
        self._active_drive_mode = MECANUM_DRIVE
        self._applied_vx = 0.0
        self._applied_vy = 0.0
        self._applied_wz = 0.0
        self._last_guard_mode = "init"
        self._pedestrian_hard_guard_enabled = _env_bool(
            "ARENA_ISAAC_PEDESTRIAN_HARD_GUARD_ENABLED",
            False,
        )
        self._pedestrian_hard_guard_margin_m = max(
            0.0,
            _env_float("ARENA_ISAAC_PEDESTRIAN_HARD_GUARD_MARGIN_M", 0.04),
        )
        self._pedestrian_hard_guard_latency_sec = max(
            0.0,
            _env_float("ARENA_ISAAC_PEDESTRIAN_HARD_GUARD_LATENCY_SEC", 0.10),
        )
        self._pedestrian_hard_guard_sample_dt_sec = max(
            0.01,
            _env_float("ARENA_ISAAC_PEDESTRIAN_HARD_GUARD_SAMPLE_DT_SEC", 0.04),
        )
        self._pedestrian_hard_guard_escape_horizon_sec = max(
            0.0,
            _env_float("ARENA_ISAAC_PEDESTRIAN_HARD_GUARD_ESCAPE_HORIZON_SEC", 0.0),
        )
        self._pedestrian_hard_guard_overlap_deadband_m = max(
            0.0,
            _env_float("ARENA_ISAAC_PEDESTRIAN_HARD_GUARD_OVERLAP_DEADBAND_M", 0.0),
        )
        self._last_pedestrian_hard_guard_log = 0.0
        self._last_pedestrian_hard_guard_event = 0.0
        self._active_pedestrian_contacts: set[str] = set()
        self._articulation = None
        self._diff_base_link_index: Optional[int] = None
        self._last_diff_force_view_attempt = 0.0
        self._warned_diff_force_view = False
        self._xform = None
        self._joint_indices: Optional[List[int]] = None
        # _heading is the incremental yaw applied by the kinematic/hybrid
        # controller. _spawn_orientation_wxyz preserves the import/spawn
        # alignment quaternion (for robots whose URDF needs roll/pitch to stand
        # upright).  Previous versions rebuilt orientation from yaw only, which
        # discarded roll/pitch and made such robots fall back onto their side.
        self._heading = 0.0
        self._spawn_orientation_wxyz = None
        self._kinematic_pos = None
        self._initialized = False
        self._smooth_vx = 0.0
        self._smooth_vy = 0.0
        self._smooth_wz = 0.0
        self._physx_nominal_z = None
        self._warned_physx_velocity_fallback = False
        self._arm_joint_indices: Optional[List[int]] = None
        self._last_cmd_log_t = 0.0
        self._last_wheel_target_log_t = 0.0
        self._last_diff_contact_sample_t = 0.0
        self._diff_contact_samples: List[Dict] = []
        self._measured_vx = 0.0
        self._measured_vy = 0.0
        self._measured_wz = 0.0
        self._diff_diagnostics_output = os.environ.get(
            "ARENA_ISAAC_DIFF_DIAGNOSTICS_OUTPUT",
            "/tmp/arena_physx_diff_diagnostics.jsonl",
        ).strip()
        self._diff_diagnostics_run_label = os.environ.get(
            "ARENA_ISAAC_DIFF_DIAGNOSTICS_RUN_LABEL",
            "default",
        ).strip() or "default"
        self._gripper_joint_indices: Optional[List[int]] = None
        self._roller_joint_indices: Optional[List[int]] = None
        self._roller_hold_positions: Optional[np.ndarray] = None
        self._settling_state = "pending"
        self._settling_started_at: Optional[float] = None
        self._settling_warmup_deadline: Optional[float] = None
        self._settling_timeout_deadline: Optional[float] = None
        self._settling_stable_started_at: Optional[float] = None
        self._settling_prev_pos: Optional[np.ndarray] = None
        self._settling_prev_yaw: Optional[float] = None
        self._warned_missing_articulation_controller = False
        self._warned_joint_hold_failure = False
        self._warned_wheel_target_failure = False
        self._hold_drive_configured = False
        self._warned_hold_drive_no_match = False
        self._hold_drive_configured_paths = set()
        self._hold_gravity_configured_paths = set()
        self._last_hold_scene_constraint_time = 0.0
        self._pending_episode_reset = None
        self._episode_reset_generation = 0
        self._episode_reset_requested_generation = 0
        self._episode_control_hold = False

    def _clock_cb(self, msg):
        try:
            stamp = msg.clock
            t = _time_msg_to_float(stamp)
            # Ignore backward /clock jumps inside this bridge process to avoid
            # publishing TF_OLD_DATA after a timeline reset. A full bridge restart
            # is still recommended after resetting the Isaac timeline.
            if self._last_clock_time_float >= 0.0 and t + 1e-9 < self._last_clock_time_float:
                self._log_warn(
                    f"[{self.name}] /clock moved backwards from {self._last_clock_time_float:.3f}s to {t:.3f}s; "
                    "ignoring this clock sample. Restart bridge after timeline reset."
                )
                return
            self._last_clock_stamp = stamp
            self._last_clock_time_float = t
        except Exception:
            pass

    def _current_tf_stamp(self):
        if self._use_clock_topic_for_tf_stamps:
            if self._last_clock_stamp is not None:
                return self._last_clock_stamp
            if not self._warned_waiting_for_clock:
                self._warned_waiting_for_clock = True
                self._log_warn(f"[{self.name}] waiting for /clock before publishing dynamic odom TF")
            return None
        return _stamp_from_node(self.node)

    def set_cmd(self, msg: Twist):
        if self._settling_state != "ready" or self._episode_control_hold:
            return
        c = self.config
        vx = max(-c.max_linear_speed, min(c.max_linear_speed, float(msg.linear.x)))
        vy = max(-c.max_lateral_speed, min(c.max_lateral_speed, float(msg.linear.y)))
        wz = max(-c.max_angular_speed, min(c.max_angular_speed, float(msg.angular.z)))
        with self._lock:
            self._vx, self._vy, self._wz = vx, vy, wz
            self._last_cmd_time = time.monotonic()
        now = time.monotonic()
        if (abs(vx) > 1e-4 or abs(vy) > 1e-4 or abs(wz) > 1e-4) and (now - self._last_cmd_log_t) > 1.0:
            self._last_cmd_log_t = now
            self._log_info(f"[{self.name}] cmd_vel received: vx={vx:+.3f}, vy={vy:+.3f}, wz={wz:+.3f}")

    def set_differential_cmd(self, msg: Twist):
        """Store a gamepad command; a fresh differential command has priority."""
        if self._settling_state != "ready" or self._episode_control_hold:
            return
        c = self.config
        vx = max(-c.max_linear_speed, min(c.max_linear_speed, float(msg.linear.x)))
        wz = max(-c.max_angular_speed, min(c.max_angular_speed, float(msg.angular.z)))
        with self._lock:
            self._diff_vx, self._diff_wz = vx, wz
            self._last_diff_cmd_time = time.monotonic()
        now = time.monotonic()
        if (abs(vx) > 1e-4 or abs(wz) > 1e-4) and (now - self._last_cmd_log_t) > 1.0:
            self._last_cmd_log_t = now
            self._log_info(
                f"[{self.name}] differential cmd received: vx={vx:+.3f}, wz={wz:+.3f}"
            )

    def set_episode_control_hold(self, enabled: bool) -> tuple[bool, str]:
        """Gate operator commands without changing legacy motion modes."""
        if self.config.mode != "physx_diff_contact":
            return False, f"robot {self.name} is in mode {self.config.mode}, not physx_diff_contact"
        enabled = bool(enabled)
        with self._lock:
            changed = self._episode_control_hold != enabled
            self._episode_control_hold = enabled
            self._clear_motion_commands_locked()
            self._last_guard_mode = "episode_control_hold" if enabled else "episode_control_released"
        if changed:
            self._log_info(f"[{self.name}] episode control hold {'enabled' if enabled else 'released'}")
        return True, f"episode control hold {'enabled' if enabled else 'released'} for {self.name}"

    def request_episode_reset(self, position, orientation_wxyz) -> tuple[bool, str, int]:
        """Queue a contact-mode reset for execution on the simulation thread."""
        if self.config.mode != "physx_diff_contact":
            return False, f"robot {self.name} is in mode {self.config.mode}, not physx_diff_contact", 0
        try:
            position = np.asarray(position, dtype=np.float32).reshape(3)
            orientation = np.asarray(orientation_wxyz, dtype=np.float32).reshape(4)
        except Exception as exc:
            return False, f"invalid reset pose: {exc}", 0
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(orientation)):
            return False, "reset pose contains non-finite values", 0
        norm = float(np.linalg.norm(orientation))
        if norm <= 1.0e-6:
            return False, "reset orientation has zero norm", 0
        orientation /= norm
        with self._lock:
            self._episode_reset_requested_generation += 1
            generation = self._episode_reset_requested_generation
            self._pending_episode_reset = (generation, position.copy(), orientation.copy())
            self._clear_motion_commands_locked()
        return True, f"queued reset generation {generation} for {self.name}", generation

    def _publish_reset_status(self, generation: int, status: str, message: str) -> None:
        if self._reset_status_pub is None:
            return
        payload = {
            "robot": self.name,
            "generation": int(generation),
            "status": str(status),
            "message": str(message),
            "sim_time_sec": float(self._last_clock_time_float),
        }
        self._reset_status_pub.publish(String(data=json.dumps(payload, sort_keys=True)))

    def _clear_motion_commands_locked(self) -> None:
        self._vx = self._vy = self._wz = 0.0
        self._diff_vx = self._diff_wz = 0.0
        self._last_cmd_time = 0.0
        self._last_diff_cmd_time = 0.0
        self._smooth_vx = self._smooth_vy = self._smooth_wz = 0.0
        self._applied_vx = self._applied_vy = self._applied_wz = 0.0
        self._active_drive_mode = MECANUM_DRIVE
        self._last_guard_mode = "episode_reset"

    def _write_episode_reset_pose(self, position: np.ndarray, orientation: np.ndarray) -> bool:
        wrote_pose = False
        if self._articulation is not None and hasattr(self._articulation, "write_root_pose_to_sim"):
            try:
                values = [[
                    float(position[0]),
                    float(position[1]),
                    float(position[2]),
                    float(orientation[0]),
                    float(orientation[1]),
                    float(orientation[2]),
                    float(orientation[3]),
                ]]
                try:
                    import torch  # type: ignore

                    root_pose = torch.tensor(values, dtype=torch.float32)
                except Exception:
                    root_pose = np.asarray(values, dtype=np.float32)
                self._articulation.write_root_pose_to_sim(root_pose)
                wrote_pose = True
            except Exception as exc:
                self._log_warn(f"[{self.name}] episode reset root-pose write failed: {exc}")
        if not wrote_pose:
            try:
                _set_usd_xform_pose(self.prim_path, position, orientation)
                wrote_pose = True
            except Exception as exc:
                self._log_warn(f"[{self.name}] episode reset USD pose fallback failed: {exc}")
        return wrote_pose

    def _apply_pending_episode_reset(self) -> bool:
        with self._lock:
            pending = self._pending_episode_reset
            if pending is not None:
                self._pending_episode_reset = None
                self._clear_motion_commands_locked()
        if pending is None:
            return False
        generation, position, orientation = pending
        zero_wheels = np.zeros(len(self._joint_indices or []), dtype=np.float32)
        if len(zero_wheels):
            self._apply_joint_velocity_targets(zero_wheels)
            try:
                self._articulation.set_joint_velocities(
                    velocities=single_articulation_targets(zero_wheels),
                    joint_indices=np.asarray(self._joint_indices, dtype=np.int32),
                )
            except Exception:
                try:
                    self._articulation.set_joint_velocities(
                        velocities=zero_wheels,
                        joint_indices=np.asarray(self._joint_indices, dtype=np.int32),
                    )
                except Exception:
                    pass
        self._write_root_velocity_command(np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
        if not self._write_episode_reset_pose(position, orientation):
            message = "episode reset rejected because no pose API succeeded"
            self._log_warn(f"[{self.name}] {message}")
            self._publish_reset_status(generation, "failed", message)
            return True
        self._write_root_velocity_command(np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
        self._actual_state_estimator.reset()
        self._measured_vx = self._measured_vy = self._measured_wz = 0.0
        self._physx_nominal_z = float(position[2])
        self._kinematic_pos = position.copy()
        self._spawn_orientation_wxyz = orientation.copy()
        self._heading = 0.0
        self._diff_contact_samples = []
        self._last_diff_contact_sample_t = 0.0
        if self.config.mode == "physx_diff_contact":
            if not self._restore_physx_diff_after_episode_reset():
                message = "episode reset could not restore PhysX differential drive state"
                self._log_warn(f"[{self.name}] {message}")
                self._publish_reset_status(generation, "failed", message)
                return True
        self._episode_reset_generation = int(generation)
        self._restart_settling_state(
            warmup_sec=self.config.spawn_settling_sec,
            min_stable_sec=self.config.spawn_settling_min_stable_sec,
            timeout_sec=self.config.spawn_settling_timeout_sec,
        )
        self._log_info(
            f"[{self.name}] episode reset generation {self._episode_reset_generation} applied: "
            f"position={position.tolist()}, yaw={_quat_wxyz_to_yaw(orientation):.3f}; entering settling"
        )
        self._publish_reset_status(generation, "applied", "pose and zero velocity written; settling started")
        return True

    def _restore_physx_diff_after_episode_reset(self) -> bool:
        """Rebind and wake PhysX state invalidated by an episode teleport."""
        if self._articulation is None:
            return False

        # A pose teleport can leave a valid-looking tensor view attached to a
        # sleeping or stale articulation.  Never reuse the cached link index
        # across reset generations.
        self._diff_base_link_index = None
        self._last_diff_force_view_attempt = 0.0
        self._warned_diff_force_view = False
        try:
            handle_valid = bool(self._articulation.is_physics_handle_valid())
        except Exception:
            handle_valid = False
        if not handle_valid and not self._recreate_articulation_wrapper():
            return False

        try:
            if hasattr(self._articulation, "wake_up"):
                self._articulation.wake_up()
        except Exception as exc:
            self._log_warn(f"[{self.name}] PhysX reset articulation wake-up failed: {exc}")
            return False

        drives_ready = self._configure_physx_diff_wheel_drives()
        force_view_ready = (
            not self.config.separated_tire_force_enabled
            or self._ensure_diff_articulation_force_view(force=True)
        )
        self._diff_contact_samples = []
        self._last_diff_contact_sample_t = 0.0
        self._log_info(
            f"[{self.name}] PhysX differential reset recovery: "
            f"drives_ready={drives_ready}, force_view_ready={force_view_ready}, "
            "articulation_woken=True"
        )
        return bool(drives_ready and force_view_ready)

    def _recreate_articulation_wrapper(self) -> bool:
        """Replace tensor-backed wrappers after a timeline lifecycle invalidation."""
        self._log_warn(
            f"[{self.name}] articulation physics handle is stale; rebuilding the robot wrapper"
        )
        self._initialized = False
        self._articulation = None
        self._joint_indices = None
        self._arm_joint_indices = None
        self._gripper_joint_indices = None
        self._roller_joint_indices = None
        self._diff_base_link_index = None
        self._last_diff_force_view_attempt = 0.0
        self._diff_contact_samples = []
        self._last_diff_contact_sample_t = 0.0
        if not self._ensure_initialized():
            self._log_warn(f"[{self.name}] failed to rebuild the robot articulation wrapper")
            return False
        self._log_info(f"[{self.name}] robot articulation wrapper rebuilt successfully")
        return True

    def _log_info(self, text: str):
        if self.logger:
            self.logger.info(text)
        else:
            print(text)

    def _log_warn(self, text: str):
        if self.logger:
            self.logger.warning(text)
        else:
            print("WARNING:", text)

    def _ensure_initialized(self):
        if self._initialized:
            return True
        try:
            self._xform = None
            try:
                _pos, quat = _get_usd_xform_pose(self.prim_path)
                self._kinematic_pos = np.array(_pos, dtype=np.float32)
                # Preserve the full initial alignment, not just yaw.
                self._spawn_orientation_wxyz = _quat_to_wxyz_np(quat)
            except Exception:
                self._kinematic_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                self._spawn_orientation_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            self._heading = 0.0

            # Pure kinematic mode intentionally does not create an Isaac wrapper
            # or an Articulation. Previous versions instantiated XFormPrim, which
            # can trigger PhysX tensor lookups on Isaac 4.5 even though physics has
            # been disabled.  Here we author USD xform ops directly.
            if self.config.mode == "kinematic":
                self._articulation = None
                self._joint_indices = []
                self._initialized = True
                self._initialize_settling_state()
                self._log_info(
                    f"[{self.name}] mecanum controller initialized: mode={self.config.mode}, "
                    f"prim={self.prim_path}, direct USD kinematic root, "
                    f"smoothing={'enabled' if self.config.smoothing_enabled else 'disabled'}"
                )
                return True

            self._xform = XFormPrim(self.prim_path)

            try:
                self._articulation = Articulation(self.articulation_path)
                try:
                    self._articulation.initialize()
                except Exception:
                    # Some Isaac prim wrappers initialize lazily after timeline playback.
                    pass
                dof_names = list(getattr(self._articulation, "dof_names", []) or [])

                joint_indices = []
                for joint_name in self.config.wheel_joints:
                    idx = self._resolve_dof_index(joint_name, dof_names)
                    if idx is None:
                        raise RuntimeError(f"cannot find DOF index for joint {joint_name}")
                    joint_indices.append(int(idx))
                self._joint_indices = joint_indices
                arm_joint_indices = []
                for joint_name in self.config.arm_hold_joints:
                    idx = self._resolve_dof_index(joint_name, dof_names)
                    if idx is not None:
                        arm_joint_indices.append(int(idx))
                self._arm_joint_indices = arm_joint_indices or None
                gripper_joint_indices = []
                for joint_name in self.config.gripper_hold_joints:
                    idx = self._resolve_dof_index(joint_name, dof_names)
                    if idx is not None:
                        gripper_joint_indices.append(int(idx))
                self._gripper_joint_indices = gripper_joint_indices or None
                roller_joint_indices = []
                keywords = tuple(str(x).lower() for x in self.config.roller_freeze_keywords if str(x).strip())
                for idx, joint_name in enumerate(dof_names):
                    low = str(joint_name).lower()
                    if any(keyword in low for keyword in keywords):
                        roller_joint_indices.append(int(idx))
                self._roller_joint_indices = roller_joint_indices or None
                if self._roller_joint_indices:
                    self._roller_hold_positions = self._read_joint_positions(self._roller_joint_indices)
                if self.config.mode == "physx_diff_contact":
                    self._configure_physx_diff_wheel_drives()
                    self._ensure_diff_articulation_force_view(force=True)
                else:
                    self._ensure_hold_scene_constraints(force=True)
            except Exception as exc:
                if self.config.mode == "hybrid":
                    self._log_warn(f"[{self.name}] articulation wheel control unavailable; hybrid will run as kinematic root control: {exc}")
                    self._articulation = None
                    self._joint_indices = []
                else:
                    self._log_warn(f"[{self.name}] mecanum controller init failed: {exc}")
                    return False

            self._initialized = True
            self._initialize_settling_state()
            self._log_info(
                f"[{self.name}] mecanum controller initialized: mode={self.config.mode}, "
                f"prim={self.prim_path}, articulation={self.articulation_path}, joints={self.config.wheel_joints}, indices={self._joint_indices}, "
                f"arm_hold={self.config.arm_hold_positions}, gripper_hold={self.config.gripper_hold_positions}, "
                f"roller_freeze_count={0 if self._roller_joint_indices is None else len(self._roller_joint_indices)}, "
                f"wheel_radius={self.config.wheel_radius:.3f}, wheel_signs={self.config.wheel_signs}, "
                f"max_wheel_speed={self.config.max_wheel_speed:.3f}, "
                f"diff_gains=({self.config.differential_linear_gain:.3f}, "
                f"{self.config.differential_angular_gain:.3f}), "
                f"smoothing={'enabled' if self.config.smoothing_enabled else 'disabled'}"
            )
            return True
        except Exception as exc:
            self._log_warn(f"[{self.name}] mecanum controller init failed: {exc}")
            return False

    def _initialize_settling_state(self):
        if self.config.mode == "kinematic" or self._articulation is None or self.config.spawn_settling_timeout_sec <= 0.0:
            self._settling_state = "ready"
            self._settling_started_at = None
            self._settling_warmup_deadline = None
            self._settling_timeout_deadline = None
            self._settling_stable_started_at = None
            self._settling_prev_pos = None
            self._settling_prev_yaw = None
            return
        self._settling_state = "pending"
        self._settling_started_at = None
        self._settling_warmup_deadline = None
        self._settling_timeout_deadline = None
        self._settling_stable_started_at = None
        self._settling_prev_pos = None
        self._settling_prev_yaw = None

    def _restart_settling_state(self, *, warmup_sec: float, min_stable_sec: float, timeout_sec: float):
        self._settling_state = "pending"
        self._settling_started_at = None
        self._settling_warmup_deadline = None
        self._settling_timeout_deadline = None
        self._settling_stable_started_at = None
        self._settling_prev_pos = None
        self._settling_prev_yaw = None
        self.config.spawn_settling_sec = max(0.0, float(warmup_sec))
        self.config.spawn_settling_min_stable_sec = max(0.0, float(min_stable_sec))
        self.config.spawn_settling_timeout_sec = max(self.config.spawn_settling_sec, float(timeout_sec))

    def _finish_settling(self, *, reason: str, warn: bool = False):
        self._settling_state = "ready"
        self._settling_started_at = None
        self._settling_warmup_deadline = None
        self._settling_timeout_deadline = None
        self._settling_stable_started_at = None
        self._settling_prev_pos = None
        self._settling_prev_yaw = None
        self._physx_nominal_z = None
        self._warned_physx_velocity_fallback = False
        msg = f"[{self.name}] spawn settling complete: {reason}; controller entering ready state"
        if warn:
            self._log_warn(msg)
        else:
            self._log_info(msg)

    def _settling_metrics(self, pos: np.ndarray, quat: np.ndarray, dt: float):
        yaw = _quat_wxyz_to_yaw(quat)
        if self._settling_prev_pos is None or self._settling_prev_yaw is None or dt <= 0.0:
            lin_speed = 0.0
            z_speed = 0.0
            ang_speed = 0.0
        else:
            delta = (np.array(pos, dtype=np.float32) - self._settling_prev_pos) / float(dt)
            lin_speed = float(np.hypot(delta[0], delta[1]))
            z_speed = float(abs(delta[2]))
            yaw_delta = math.atan2(math.sin(yaw - self._settling_prev_yaw), math.cos(yaw - self._settling_prev_yaw))
            ang_speed = float(abs(yaw_delta) / float(dt))
        self._settling_prev_pos = np.array(pos, dtype=np.float32)
        self._settling_prev_yaw = float(yaw)
        roll, pitch = _quat_wxyz_to_roll_pitch(quat)
        return {
            "roll_deg": abs(math.degrees(roll)),
            "pitch_deg": abs(math.degrees(pitch)),
            "lin_speed": lin_speed,
            "z_speed": z_speed,
            "ang_speed": ang_speed,
        }

    def _maybe_update_settling_state(self, now: float, dt: float) -> bool:
        if self._settling_state == "ready":
            return False

        if self._settling_state == "pending":
            self._settling_state = "settling"
            self._settling_started_at = float(now)
            self._settling_warmup_deadline = float(now) + float(self.config.spawn_settling_sec)
            self._settling_timeout_deadline = float(now) + float(self.config.spawn_settling_timeout_sec)
            self._settling_stable_started_at = None
            self._settling_prev_pos = None
            self._settling_prev_yaw = None
            self._smooth_vx = self._smooth_vy = self._smooth_wz = 0.0
            self._applied_vx = self._applied_vy = self._applied_wz = 0.0
            self._log_info(
                f"[{self.name}] spawn settling start: warmup={self.config.spawn_settling_sec:.2f}s, "
                f"stable_for={self.config.spawn_settling_min_stable_sec:.2f}s, "
                f"timeout={self.config.spawn_settling_timeout_sec:.2f}s, "
                "holding arm/gripper/rollers and commanding zero base motion"
            )

        if self._settling_state != "settling":
            return False

        self._applied_vx = self._applied_vy = self._applied_wz = 0.0
        self._publish_applied_cmd_vel()
        self._apply_navigation_hold_targets()
        if self._joint_indices is not None:
            self._apply_joint_velocity_targets(np.zeros(len(self._joint_indices), dtype=np.float32))
        self._publish_actual_odom_tf()

        pos = quat = None
        try:
            pos, quat = _get_usd_xform_pose(self.prim_path)
        except Exception:
            pos = quat = None

        if self._settling_timeout_deadline is not None and now >= self._settling_timeout_deadline:
            self._finish_settling(reason="timeout reached before stable pose was confirmed", warn=True)
            return True

        if pos is None or quat is None:
            return True

        metrics = self._settling_metrics(pos, quat, dt)
        if self._settling_warmup_deadline is not None and now < self._settling_warmup_deadline:
            self._settling_stable_started_at = None
            return True

        stable = (
            metrics["roll_deg"] <= self.config.spawn_settling_max_roll_pitch_deg
            and metrics["pitch_deg"] <= self.config.spawn_settling_max_roll_pitch_deg
            and metrics["lin_speed"] <= self.config.spawn_settling_max_lin_speed
            and metrics["z_speed"] <= self.config.spawn_settling_max_z_speed
            and metrics["ang_speed"] <= self.config.spawn_settling_max_ang_speed
        )
        if stable:
            if self._settling_stable_started_at is None:
                self._settling_stable_started_at = float(now)
            elif (now - self._settling_stable_started_at) >= self.config.spawn_settling_min_stable_sec:
                self._finish_settling(
                    reason=(
                        f"stable pose confirmed "
                        f"(roll={metrics['roll_deg']:.2f}deg, pitch={metrics['pitch_deg']:.2f}deg, "
                        f"lin={metrics['lin_speed']:.3f}m/s, ang={metrics['ang_speed']:.3f}rad/s)"
                    )
                )
        else:
            self._settling_stable_started_at = None
        return True


    def _guard_extents(self):
        return footprint_extents(
            length=_env_float(
                "ARENA_ISAAC_ROBOT_FOOTPRINT_LENGTH",
                2.0 * float(self.config.half_length),
            ),
            width=_env_float(
                "ARENA_ISAAC_ROBOT_FOOTPRINT_WIDTH",
                2.0 * float(self.config.half_width),
            ),
            margin=_env_float("ARENA_ISAAC_ROBOT_FOOTPRINT_MARGIN", 0.05),
            footprint_forward=_env_float("ARENA_ISAAC_ROBOT_FOOTPRINT_FORWARD", -1.0),
            footprint_rear=_env_float("ARENA_ISAAC_ROBOT_FOOTPRINT_REAR", -1.0),
            footprint_left=_env_float("ARENA_ISAAC_ROBOT_FOOTPRINT_LEFT", -1.0),
            footprint_right=_env_float("ARENA_ISAAC_ROBOT_FOOTPRINT_RIGHT", -1.0),
        )

    def _current_guard_pose(self):
        try:
            pos, quat = _get_usd_xform_pose(self.prim_path)
            return np.array(pos, dtype=np.float32), float(_quat_wxyz_to_yaw(quat))
        except Exception:
            if self._kinematic_pos is None:
                return None, None
            return np.array(self._kinematic_pos, dtype=np.float32), float(self._heading)

    def get_dynamic_guard_state(self, dt: float) -> Optional[RobotGuardState]:
        pos, heading = self._current_guard_pose()
        if pos is None or heading is None:
            return None
        if self._settling_state != "ready":
            vx = vy = wz = 0.0
        elif self.config.mode == "physx_diff_contact":
            vx = float(self._measured_vx)
            vy = float(self._measured_vy)
            wz = float(self._measured_wz)
        else:
            vx, vy, wz = self._current_cmd()
        next_pos_xy, next_heading = predict_robot_pose(
            pos_xy=(float(pos[0]), float(pos[1])),
            heading=float(heading),
            vx=float(vx),
            vy=float(vy),
            wz=float(wz),
            dt=float(max(0.0, dt)),
        )
        forward, rear, left, right = self._guard_extents()
        return RobotGuardState(
            name=self.name,
            pos_xy=(float(pos[0]), float(pos[1])),
            heading=float(heading),
            next_pos_xy=next_pos_xy,
            next_heading=float(next_heading),
            forward=forward,
            rear=rear,
            left=left,
            right=right,
        )

    def _pedestrian_hard_guard_horizon(self, vx: float, vy: float, wz: float) -> float:
        linear_speed = max(
            math.hypot(float(vx), float(vy)),
            math.hypot(float(self._measured_vx), float(self._measured_vy)),
        )
        angular_speed = max(abs(float(wz)), abs(float(self._measured_wz)))
        linear_time = max(
            linear_speed / max(float(self.config.max_linear_decel), 1e-3),
            linear_speed / max(float(self.config.max_lateral_decel), 1e-3),
        )
        angular_time = angular_speed / max(float(self.config.max_angular_decel), 1e-3)
        return max(
            self._pedestrian_hard_guard_sample_dt_sec,
            self._pedestrian_hard_guard_latency_sec + max(linear_time, angular_time),
        )

    def _active_pedestrian_guard_states(self, horizon_sec: float):
        try:
            from pedestrian.simulator.logic.people_manager import PeopleManager
        except Exception:
            return []
        states = []
        seen = set()
        for person in list(PeopleManager.get_people_manager().people.values()):
            identity = id(person)
            if identity in seen:
                continue
            seen.add(identity)
            if not bool(getattr(person, "_active", False)):
                continue
            if bool(getattr(person, "is_parked", False)):
                continue
            if not bool(getattr(person, "_pose_valid", False)):
                continue
            getter = getattr(person, "get_dynamic_guard_state", None)
            if not callable(getter):
                continue
            state = getter(float(horizon_sec))
            if state is not None:
                states.append(state)
        return states

    def _apply_pedestrian_hard_guard(
        self,
        vx: float,
        vy: float,
        wz: float,
    ) -> tuple[float, float, float]:
        if not self._pedestrian_hard_guard_enabled:
            return float(vx), float(vy), float(wz)
        horizon = self._pedestrian_hard_guard_horizon(vx, vy, wz)
        pedestrians = self._active_pedestrian_guard_states(horizon)
        if not pedestrians:
            return float(vx), float(vy), float(wz)
        robot_state = self.get_dynamic_guard_state(0.0)
        if robot_state is None:
            return float(vx), float(vy), float(wz)
        result = scale_robot_command_for_pedestrians(
            robot=robot_state,
            pedestrians=pedestrians,
            vx=float(vx),
            vy=float(vy),
            wz=float(wz),
            horizon_sec=horizon,
            sample_dt_sec=self._pedestrian_hard_guard_sample_dt_sec,
            margin_m=self._pedestrian_hard_guard_margin_m,
            overlap_escape_horizon_sec=self._pedestrian_hard_guard_escape_horizon_sec,
            overlap_deadband_m=self._pedestrian_hard_guard_overlap_deadband_m,
        )
        now = time.monotonic()
        self._last_guard_mode = f"pedestrian_{result.mode}"
        command_modified = any(
            abs(applied - requested) > 1e-6
            for applied, requested in (
                (float(result.vx), float(vx)),
                (float(result.vy), float(vy)),
                (float(result.wz), float(wz)),
            )
        )
        requested_motion = (
            abs(float(vx)) > 1e-6
            or abs(float(vy)) > 1e-6
            or abs(float(wz)) > 1e-6
        )
        if command_modified or (result.scale < 0.999 and requested_motion):
            if self._hard_guard_event_pub is not None and now - self._last_pedestrian_hard_guard_event >= 0.1:
                self._last_pedestrian_hard_guard_event = now
                event = {
                    "event": "hard_guard_intervention",
                    "robot": self.name,
                    "mode": result.mode,
                    "scale": float(result.scale),
                    "blocked_by": result.blocked_by,
                    "input": {"vx": float(vx), "vy": float(vy), "wz": float(wz)},
                    "applied": {
                        "vx": float(result.vx),
                        "vy": float(result.vy),
                        "wz": float(result.wz),
                    },
                    "horizon_sec": float(horizon),
                    "sim_time_sec": float(self._last_clock_time_float),
                }
                self._hard_guard_event_pub.publish(String(data=json.dumps(event, sort_keys=True)))
            if now - self._last_pedestrian_hard_guard_log >= 0.5:
                self._last_pedestrian_hard_guard_log = now
                self._log_warn(
                    f"[{self.name}] pedestrian hard guard {result.mode}: "
                    f"scale={result.scale:.3f}, blocked_by={result.blocked_by}, "
                    f"horizon={horizon:.2f}s"
                )
        return result.vx, result.vy, result.wz

    def set_pedestrian_hard_guard_enabled(self, enabled: bool) -> tuple[bool, str]:
        self._pedestrian_hard_guard_enabled = bool(enabled)
        state = "enabled" if self._pedestrian_hard_guard_enabled else "detect_only"
        self._last_guard_mode = f"pedestrian_guard_{state}"
        return True, f"pedestrian hard guard is {state}"

    def _publish_pedestrian_contacts(self) -> None:
        if self._pedestrian_contact_pub is None:
            return
        robot_state = self.get_dynamic_guard_state(0.0)
        if robot_state is None:
            return
        contacts = current_pedestrian_contacts(
            robot_state,
            self._active_pedestrian_guard_states(0.0),
        )
        current_contacts = set(contacts)
        for pedestrian_name, penetration in contacts.items():
            if pedestrian_name in self._active_pedestrian_contacts:
                continue
            event = {
                "event": "pedestrian_robot_contact",
                "robot": self.name,
                "pedestrian": pedestrian_name,
                "penetration_m": float(penetration),
                "geometry_source": "zero_margin_runtime_envelope",
                "hard_guard_enabled": bool(self._pedestrian_hard_guard_enabled),
                "sim_time_sec": float(self._last_clock_time_float),
            }
            self._pedestrian_contact_pub.publish(String(data=json.dumps(event, sort_keys=True)))
        self._active_pedestrian_contacts = current_contacts

    def _selected_cmd(self):
        with self._lock:
            if self._episode_control_hold:
                return 0.0, 0.0, 0.0, DIFFERENTIAL_DRIVE
            now = time.monotonic()
            if now - self._last_diff_cmd_time <= self.config.differential_timeout_sec:
                return self._diff_vx, 0.0, self._diff_wz, DIFFERENTIAL_DRIVE
            if now - self._last_cmd_time <= self.config.timeout_sec:
                return self._vx, self._vy, self._wz, MECANUM_DRIVE
            return 0.0, 0.0, 0.0, MECANUM_DRIVE

    def _current_cmd(self):
        vx, vy, wz, _drive_mode = self._selected_cmd()
        return vx, vy, wz

    def _wheel_speeds(self, vx: float, vy: float, wz: float, drive_mode: str = MECANUM_DRIVE):
        c = self.config
        half_width = (
            0.5 * c.differential_track_width
            if drive_mode == DIFFERENTIAL_DRIVE
            else c.half_width
        )
        return wheel_angular_speeds(
            vx=vx,
            vy=vy,
            wz=wz,
            drive_mode=drive_mode,
            half_length=c.half_length,
            half_width=half_width,
            wheel_radius=c.wheel_radius,
            wheel_signs=c.wheel_signs,
            max_wheel_speed=c.max_wheel_speed,
        )

    def _resolve_dof_index(self, joint_name: str, dof_names: Optional[List[str]] = None) -> Optional[int]:
        idx = None
        try:
            idx = self._articulation.get_dof_index(joint_name) if self._articulation is not None else None
        except Exception:
            idx = None
        if idx is None and dof_names:
            try:
                idx = dof_names.index(joint_name)
            except Exception:
                idx = None
        return None if idx is None else int(idx)

    def _read_joint_positions(self, joint_indices: Optional[List[int]]) -> Optional[np.ndarray]:
        if self._articulation is None or not joint_indices:
            return None
        idx_arr = np.array(joint_indices, dtype=np.int32)
        try:
            vals = self._articulation.get_joint_positions(joint_indices=idx_arr)
            return np.array(vals, dtype=np.float32)
        except Exception:
            pass
        try:
            vals = self._articulation.get_joint_positions()
            vals = np.array(vals, dtype=np.float32)
            return vals[idx_arr]
        except Exception:
            return None

    def _read_joint_velocities(self, joint_indices: Optional[List[int]]) -> Optional[np.ndarray]:
        if self._articulation is None or not joint_indices:
            return None
        idx_arr = np.array(joint_indices, dtype=np.int32)
        try:
            values = self._articulation.get_joint_velocities(joint_indices=idx_arr)
            return np.array(values, dtype=np.float32).reshape(-1)
        except Exception:
            pass
        try:
            values = np.array(self._articulation.get_joint_velocities(), dtype=np.float32)
            if values.ndim == 2 and values.shape[0] == 1:
                values = values[0]
            return values[idx_arr].reshape(-1)
        except Exception:
            return None

    def _sample_physx_diff_wheel_contacts(self) -> List[Dict]:
        """Use small overlap boxes to identify non-robot shapes touching each tire."""
        try:
            import carb
            import omni.physx

            stage = get_current_stage()
            if stage is None:
                return []
            roots = self._hold_scene_root_prims()
            robot_prefixes = tuple(str(root.GetPath()).rstrip("/") for root in roots)
            wheel_names = [name.replace("_joint", "_link") for name in self.config.wheel_joints]
            wheel_prims = {}
            for root in roots:
                for prim in Usd.PrimRange(root):
                    if prim.GetName() in wheel_names and prim.GetName() not in wheel_prims:
                        wheel_prims[prim.GetName()] = prim
            xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            scene_query = omni.physx.get_physx_scene_query_interface()
            samples = []
            for wheel_name in wheel_names:
                prim = wheel_prims.get(wheel_name)
                if prim is None:
                    samples.append({"wheel": wheel_name, "contact": None, "external_hits": []})
                    continue
                center_vec = xform_cache.GetLocalToWorldTransform(prim).ExtractTranslation()
                center = tuple(float(center_vec[index]) for index in range(3))
                hits = []

                def _on_hit(hit):
                    paths = []
                    for attribute in ("collision", "rigid_body"):
                        try:
                            value = str(getattr(hit, attribute))
                        except Exception:
                            continue
                        if value and value != "None":
                            paths.append(value)
                    hits.extend(paths)
                    return True

                scene_query.overlap_box(
                    carb.Float3(0.035, 0.070, 0.085),
                    carb.Float3(*center),
                    carb.Float4(0.0, 0.0, 0.0, 1.0),
                    _on_hit,
                    False,
                )
                external_hits = sorted(
                    {
                        path
                        for path in hits
                        if not any(
                            path == prefix or path.startswith(prefix + "/")
                            for prefix in robot_prefixes
                        )
                    }
                )
                samples.append(
                    {
                        "wheel": wheel_name,
                        "center": list(center),
                        "contact": bool(external_hits),
                        "external_hits": external_hits,
                    }
                )
            return samples
        except Exception as exc:
            return [{"wheel": "query", "contact": None, "external_hits": [], "error": str(exc)}]

    def _append_physx_diff_diagnostic(self, payload: Dict) -> None:
        if not self._diff_diagnostics_output:
            return
        try:
            path = Path(self._diff_diagnostics_output).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, sort_keys=True) + "\n")
        except Exception as exc:
            self._log_warn(f"[{self.name}] failed to append differential diagnostics: {exc}")

    def _ensure_diff_articulation_force_view(self, *, force: bool = False) -> bool:
        if not self.config.separated_tire_force_enabled:
            return False
        try:
            handle_valid = bool(
                self._articulation is not None
                and self._articulation.is_physics_handle_valid()
            )
        except Exception:
            handle_valid = False
        if not handle_valid and self._initialized:
            if not self._recreate_articulation_wrapper():
                return False
        try:
            physics_view = getattr(self._articulation, "_physics_view", None)
            if (
                not force
                and self._diff_base_link_index is not None
                and self._articulation is not None
                and self._articulation.is_physics_handle_valid()
                and physics_view is not None
            ):
                return True
        except Exception:
            self._diff_base_link_index = None

        now = time.monotonic()
        if not force and now - self._last_diff_force_view_attempt < 1.0:
            return False
        self._last_diff_force_view_attempt = now
        try:
            if self._articulation is None:
                raise RuntimeError("articulation is unavailable")
            self._articulation.initialize()
            if not self._articulation.is_physics_handle_valid():
                raise RuntimeError("articulation physics handle is not ready")
            physics_view = getattr(self._articulation, "_physics_view", None)
            if physics_view is None:
                raise RuntimeError("articulation tensor view is unavailable")
            base_link_index = self._articulation.get_link_index("base_link")
            if base_link_index is None:
                raise RuntimeError(
                    f"base_link is absent from articulation links {getattr(self._articulation, 'body_names', [])}"
                )
            if int(base_link_index) >= int(physics_view.max_links):
                raise RuntimeError("base_link index is outside the articulation tensor view")
            self._diff_base_link_index = int(base_link_index)
            self._warned_diff_force_view = False
            self._log_info(
                f"[{self.name}] separated tire-force model attached to articulation link "
                f"base_link[{self._diff_base_link_index}]: "
                f"longitudinal=(k={self.config.tire_longitudinal_stiffness:.1f}, "
                f"max={self.config.tire_max_longitudinal_force:.1f}N), "
                f"lateral=(k={self.config.tire_lateral_stiffness:.1f}, "
                f"max={self.config.tire_max_lateral_force:.1f}N)"
            )
            return True
        except Exception as exc:
            self._diff_base_link_index = None
            if not self._warned_diff_force_view:
                self._warned_diff_force_view = True
                self._log_warn(
                    f"[{self.name}] separated tire-force model is waiting for articulation physics: {exc}"
                )
            return False

    def _current_physx_diff_contacts(self, *, force_refresh: bool = False) -> List[Dict]:
        now = time.monotonic()
        if (
            force_refresh
            or not self._diff_contact_samples
            or now - self._last_diff_contact_sample_t >= self.config.tire_contact_refresh_sec
        ):
            self._diff_contact_samples = self._sample_physx_diff_wheel_contacts()
            self._last_diff_contact_sample_t = now
        return self._diff_contact_samples

    def _apply_separated_tire_forces(
        self,
        actual_wheel_velocities: Optional[np.ndarray],
        contacts: List[Dict],
    ) -> Optional[Dict]:
        if not self.config.separated_tire_force_enabled:
            return None
        if actual_wheel_velocities is None or np.asarray(actual_wheel_velocities).size != 4:
            return None
        if not self._ensure_diff_articulation_force_view():
            return None

        contact_mask = [sample.get("contact") is True for sample in contacts]
        wrench = separated_tire_wrench(
            joint_velocities=np.asarray(actual_wheel_velocities).reshape(-1),
            wheel_radius=self.config.wheel_radius,
            wheel_signs=self.config.wheel_signs,
            half_length=self.config.half_length,
            half_width=0.5 * self.config.differential_track_width,
            measured_vx=self._measured_vx,
            measured_vy=self._measured_vy,
            measured_wz=self._measured_wz,
            longitudinal_stiffness=self.config.tire_longitudinal_stiffness,
            lateral_stiffness=self.config.tire_lateral_stiffness,
            max_longitudinal_force=self.config.tire_max_longitudinal_force,
            max_lateral_force=self.config.tire_max_lateral_force,
            contact_mask=contact_mask,
        )
        try:
            physics_view = self._articulation._physics_view
            forces, torques, indices = articulation_link_wrench_arrays(
                articulation_count=int(physics_view.count),
                max_links=int(physics_view.max_links),
                link_index=int(self._diff_base_link_index),
                force=wrench["net_force_body_n"],
                torque=wrench["net_torque_body_nm"],
            )
            physics_view.apply_forces_and_torques_at_position(
                forces,
                torques,
                None,
                indices,
                False,
            )
            return wrench
        except Exception as exc:
            self._diff_base_link_index = None
            if not self._warned_diff_force_view:
                self._warned_diff_force_view = True
                self._log_warn(f"[{self.name}] failed to apply separated tire forces: {exc}")
            return None

    def _hold_scene_root_prims(self):
        stage = get_current_stage()
        if stage is None:
            return []
        roots = []
        seen = set()
        # Prefer wide asset roots first. Articulation roots can be narrow wrapper
        # prims that do not contain the imported joint/link prims.
        for path in (
            self.asset_root_path,
            self.prim_path,
            self.nav_base_path,
            self.articulation_path,
        ):
            if not path:
                continue
            path = str(path)
            if path in seen:
                continue
            seen.add(path)
            prim = stage.GetPrimAtPath(str(path)) if path else None
            if prim is not None and prim.IsValid():
                roots.append(prim)
        return roots

    def _set_drive_attrs(
        self,
        prim,
        drive_type: str,
        *,
        target_position=None,
        target_velocity=None,
        stiffness=None,
        damping=None,
        max_force=None,
    ) -> bool:
        if UsdPhysics is None:
            return False
        try:
            drive = UsdPhysics.DriveAPI.Get(prim, drive_type)
            if not drive:
                drive = UsdPhysics.DriveAPI.Apply(prim, drive_type)
            for getter, creator, value in (
                (drive.GetTargetPositionAttr, drive.CreateTargetPositionAttr, target_position),
                (drive.GetTargetVelocityAttr, drive.CreateTargetVelocityAttr, target_velocity),
                (drive.GetStiffnessAttr, drive.CreateStiffnessAttr, stiffness),
                (drive.GetDampingAttr, drive.CreateDampingAttr, damping),
                (drive.GetMaxForceAttr, drive.CreateMaxForceAttr, max_force),
            ):
                if value is None:
                    continue
                attr = getter()
                if not attr:
                    creator(float(value))
                else:
                    attr.Set(float(value))
            return True
        except Exception:
            return False

    def _configure_hold_joint_drives(self) -> bool:
        if UsdPhysics is None:
            return False
        roots = self._hold_scene_root_prims()
        if not roots:
            return False
        hold_targets = {
            str(name): float(pos)
            for name, pos in zip(self.config.arm_hold_joints, self.config.arm_hold_positions)
        }
        hold_targets.update(
            {
                str(name): float(pos)
                for name, pos in zip(self.config.gripper_hold_joints, self.config.gripper_hold_positions)
            }
        )
        if not hold_targets:
            return False

        updated = 0
        matched_paths = set()
        sample_joint_names = []
        for root in roots:
            for prim in Usd.PrimRange(root):
                joint_name = prim.GetName()
                drive_type = None
                if prim.IsA(UsdPhysics.RevoluteJoint):
                    drive_type = "angular"
                elif prim.IsA(UsdPhysics.PrismaticJoint):
                    drive_type = "linear"
                if drive_type is not None and len(sample_joint_names) < 12:
                    sample_joint_names.append(joint_name)
                if joint_name not in hold_targets:
                    continue
                path = str(prim.GetPath())
                if path in matched_paths:
                    continue
                target_position = hold_targets[joint_name]
                if drive_type == "angular":
                    target_position = math.degrees(target_position)
                elif drive_type != "linear":
                    continue
                if self._set_drive_attrs(
                    prim,
                    drive_type,
                    target_position=target_position,
                    target_velocity=0.0,
                    stiffness=self.config.hold_drive_stiffness,
                    damping=self.config.hold_drive_damping,
                    max_force=self.config.hold_drive_max_force,
                ):
                    matched_paths.add(path)
                    updated += 1
        if updated:
            new_paths = matched_paths - self._hold_drive_configured_paths
            self._hold_drive_configured_paths.update(matched_paths)
            if new_paths or not self._hold_drive_configured:
                self._log_info(
                    f"[{self.name}] configured hold joint drives: joints={updated}, roots={len(roots)}, "
                    f"stiffness={self.config.hold_drive_stiffness:g}, damping={self.config.hold_drive_damping:g}, "
                    f"max_force={self.config.hold_drive_max_force:g}"
                )
            return True
        if not self._warned_hold_drive_no_match:
            self._warned_hold_drive_no_match = True
            roots_text = ", ".join(str(root.GetPath()) for root in roots)
            sample_text = ", ".join(sample_joint_names[:12]) if sample_joint_names else "none"
            self._log_warn(
                f"[{self.name}] no USD joint drives matched arm/gripper hold joints; "
                f"roots=[{roots_text}], sample_joint_prims=[{sample_text}]"
            )
        return False

    def _configure_physx_diff_wheel_drives(self) -> bool:
        """Configure imported wheel joints as finite-force angular velocity drives."""
        if UsdPhysics is None:
            return False
        wheel_names = set(self.config.wheel_joints)
        updated_paths = set()
        for root in self._hold_scene_root_prims():
            for prim in Usd.PrimRange(root):
                if prim.GetName() not in wheel_names or not prim.IsA(UsdPhysics.RevoluteJoint):
                    continue
                if self._set_drive_attrs(
                    prim,
                    "angular",
                    target_velocity=0.0,
                    stiffness=0.0,
                    damping=self.config.wheel_drive_damping,
                    max_force=self.config.wheel_drive_max_force,
                ):
                    updated_paths.add(str(prim.GetPath()))
        runtime_configured = False
        if self._articulation is not None and self._joint_indices:
            indices = np.asarray(self._joint_indices, dtype=np.int32)
            kps = single_articulation_targets(np.zeros(len(indices), dtype=np.float32))
            kds = single_articulation_targets(
                np.full(len(indices), self.config.wheel_drive_damping, dtype=np.float32)
            )
            max_efforts = single_articulation_targets(
                np.full(len(indices), self.config.wheel_drive_max_force, dtype=np.float32)
            )
            try:
                self._articulation.set_gains(kps=kps, kds=kds, joint_indices=indices)
                self._articulation.set_max_efforts(values=max_efforts, joint_indices=indices)
                gains = self._articulation.get_gains(joint_indices=indices)
                efforts = self._articulation.get_max_efforts(joint_indices=indices)
                if gains is not None and efforts is not None:
                    read_kps, read_kds = gains
                    runtime_configured = (
                        np.allclose(np.asarray(read_kps), 0.0, atol=1.0e-4)
                        and np.allclose(
                            np.asarray(read_kds),
                            self.config.wheel_drive_damping,
                            rtol=1.0e-4,
                            atol=1.0e-4,
                        )
                        and np.allclose(
                            np.asarray(efforts),
                            self.config.wheel_drive_max_force,
                            rtol=1.0e-4,
                            atol=1.0e-4,
                        )
                    )
            except Exception as exc:
                self._log_warn(f"[{self.name}] failed to configure runtime wheel drive gains: {exc}")
        if updated_paths and runtime_configured:
            self._log_info(
                f"[{self.name}] configured {len(updated_paths)} PhysX differential wheel drives: "
                f"damping={self.config.wheel_drive_damping:.3f}, "
                f"max_force={self.config.wheel_drive_max_force:.3f}, runtime_readback=ok"
            )
        elif not updated_paths:
            self._log_warn(f"[{self.name}] no PhysX differential wheel drive prims were configured")
        else:
            self._log_warn(f"[{self.name}] PhysX wheel drive USD attributes exist but runtime readback failed")
        return len(updated_paths) == len(wheel_names) and runtime_configured

    def _configure_hold_link_gravity(self) -> bool:
        if UsdPhysics is None or not self.config.arm_hold_disable_gravity:
            return False
        roots = self._hold_scene_root_prims()
        if not roots:
            return False

        updated_paths = set()
        for root in roots:
            for prim in Usd.PrimRange(root):
                if prim.GetName() not in _MECANUM_HOLD_LINK_NAMES:
                    continue
                if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue
                try:
                    attr = prim.GetAttribute("physxRigidBody:disableGravity")
                    if not attr or not attr.IsValid():
                        attr = prim.CreateAttribute(
                            "physxRigidBody:disableGravity",
                            Sdf.ValueTypeNames.Bool,
                            custom=False,
                        )
                    attr.Set(True)
                    updated_paths.add(str(prim.GetPath()))
                except Exception:
                    continue
        if not updated_paths:
            return False
        new_paths = updated_paths - self._hold_gravity_configured_paths
        self._hold_gravity_configured_paths.update(updated_paths)
        if new_paths:
            self._log_info(
                f"[{self.name}] disabled gravity on navigation-held arm links: links={len(updated_paths)}, roots={len(roots)}"
            )
        return True

    def _ensure_hold_scene_constraints(self, *, force: bool = False):
        now = time.monotonic()
        if not force and now - self._last_hold_scene_constraint_time < 1.0:
            return
        self._last_hold_scene_constraint_time = now
        self._configure_hold_link_gravity()
        self._hold_drive_configured = self._configure_hold_joint_drives()

    def _apply_articulation_action(self, action: ArticulationAction, *, context: str) -> bool:
        if self._articulation is None:
            return False
        apply_action = getattr(self._articulation, "apply_action", None)
        if callable(apply_action):
            try:
                apply_action(action)
                return True
            except Exception:
                pass
        controller_getter = getattr(self._articulation, "get_articulation_controller", None)
        if callable(controller_getter):
            try:
                controller_getter().apply_action(action)
                return True
            except Exception as exc:
                self._log_warn(f"[{self.name}] failed to apply {context} targets: {exc}")
                return False
        if not self._warned_missing_articulation_controller:
            self._warned_missing_articulation_controller = True
            self._log_warn(
                f"[{self.name}] articulation action fallback unavailable; "
                "joint control will rely on direct joint APIs"
            )
        return False

    def _apply_joint_hold(self, joint_indices: Optional[List[int]], positions: Optional[np.ndarray], *, hard_sync: bool = True):
        if self._articulation is None or not joint_indices or positions is None:
            return
        indices = np.array(joint_indices, dtype=np.int32)
        targets = np.array(positions[: len(joint_indices)], dtype=np.float32)
        zeros = np.zeros_like(targets)
        position_applied = False
        velocity_applied = False
        try:
            self._articulation.set_joint_position_targets(
                positions=targets,
                joint_indices=indices,
            )
            position_applied = True
        except Exception:
            pass
        try:
            self._articulation.set_joint_velocity_targets(
                velocities=zeros,
                joint_indices=indices,
            )
            velocity_applied = True
        except Exception:
            pass
        if (position_applied or velocity_applied) and not hard_sync:
            return
        try:
            self._articulation.set_joint_positions(
                positions=targets,
                joint_indices=indices,
            )
            position_applied = True
        except Exception:
            pass
        try:
            self._articulation.set_joint_velocities(
                velocities=zeros,
                joint_indices=indices,
            )
            velocity_applied = True
        except Exception:
            pass
        if position_applied:
            return
        self._ensure_hold_scene_constraints()
        if self._hold_drive_configured:
            return
        try:
            action = ArticulationAction(
                joint_positions=targets,
                joint_velocities=zeros,
                joint_indices=indices,
            )
            if self._apply_articulation_action(action, context="joint hold"):
                return
        except Exception as exc:
            self._log_warn(f"[{self.name}] failed to apply joint hold targets: {exc}")
        if not velocity_applied and not self._warned_joint_hold_failure:
            self._warned_joint_hold_failure = True
            self._log_warn(f"[{self.name}] failed to apply joint hold targets")

    def _apply_joint_velocity_targets(self, speeds: np.ndarray):
        if self._articulation is None or self._joint_indices is None:
            return
        indices = np.array(self._joint_indices, dtype=np.int32)
        try:
            # Isaac 4.5's Articulation is a view API and requires (M, K), even
            # for one robot. A flat (K,) array can be accepted without raising
            # while failing to update the intended DOF targets.
            self._articulation.set_joint_velocity_targets(
                velocities=single_articulation_targets(speeds),
                joint_indices=indices,
            )
            return
        except Exception:
            pass
        try:
            # Compatibility with legacy single-articulation wrappers.
            self._articulation.set_joint_velocity_targets(
                velocities=np.asarray(speeds, dtype=np.float32),
                joint_indices=indices,
            )
            return
        except Exception:
            pass
        if self.config.mode == "physx_diff_contact":
            if not self._warned_wheel_target_failure:
                self._warned_wheel_target_failure = True
                self._log_warn(
                    f"[{self.name}] failed to apply PhysX wheel drive targets; "
                    "direct joint-state fallback is disabled in contact mode"
                )
            return
        try:
            # Fallback: directly set the joint velocity state. Less ideal than a
            # drive target, but useful across Isaac API variants.
            self._articulation.set_joint_velocities(
                velocities=speeds,
                joint_indices=indices,
            )
            return
        except Exception:
            pass
        try:
            action = ArticulationAction(
                joint_velocities=speeds,
                joint_indices=indices,
            )
            self._apply_articulation_action(action, context="wheel velocity")
        except Exception as exc:
            self._log_warn(f"[{self.name}] failed to apply wheel velocity targets: {exc}")

    def _apply_navigation_hold_targets(self):
        if self._motion_backend is not None and not self._motion_backend.applies_navigation_hold:
            return
        self._ensure_hold_scene_constraints()
        self._apply_joint_hold(
            self._arm_joint_indices,
            np.array(self.config.arm_hold_positions, dtype=np.float32),
            hard_sync=self.config.arm_hold_hard_sync,
        )
        self._apply_joint_hold(
            self._gripper_joint_indices,
            np.array(self.config.gripper_hold_positions, dtype=np.float32),
            hard_sync=True,
        )
        if self._roller_joint_indices:
            if self._roller_hold_positions is None or len(self._roller_hold_positions) != len(self._roller_joint_indices):
                self._roller_hold_positions = self._read_joint_positions(self._roller_joint_indices)
            if self._roller_hold_positions is not None:
                self._apply_joint_hold(
                    self._roller_joint_indices,
                    self._roller_hold_positions,
                    hard_sync=False,
                )


    def _publish_applied_cmd_vel(self):
        if self._applied_pub is None:
            return
        try:
            msg = Twist()
            msg.linear.x = float(self._applied_vx)
            msg.linear.y = float(self._applied_vy)
            msg.angular.z = float(self._applied_wz)
            self._applied_pub.publish(msg)
        except Exception as exc:
            self._log_warn(f"[{self.name}] failed to publish applied cmd_vel: {exc}")

    def _publish_static_sensor_tf_once(self, stamp):
        if self._static_tf_pub is None or self._static_tf_sent:
            return
        # Static sensor transforms must be valid for all simulation times.
        # Leaving header.stamp at the default zero value avoids mixing wall time
        # with Isaac /clock time and prevents RViz MessageFilter drops.
        static_stamp = None
        try:
            transforms = [
                _sensor_tf_from_stage_or_env(
                    robot_root_path=self.prim_path,
                    sensor_prim_name="front_laser",
                    prefix="FRONT",
                    default_x=0.42,
                    default_y=0.0,
                    default_z=0.20,
                    default_yaw=0.0,
                    base_frame=self.base_frame,
                    frame=self.front_laser_frame,
                    stamp=static_stamp,
                    logger=self.logger,
                ),
                _sensor_tf_from_stage_or_env(
                    robot_root_path=self.prim_path,
                    sensor_prim_name="rear_laser",
                    prefix="REAR",
                    default_x=-0.42,
                    default_y=0.0,
                    default_z=0.20,
                    default_yaw=math.pi,
                    base_frame=self.base_frame,
                    frame=self.rear_laser_frame,
                    stamp=static_stamp,
                    logger=self.logger,
                ),
            ]
            self._static_tf_pub.sendTransform(transforms)
            self._static_tf_sent = True
            self._log_info(
                f"[{self.name}] published static sensor TFs from USD prims when available: "
                f"{self.base_frame}->{self.front_laser_frame}, {self.base_frame}->{self.rear_laser_frame}"
            )
        except Exception as exc:
            self._log_warn(f"[{self.name}] failed to publish static lidar TF: {exc}")

    def _publish_actual_odom_tf(self):
        try:
            pos, quat = _get_usd_xform_pose(self.prim_path)
        except Exception:
            if self._kinematic_pos is None:
                return
            pos = self._kinematic_pos
            align = self._spawn_orientation_wxyz if self._spawn_orientation_wxyz is not None else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            quat = _quat_multiply_wxyz(_yaw_to_quat_wxyz(self._heading), align)
        state = self._actual_state_estimator.sample(pos, quat)
        self._measured_vx = state.vx
        self._measured_vy = state.vy
        self._measured_wz = state.wz

        if not self.publish_actual_odom_tf or self.node is None:
            return

        stamp = self._current_tf_stamp()
        self._publish_static_sensor_tf_once(None)
        if stamp is None:
            return
        try:
            if self._tf_pub is not None:
                self._tf_pub.sendTransform(_make_transform(self.odom_frame, self.base_frame, stamp, pos, quat))
            if self._odom_pub is not None:
                odom = Odometry()
                if stamp is not None:
                    odom.header.stamp = stamp
                odom.header.frame_id = self.odom_frame
                odom.child_frame_id = self.base_frame
                odom.pose.pose.position.x = float(pos[0])
                odom.pose.pose.position.y = float(pos[1])
                odom.pose.pose.position.z = float(pos[2])
                qx, qy, qz, qw = _quat_wxyz_to_xyzw_fields(quat)
                odom.pose.pose.orientation.x = qx
                odom.pose.pose.orientation.y = qy
                odom.pose.pose.orientation.z = qz
                odom.pose.pose.orientation.w = qw
                if self._motion_backend is not None and self._motion_backend.uses_measured_odom_twist:
                    odom.twist.twist.linear.x = state.vx
                    odom.twist.twist.linear.y = state.vy
                    odom.twist.twist.angular.z = state.wz
                else:
                    odom.twist.twist.linear.x = float(self._applied_vx)
                    odom.twist.twist.linear.y = float(self._applied_vy)
                    odom.twist.twist.angular.z = float(self._applied_wz)
                # Conservative covariance for simulated kinematic odometry.
                odom.pose.covariance[0] = 0.01
                odom.pose.covariance[7] = 0.01
                odom.pose.covariance[35] = 0.02
                odom.twist.covariance[0] = 0.02
                odom.twist.covariance[7] = 0.02
                odom.twist.covariance[35] = 0.05
                self._odom_pub.publish(odom)
        except Exception as exc:
            self._log_warn(f"[{self.name}] failed to publish actual odom/tf: {exc}")

    def _apply_velocity_smoothing(self, vx: float, vy: float, wz: float, dt: float):
        if not self.config.smoothing_enabled:
            self._smooth_vx, self._smooth_vy, self._smooth_wz = float(vx), float(vy), float(wz)
            return float(vx), float(vy), float(wz)
        c = self.config
        self._smooth_vx = _slew_axis(self._smooth_vx, vx, dt, c.max_linear_accel, c.max_linear_decel)
        self._smooth_vy = _slew_axis(self._smooth_vy, vy, dt, c.max_lateral_accel, c.max_lateral_decel)
        self._smooth_wz = _slew_axis(self._smooth_wz, wz, dt, c.max_angular_accel, c.max_angular_decel)
        return self._smooth_vx, self._smooth_vy, self._smooth_wz

    def _compute_physx_root_velocity_world(self, vx: float, vy: float, wz: float, pos, quat):
        c = self.config
        yaw = _quat_wxyz_to_yaw(quat)
        planar_quat = _yaw_to_quat_wxyz(yaw)
        lin = _quat_apply_wxyz(planar_quat, (vx, vy, 0.0))
        if self._physx_nominal_z is None:
            self._physx_nominal_z = float(pos[2])
        z_error = float(self._physx_nominal_z) - float(pos[2])
        lin[2] = _clamp(c.physx_height_kp * z_error, -c.physx_height_max_vel, c.physx_height_max_vel)

        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        body_up = _quat_apply_wxyz(quat, world_up)
        upright_axis = np.cross(body_up, world_up)
        ang = np.array([
            _clamp(c.physx_upright_kp * float(upright_axis[0]), -c.physx_upright_max_ang_vel, c.physx_upright_max_ang_vel),
            _clamp(c.physx_upright_kp * float(upright_axis[1]), -c.physx_upright_max_ang_vel, c.physx_upright_max_ang_vel),
            float(wz),
        ], dtype=np.float32)
        return lin.astype(np.float32), ang.astype(np.float32), yaw

    def _write_root_velocity_command(self, lin_vel_w, ang_vel_w) -> bool:
        lin_vel_w = np.array(lin_vel_w, dtype=np.float32)
        ang_vel_w = np.array(ang_vel_w, dtype=np.float32)
        if self._articulation is not None:
            # IsaacLab Articulation API path.  The uploaded reference project uses
            # write_root_velocity_to_sim([vx,vy,vz,wx,wy,wz]) every frame.
            try:
                if hasattr(self._articulation, "write_root_velocity_to_sim"):
                    try:
                        import torch  # type: ignore
                        root_vel = torch.tensor([[*lin_vel_w.tolist(), *ang_vel_w.tolist()]], dtype=torch.float32)
                    except Exception:
                        root_vel = np.array([[*lin_vel_w.tolist(), *ang_vel_w.tolist()]], dtype=np.float32)
                    self._articulation.write_root_velocity_to_sim(root_vel)
                    return True
            except Exception as exc:
                self._log_warn(f"[{self.name}] write_root_velocity_to_sim failed: {exc}")
            # Isaac Core-style APIs, if available.
            try:
                if hasattr(self._articulation, "set_linear_velocity"):
                    self._articulation.set_linear_velocity(lin_vel_w)
                    if hasattr(self._articulation, "set_angular_velocity"):
                        self._articulation.set_angular_velocity(ang_vel_w)
                    return True
            except Exception as exc:
                self._log_warn(f"[{self.name}] set_linear/angular_velocity failed: {exc}")
        try:
            if self._xform is not None and hasattr(self._xform, "set_linear_velocity"):
                self._xform.set_linear_velocity(lin_vel_w)
                if hasattr(self._xform, "set_angular_velocity"):
                    self._xform.set_angular_velocity(ang_vel_w)
                return True
        except Exception as exc:
            self._log_warn(f"[{self.name}] XFormPrim velocity API failed: {exc}")
        # USD Physics fallback.  This is less authoritative than a live PhysX API,
        # but can work for some rigid-body authored roots and keeps the prototype
        # self-contained in Isaac Sim without IsaacLab imports.
        if UsdPhysics is not None:
            try:
                stage = get_current_stage()
                prim = stage.GetPrimAtPath(self.prim_path) if stage is not None else None
                if prim is not None and prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    rb = UsdPhysics.RigidBodyAPI(prim)
                    rb.CreateVelocityAttr().Set(Gf.Vec3f(float(lin_vel_w[0]), float(lin_vel_w[1]), float(lin_vel_w[2])))
                    rb.CreateAngularVelocityAttr().Set(Gf.Vec3f(float(ang_vel_w[0]), float(ang_vel_w[1]), float(ang_vel_w[2])))
                    return True
            except Exception as exc:
                self._log_warn(f"[{self.name}] USD Physics velocity attr write failed: {exc}")
        return False

    def _apply_physx_root_velocity_base(self, vx: float, vy: float, wz: float, dt: float):
        try:
            pos, quat = _get_usd_xform_pose(self.prim_path)
        except Exception as exc:
            self._log_warn(f"[{self.name}] cannot read root pose for PhysX velocity mode: {exc}")
            return
        self._applied_vx, self._applied_vy, self._applied_wz = float(vx), float(vy), float(wz)
        self._last_guard_mode = "unguarded"
        self._publish_applied_cmd_vel()

        lin_w, ang_w, _yaw = self._compute_physx_root_velocity_world(vx, vy, wz, pos, quat)
        wrote = self._write_root_velocity_command(lin_w, ang_w)
        if not wrote:
            if not self._warned_physx_velocity_fallback:
                self._warned_physx_velocity_fallback = True
                self._log_warn(
                    f"[{self.name}] PhysX root velocity API unavailable for prim={self.prim_path}; "
                    f"fallback_kinematic={self.config.physx_fallback_kinematic}."
                )
            if self.config.physx_fallback_kinematic:
                # Emergency fallback keeps teleop usable, but the bridge log makes
                # it explicit that this is no longer a real PhysX root-velocity run.
                self._apply_kinematic_base(vx, vy, wz, dt)
                return
        # Odom is always read back from the actual stage pose, after PhysX or the
        # fallback has had a chance to update the root.
        self._publish_actual_odom_tf()

    def _apply_kinematic_base(self, vx: float, vy: float, wz: float, dt: float):
        # Direct USD kinematic control.  Do not instantiate XFormPrim or
        # Articulation here; the robot is intentionally non-physical in this
        # mode, so PhysX tensor APIs should never be involved.
        if self._kinematic_pos is None:
            try:
                pos, quat = _get_usd_xform_pose(self.prim_path)
                self._kinematic_pos = np.array(pos, dtype=np.float32)
                if self._spawn_orientation_wxyz is None:
                    self._spawn_orientation_wxyz = _quat_to_wxyz_np(quat)
            except Exception:
                return

        self._applied_vx, self._applied_vy, self._applied_wz = float(vx), float(vy), float(wz)
        self._last_guard_mode = "unguarded"
        self._publish_applied_cmd_vel()

        self._heading += wz * dt
        c = math.cos(self._heading)
        s = math.sin(self._heading)
        wx = c * vx - s * vy
        wy = s * vx + c * vy
        self._kinematic_pos = np.array([
            float(self._kinematic_pos[0]) + wx * dt,
            float(self._kinematic_pos[1]) + wy * dt,
            float(self._kinematic_pos[2]),
        ], dtype=np.float32)

        yaw_quat = _yaw_to_quat_wxyz(self._heading)
        align_quat = self._spawn_orientation_wxyz
        if align_quat is None:
            align_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        new_quat = _quat_multiply_wxyz(yaw_quat, align_quat)
        try:
            _set_usd_xform_pose(self.prim_path, self._kinematic_pos, new_quat)
        except Exception as exc:
            self._log_warn(f"[{self.name}] failed to apply direct USD kinematic pose: {exc}")
        self._publish_actual_odom_tf()

    def update(self):
        if not self._ensure_initialized():
            return
        if self._apply_pending_episode_reset():
            self._last_update_time = time.monotonic()
            return
        now = time.monotonic()
        if self._last_update_time is None:
            self._last_update_time = now
            return
        dt = max(0.0, min(0.05, now - self._last_update_time))
        self._last_update_time = now

        if self._maybe_update_settling_state(now, dt):
            return

        vx, vy, wz, drive_mode = self._selected_cmd()
        if drive_mode != self._active_drive_mode:
            self._active_drive_mode = drive_mode
            if drive_mode == DIFFERENTIAL_DRIVE:
                self._smooth_vy = 0.0
            self._log_info(f"[{self.name}] active drive input: {drive_mode}")
        vx, vy, wz = self._apply_velocity_smoothing(vx, vy, wz, dt)
        if drive_mode == DIFFERENTIAL_DRIVE:
            vy = 0.0
            self._smooth_vy = 0.0
        self._publish_pedestrian_contacts()
        vx, vy, wz = self._apply_pedestrian_hard_guard(vx, vy, wz)
        if self._pedestrian_hard_guard_enabled:
            self._smooth_vx = float(vx)
            self._smooth_vy = float(vy)
            self._smooth_wz = float(wz)
        wheel_speeds = self._wheel_speeds(vx, vy, wz, drive_mode)

        if self.config.mode != "joint":
            self._apply_navigation_hold_targets()

        if self.config.mode in ("joint", "hybrid") and self._articulation is not None:
            self._apply_joint_velocity_targets(wheel_speeds)
        elif self._motion_backend is not None:
            self._motion_backend.apply(self, vx, vy, wz, dt, drive_mode)
        elif self.config.mode == "physx_root_velocity" and self._articulation is not None:
            # Visual wheel rolling only.  Chassis traction comes from root velocity.
            self._apply_joint_velocity_targets(wheel_speeds)

        if self.config.mode in ("hybrid", "kinematic"):
            self._apply_kinematic_base(vx, vy, wz, dt)
        elif self.config.mode == "physx_root_velocity":
            self._apply_physx_root_velocity_base(vx, vy, wz, dt)


class MecanumTeleopManager:
    def __init__(self):
        self.node = None
        self.robots: Dict[str, MecanumRobot] = {}
        self.subscriptions = []
        self._reset_service = None
        self._control_hold_service = None
        self._pedestrian_hard_guard_service = None
        self.pedestrian_hard_guard_enabled: bool | None = None

    def register_node(self, node):
        self.node = node
        if self._reset_service is None:
            self._reset_service = node.create_service(
                ResetRobot,
                "/isaac/reset_mecanum_episode",
                self._reset_robot_callback,
            )
        if self._control_hold_service is None:
            self._control_hold_service = node.create_service(
                SetBool,
                "/isaac/set_mecanum_control_hold",
                self._control_hold_callback,
            )
        if self._pedestrian_hard_guard_service is None:
            self._pedestrian_hard_guard_service = node.create_service(
                SetBool,
                "/isaac/set_pedestrian_hard_guard",
                self._pedestrian_hard_guard_callback,
            )

    def _pedestrian_hard_guard_callback(self, request, response):
        if len(self.robots) != 1:
            response.success = False
            response.message = (
                "pedestrian hard guard control requires exactly one registered robot; "
                f"found {len(self.robots)}"
            )
            return response
        robot = next(iter(self.robots.values()))
        success, message = robot.set_pedestrian_hard_guard_enabled(bool(request.data))
        if success:
            self.pedestrian_hard_guard_enabled = bool(request.data)
        response.success = bool(success)
        response.message = str(message)
        return response

    def _control_hold_callback(self, request, response):
        if len(self.robots) != 1:
            response.success = False
            response.message = f"control hold requires exactly one registered robot; found {len(self.robots)}"
            return response
        robot = next(iter(self.robots.values()))
        success, message = robot.set_episode_control_hold(bool(request.data))
        response.success = bool(success)
        response.message = str(message)
        return response

    def _reset_robot_callback(self, request, response):
        name = str(getattr(request, "name", "") or "").strip()
        robot = self.robots.get(name)
        if robot is None and not name and len(self.robots) == 1:
            robot = next(iter(self.robots.values()))
        if robot is None:
            response.accepted = False
            response.message = f"unknown mecanum robot: {name or '<empty>'}"
            response.generation = 0
            return response
        pose = request.pose
        accepted, message, generation = robot.request_episode_reset(
            [pose.position.x, pose.position.y, pose.position.z],
            [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z],
        )
        response.accepted = bool(accepted)
        response.message = str(message)
        response.generation = int(generation)
        return response

    def add_robot(
        self,
        *,
        name: str,
        prim_path: str,
        articulation_path: Optional[str] = None,
        nav_base_path: Optional[str] = None,
        cmd_vel_topic: str,
        robot_model: str,
        odom_frame: str = "odom",
        base_frame: str = "base_link",
        asset_root_path: Optional[str] = None,
    ):

        if not self.node:
            raise RuntimeError("MecanumTeleopManager.register_node() must be called before add_robot().")
        config = parse_mecanum_config(robot_model)
        differential_topic = str(
            os.environ.get("ARENA_ISAAC_DIFF_CMD_VEL_TOPIC", "/cmd_vel_gamepad_diff")
        ).strip()
        robot = MecanumRobot(
            name=name,
            prim_path=prim_path,
            articulation_path=articulation_path,
            nav_base_path=nav_base_path,
            cmd_vel_topic=cmd_vel_topic,
            config=config,
            logger=self.node.get_logger(),
            node=self.node,
            odom_frame=odom_frame,
            base_frame=base_frame,
            asset_root_path=asset_root_path,
        )
        self.robots[name] = robot

        def _cb(msg, robot_name=name):
            self.robots[robot_name].set_cmd(msg)

        sub = self.node.create_subscription(Twist, cmd_vel_topic, _cb, 10)
        self.subscriptions.append(sub)

        if differential_topic and differential_topic != cmd_vel_topic:
            def _diff_cb(msg, robot_name=name):
                self.robots[robot_name].set_differential_cmd(msg)

            diff_sub = self.node.create_subscription(
                Twist,
                differential_topic,
                _diff_cb,
                10,
            )
            self.subscriptions.append(diff_sub)
        self.node.get_logger().info(
            f"Registered mecanum teleop robot {name}: prim={prim_path}, nav_base={nav_base_path or prim_path}, "
            f"articulation={articulation_path or prim_path}, topic={cmd_vel_topic}, "
            f"diff_topic={differential_topic or 'disabled'}, mode={config.mode}, "
            f"actual_odom_tf={robot.publish_actual_odom_tf}, odom_topic={robot.odom_topic}, applied_topic={robot.applied_cmd_vel_topic}"
        )

    def update(self):
        for robot in list(self.robots.values()):
            robot.update()


mecanum_teleop_manager = MecanumTeleopManager()


def _get_usd_xform_pose(prim_path: str):
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path) if stage is not None else None
    if prim is None or not prim.IsValid():
        raise RuntimeError(f"invalid prim path: {prim_path}")
    xformable = UsdGeom.Xformable(prim)
    pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for op in xformable.GetOrderedXformOps():
        try:
            v = op.Get()
        except Exception:
            continue
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate and v is not None:
            pos = np.array([float(v[0]), float(v[1]), float(v[2])], dtype=np.float32)
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient and v is not None:
            try:
                im = v.GetImaginary()
                quat = np.array([float(v.GetReal()), float(im[0]), float(im[1]), float(im[2])], dtype=np.float32)
            except Exception:
                pass
    return pos, quat


def _set_usd_xform_pose(prim_path: str, position, quat_wxyz):
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path) if stage is not None else None
    if prim is None or not prim.IsValid():
        raise RuntimeError(f"invalid prim path: {prim_path}")
    pos = np.array(position, dtype=np.float32)
    quat = np.array(quat_wxyz, dtype=np.float32)
    xformable = UsdGeom.Xformable(prim)
    ops = list(xformable.GetOrderedXformOps())
    translate_op = next((op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
    orient_op = next((op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeOrient), None)
    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    if orient_op is None:
        orient_op = xformable.AddOrientOp()
    xformable.SetXformOpOrder([translate_op, orient_op])
    translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    orient_op.Set(Gf.Quatd(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])))



def _quat_to_wxyz_np(quat):
    try:
        return np.array([float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])], dtype=np.float32)
    except Exception:
        if isinstance(quat, Gf.Quatd) or isinstance(quat, Gf.Quatf):
            im = quat.GetImaginary()
            return np.array([float(quat.GetReal()), float(im[0]), float(im[1]), float(im[2])], dtype=np.float32)
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _quat_multiply_wxyz(a, b):
    aw, ax, ay, az = [float(x) for x in a]
    bw, bx, by, bz = [float(x) for x in b]
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float32)


def _yaw_to_quat_wxyz(yaw: float):
    half = 0.5 * yaw
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


def _quat_wxyz_to_yaw(quat) -> float:
    # Accept numpy/list/Gf.Quat-like wxyz quaternions.
    try:
        w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    except Exception:
        if isinstance(quat, Gf.Quatd) or isinstance(quat, Gf.Quatf):
            w = float(quat.GetReal())
            im = quat.GetImaginary()
            x, y, z = float(im[0]), float(im[1]), float(im[2])
        else:
            return 0.0
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quat_wxyz_to_roll_pitch(quat) -> tuple[float, float]:
    try:
        w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    except Exception:
        if isinstance(quat, Gf.Quatd) or isinstance(quat, Gf.Quatf):
            w = float(quat.GetReal())
            im = quat.GetImaginary()
            x, y, z = float(im[0]), float(im[1]), float(im[2])
        else:
            return 0.0, 0.0
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)
    return roll, pitch
