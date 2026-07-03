"""ROS /cmd_vel driven mecanum controller for Isaac Sim 4.5.

This module is intentionally independent from Arena's nav2 controller configs.  It is
used for Milestone B: import a custom URDF robot and drive its mecanum wheel joints
from geometry_msgs/Twist.

Modes:
- joint:     send wheel joint velocity targets only. This is the most physically
             faithful path, but it requires valid wheel/roller collision geometry.
- physx_wheels:
             use motion-based root control for chassis locomotion while keeping
             wheel motion as visual rolling only. This is the recommended
             task-level-realistic mode when the URDF lacks usable wheel/roller
             collision geometry.
- hybrid:    send wheel joint velocity targets and kinematically advance the root
             prim. Useful when visual wheel animation is desired but the URDF lacks
             roller collision/contact geometry.
- kinematic: kinematically advance the root prim only.
- physx_root_velocity:
             command planar root velocity and let PhysX integrate/contact-resolve
             the chassis motion. Wheels are treated as visual rolling only.
"""
from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
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
try:
    from isaac_utils.collision_guard import KinematicCollisionGuard, config_from_env as collision_guard_config_from_env
except Exception:  # pragma: no cover - imported inside Isaac Sim normally
    KinematicCollisionGuard = None  # type: ignore
    collision_guard_config_from_env = None  # type: ignore
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, Usd, UsdGeom
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
    # URDF wheel joint origins are approximately +/-0.1575, +/-0.1725 m.
    half_length: float = 0.1575
    half_width: float = 0.1725
    # Estimated wheel radius. If the robot is too slow/fast, tune this first.
    wheel_radius: float = 0.060
    max_wheel_speed: float = 20.0
    max_linear_speed: float = 0.8
    max_lateral_speed: float = 0.5
    max_angular_speed: float = 1.2
    # Joint sign convention may need tuning after first physical test.
    # Default is chosen for a common X mecanum layout.
    wheel_signs: List[float] = field(default_factory=lambda: [1.0, -1.0, 1.0, -1.0])
    mode: str = "joint"  # joint | physx_wheels | hybrid | kinematic | physx_root_velocity
    timeout_sec: float = 0.5
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
    full_asset_handoff_enabled: bool = False
    full_asset_handoff_settling_sec: float = 0.6
    full_asset_handoff_min_stable_sec: float = 0.2
    full_asset_handoff_timeout_sec: float = 2.0


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
      - mecanum730_xms5_lidar_physx_wheels
      - mecanum730_xms5_physx
    """
    config = MecanumConfig()
    model = (robot_model or "").lower()
    if "physx_wheels" in model:
        config.mode = "physx_wheels"
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
    config.spawn_settling_sec = max(0.0, _env_float("ARENA_ISAAC_SPAWN_SETTLING_SEC", config.spawn_settling_sec))
    config.spawn_settling_min_stable_sec = max(0.0, _env_float("ARENA_ISAAC_SPAWN_SETTLING_MIN_STABLE_SEC", config.spawn_settling_min_stable_sec))
    config.spawn_settling_timeout_sec = max(config.spawn_settling_sec, _env_float("ARENA_ISAAC_SPAWN_SETTLING_TIMEOUT_SEC", config.spawn_settling_timeout_sec))
    config.spawn_settling_max_roll_pitch_deg = max(0.0, _env_float("ARENA_ISAAC_SPAWN_SETTLING_MAX_ROLL_PITCH_DEG", config.spawn_settling_max_roll_pitch_deg))
    config.spawn_settling_max_lin_speed = max(0.0, _env_float("ARENA_ISAAC_SPAWN_SETTLING_MAX_LIN_SPEED", config.spawn_settling_max_lin_speed))
    config.spawn_settling_max_ang_speed = max(0.0, _env_float("ARENA_ISAAC_SPAWN_SETTLING_MAX_ANG_SPEED", config.spawn_settling_max_ang_speed))
    config.spawn_settling_max_z_speed = max(0.0, _env_float("ARENA_ISAAC_SPAWN_SETTLING_MAX_Z_SPEED", config.spawn_settling_max_z_speed))
    config.full_asset_handoff_enabled = _env_bool("ARENA_ISAAC_FULL_ASSET_HANDOFF_ENABLED", config.full_asset_handoff_enabled)
    config.full_asset_handoff_settling_sec = max(0.0, _env_float("ARENA_ISAAC_FULL_ASSET_HANDOFF_SETTLING_SEC", config.full_asset_handoff_settling_sec))
    config.full_asset_handoff_min_stable_sec = max(0.0, _env_float("ARENA_ISAAC_FULL_ASSET_HANDOFF_MIN_STABLE_SEC", config.full_asset_handoff_min_stable_sec))
    config.full_asset_handoff_timeout_sec = max(config.full_asset_handoff_settling_sec, _env_float("ARENA_ISAAC_FULL_ASSET_HANDOFF_TIMEOUT_SEC", config.full_asset_handoff_timeout_sec))
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
            base_inv = base_world.GetInverse()
            sensor_local = base_inv * sensor_world
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
        handoff_asset_root_path: Optional[str] = None,
        handoff_prim_path: Optional[str] = None,
        handoff_articulation_path: Optional[str] = None,
        handoff_nav_base_path: Optional[str] = None,
    ):

        self.name = name
        self.prim_path = prim_path
        self.articulation_path = articulation_path or prim_path
        self.nav_base_path = nav_base_path or prim_path
        self.asset_root_path = asset_root_path or prim_path
        self.handoff_asset_root_path = handoff_asset_root_path
        self.handoff_prim_path = handoff_prim_path
        self.handoff_articulation_path = handoff_articulation_path or handoff_prim_path
        self.handoff_nav_base_path = handoff_nav_base_path or handoff_prim_path
        self.cmd_vel_topic = cmd_vel_topic
        self.config = config
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
        self._applied_vx = 0.0
        self._applied_vy = 0.0
        self._applied_wz = 0.0
        self._last_guard_mode = "init"
        self._articulation = None
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
        self._collision_guard = None
        self._smooth_vx = 0.0
        self._smooth_vy = 0.0
        self._smooth_wz = 0.0
        self._physx_nominal_z = None
        self._warned_physx_velocity_fallback = False
        self._arm_joint_indices: Optional[List[int]] = None
        self._last_cmd_log_t = 0.0
        self._motion_nominal_z = None
        self._warned_motion_root_fallback = False
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
        self._handoff_completed = False

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
        if self._settling_state != "ready":
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
                self._maybe_create_collision_guard()
                self._initialized = True
                self._initialize_settling_state()
                guard_state = "enabled" if self._collision_guard is not None else "disabled"
                self._log_info(
                    f"[{self.name}] mecanum controller initialized: mode={self.config.mode}, "
                    f"prim={self.prim_path}, direct USD kinematic root, collision_guard={guard_state}, "
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
            except Exception as exc:
                if self.config.mode == "hybrid":
                    self._log_warn(f"[{self.name}] articulation wheel control unavailable; hybrid will run as kinematic root control: {exc}")
                    self._articulation = None
                    self._joint_indices = []
                else:
                    self._log_warn(f"[{self.name}] mecanum controller init failed: {exc}")
                    return False

            self._maybe_create_collision_guard()
            self._initialized = True
            self._initialize_settling_state()
            guard_state = "enabled" if self._collision_guard is not None else "disabled"
            self._log_info(
                f"[{self.name}] mecanum controller initialized: mode={self.config.mode}, "
                f"prim={self.prim_path}, articulation={self.articulation_path}, joints={self.config.wheel_joints}, indices={self._joint_indices}, "
                f"arm_hold={self.config.arm_hold_positions}, gripper_hold={self.config.gripper_hold_positions}, "
                f"roller_freeze_count={0 if self._roller_joint_indices is None else len(self._roller_joint_indices)}, "
                f"collision_guard={guard_state}, smoothing={'enabled' if self.config.smoothing_enabled else 'disabled'}"
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
        self._motion_nominal_z = None
        self._warned_physx_velocity_fallback = False
        self._warned_motion_root_fallback = False
        msg = f"[{self.name}] spawn settling complete: {reason}; controller entering ready state"
        if warn:
            self._log_warn(msg)
        else:
            self._log_info(msg)

    def _perform_full_asset_handoff(self) -> bool:
        if self._handoff_completed or not self.config.full_asset_handoff_enabled:
            return False
        if not self.handoff_asset_root_path or not self.handoff_prim_path:
            return False
        stage = get_current_stage()
        if stage is None:
            return False
        nav_root = stage.GetPrimAtPath(self.asset_root_path)
        full_root = stage.GetPrimAtPath(self.handoff_asset_root_path)
        if full_root is None or not full_root.IsValid():
            self._log_warn(f"[{self.name}] full-asset handoff requested but target root is invalid: {self.handoff_asset_root_path}")
            return False
        try:
            pos, quat = _get_usd_xform_pose(self.prim_path)
        except Exception as exc:
            self._log_warn(f"[{self.name}] failed to capture nav pose before handoff: {exc}")
            return False
        old_asset_root = self.asset_root_path
        try:
            full_root.SetActive(True)
            _set_usd_xform_pose(self.handoff_prim_path, pos, quat)
            if nav_root is not None and nav_root.IsValid():
                nav_root.SetActive(False)
        except Exception as exc:
            self._log_warn(f"[{self.name}] failed during full-asset handoff activation: {exc}")
            return False

        self.asset_root_path = self.handoff_asset_root_path
        self.prim_path = self.handoff_prim_path
        self.articulation_path = self.handoff_articulation_path or self.handoff_prim_path
        self.nav_base_path = self.handoff_nav_base_path or self.handoff_prim_path
        self._articulation = None
        self._xform = None
        self._joint_indices = None
        self._arm_joint_indices = None
        self._gripper_joint_indices = None
        self._roller_joint_indices = None
        self._roller_hold_positions = None
        self._collision_guard = None
        self._initialized = False
        self._static_tf_sent = False
        self._handoff_completed = True
        self._smooth_vx = self._smooth_vy = self._smooth_wz = 0.0
        self._applied_vx = self._applied_vy = self._applied_wz = 0.0
        self._restart_settling_state(
            warmup_sec=self.config.full_asset_handoff_settling_sec,
            min_stable_sec=self.config.full_asset_handoff_min_stable_sec,
            timeout_sec=self.config.full_asset_handoff_timeout_sec,
        )
        self._log_info(
            f"[{self.name}] full-asset handoff complete: nav_root={old_asset_root} "
            f"-> full_root={self.handoff_asset_root_path}; re-entering settling"
        )
        return True

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
                if self._perform_full_asset_handoff():
                    return True
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


    def _maybe_create_collision_guard(self):
        if self._collision_guard is not None:
            return
        enabled = str(os.environ.get("ARENA_ISAAC_ENABLE_KINEMATIC_COLLISION_GUARD", "true")).strip().lower() in {"1", "true", "yes", "on"}
        if not enabled:
            self._log_info(f"[{self.name}] collision_guard disabled by ARENA_ISAAC_ENABLE_KINEMATIC_COLLISION_GUARD")
            return
        if KinematicCollisionGuard is None or collision_guard_config_from_env is None:
            self._log_warn(f"[{self.name}] collision guard requested but isaac_utils.collision_guard is unavailable")
            return
        try:
            cfg = collision_guard_config_from_env()
            if not cfg.enabled:
                self._log_info(f"[{self.name}] collision_guard config disabled")
                return
            self._collision_guard = KinematicCollisionGuard(robot_name=self.name, logger=self.logger, config=cfg)
            self._collision_guard.refresh(force=True)
            backend = os.environ.get("ARENA_ISAAC_COLLISION_GUARD_BACKEND", "proxy")
            self._log_info(f"[{self.name}] collision_guard initialized: backend={backend}, length={cfg.length:.3f}, width={cfg.width:.3f}, margin={cfg.margin:.3f}")
        except Exception as exc:
            self._collision_guard = None
            self._log_warn(f"[{self.name}] failed to initialize collision guard: {exc}")

    def _current_cmd(self):
        with self._lock:
            if time.monotonic() - self._last_cmd_time > self.config.timeout_sec:
                return 0.0, 0.0, 0.0
            return self._vx, self._vy, self._wz

    def _wheel_speeds(self, vx: float, vy: float, wz: float):
        c = self.config
        k = c.half_length + c.half_width
        r = c.wheel_radius
        # Robot frame: x forward, y left, z yaw CCW.
        omega = np.array([
            (vx - vy - k * wz) / r,  # front-left
            (vx + vy + k * wz) / r,  # front-right
            (vx + vy - k * wz) / r,  # rear-left
            (vx - vy + k * wz) / r,  # rear-right
        ], dtype=np.float32)
        omega *= np.array(c.wheel_signs, dtype=np.float32)
        omega = np.clip(omega, -c.max_wheel_speed, c.max_wheel_speed)
        return omega

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

    def _apply_joint_hold(self, joint_indices: Optional[List[int]], positions: Optional[np.ndarray], *, hard_sync: bool = True):
        if self._articulation is None or not joint_indices or positions is None:
            return
        indices = np.array(joint_indices, dtype=np.int32)
        targets = np.array(positions[: len(joint_indices)], dtype=np.float32)
        zeros = np.zeros_like(targets)
        target_applied = False
        try:
            self._articulation.set_joint_position_targets(
                positions=targets,
                joint_indices=indices,
            )
            target_applied = True
        except Exception:
            pass
        try:
            self._articulation.set_joint_velocity_targets(
                velocities=zeros,
                joint_indices=indices,
            )
            target_applied = True
        except Exception:
            pass
        if target_applied and not hard_sync:
            return
        try:
            self._articulation.set_joint_positions(
                positions=targets,
                joint_indices=indices,
            )
        except Exception:
            pass
        try:
            self._articulation.set_joint_velocities(
                velocities=zeros,
                joint_indices=indices,
            )
            return
        except Exception:
            pass
        try:
            action = ArticulationAction(
                joint_positions=targets,
                joint_velocities=zeros,
                joint_indices=indices,
            )
            self._articulation.get_articulation_controller().apply_action(action)
        except Exception as exc:
            self._log_warn(f"[{self.name}] failed to apply joint hold targets: {exc}")

    def _apply_joint_velocity_targets(self, speeds: np.ndarray):
        if self._articulation is None or self._joint_indices is None:
            return
        try:
            # Preferred modern API.
            self._articulation.set_joint_velocity_targets(
                velocities=speeds,
                joint_indices=np.array(self._joint_indices, dtype=np.int32),
            )
            return
        except Exception:
            pass
        try:
            # Fallback: directly set the joint velocity state. Less ideal than a
            # drive target, but useful across Isaac API variants.
            self._articulation.set_joint_velocities(
                velocities=speeds,
                joint_indices=np.array(self._joint_indices, dtype=np.int32),
            )
            return
        except Exception:
            pass
        try:
            action = ArticulationAction(
                joint_velocities=speeds,
                joint_indices=np.array(self._joint_indices, dtype=np.int32),
            )
            self._articulation.get_articulation_controller().apply_action(action)
        except Exception as exc:
            self._log_warn(f"[{self.name}] failed to apply wheel velocity targets: {exc}")

    def _apply_navigation_hold_targets(self):
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
        if not self.publish_actual_odom_tf or self.node is None:
            return
        try:
            pos, quat = _get_usd_xform_pose(self.prim_path)
        except Exception:
            if self._kinematic_pos is None:
                return
            pos = self._kinematic_pos
            align = self._spawn_orientation_wxyz if self._spawn_orientation_wxyz is not None else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            quat = _quat_multiply_wxyz(_yaw_to_quat_wxyz(self._heading), align)
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

    def _filter_with_collision_guard(self, pos_xyz, heading: float, vx: float, vy: float, wz: float, dt: float, label: str):
        guard_mode = "unguarded"
        if self._collision_guard is not None and dt > 0.0:
            try:
                vx, vy, wz, guard_mode = self._collision_guard.filter_motion(
                    pos_xyz=pos_xyz,
                    heading=heading,
                    vx=vx,
                    vy=vy,
                    wz=wz,
                    dt=dt,
                )
            except Exception as exc:
                self._log_warn(f"[{self.name}] collision guard failed; allowing {label} step: {exc}")
                guard_mode = "guard_error_allow"
        return float(vx), float(vy), float(wz), str(guard_mode)

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
        heading = _quat_wxyz_to_yaw(quat)
        vx, vy, wz, guard_mode = self._filter_with_collision_guard(pos, heading, vx, vy, wz, dt, "physx_root_velocity")
        self._applied_vx, self._applied_vy, self._applied_wz = float(vx), float(vy), float(wz)
        self._last_guard_mode = guard_mode
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

    def _apply_motion_based_root(self, pos, quat, vx: float, vy: float, wz: float, dt: float) -> bool:
        if dt <= 0.0:
            return False
        if self._motion_nominal_z is None:
            self._motion_nominal_z = float(pos[2])

        yaw = _quat_wxyz_to_yaw(quat)
        planar_quat = _yaw_to_quat_wxyz(yaw)
        lin_w = _quat_apply_wxyz(planar_quat, (vx, vy, 0.0))
        next_yaw = yaw + float(wz) * float(dt)
        target_pos = np.array([
            float(pos[0]) + float(lin_w[0]) * float(dt),
            float(pos[1]) + float(lin_w[1]) * float(dt),
            float(self._motion_nominal_z),
        ], dtype=np.float32)
        target_quat = _yaw_to_quat_wxyz(next_yaw)

        wrote_pose = False
        if self._articulation is not None and hasattr(self._articulation, "write_root_pose_to_sim"):
            try:
                try:
                    import torch  # type: ignore
                    root_pose = torch.tensor(
                        [[
                            float(target_pos[0]),
                            float(target_pos[1]),
                            float(target_pos[2]),
                            float(target_quat[0]),
                            float(target_quat[1]),
                            float(target_quat[2]),
                            float(target_quat[3]),
                        ]],
                        dtype=torch.float32,
                    )
                except Exception:
                    root_pose = np.array([[
                        float(target_pos[0]),
                        float(target_pos[1]),
                        float(target_pos[2]),
                        float(target_quat[0]),
                        float(target_quat[1]),
                        float(target_quat[2]),
                        float(target_quat[3]),
                    ]], dtype=np.float32)
                self._articulation.write_root_pose_to_sim(root_pose)
                wrote_pose = True
            except Exception as exc:
                self._log_warn(f"[{self.name}] write_root_pose_to_sim failed: {exc}")

        lin_cmd, ang_cmd, _ = self._compute_physx_root_velocity_world(vx, vy, wz, pos, quat)
        wrote_velocity = self._write_root_velocity_command(lin_cmd, ang_cmd)

        if not wrote_pose:
            try:
                _set_usd_xform_pose(self.prim_path, target_pos, target_quat)
                wrote_pose = True
            except Exception as exc:
                if not self._warned_motion_root_fallback:
                    self._warned_motion_root_fallback = True
                    self._log_warn(f"[{self.name}] motion-based root pose fallback failed: {exc}")

        return bool(wrote_pose or wrote_velocity)

    def _apply_physx_wheels_base(self, vx: float, vy: float, wz: float, dt: float):
        try:
            pos, quat = _get_usd_xform_pose(self.prim_path)
            heading = _quat_wxyz_to_yaw(quat)
        except Exception:
            pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            heading = 0.0

        vx, vy, wz, guard_mode = self._filter_with_collision_guard(pos, heading, vx, vy, wz, dt, "physx_wheels")
        self._applied_vx, self._applied_vy, self._applied_wz = float(vx), float(vy), float(wz)
        self._last_guard_mode = guard_mode
        self._publish_applied_cmd_vel()

        wheel_speeds = self._wheel_speeds(vx, vy, wz)
        self._apply_navigation_hold_targets()
        self._apply_joint_velocity_targets(wheel_speeds)
        wrote = self._apply_motion_based_root(pos, quat, vx, vy, wz, dt)
        if not wrote and self.config.physx_fallback_kinematic:
            self._apply_kinematic_base(vx, vy, wz, dt)
            return
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

        vx, vy, wz, guard_mode = self._filter_with_collision_guard(self._kinematic_pos, self._heading, vx, vy, wz, dt, "kinematic")
        self._applied_vx, self._applied_vy, self._applied_wz = float(vx), float(vy), float(wz)
        self._last_guard_mode = str(guard_mode)
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
        now = time.monotonic()
        if self._last_update_time is None:
            self._last_update_time = now
            return
        dt = max(0.0, min(0.05, now - self._last_update_time))
        self._last_update_time = now

        if self._maybe_update_settling_state(now, dt):
            return

        vx, vy, wz = self._current_cmd()
        vx, vy, wz = self._apply_velocity_smoothing(vx, vy, wz, dt)
        wheel_speeds = self._wheel_speeds(vx, vy, wz)

        if self.config.mode != "joint":
            self._apply_navigation_hold_targets()

        if self.config.mode in ("joint", "hybrid") and self._articulation is not None:
            self._apply_joint_velocity_targets(wheel_speeds)
        elif self.config.mode == "physx_wheels":
            self._apply_physx_wheels_base(vx, vy, wz, dt)
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

    def register_node(self, node):
        self.node = node

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
        handoff_asset_root_path: Optional[str] = None,
        handoff_prim_path: Optional[str] = None,
        handoff_articulation_path: Optional[str] = None,
        handoff_nav_base_path: Optional[str] = None,
    ):

        if not self.node:
            raise RuntimeError("MecanumTeleopManager.register_node() must be called before add_robot().")
        config = parse_mecanum_config(robot_model)
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
            handoff_asset_root_path=handoff_asset_root_path,
            handoff_prim_path=handoff_prim_path,
            handoff_articulation_path=handoff_articulation_path,
            handoff_nav_base_path=handoff_nav_base_path,
        )
        self.robots[name] = robot

        def _cb(msg, robot_name=name):
            self.robots[robot_name].set_cmd(msg)

        sub = self.node.create_subscription(Twist, cmd_vel_topic, _cb, 10)
        self.subscriptions.append(sub)
        self.node.get_logger().info(
            f"Registered mecanum teleop robot {name}: prim={prim_path}, nav_base={nav_base_path or prim_path}, "
            f"articulation={articulation_path or prim_path}, topic={cmd_vel_topic}, mode={config.mode}, "
            f"actual_odom_tf={robot.publish_actual_odom_tf}, odom_topic={robot.odom_topic}, applied_topic={robot.applied_cmd_vel_topic}, "
            f"handoff={'enabled' if robot.config.full_asset_handoff_enabled and handoff_prim_path else 'disabled'}"
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
