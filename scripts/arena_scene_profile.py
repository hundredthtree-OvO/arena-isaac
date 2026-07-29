#!/usr/bin/env python3
"""YAML-driven helper for the Arena Isaac scene workflow.

Examples, run from ~/resources/arena_ws/src/arena/arena-isaac:
  python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml bridge edit
  python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml spawn
  python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml export
  python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml scheme1 steps
  python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml scheme1 check
  python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml collision2d init
  python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml collision2d render
  python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml bridge guard2d
  python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml bridge voxel_build
  python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml voxel build
  python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml bridge voxel_guard
  python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml bridge rtx_scan
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_profile(path: str) -> Dict[str, Any]:
    profile_path = Path(path).expanduser().resolve()
    with open(profile_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"invalid profile: {path}")
    mode_files = data.get("mode_files", {}) or {}
    if not isinstance(mode_files, dict):
        raise RuntimeError(f"invalid mode_files mapping: {path}")
    modes = dict(data.get("modes", {}) or {})
    for mode, mode_file in mode_files.items():
        mode_path = Path(_expand(mode_file))
        if not mode_path.is_absolute():
            mode_path = profile_path.parent / mode_path
        with open(mode_path.resolve(), "r", encoding="utf-8") as f:
            mode_data = yaml.safe_load(f) or {}
        if not isinstance(mode_data, dict):
            raise RuntimeError(f"invalid mode profile '{mode}': {mode_path}")
        modes[str(mode)] = mode_data
    data["modes"] = modes
    return data


def as_bool(v: Any) -> str:
    return "true" if bool(v) else "false"


def cmd_str(cmd: List[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def run(cmd: List[str], *, env=None, dry_run: bool = False) -> int:
    print("\n$ " + cmd_str(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.call(cmd, env=env)


def _expand(path: Any) -> str:
    return str(path).replace("$HOME", os.path.expanduser("~"))


def _phase_mode(profile: Dict[str, Any], phase: str | None) -> str | None:
    if phase:
        phases = profile.get("phases", {}) or {}
        if phase not in phases:
            raise RuntimeError(f"unknown phase '{phase}'. Available phases: {', '.join(sorted(phases))}")
        mode = (phases[phase] or {}).get("mode")
        if mode:
            return str(mode)
    default_mode = profile.get("default_mode")
    return str(default_mode) if default_mode else None


def _phase_section(profile: Dict[str, Any], phase: str | None, section: str) -> Dict[str, Any]:
    merged = dict(profile.get(section, {}) or {})
    mode = _phase_mode(profile, phase)
    if mode:
        modes = profile.get("modes", {}) or {}
        if mode not in modes:
            raise RuntimeError(f"unknown mode '{mode}'. Available modes: {', '.join(sorted(modes))}")
        merged.update((modes[mode] or {}).get(section, {}) or {})
    if phase:
        phases = profile.get("phases", {}) or {}
        merged.update((phases[phase] or {}).get(section, {}) or {})
    return merged


def _phase_options(profile: Dict[str, Any], phase: str) -> Dict[str, Any]:
    mode = _phase_mode(profile, phase)
    options: Dict[str, Any] = {}
    if mode:
        options.update(((profile.get("modes", {}) or {}).get(mode, {}) or {}).get("runtime", {}) or {})
    options.update((profile.get("phases", {}) or {}).get(phase, {}) or {})
    return options


def bridge_cmd(profile: Dict[str, Any], phase: str) -> tuple[List[str], Dict[str, str]]:
    bridge = profile.get("bridge", {})
    proxy = _phase_section(profile, phase, "proxy")
    guard = _phase_section(profile, phase, "guard")
    collision2d = profile.get("collision2d", {})
    voxel = profile.get("voxel", {})
    pedestrians = _phase_section(profile, phase, "pedestrians")
    robot = _phase_section(profile, phase, "robot")
    robot_geometry = _phase_section(profile, phase, "robot_geometry")
    lidar = _phase_section(profile, phase, "lidar")
    odom_tf = _phase_section(profile, phase, "odom_tf")
    motion = _phase_section(profile, phase, "motion")
    physx = _phase_section(profile, phase, "physx_root_velocity")
    arm_hold = _phase_section(profile, phase, "arm_hold")
    gamepad = _phase_section(profile, phase, "gamepad")
    phases = profile.get("phases", {})
    phase_cfg = _phase_options(profile, phase)
    if not phase_cfg:
        raise RuntimeError(f"unknown phase '{phase}'. Available phases: {', '.join(sorted(phases.keys()))}")

    env = os.environ.copy()
    env["ISAAC_PATH"] = _expand(bridge.get("isaac_path", os.environ.get("ISAAC_PATH", "$HOME/resources/isaac-sim-4.5.0")))
    env["ARENA_ISAAC_COLLISION_GUARD_OVERLAP_POLICY"] = str(phase_cfg.get("overlap_policy", guard.get("overlap_policy", "escape")))
    env["ARENA_ISAAC_COLLISION_GUARD_BACKEND"] = str(phase_cfg.get("guard_backend", guard.get("backend", "proxy")))
    env["ARENA_ISAAC_SCENE_DOOR_COLLISION_POLICY"] = str(
        proxy.get("door_collision_policy", "disabled")
    )
    env["ARENA_ISAAC_SCENE_DOOR_COLLISION_TARGETS"] = ",".join(
        str(value) for value in proxy.get("door_collision_targets", [])
    )
    env["ARENA_ISAAC_SCENE_DOOR_COLLISION_MESHES"] = ",".join(
        str(value) for value in proxy.get("door_collision_meshes", [])
    )
    env["ARENA_ISAAC_SCENE_DOOR_LEAF_APPROXIMATION"] = str(
        proxy.get("door_leaf_approximation", "convexHull")
    )
    env["ARENA_ISAAC_SCENE_DOOR_FRAME_APPROXIMATION"] = str(
        proxy.get("door_frame_approximation", "sdf")
    )
    env["ARENA_ISAAC_SCENE_DOOR_FRAME_SDF_RESOLUTION"] = str(
        int(proxy.get("door_frame_sdf_resolution", 256))
    )
    if "escape_epsilon" in guard:
        env["ARENA_ISAAC_COLLISION_GUARD_ESCAPE_EPS"] = str(guard.get("escape_epsilon"))
    if "overlap_deadband" in guard:
        env["ARENA_ISAAC_COLLISION_GUARD_OVERLAP_DEADBAND"] = str(guard.get("overlap_deadband"))
    if "escape_tolerance" in guard:
        env["ARENA_ISAAC_COLLISION_GUARD_ESCAPE_TOLERANCE"] = str(guard.get("escape_tolerance"))
    if "last_free_max_age_sec" in guard:
        env["ARENA_ISAAC_COLLISION_GUARD_LAST_FREE_MAX_AGE_SEC"] = str(guard.get("last_free_max_age_sec"))
    for key, env_name in (
        ("footprint_forward", "ARENA_ISAAC_COLLISION_GUARD_FOOTPRINT_FORWARD"),
        ("footprint_rear", "ARENA_ISAAC_COLLISION_GUARD_FOOTPRINT_REAR"),
        ("footprint_left", "ARENA_ISAAC_COLLISION_GUARD_FOOTPRINT_LEFT"),
        ("footprint_right", "ARENA_ISAAC_COLLISION_GUARD_FOOTPRINT_RIGHT"),
    ):
        if guard.get(key) is not None:
            env[env_name] = str(guard.get(key))
    if collision2d.get("config_path"):
        env["ARENA_ISAAC_COLLISION2D_CONFIG"] = _expand(collision2d.get("config_path"))
    if collision2d.get("refresh_sec") is not None:
        env["ARENA_ISAAC_COLLISION2D_REFRESH_SEC"] = str(collision2d.get("refresh_sec"))
    if voxel.get("map_path"):
        env["ARENA_ISAAC_VOXEL_MAP_PATH"] = _expand(voxel.get("map_path"))
    if voxel.get("resolution") is not None:
        env["ARENA_ISAAC_VOXEL_RESOLUTION"] = str(voxel.get("resolution"))
    if voxel.get("sample_step") is not None:
        env["ARENA_ISAAC_VOXEL_SAMPLE_STEP"] = str(voxel.get("sample_step"))
    if voxel.get("z_min") is not None:
        env["ARENA_ISAAC_VOXEL_Z_MIN"] = str(voxel.get("z_min"))
        env["ARENA_ISAAC_VOXEL_GUARD_Z_MIN"] = str(voxel.get("z_min"))
    if voxel.get("z_max") is not None:
        env["ARENA_ISAAC_VOXEL_Z_MAX"] = str(voxel.get("z_max"))
        env["ARENA_ISAAC_VOXEL_GUARD_Z_MAX"] = str(voxel.get("z_max"))
    if voxel.get("refresh_sec") is not None:
        env["ARENA_ISAAC_VOXEL_GUARD_REFRESH_SEC"] = str(voxel.get("refresh_sec"))
    if voxel.get("skip_keywords"):
        env["ARENA_ISAAC_VOXEL_SKIP_KEYWORDS"] = ",".join(str(x) for x in voxel.get("skip_keywords", []))
    if voxel.get("include_keywords"):
        env["ARENA_ISAAC_VOXEL_INCLUDE_KEYWORDS"] = ",".join(str(x) for x in voxel.get("include_keywords", []))
    if pedestrians.get("stop_radius_m") is not None:
        env["ARENA_ISAAC_PEDESTRIAN_STOP_RADIUS_M"] = str(pedestrians.get("stop_radius_m"))
    if pedestrians.get("constrained_waypoint_radius_m") is not None:
        env["ARENA_ISAAC_CONSTRAINED_WAYPOINT_RADIUS_M"] = str(
            pedestrians.get("constrained_waypoint_radius_m")
        )
    if pedestrians.get("robot_interaction_policy") is not None:
        env["ARENA_ISAAC_PEDESTRIAN_ROBOT_POLICY"] = str(
            pedestrians.get("robot_interaction_policy")
        )
    if pedestrians.get("physics_proxy_enabled") is not None:
        env["ARENA_ISAAC_PEDESTRIAN_PHYSICS_PROXY_ENABLED"] = (
            "true" if bool(pedestrians.get("physics_proxy_enabled")) else "false"
        )
    if pedestrians.get("hard_guard_enabled") is not None:
        env["ARENA_ISAAC_PEDESTRIAN_HARD_GUARD_ENABLED"] = (
            "true" if bool(pedestrians.get("hard_guard_enabled")) else "false"
        )
    for key, env_name in (
        ("hard_guard_margin_m", "ARENA_ISAAC_PEDESTRIAN_HARD_GUARD_MARGIN_M"),
        ("hard_guard_control_latency_sec", "ARENA_ISAAC_PEDESTRIAN_HARD_GUARD_LATENCY_SEC"),
        ("hard_guard_sample_dt_sec", "ARENA_ISAAC_PEDESTRIAN_HARD_GUARD_SAMPLE_DT_SEC"),
        ("hard_guard_pedestrian_horizon_sec", "ARENA_ISAAC_PEDESTRIAN_HARD_GUARD_HORIZON_SEC"),
        ("hard_guard_release_margin_m", "ARENA_ISAAC_PEDESTRIAN_HARD_GUARD_RELEASE_MARGIN_M"),
        ("hard_guard_release_hold_sec", "ARENA_ISAAC_PEDESTRIAN_HARD_GUARD_RELEASE_HOLD_SEC"),
        ("hard_guard_escape_horizon_sec", "ARENA_ISAAC_PEDESTRIAN_HARD_GUARD_ESCAPE_HORIZON_SEC"),
        ("hard_guard_overlap_deadband_m", "ARENA_ISAAC_PEDESTRIAN_HARD_GUARD_OVERLAP_DEADBAND_M"),
    ):
        if pedestrians.get(key) is not None:
            env[env_name] = str(pedestrians.get(key))
        else:
            env.pop(env_name, None)
    for key, env_name in (
        (
            "external_motion_animation_full_speed_mps",
            "ARENA_ISAAC_EXTERNAL_MOTION_ANIMATION_FULL_SPEED_MPS",
        ),
        (
            "external_motion_animation_speed_exponent",
            "ARENA_ISAAC_EXTERNAL_MOTION_ANIMATION_SPEED_EXPONENT",
        ),
        ("external_turn_slow_angle_deg", "ARENA_ISAAC_EXTERNAL_TURN_SLOW_ANGLE_DEG"),
        (
            "external_turn_full_slow_angle_deg",
            "ARENA_ISAAC_EXTERNAL_TURN_FULL_SLOW_ANGLE_DEG",
        ),
        ("external_turn_min_speed_scale", "ARENA_ISAAC_EXTERNAL_TURN_MIN_SPEED_SCALE"),
        ("external_turn_yaw_rate_radps", "ARENA_ISAAC_EXTERNAL_MOTION_YAW_RATE_RADPS"),
    ):
        if pedestrians.get(key) is not None:
            env[env_name] = str(pedestrians.get(key))
        else:
            env.pop(env_name, None)


    # V16 robot geometry / lidar / actual odom-tf settings.  These are consumed
    # inside the Isaac bridge process when the robot is imported.
    if robot_geometry.get("auto_footprint", False):
        env["ARENA_ISAAC_ROBOT_AUTO_FOOTPRINT"] = "true"
    else:
        env["ARENA_ISAAC_ROBOT_AUTO_FOOTPRINT"] = "false"
    if robot_geometry.get("include_keywords") is not None:
        env["ARENA_ISAAC_ROBOT_GEOMETRY_INCLUDE_KEYWORDS"] = ",".join(str(x) for x in robot_geometry.get("include_keywords", []))
    if robot_geometry.get("exclude_keywords") is not None:
        env["ARENA_ISAAC_ROBOT_GEOMETRY_EXCLUDE_KEYWORDS"] = ",".join(str(x) for x in robot_geometry.get("exclude_keywords", []))
    if robot_geometry.get("z_min") is not None:
        env["ARENA_ISAAC_ROBOT_GEOMETRY_Z_MIN"] = str(robot_geometry.get("z_min"))
    if robot_geometry.get("z_max") is not None:
        env["ARENA_ISAAC_ROBOT_GEOMETRY_Z_MAX"] = str(robot_geometry.get("z_max"))

    env["ARENA_ISAAC_LIDAR_BACKEND"] = str(phase_cfg.get("lidar_backend", lidar.get("backend", "rtx")))
    lidar_backend = env["ARENA_ISAAC_LIDAR_BACKEND"].strip().lower()
    env["ARENA_ISAAC_LIDAR_MOUNT_SOURCE"] = str(lidar.get("mount_source", "manual"))
    env["ARENA_ISAAC_LIDAR_FRONT_TOPIC"] = str(lidar.get("front_topic", "/front_scan"))
    env["ARENA_ISAAC_LIDAR_REAR_TOPIC"] = str(lidar.get("rear_topic", "/rear_scan"))
    env["ARENA_ISAAC_LIDAR_DEDUPLICATE"] = (
        "true"
        if lidar_backend == "rtx" and bool(lidar.get("deduplicate", True))
        else "false"
    )
    if lidar.get("calibration_dir") is not None:
        env["ARENA_ISAAC_LIDAR_CALIBRATION_DIR"] = str(lidar.get("calibration_dir"))
    if lidar.get("range_offset_m") is not None:
        env["ARENA_ISAAC_LIDAR_RANGE_OFFSET_M"] = str(lidar.get("range_offset_m"))
    env["ARENA_ISAAC_LIDAR_PUBLISH_POINTS"] = "true" if bool(lidar.get("publish_points", False)) else "false"
    env["ARENA_ISAAC_LIDAR_VISUALIZE"] = "true" if bool(lidar.get("visualize", True)) else "false"
    synth_lidar = lidar.get("synthetic_2d", {}) or {}
    for key, env_name in (
        ("samples", "ARENA_ISAAC_LIDAR_SAMPLES"),
        ("fov_deg", "ARENA_ISAAC_LIDAR_FOV_DEG"),
        ("range_min", "ARENA_ISAAC_LIDAR_RANGE_MIN"),
        ("range_max", "ARENA_ISAAC_LIDAR_RANGE_MAX"),
        ("update_rate", "ARENA_ISAAC_LIDAR_UPDATE_RATE"),
    ):
        value = phase_cfg.get(key, synth_lidar.get(key))
        if value is not None:
            env[env_name] = str(value)

    env["ARENA_ISAAC_PUBLISH_ACTUAL_ODOM_TF"] = "true" if bool(odom_tf.get("publish_actual", True)) else "false"
    env["ARENA_ISAAC_PUBLISH_SENSOR_STATIC_TF"] = "true" if bool(odom_tf.get("publish_sensor_static_tf", True)) else "false"
    env["ARENA_ISAAC_ODOM_TOPIC"] = str(robot.get("odom_topic", "/odom"))
    env["ARENA_ISAAC_ODOM_FRAME"] = str(robot.get("odom_frame", "odom"))
    env["ARENA_ISAAC_BASE_FRAME"] = str(robot.get("base_frame", "base_link"))
    env["ARENA_ISAAC_CMD_VEL_APPLIED_TOPIC"] = str(robot.get("cmd_vel_applied_topic", "/cmd_vel_applied"))
    env["ARENA_ISAAC_DIFF_CMD_VEL_TOPIC"] = str(
        gamepad.get("cmd_vel_topic", "/cmd_vel_gamepad_diff")
    )
    env["ARENA_ISAAC_DIFF_CMD_TIMEOUT_SEC"] = str(
        gamepad.get("priority_timeout_sec", 0.30)
    )
    env["ARENA_ISAAC_DIFF_TRACK_WIDTH"] = str(gamepad.get("track_width_m", 0.345))
    contact_env_keys = {
        "wheel_drive_damping": "ARENA_ISAAC_DIFF_WHEEL_DRIVE_DAMPING",
        "wheel_drive_max_force": "ARENA_ISAAC_DIFF_WHEEL_DRIVE_MAX_FORCE",
        "wheel_static_friction": "ARENA_ISAAC_DIFF_WHEEL_STATIC_FRICTION",
        "wheel_dynamic_friction": "ARENA_ISAAC_DIFF_WHEEL_DYNAMIC_FRICTION",
        "wheel_friction_combine_mode": "ARENA_ISAAC_DIFF_WHEEL_FRICTION_COMBINE_MODE",
        "wheel_radius": "ARENA_ISAAC_DIFF_WHEEL_RADIUS",
        "max_wheel_speed": "ARENA_ISAAC_DIFF_MAX_WHEEL_SPEED",
        "wheel_signs": "ARENA_ISAAC_DIFF_WHEEL_SIGNS",
        "linear_gain": "ARENA_ISAAC_DIFF_LINEAR_GAIN",
        "angular_gain": "ARENA_ISAAC_DIFF_ANGULAR_GAIN",
        "tire_force_enabled": "ARENA_ISAAC_DIFF_TIRE_FORCE_ENABLED",
        "tire_longitudinal_stiffness": "ARENA_ISAAC_DIFF_TIRE_LONGITUDINAL_STIFFNESS",
        "tire_lateral_stiffness": "ARENA_ISAAC_DIFF_TIRE_LATERAL_STIFFNESS",
        "tire_max_longitudinal_force": "ARENA_ISAAC_DIFF_TIRE_MAX_LONGITUDINAL_FORCE",
        "tire_max_lateral_force": "ARENA_ISAAC_DIFF_TIRE_MAX_LATERAL_FORCE",
        "tire_contact_refresh_sec": "ARENA_ISAAC_DIFF_TIRE_CONTACT_REFRESH_SEC",
        "diagnostics_output": "ARENA_ISAAC_DIFF_DIAGNOSTICS_OUTPUT",
        "diagnostics_run_label": "ARENA_ISAAC_DIFF_DIAGNOSTICS_RUN_LABEL",
    }
    if "physx_diff_contact" in str(robot.get("model", "")).lower():
        for key, env_name in contact_env_keys.items():
            value = motion.get(key)
            if value is None:
                continue
            if key == "wheel_signs":
                value = ",".join(str(float(x)) for x in value)
            elif isinstance(value, bool):
                value = "true" if value else "false"
            env[env_name] = str(value)
    else:
        # A parent shell may retain values from an earlier contact run. Never
        # let those parameters alter the legacy physx_wheels controller.
        for env_name in contact_env_keys.values():
            env.pop(env_name, None)
    env["ARENA_ISAAC_FRONT_LASER_FRAME"] = str(lidar.get("front_frame", robot.get("front_laser_frame", "front_laser_link")))
    env["ARENA_ISAAC_REAR_LASER_FRAME"] = str(lidar.get("rear_frame", robot.get("rear_laser_frame", "rear_laser_link")))
    if robot.get("auto_ground_align") is not None:
        env["ARENA_ISAAC_ROBOT_AUTO_GROUND_ALIGN"] = "true" if bool(robot.get("auto_ground_align")) else "false"
    for key, env_name in (
        ("ground_z", "ARENA_ISAAC_ROBOT_GROUND_Z"),
        ("ground_clearance", "ARENA_ISAAC_ROBOT_GROUND_CLEARANCE"),
        ("ground_lift_offset", "ARENA_ISAAC_ROBOT_GROUND_LIFT_OFFSET"),
        ("spawn_settling_sec", "ARENA_ISAAC_SPAWN_SETTLING_SEC"),
        ("spawn_settling_min_stable_sec", "ARENA_ISAAC_SPAWN_SETTLING_MIN_STABLE_SEC"),
        ("spawn_settling_timeout_sec", "ARENA_ISAAC_SPAWN_SETTLING_TIMEOUT_SEC"),
        ("spawn_settling_max_roll_pitch_deg", "ARENA_ISAAC_SPAWN_SETTLING_MAX_ROLL_PITCH_DEG"),
        ("spawn_settling_max_lin_speed", "ARENA_ISAAC_SPAWN_SETTLING_MAX_LIN_SPEED"),
        ("spawn_settling_max_ang_speed", "ARENA_ISAAC_SPAWN_SETTLING_MAX_ANG_SPEED"),
        ("spawn_settling_max_z_speed", "ARENA_ISAAC_SPAWN_SETTLING_MAX_Z_SPEED"),
        ("full_asset_handoff_settling_sec", "ARENA_ISAAC_FULL_ASSET_HANDOFF_SETTLING_SEC"),
        ("full_asset_handoff_min_stable_sec", "ARENA_ISAAC_FULL_ASSET_HANDOFF_MIN_STABLE_SEC"),
        ("full_asset_handoff_timeout_sec", "ARENA_ISAAC_FULL_ASSET_HANDOFF_TIMEOUT_SEC"),
    ):
        if robot.get(key) is not None:
            env[env_name] = str(robot.get(key))
    if robot.get("full_asset_handoff") is not None:
        env["ARENA_ISAAC_FULL_ASSET_HANDOFF_ENABLED"] = "true" if bool(robot.get("full_asset_handoff")) else "false"
    if robot.get("ground_contact_include_keywords") is not None:
        env["ARENA_ISAAC_ROBOT_GROUND_INCLUDE_KEYWORDS"] = ",".join(str(x) for x in robot.get("ground_contact_include_keywords", []))
    if robot.get("ground_contact_exclude_keywords") is not None:
        env["ARENA_ISAAC_ROBOT_GROUND_EXCLUDE_KEYWORDS"] = ",".join(str(x) for x in robot.get("ground_contact_exclude_keywords", []))

    # v18 motion smoothing / PhysX-root-velocity controller settings.
    if motion.get("smoothing_enabled") is not None:
        env["ARENA_ISAAC_MOTION_SMOOTHING_ENABLED"] = "true" if bool(motion.get("smoothing_enabled")) else "false"
    for key, env_name in (
        ("max_linear_speed", "ARENA_ISAAC_MAX_LINEAR_SPEED"),
        ("max_lateral_speed", "ARENA_ISAAC_MAX_LATERAL_SPEED"),
        ("max_angular_speed", "ARENA_ISAAC_MAX_ANGULAR_SPEED"),
        ("max_linear_accel", "ARENA_ISAAC_MOTION_MAX_LINEAR_ACCEL"),
        ("max_lateral_accel", "ARENA_ISAAC_MOTION_MAX_LATERAL_ACCEL"),
        ("max_angular_accel", "ARENA_ISAAC_MOTION_MAX_ANGULAR_ACCEL"),
        ("max_linear_decel", "ARENA_ISAAC_MOTION_MAX_LINEAR_DECEL"),
        ("max_lateral_decel", "ARENA_ISAAC_MOTION_MAX_LATERAL_DECEL"),
        ("max_angular_decel", "ARENA_ISAAC_MOTION_MAX_ANGULAR_DECEL"),
    ):
        if motion.get(key) is not None:
            env[env_name] = str(motion.get(key))
    for key, env_name in (
        ("height_kp", "ARENA_ISAAC_PHYSX_ROOT_HEIGHT_KP"),
        ("height_max_vel", "ARENA_ISAAC_PHYSX_ROOT_HEIGHT_MAX_VEL"),
        ("upright_kp", "ARENA_ISAAC_PHYSX_ROOT_UPRIGHT_KP"),
        ("upright_max_ang_vel", "ARENA_ISAAC_PHYSX_ROOT_UPRIGHT_MAX_ANG_VEL"),
    ):
        if physx.get(key) is not None:
            env[env_name] = str(physx.get(key))
    if physx.get("fallback_kinematic") is not None:
        env["ARENA_ISAAC_PHYSX_ROOT_FALLBACK_KINEMATIC"] = "true" if bool(physx.get("fallback_kinematic")) else "false"
    hold_positions = arm_hold.get("positions")
    if hold_positions is not None:
        env["ARENA_ISAAC_ARM_HOLD_POSITIONS"] = ",".join(str(float(x)) for x in hold_positions)
    gripper_hold_positions = arm_hold.get("gripper_positions")
    if gripper_hold_positions is not None:
        env["ARENA_ISAAC_GRIPPER_HOLD_POSITIONS"] = ",".join(str(float(x)) for x in gripper_hold_positions)
    if arm_hold.get("hard_sync") is not None:
        env["ARENA_ISAAC_ARM_HOLD_HARD_SYNC"] = "true" if bool(arm_hold.get("hard_sync")) else "false"
    if arm_hold.get("disable_gravity") is not None:
        env["ARENA_ISAAC_ARM_HOLD_DISABLE_GRAVITY"] = "true" if bool(arm_hold.get("disable_gravity")) else "false"
    for key, env_name in (
        ("drive_stiffness", "ARENA_ISAAC_HOLD_DRIVE_STIFFNESS"),
        ("drive_damping", "ARENA_ISAAC_HOLD_DRIVE_DAMPING"),
        ("drive_max_force", "ARENA_ISAAC_HOLD_DRIVE_MAX_FORCE"),
    ):
        if arm_hold.get(key) is not None:
            env[env_name] = str(arm_hold.get(key))

    enable_people_stack = phase_cfg.get("enable_people_stack", bridge.get("enable_people_stack", False))
    enable_material_stack = phase_cfg.get("enable_material_stack", bridge.get("enable_material_stack", False))
    enable_navmesh = phase_cfg.get("enable_navmesh", bridge.get("enable_navmesh", False))
    enable_character_services = phase_cfg.get(
        "enable_character_services",
        bridge.get("enable_character_services", False),
    )
    people_extension_mode = str(
        phase_cfg.get(
            "people_extension_mode",
            bridge.get("people_extension_mode", "minimal"),
        )
    )
    env["ARENA_ISAAC_ENABLE_CHARACTER_SERVICES"] = as_bool(enable_character_services)
    env["ARENA_ISAAC_PEOPLE_EXTENSION_MODE"] = people_extension_mode

    args = [
        "ros2", "launch", "ros2isaacsim", "run_isaacsim.launch.py",
        f"isaac_path:={env['ISAAC_PATH']}",
        f"log_level:={bridge.get('log_level', 'info')}",
        f"idle_fps:={bridge.get('idle_fps', 5)}",
        f"active_fps:={bridge.get('active_fps', 30)}",
        f"mem_log_sec:={bridge.get('mem_log_sec', 30)}",
        f"enable_people_stack:={as_bool(enable_people_stack)}",
        f"enable_character_services:={as_bool(enable_character_services)}",
        f"people_extension_mode:={people_extension_mode}",
        f"enable_material_stack:={as_bool(enable_material_stack)}",
        f"enable_navmesh:={as_bool(enable_navmesh)}",
        f"enable_scene_collision_repair:={as_bool(phase_cfg.get('enable_scene_collision_repair', proxy.get('enable_scene_collision_repair', True)))}",
        f"scene_collision_repair_mode:={proxy.get('repair_mode', 'safe')}",
        f"scene_collision_proxy_source:={phase_cfg.get('proxy_source', proxy.get('source', 'auto'))}",
        f"scene_collision_proxy_config:={_expand(proxy.get('config_path'))}",
        f"scene_collision_auto_write_config:={as_bool(phase_cfg.get('auto_write_config', proxy.get('auto_write_config', True)))}",
        f"scene_collision_proxy_visible:={as_bool(phase_cfg.get('proxy_visible', proxy.get('visible', True)))}",
        f"scene_collision_enable_wall_proxy:={as_bool(proxy.get('enable_wall_proxy', True))}",
        f"scene_collision_enable_door_proxy:={as_bool(proxy.get('enable_door_proxy', True))}",
        f"scene_collision_door_default_enabled:={as_bool(proxy.get('door_default_enabled', False))}",
        f"scene_collision_wall_default_enabled:={as_bool(proxy.get('wall_default_enabled', True))}",
        f"enable_kinematic_collision_guard:={as_bool(phase_cfg.get('enable_guard', False))}",
        f"collision_guard_length:={guard.get('length', 0.80)}",
        f"collision_guard_width:={guard.get('width', 0.60)}",
        f"collision_guard_margin:={guard.get('margin', 0.05)}",
        f"collision_guard_z_max:={guard.get('z_max', 1.00)}",
        f"collision_guard_log_sec:={guard.get('log_sec', 5.0)}",
        f"collision_guard_block_log_sec:={guard.get('block_log_sec', 2.0)}",
        f"scene_collision_proxy_export_path:={_expand(proxy.get('config_path'))}",
    ]
    return args, env


def spawn_cmd(profile: Dict[str, Any], phase: str | None = None) -> List[str]:
    s = profile.get("scene", {})
    r = _phase_section(profile, phase, "robot")
    cmd = [
        "ros2", "run", "ros2isaacsim", "spawn_v10_scene_lidar_validation",
        "--scene-mode", str(s.get("mode", "usd")),
        "--scene-usd", _expand(s.get("usd_path")),
        "--scene-name", str(s.get("name", "scene")),
        "--scene-x", str(s.get("x", 0.0)),
        "--scene-y", str(s.get("y", 0.0)),
        "--scene-z", str(s.get("z", 0.0)),
        "--scene-yaw", str(s.get("yaw", 0.0)),
        "--robot-name", str(r.get("name", "xms_mecanum")),
        "--robot-model", str(r.get("model", "mecanum730_xms5_lidar_physx_wheels")),
        "--cmd-vel-topic", str(r.get("cmd_vel_topic", "/cmd_vel")),
        "--x", str(r.get("x", 0.0)),
        "--y", str(r.get("y", 0.0)),
        "--z", str(r.get("z", 0.05)),
        "--yaw", str(r.get("yaw", 0.0)),
    ]
    if r.get("urdf_path") is not None:
        cmd.extend(["--urdf-path", _expand(r.get("urdf_path"))])
    return cmd


def export_cmds(profile: Dict[str, Any]) -> List[List[str]]:
    s = profile.get("scene", {})
    p = profile.get("proxy", {})
    root = s.get("root_path", f"/World/{s.get('name', 'scene')}")
    out = _expand(p.get("config_path"))
    return [
        ["ros2", "param", "set", "/isaac_controller", "collision_proxy_export_scene_root", str(root)],
        ["ros2", "param", "set", "/isaac_controller", "collision_proxy_export_path", str(out)],
        ["ros2", "service", "call", "/isaac/export_collision_proxies", "std_srvs/srv/Trigger", "{}"],
    ]


def collision2d_cmd(profile: Dict[str, Any], action: str) -> List[str]:
    here = Path(__file__).resolve().parent
    tool = here / "collision2d_tool.py"
    s = profile.get("scene", {})
    p = profile.get("proxy", {})
    c = profile.get("collision2d", {})
    config_path = _expand(c.get("config_path", "/home/stardust/resources/arena_ws/arena_assets/collision_configs/shenxinfu_841837.collision2d.yaml"))
    if action == "init":
        return [
            "python3", str(tool), "init-from-proxies",
            "--proxy-yaml", _expand(p.get("config_path")),
            "--out", config_path,
            "--scene", str(s.get("name", "scene")),
            "--root-path", str(s.get("root_path", f"/World/{s.get('name', 'scene')}")),
            "--resolution", str(c.get("resolution", 0.03)),
        ]
    if action == "render":
        return ["python3", str(tool), "render", "--config", config_path, "--out", _expand(c.get("debug_svg", config_path.replace(".yaml", ".svg")))]
    if action == "summary":
        return ["python3", str(tool), "summary", "--config", config_path]
    raise RuntimeError(f"unknown collision2d action: {action}")


def voxel_cmds(profile: Dict[str, Any], action: str) -> List[List[str]]:
    here = Path(__file__).resolve().parent
    tool = here / "voxel_map_tool.py"
    s = profile.get("scene", {})
    v = profile.get("voxel", {})
    map_path = _expand(v.get("map_path", "/home/stardust/resources/arena_ws/arena_assets/collision_configs/shenxinfu_841837.voxel.json.gz"))
    debug_svg = _expand(v.get("debug_svg", map_path.replace(".json.gz", ".svg").replace(".json", ".svg")))
    debug_pcd = _expand(v.get("debug_pcd", map_path.replace(".json.gz", ".pcd").replace(".json", ".pcd")))
    if action == "build":
        # Requires bridge running and scene already imported.
        skip = ",".join(str(x) for x in v.get("skip_keywords", []))
        include = ",".join(str(x) for x in v.get("include_keywords", []))
        return [[
            "ros2", "run", "ros2isaacsim", "export_voxel_map",
            "--scene-root", str(s.get("root_path", f"/World/{s.get('name', 'scene')}")),
            "--output", map_path,
            "--debug-svg", debug_svg,
            "--debug-pcd", debug_pcd,
            "--resolution", str(v.get("resolution", 0.05)),
            "--sample-step", str(v.get("sample_step", v.get("resolution", 0.05))),
            "--z-min", str(v.get("z_min", 0.05)),
            "--z-max", str(v.get("z_max", 1.20)),
            "--skip-keywords", skip,
            "--include-keywords", include,
            "--max-faces-per-mesh", str(v.get("max_faces_per_mesh", 100000)),
            "--max-samples-per-mesh", str(v.get("max_samples_per_mesh", 250000)),
            "--max-stored-voxels", str(v.get("max_stored_voxels", 300000)),
            "--max-debug-stage-points", str(v.get("max_debug_stage_points", 30000)),
            "--create-stage-debug-points", str(v.get("create_stage_debug_points", True)).lower(),
            "--stage-debug-path", str(v.get("stage_debug_path", "")),
            "--timeout", str(v.get("build_timeout", 900.0)),
        ]]
    if action == "summary":
        return [["python3", str(tool), "summary", "--map", map_path]]
    if action == "render":
        return [["python3", str(tool), "render", "--map", map_path, "--out", debug_svg]]
    if action == "pcd":
        return [["python3", str(tool), "pcd", "--map", map_path, "--out", debug_pcd]]
    raise RuntimeError(f"unknown voxel action: {action}")



def teleop_cmd(profile: Dict[str, Any]) -> List[str]:
    t = profile.get("teleop", {})
    r = profile.get("robot", {})
    return [
        "ros2", "run", "ros2isaacsim", "wasd_combo_teleop",
        "--topic", str(r.get("cmd_vel_topic", "/cmd_vel")),
        "--linear", str(t.get("linear", 0.20)),
        "--lateral", str(t.get("lateral", 0.20)),
        "--angular", str(t.get("angular", 0.50)),
        "--debug",
    ]


def gamepad_cmd(profile: Dict[str, Any], phase: str | None = None) -> List[str]:
    gamepad = _phase_section(profile, phase, "gamepad")
    return [
        "ros2", "launch", "ros2isaacsim", "gamepad_diff_teleop.launch.py",
        f"device_id:={gamepad.get('device_id', 0)}",
        f"joy_topic:={gamepad.get('joy_topic', '/joy')}",
        f"output_topic:={gamepad.get('cmd_vel_topic', '/cmd_vel_gamepad_diff')}",
        f"linear_axis:={gamepad.get('linear_axis', 1)}",
        f"angular_axis:={gamepad.get('angular_axis', 0)}",
        f"enable_button:={gamepad.get('enable_button', 4)}",
        f"linear_scale:={gamepad.get('linear_scale', 0.20)}",
        f"angular_scale:={gamepad.get('angular_scale', 0.50)}",
        f"deadzone:={gamepad.get('deadzone', 0.10)}",
        f"joy_timeout_sec:={gamepad.get('joy_timeout_sec', 0.50)}",
    ]


def synthetic_laser_cmd(profile: Dict[str, Any]) -> List[str]:
    lidar = profile.get("lidar", {}) or {}
    synth = lidar.get("synthetic_2d", {}) or {}
    robot = profile.get("robot", {}) or {}
    voxel = profile.get("voxel", {}) or {}
    params = {
        "use_sim_time": bool(robot.get("use_sim_time", True)),
        "map_yaml_path": _expand(synth.get("map_yaml_path", "")),
        "map_topic": synth.get("map_topic", "/map"),
        "auto_use_map_topic_frame": bool(synth.get("auto_use_map_topic_frame", True)),
        "map_occupied_threshold": synth.get("map_occupied_threshold", 50),
        "map_unknown_is_occupied": bool(synth.get("map_unknown_is_occupied", False)),
        "map_path": _expand(synth.get("map_path", voxel.get("map_path", ""))),
        "map_frame": synth.get("map_frame", robot.get("odom_frame", "odom")),
        "front_frame": lidar.get("front_frame", "base_scan_01"),
        "rear_frame": lidar.get("rear_frame", "base_scan_02"),
        "front_topic": lidar.get("front_topic", "/front_scan"),
        "rear_topic": lidar.get("rear_topic", "/rear_scan"),
        "front_enabled": bool(synth.get("front_enabled", True)),
        "rear_enabled": bool(synth.get("rear_enabled", True)),
        "samples": synth.get("samples", 721),
        "fov_deg": synth.get("fov_deg", 270.0),
        "range_min": synth.get("range_min", 0.05),
        "range_max": synth.get("range_max", 12.0),
        "update_rate": synth.get("update_rate", 10.0),
        "z_min": synth.get("z_min", voxel.get("z_min", 0.05)),
        "z_max": synth.get("z_max", voxel.get("z_max", 1.20)),
        "tf_timeout_sec": synth.get("tf_timeout_sec", 0.05),
        "front_angle_offset_deg": synth.get("front_angle_offset_deg", 0.0),
        "rear_angle_offset_deg": synth.get("rear_angle_offset_deg", 0.0),
        "front_reverse": bool(synth.get("front_reverse", False)),
        "rear_reverse": bool(synth.get("rear_reverse", False)),
        "dynamic_enabled": bool(synth.get("dynamic_enabled", True)),
        "people_frame": synth.get("people_frame", ""),
        "people_topic": synth.get("people_topic", "/task_generator_node/people"),
        "arena_people_topic": synth.get("arena_people_topic", "/task_generator_node/arena_peds"),
        "pedestrian_radius": synth.get("pedestrian_radius", 0.35),
        "pedestrian_timeout_sec": synth.get("pedestrian_timeout_sec", 1.0),
        "log_sec": synth.get("log_sec", 2.0),
    }
    cmd = ["ros2", "run", "ros2isaacsim", "synthetic_2d_laser", "--ros-args"]
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        cmd.extend(["-p", f"{key}:={str(value).lower() if isinstance(value, bool) else value}"])
    return [str(x) for x in cmd]


def pedestrians_cmd(profile: Dict[str, Any]) -> List[str]:
    ped = profile.get("pedestrians", {})
    voxel = profile.get("voxel", {})
    cmd = ["ros2", "run", "ros2isaacsim", "client_pub_ped"]
    if bool(ped.get("auto_sample", False)):
        map_path = _expand(
            ped.get(
                "voxel_map_path",
                voxel.get("map_path", "/home/stardust/resources/arena_ws/arena_assets/collision_configs/shenxinfu_841837.voxel.json.gz"),
            )
        )
        cmd.extend([
            "--auto-map", map_path,
            "--auto-count", str(ped.get("count", 2)),
            "--auto-seed", str(ped.get("seed", 12345)),
            "--auto-clearance", str(ped.get("clearance", 0.35)),
            "--auto-min-path-length", str(ped.get("min_path_length", 1.2)),
            "--auto-max-path-length", str(ped.get("max_path_length", 5.5)),
            "--auto-boundary-margin", str(ped.get("boundary_margin", 0.20)),
            "--auto-velocity-min", str(ped.get("velocity_min", 0.35)),
            "--auto-velocity-max", str(ped.get("velocity_max", 0.50)),
            "--auto-stage-prefix", str(ped.get("stage_prefix", "social_ped_")),
        ])
        character_names = ped.get("character_names") or []
        if character_names:
            cmd.extend(["--auto-character-names", ",".join(str(x) for x in character_names)])
        if ped.get("dump_generated_yaml"):
            cmd.extend(["--dump-generated-yaml", _expand(ped.get("dump_generated_yaml"))])
        return cmd

    here = Path(__file__).resolve().parent
    agents_yaml = _expand(
        ped.get(
            "agents_yaml",
            str(here / "pedestrians" / "shenxinfu_841837.social_nav.yaml"),
        )
    )
    cmd.extend(["--agents-yaml", agents_yaml])
    return cmd


def scheme1_step_cmds(profile: Dict[str, Any]) -> List[tuple[str, List[str]]]:
    return [
        ("Bridge (keep this terminal running)", bridge_cmd(profile, "social_nav")[0]),
        ("Spawn scene + robot after bridge is ready", spawn_cmd(profile)),
        ("Start synthetic 2D laser after TF is available", synthetic_laser_cmd(profile)),
        ("Spawn pedestrians overlay (optional)", pedestrians_cmd(profile)),
        ("Teleop /cmd_vel for scan sanity check (optional)", teleop_cmd(profile)),
    ]


def scheme1_check_cmd(profile: Dict[str, Any]) -> List[str]:
    here = Path(__file__).resolve().parent
    tool = here / "voxel_map_tool.py"
    lidar = profile.get("lidar", {}) or {}
    synth = lidar.get("synthetic_2d", {}) or {}
    voxel = profile.get("voxel", {}) or {}
    map_yaml_path = _expand(synth.get("map_yaml_path", ""))
    voxel_map_path = _expand(synth.get("map_path", voxel.get("map_path", "")))
    if not map_yaml_path:
        raise RuntimeError("scheme1 check requires lidar.synthetic_2d.map_yaml_path in the profile")
    if not voxel_map_path:
        raise RuntimeError("scheme1 check requires voxel.map_path or lidar.synthetic_2d.map_path in the profile")
    return [
        "python3", str(tool), "check-occ-map",
        "--map", voxel_map_path,
        "--yaml", map_yaml_path,
        "--sample-poses", "6",
        "--beams", "72",
        "--range-max", str(synth.get("range_max", 12.0)),
    ]


def scheme1_steps_text(profile: Dict[str, Any]) -> str:
    lines = ["Scheme 1 startup order:"]
    for idx, (label, cmd) in enumerate(scheme1_step_cmds(profile), 1):
        lines.append(f"{idx}. {label}")
        lines.append(f"   {cmd_str(cmd)}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="scripts/profiles/shenxinfu_841837.yaml")
    parser.add_argument("--dry-run", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_bridge = sub.add_parser("bridge")
    p_bridge.add_argument("phase", choices=["edit", "verify", "guard", "strict", "guard2d", "strict2d", "debug2d", "voxel_build", "voxel_guard", "voxel_strict", "voxel_debug", "social_nav", "rtx_scan", "physx_diff_contact"])
    p_spawn = sub.add_parser("spawn")
    p_spawn.add_argument("--phase")
    sub.add_parser("export")
    p2d = sub.add_parser("collision2d")
    p2d.add_argument("action", choices=["init", "render", "summary"])
    pv = sub.add_parser("voxel")
    pv.add_argument("action", choices=["build", "render", "summary", "pcd"])
    sub.add_parser("teleop")
    p_gamepad = sub.add_parser("gamepad")
    p_gamepad.add_argument("--phase")
    sub.add_parser("synthetic_laser")
    sub.add_parser("pedestrians")
    p_scheme1 = sub.add_parser("scheme1")
    p_scheme1.add_argument("action", choices=["steps", "check"])
    args = parser.parse_args(argv)

    profile = load_profile(args.profile)
    if args.cmd == "bridge":
        cmd, env = bridge_cmd(profile, args.phase)
        return run(cmd, env=env, dry_run=args.dry_run)
    if args.cmd == "spawn":
        return run(spawn_cmd(profile, args.phase), dry_run=args.dry_run)
    if args.cmd == "export":
        rc = 0
        for cmd in export_cmds(profile):
            rc = run(cmd, dry_run=args.dry_run)
            if rc != 0:
                return rc
        return rc
    if args.cmd == "collision2d":
        return run(collision2d_cmd(profile, args.action), dry_run=args.dry_run)
    if args.cmd == "voxel":
        rc = 0
        for cmd in voxel_cmds(profile, args.action):
            rc = run(cmd, dry_run=args.dry_run)
            if rc != 0:
                return rc
        return rc
    if args.cmd == "teleop":
        return run(teleop_cmd(profile), dry_run=args.dry_run)
    if args.cmd == "gamepad":
        return run(gamepad_cmd(profile, args.phase), dry_run=args.dry_run)
    if args.cmd == "synthetic_laser":
        return run(synthetic_laser_cmd(profile), dry_run=args.dry_run)
    if args.cmd == "pedestrians":
        return run(pedestrians_cmd(profile), dry_run=args.dry_run)
    if args.cmd == "scheme1":
        if args.action == "steps":
            print(scheme1_steps_text(profile))
            return 0
        if args.action == "check":
            return run(scheme1_check_cmd(profile), dry_run=args.dry_run)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
