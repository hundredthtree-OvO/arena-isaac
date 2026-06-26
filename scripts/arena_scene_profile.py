#!/usr/bin/env python3
"""YAML-driven helper for the Arena Isaac scene workflow.

Examples, run from ~/resources/arena_ws/src/arena/arena-isaac:
  python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml bridge edit
  python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml spawn
  python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml export
  python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml collision2d init
  python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml collision2d render
  python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml bridge guard2d
  python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml bridge voxel_build
  python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml voxel build
  python3 scripts/arena_scene_profile.py --profile scripts/profiles/shenxinfu_841837.yaml bridge voxel_guard
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
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"invalid profile: {path}")
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


def bridge_cmd(profile: Dict[str, Any], phase: str) -> tuple[List[str], Dict[str, str]]:
    bridge = profile.get("bridge", {})
    scene = profile.get("scene", {})
    proxy = profile.get("proxy", {})
    guard = profile.get("guard", {})
    collision2d = profile.get("collision2d", {})
    voxel = profile.get("voxel", {})
    robot = profile.get("robot", {})
    robot_geometry = profile.get("robot_geometry", {})
    lidar = profile.get("lidar", {})
    odom_tf = profile.get("odom_tf", {})
    phases = profile.get("phases", {})
    phase_cfg = phases.get(phase, {})
    if not phase_cfg:
        raise RuntimeError(f"unknown phase '{phase}'. Available phases: {', '.join(sorted(phases.keys()))}")

    env = os.environ.copy()
    env["ISAAC_PATH"] = _expand(bridge.get("isaac_path", os.environ.get("ISAAC_PATH", "$HOME/resources/isaac-sim-4.5.0")))
    env["ARENA_ISAAC_COLLISION_GUARD_OVERLAP_POLICY"] = str(phase_cfg.get("overlap_policy", guard.get("overlap_policy", "escape")))
    env["ARENA_ISAAC_COLLISION_GUARD_BACKEND"] = str(phase_cfg.get("guard_backend", guard.get("backend", "proxy")))
    if "escape_epsilon" in guard:
        env["ARENA_ISAAC_COLLISION_GUARD_ESCAPE_EPS"] = str(guard.get("escape_epsilon"))
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

    env["ARENA_ISAAC_LIDAR_MOUNT_SOURCE"] = str(lidar.get("mount_source", "manual"))
    if lidar.get("inset_x") is not None:
        env["ARENA_ISAAC_LIDAR_INSET_X"] = str(lidar.get("inset_x"))
    if lidar.get("y") is not None:
        env["ARENA_ISAAC_LIDAR_Y"] = str(lidar.get("y"))
    if lidar.get("z") is not None:
        env["ARENA_ISAAC_LIDAR_Z"] = str(lidar.get("z"))
    if lidar.get("front_x") is not None:
        env["ARENA_ISAAC_LIDAR_FRONT_X"] = str(lidar.get("front_x"))
    if lidar.get("rear_x") is not None:
        env["ARENA_ISAAC_LIDAR_REAR_X"] = str(lidar.get("rear_x"))
    if lidar.get("front_yaw") is not None:
        env["ARENA_ISAAC_LIDAR_FRONT_YAW"] = str(lidar.get("front_yaw"))
    if lidar.get("rear_yaw") is not None:
        env["ARENA_ISAAC_LIDAR_REAR_YAW"] = str(lidar.get("rear_yaw"))
    env["ARENA_ISAAC_LIDAR_VISUALIZE"] = "true" if bool(lidar.get("visualize", True)) else "false"

    env["ARENA_ISAAC_PUBLISH_ACTUAL_ODOM_TF"] = "true" if bool(odom_tf.get("publish_actual", True)) else "false"
    env["ARENA_ISAAC_PUBLISH_SENSOR_STATIC_TF"] = "true" if bool(odom_tf.get("publish_sensor_static_tf", True)) else "false"
    env["ARENA_ISAAC_ODOM_TOPIC"] = str(robot.get("odom_topic", "/odom"))
    env["ARENA_ISAAC_ODOM_FRAME"] = str(robot.get("odom_frame", "odom"))
    env["ARENA_ISAAC_BASE_FRAME"] = str(robot.get("base_frame", "base_link"))
    env["ARENA_ISAAC_CMD_VEL_APPLIED_TOPIC"] = str(robot.get("cmd_vel_applied_topic", "/cmd_vel_applied"))
    env["ARENA_ISAAC_FRONT_LASER_FRAME"] = str(lidar.get("front_frame", robot.get("front_laser_frame", "front_laser_link")))
    env["ARENA_ISAAC_REAR_LASER_FRAME"] = str(lidar.get("rear_frame", robot.get("rear_laser_frame", "rear_laser_link")))
    args = [
        "ros2", "launch", "ros2isaacsim", "run_isaacsim.launch.py",
        f"isaac_path:={env['ISAAC_PATH']}",
        f"log_level:={bridge.get('log_level', 'info')}",
        f"idle_fps:={bridge.get('idle_fps', 5)}",
        f"active_fps:={bridge.get('active_fps', 30)}",
        f"mem_log_sec:={bridge.get('mem_log_sec', 30)}",
        f"enable_people_stack:={as_bool(bridge.get('enable_people_stack', False))}",
        f"enable_material_stack:={as_bool(bridge.get('enable_material_stack', False))}",
        f"enable_navmesh:={as_bool(bridge.get('enable_navmesh', False))}",
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


def spawn_cmd(profile: Dict[str, Any]) -> List[str]:
    s = profile.get("scene", {})
    r = profile.get("robot", {})
    return [
        "ros2", "run", "ros2isaacsim", "spawn_v10_scene_lidar_validation",
        "--scene-mode", str(s.get("mode", "usd")),
        "--scene-usd", _expand(s.get("usd_path")),
        "--scene-name", str(s.get("name", "scene")),
        "--scene-x", str(s.get("x", 0.0)),
        "--scene-y", str(s.get("y", 0.0)),
        "--scene-z", str(s.get("z", 0.0)),
        "--scene-yaw", str(s.get("yaw", 0.0)),
        "--robot-name", str(r.get("name", "xms_mecanum")),
        "--cmd-vel-topic", str(r.get("cmd_vel_topic", "/cmd_vel")),
        "--x", str(r.get("x", 0.0)),
        "--y", str(r.get("y", 0.0)),
        "--z", str(r.get("z", 0.05)),
        "--yaw", str(r.get("yaw", 0.0)),
    ]


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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="scripts/profiles/shenxinfu_841837.yaml")
    parser.add_argument("--dry-run", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_bridge = sub.add_parser("bridge")
    p_bridge.add_argument("phase", choices=["edit", "verify", "guard", "strict", "guard2d", "strict2d", "debug2d", "voxel_build", "voxel_guard", "voxel_strict", "voxel_debug"])
    sub.add_parser("spawn")
    sub.add_parser("export")
    p2d = sub.add_parser("collision2d")
    p2d.add_argument("action", choices=["init", "render", "summary"])
    pv = sub.add_parser("voxel")
    pv.add_argument("action", choices=["build", "render", "summary", "pcd"])
    sub.add_parser("teleop")
    args = parser.parse_args(argv)

    profile = load_profile(args.profile)
    if args.cmd == "bridge":
        cmd, env = bridge_cmd(profile, args.phase)
        return run(cmd, env=env, dry_run=args.dry_run)
    if args.cmd == "spawn":
        return run(spawn_cmd(profile), dry_run=args.dry_run)
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
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
