#!/usr/bin/env python3
"""Spawn a static validation scene + kinematic mecanum robot + dual lidar.

This is the Milestone-C/V10 entry point used before pedestrian tests:
  1. Load a simple USD scene (default: bundled small_warehouse.usd).
  2. Spawn the custom mecanum robot in pure kinematic mode.
  3. Request Isaac-side front/rear RTX 2D lidar (/front_scan, /rear_scan).
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Optional

import rclpy
from geometry_msgs.msg import Pose
from isaacsim_msgs.srv import ImportUsd, UrdfToUsd, SpawnFloor, SpawnWall
from rclpy.node import Node

try:
    from ament_index_python.packages import get_package_share_directory
except Exception:  # pragma: no cover
    get_package_share_directory = None


def _quat_from_rpy(roll: float, pitch: float, yaw: float):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _fill_pose(pose: Pose, x: float, y: float, z: float, roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0):
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    qx, qy, qz, qw = _quat_from_rpy(roll, pitch, yaw)
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw


def _default_scene_path() -> Optional[Path]:
    candidates = []
    if get_package_share_directory is not None:
        try:
            share = Path(get_package_share_directory("ros2isaacsim"))
            candidates.append(share / "assets" / "scenes" / "small_warehouse.usd")
        except Exception:
            pass
    candidates.append(Path(__file__).resolve().parent / "assets" / "scenes" / "small_warehouse.usd")
    # Source-tree fallback when running with --symlink-install.
    candidates.append(Path(__file__).resolve().parents[1] / "assets" / "scenes" / "small_warehouse.usd")
    for p in candidates:
        if p.exists():
            return p
    return None


class V10Spawner(Node):
    def __init__(self):
        super().__init__("spawn_v10_scene_lidar_validation")
        self.import_usd_cli = self.create_client(ImportUsd, "/isaac/import_usd")
        self.urdf_cli = self.create_client(UrdfToUsd, "/isaac/urdf_to_usd")
        self.floor_cli = self.create_client(SpawnFloor, "/isaac/spawn_floor")
        self.wall_cli = self.create_client(SpawnWall, "/isaac/spawn_wall")
        while not self.urdf_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("/isaac/urdf_to_usd service not available, waiting...")

    def _wait_for_generated_scene_services(self):
        while not self.floor_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("/isaac/spawn_floor service not available, waiting...")
        while not self.wall_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("/isaac/spawn_wall service not available, waiting...")

    def call_spawn_floor(self, name: str, x_len: float, y_len: float, x: float, y: float):
        req = SpawnFloor.Request()
        req.name = name
        req.x_length = float(x_len)
        req.y_length = float(y_len)
        req.pos = [float(x), float(y)]
        req.material = ""
        fut = self.floor_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        resp = fut.result()
        if resp is None or not resp.ret:
            raise RuntimeError(f"Failed to spawn floor: {name}")

    def call_spawn_wall(self, name: str, start, end, height: float = 2.0):
        req = SpawnWall.Request()
        req.name = name
        req.start = [float(start[0]), float(start[1])]
        req.end = [float(end[0]), float(end[1])]
        req.height = float(height)
        req.material = ""
        fut = self.wall_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        resp = fut.result()
        if resp is None or not resp.ret:
            raise RuntimeError(f"Failed to spawn wall: {name}")

    def call_spawn_generated_room(self, prefix: str = "v10_room"):
        """Generate a small static room/corridor scene from Arena wall/floor services.

        This avoids relying on remote USD assets and is sufficient for first
        front/rear LaserScan validation.
        """
        self._wait_for_generated_scene_services()
        self.call_spawn_floor(f"{prefix}_floor", 8.0, 6.0, 0.0, 0.0)
        # Outer rectangle.
        self.call_spawn_wall(f"{prefix}_wall_south", (-4.0, -3.0), (4.0, -3.0))
        self.call_spawn_wall(f"{prefix}_wall_north", (-4.0, 3.0), (4.0, 3.0))
        self.call_spawn_wall(f"{prefix}_wall_west", (-4.0, -3.0), (-4.0, 3.0))
        self.call_spawn_wall(f"{prefix}_wall_east", (4.0, -3.0), (4.0, 3.0))
        # Simple toilet/stall-like partitions with gaps.
        self.call_spawn_wall(f"{prefix}_partition_1", (-1.2, -3.0), (-1.2, 0.8), height=1.8)
        self.call_spawn_wall(f"{prefix}_partition_2", (1.2, -3.0), (1.2, 0.8), height=1.8)
        self.call_spawn_wall(f"{prefix}_partition_3", (-1.2, 0.8), (1.2, 0.8), height=1.8)
        self.call_spawn_wall(f"{prefix}_short_obstacle", (-3.0, 1.0), (-2.0, 1.0), height=0.9)
        self.get_logger().info("Generated V10 static room: outer walls + stall-like partitions")

    def call_import_scene(self, scene_path: Path, name: str, x: float, y: float, z: float, yaw: float):
        while not self.import_usd_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("/isaac/import_usd service not available, waiting...")
        req = ImportUsd.Request()
        req.name = name
        req.model = "v10_static_scene"
        req.usd_path = str(scene_path)
        req.control = False
        _fill_pose(req.pose, x, y, z, 0.0, 0.0, yaw)
        fut = self.import_usd_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        resp = fut.result()
        if resp is None or not resp.ret:
            raise RuntimeError(f"Failed to import scene USD: {scene_path}")
        self.get_logger().info(f"Imported validation scene {scene_path} at /World/{name}")

    def call_spawn_robot(self, parsed):
        req = UrdfToUsd.Request()
        req.name = parsed.robot_name
        req.urdf_path = str(Path(parsed.urdf_path).expanduser().resolve())
        # The mode suffix must remain last for Isaac-side mode parsing.
        req.robot_model = "mecanum730_xms5_lidar_kinematic"
        req.no_localization = True
        req.base_frame = parsed.base_frame
        req.odom_frame = parsed.odom_frame
        req.cmd_vel_topic = parsed.cmd_vel_topic
        _fill_pose(req.pose, parsed.x, parsed.y, parsed.z, parsed.roll, parsed.pitch, parsed.yaw)
        fut = self.urdf_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        resp = fut.result()
        if resp is None:
            raise RuntimeError("No response from /isaac/urdf_to_usd")
        self.get_logger().info(f"Spawned robot result: usd_path={resp.usd_path}")
        return resp


def main(args=None):
    default_scene = _default_scene_path()
    parser = argparse.ArgumentParser(description="V10 static scene + dual lidar validation spawner.")
    parser.add_argument("--scene-mode", choices=["generated", "usd", "none"], default="generated", help="generated = service-built room; usd = import --scene-usd; none = robot+lidar only.")
    parser.add_argument("--scene-usd", default=str(default_scene) if default_scene else "", help="Static scene USD to import when --scene-mode usd.")
    parser.add_argument("--scene-name", default="v10_scene")
    parser.add_argument("--scene-x", type=float, default=0.0)
    parser.add_argument("--scene-y", type=float, default=0.0)
    parser.add_argument("--scene-z", type=float, default=0.0)
    parser.add_argument("--scene-yaw", type=float, default=0.0)
    parser.add_argument("--skip-scene", action="store_true", help="Deprecated alias for --scene-mode none.")

    parser.add_argument(
        "--urdf-path",
        default="/home/stardust/resources/mcp_service/curobo_v3/urdf/motion_wheel_arm_simple_sphere_urdf/mecanum730_xms5_gripper_joint_robotbuilder_link2plus_ee_motion_safe_collision.urdf",
    )
    parser.add_argument("--robot-name", default="xms_mecanum")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--x", type=float, default=0.0)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--z", type=float, default=0.05)
    parser.add_argument("--roll", type=float, default=0.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parsed, ros_args = parser.parse_known_args(args)

    if parsed.skip_scene:
        parsed.scene_mode = "none"
    if parsed.scene_mode == "usd":
        scene_path = Path(parsed.scene_usd).expanduser().resolve()
        if not scene_path.exists():
            raise FileNotFoundError(f"Scene USD not found: {scene_path}. Use --scene-usd or --scene-mode generated/none.")
    else:
        scene_path = None
    urdf_path = Path(parsed.urdf_path).expanduser().resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    rclpy.init(args=ros_args)
    node = V10Spawner()
    try:
        if parsed.scene_mode == "generated":
            node.call_spawn_generated_room(parsed.scene_name)
        elif parsed.scene_mode == "usd":
            node.call_import_scene(scene_path, parsed.scene_name, parsed.scene_x, parsed.scene_y, parsed.scene_z, parsed.scene_yaw)
        node.call_spawn_robot(parsed)
        node.get_logger().info("V10 validation ready. Check topics: /front_scan, /rear_scan, /cmd_vel")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
