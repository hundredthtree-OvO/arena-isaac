from __future__ import annotations

import argparse
import math
from pathlib import Path

import rclpy
from geometry_msgs.msg import Pose
from isaacsim_msgs.srv import UrdfToUsd
from rclpy.node import Node


def _quat_from_rpy(roll: float, pitch: float, yaw: float):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return (qx, qy, qz, qw)


class SpawnMecanumTeleop(Node):
    def __init__(self):
        super().__init__("spawn_mecanum_teleop")
        self.cli = self.create_client(UrdfToUsd, "/isaac/urdf_to_usd")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("/isaac/urdf_to_usd service not available, waiting again...")

    def call(self, req: UrdfToUsd.Request):
        fut = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        return fut.result()


def main(args=None):
    parser = argparse.ArgumentParser(description="Spawn the mecanum730_xms5 URDF and attach /cmd_vel teleop control.")
    parser.add_argument(
        "--urdf-path",
        default="/home/stardust/resources/mcp_service/curobo_v3/urdf/motion_wheel_arm_simple_sphere_urdf/mecanum730_xms5_gripper_joint_robotbuilder_link2plus_ee_motion_safe_collision.urdf",
        help="Absolute path to mecanum730_xms5 URDF.",
    )
    parser.add_argument("--name", default="mecanum730_xms5", help="Robot instance name in Isaac stage.")
    parser.add_argument(
        "--mode",
        choices=["joint", "hybrid", "kinematic", "physx_wheels", "physx_diff_contact"],
        default="physx_wheels",
        help="joint = wheel velocity targets only; hybrid = wheel spin + kinematic base fallback; kinematic = base only; physx_wheels = PhysX articulation driven by wheel targets.",
    )
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--x", type=float, default=0.0)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--z", type=float, default=0.05)
    parser.add_argument("--roll", type=float, default=0.0, help="Initial roll in radians.")
    parser.add_argument("--pitch", type=float, default=0.0, help="Initial pitch in radians.")
    parser.add_argument("--yaw", type=float, default=0.0, help="Initial yaw in radians.")
    parser.add_argument("--no-localization", action="store_true")
    parser.add_argument("--dual-lidar", action="store_true", help="Attach front/rear RTX 2D lidar and publish /front_scan, /rear_scan.")

    parsed, ros_args = parser.parse_known_args(args)
    urdf_path = Path(parsed.urdf_path).expanduser()
    if parsed.mode != "physx_wheels" and not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    rclpy.init(args=ros_args)
    node = SpawnMecanumTeleop()

    req = UrdfToUsd.Request()
    req.name = parsed.name
    req.urdf_path = str(urdf_path)
    if parsed.mode == "joint":
        req.robot_model = "mecanum730_xms5"
    elif parsed.dual_lidar:
        # Keep the mode suffix last so Isaac-side mode parsing still works.
        req.robot_model = f"mecanum730_xms5_lidar_{parsed.mode}"
    else:
        req.robot_model = f"mecanum730_xms5_{parsed.mode}"
    req.no_localization = bool(parsed.no_localization)
    req.base_frame = parsed.base_frame
    req.odom_frame = parsed.odom_frame
    req.cmd_vel_topic = parsed.cmd_vel_topic
    req.pose = Pose()
    req.pose.position.x = parsed.x
    req.pose.position.y = parsed.y
    req.pose.position.z = parsed.z
    qx, qy, qz, qw = _quat_from_rpy(parsed.roll, parsed.pitch, parsed.yaw)
    req.pose.orientation.x = qx
    req.pose.orientation.y = qy
    req.pose.orientation.z = qz
    req.pose.orientation.w = qw

    resp = node.call(req)
    if resp is None:
        node.get_logger().error("No response from /isaac/urdf_to_usd")
    else:
        node.get_logger().info(f"spawn result usd_path={resp.usd_path}")
        node.get_logger().info(f"teleop: publish Twist to {parsed.cmd_vel_topic}, e.g. ros2 run teleop_twist_keyboard teleop_twist_keyboard")
        if parsed.mode == "joint":
            node.get_logger().info("mode=joint: if wheels spin but chassis does not move, retry with --mode hybrid because this URDF has little/no wheel collision geometry.")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
