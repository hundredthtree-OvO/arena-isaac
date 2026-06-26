#!/usr/bin/env python3
"""Deterministic /cmd_vel pulse publisher for debugging Isaac kinematic robot motion.

Use this to distinguish "teleop did not publish" from "Isaac controller did not move".
"""
from __future__ import annotations

import argparse
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class CmdVelPulse(Node):
    def __init__(self, topic: str):
        super().__init__("cmd_vel_pulse")
        self.pub = self.create_publisher(Twist, topic, 10)
        self.topic = topic

    def publish_twist(self, vx: float, vy: float, wz: float):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.angular.z = float(wz)
        self.pub.publish(msg)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/cmd_vel")
    parser.add_argument("--motion", choices=["forward", "backward", "left", "right", "turn_left", "turn_right"], default="forward")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--linear", type=float, default=0.20)
    parser.add_argument("--lateral", type=float, default=0.20)
    parser.add_argument("--angular", type=float, default=0.50)
    parsed, ros_args = parser.parse_known_args(args)

    rclpy.init(args=ros_args)
    node = CmdVelPulse(parsed.topic)

    vx = vy = wz = 0.0
    if parsed.motion == "forward":
        vx = parsed.linear
    elif parsed.motion == "backward":
        vx = -parsed.linear
    elif parsed.motion == "left":
        vy = parsed.lateral
    elif parsed.motion == "right":
        vy = -parsed.lateral
    elif parsed.motion == "turn_left":
        wz = parsed.angular
    elif parsed.motion == "turn_right":
        wz = -parsed.angular

    node.get_logger().info(f"Publishing {parsed.motion}: vx={vx}, vy={vy}, wz={wz} to {parsed.topic} for {parsed.duration}s")
    t_end = time.time() + max(0.0, parsed.duration)
    period = 1.0 / max(1.0, parsed.rate)
    try:
        while rclpy.ok() and time.time() < t_end:
            node.publish_twist(vx, vy, wz)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(period)
    finally:
        for _ in range(8):
            node.publish_twist(0.0, 0.0, 0.0)
            time.sleep(0.02)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
