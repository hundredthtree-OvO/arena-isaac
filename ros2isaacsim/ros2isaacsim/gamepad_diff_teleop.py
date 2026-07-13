#!/usr/bin/env python3
"""Deadman-controlled gamepad teleoperation for a differential-drive base."""

from __future__ import annotations

import math
import time
from typing import Sequence

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Joy


def apply_deadzone(value: float, deadzone: float) -> float:
    """Apply a continuous, rescaled joystick deadzone."""
    value = max(-1.0, min(1.0, float(value)))
    deadzone = max(0.0, min(0.99, float(deadzone)))
    magnitude = abs(value)
    if magnitude <= deadzone:
        return 0.0
    return math.copysign((magnitude - deadzone) / (1.0 - deadzone), value)


def axis_value(axes: Sequence[float], index: int) -> float:
    """Read one joystick axis, returning zero for an unavailable mapping."""
    if index < 0 or index >= len(axes):
        return 0.0
    return float(axes[index])


class GamepadDiffTeleop(Node):
    def __init__(self):
        super().__init__("gamepad_diff_teleop")
        self.declare_parameter("joy_topic", "/joy")
        self.declare_parameter("output_topic", "/cmd_vel_gamepad_diff")
        self.declare_parameter("linear_axis", 1)
        self.declare_parameter("angular_axis", 0)
        self.declare_parameter("enable_button", 4)
        self.declare_parameter("linear_scale", 0.2)
        self.declare_parameter("angular_scale", 0.5)
        self.declare_parameter("deadzone", 0.1)
        self.declare_parameter("joy_timeout_sec", 0.5)

        self.joy_topic = str(self.get_parameter("joy_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.linear_axis = int(self.get_parameter("linear_axis").value)
        self.angular_axis = int(self.get_parameter("angular_axis").value)
        self.enable_button = int(self.get_parameter("enable_button").value)
        self.linear_scale = float(self.get_parameter("linear_scale").value)
        self.angular_scale = float(self.get_parameter("angular_scale").value)
        self.deadzone = float(self.get_parameter("deadzone").value)
        self.joy_timeout_sec = max(
            0.05,
            float(self.get_parameter("joy_timeout_sec").value),
        )

        self.publisher = self.create_publisher(Twist, self.output_topic, 10)
        self.subscription = self.create_subscription(
            Joy,
            self.joy_topic,
            self._joy_callback,
            qos_profile_sensor_data,
        )
        self.timer = self.create_timer(0.05, self._watchdog)
        self._enabled = False
        self._last_joy_time = 0.0
        self._warned_mapping = False

        self.get_logger().info(
            f"Gamepad differential teleop: {self.joy_topic} -> "
            f"{self.output_topic}, axes=(linear:{self.linear_axis}, "
            f"angular:{self.angular_axis}), "
            f"deadman_button={self.enable_button}, "
            f"scales=({self.linear_scale:.3f}, "
            f"{self.angular_scale:.3f})"
        )

    def _publish(self, vx: float, wz: float) -> None:
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = 0.0
        msg.angular.z = float(wz)
        self.publisher.publish(msg)

    def _joy_callback(self, msg: Joy) -> None:
        self._last_joy_time = time.monotonic()
        mapping_valid = (
            0 <= self.linear_axis < len(msg.axes)
            and 0 <= self.angular_axis < len(msg.axes)
            and 0 <= self.enable_button < len(msg.buttons)
        )
        if not mapping_valid:
            if not self._warned_mapping:
                self._warned_mapping = True
                self.get_logger().warning(
                    "Gamepad mapping is outside the received Joy "
                    "axes/buttons; "
                    "inspect `ros2 topic echo /joy` and update the profile."
                )
            if self._enabled:
                self._publish(0.0, 0.0)
            self._enabled = False
            return

        enabled = bool(msg.buttons[self.enable_button])
        if enabled:
            vx = apply_deadzone(
                axis_value(msg.axes, self.linear_axis), self.deadzone
            )
            wz = apply_deadzone(
                axis_value(msg.axes, self.angular_axis), self.deadzone
            )
            self._publish(vx * self.linear_scale, wz * self.angular_scale)
        elif self._enabled:
            # Publish one stop on release, then become silent so keyboard
            # can resume after the controller's gamepad priority timeout.
            self._publish(0.0, 0.0)
        self._enabled = enabled

    def _watchdog(self) -> None:
        if not self._enabled:
            return
        if time.monotonic() - self._last_joy_time <= self.joy_timeout_sec:
            return
        self._publish(0.0, 0.0)
        self._enabled = False
        self.get_logger().warning(
            "Gamepad input timed out; published stop command."
        )


def main(args=None):
    rclpy.init(args=args)
    node = GamepadDiffTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
