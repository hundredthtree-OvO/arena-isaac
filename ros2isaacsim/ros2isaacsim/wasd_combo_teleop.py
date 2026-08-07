#!/usr/bin/env python3
"""Pygame-based differential WASD teleop for operator collection.

Unlike terminal character teleop, this node reads the current keyboard state each
frame, so combined commands such as W+A (forward + turn left) are represented
in one Twist message.  It intentionally emits only ``linear.x`` and
``angular.z`` so keyboard and gamepad collection share one action contract.
"""
from __future__ import annotations

import argparse
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


HELP = """
Differential keyboard teleop window must be focused.

Keys:
  Space     : hold to arm keyboard motion (deadman)
  W/S       : forward / backward           -> linear.x
  A/D       : turn left / turn right       -> angular.z
  Shift     : hold for slow mode
  Esc/close : quit

Examples:
  Space+W   = forward
  Space+W+A = forward while turning left
"""


def _approach(current: float, target: float, max_delta: float) -> float:
    if target > current:
        return min(target, current + max_delta)
    return max(target, current - max_delta)


class DifferentialKeyboardState:
    """Pure keyboard-to-virtual-stick state used by the UI and unit tests."""

    def __init__(self, rise_time_sec: float = 0.35):
        self.rise_time_sec = max(1e-3, float(rise_time_sec))
        self.linear = 0.0
        self.angular = 0.0

    def reset(self) -> None:
        self.linear = 0.0
        self.angular = 0.0

    @staticmethod
    def _axis(positive: bool, negative: bool) -> float:
        return float(bool(positive)) - float(bool(negative))

    def _advance(self, current: float, target: float, dt: float) -> float:
        # Releasing an axis is a command-side stop.  The bridge owns the
        # physical deceleration, matching gamepad deadman behavior.
        if target == 0.0:
            return 0.0
        # Never jump directly through zero into reverse motion.
        if current * target < 0.0:
            return 0.0
        return _approach(current, target, max(0.0, dt) / self.rise_time_sec)

    def update(
        self,
        *,
        dt: float,
        armed: bool,
        focused: bool,
        forward: bool,
        backward: bool,
        left: bool,
        right: bool,
    ) -> tuple[float, float]:
        if not armed or not focused:
            self.reset()
            return 0.0, 0.0
        self.linear = self._advance(
            self.linear,
            self._axis(forward, backward),
            dt,
        )
        self.angular = self._advance(
            self.angular,
            self._axis(left, right),
            dt,
        )
        return self.linear, self.angular


class ComboTeleop(Node):
    def __init__(
        self,
        topic: str,
        linear: float,
        angular: float,
        rate_hz: float,
    ):
        super().__init__("wasd_combo_teleop")
        self.pub = self.create_publisher(Twist, topic, 10)
        self.topic = topic
        self.linear = float(linear)
        self.angular = float(angular)
        self.get_logger().info(
            f"Publishing differential keyboard Twist to {topic}: "
            f"linear={self.linear:.3f}, angular={self.angular:.3f}, rate={rate_hz:.1f} Hz"
        )

    def publish_cmd(self, vx: float, wz: float):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.angular.z = float(wz)
        self.pub.publish(msg)

    def publish_stop(self, n: int = 3):
        for _ in range(max(1, n)):
            self.publish_cmd(0.0, 0.0)
            time.sleep(0.02)


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Multi-key differential teleop publishing Twist."
    )
    parser.add_argument("--topic", default="/cmd_vel_gamepad_diff")
    parser.add_argument("--linear", type=float, default=0.40, help="Forward/backward speed in m/s.")
    parser.add_argument("--angular", type=float, default=0.40, help="Yaw speed in rad/s.")
    parser.add_argument("--rate", type=float, default=50.0, help="Publish rate in Hz.")
    parser.add_argument("--rise-time", type=float, default=0.35, help="Seconds for a held direction to reach full command.")
    parser.add_argument("--slow-scale", type=float, default=0.35, help="Speed multiplier while holding Shift.")
    parser.add_argument("--debug", action="store_true", help="Print non-zero commands and command changes.")
    parsed, ros_args = parser.parse_known_args(args)

    try:
        import pygame
    except Exception as exc:
        print("ERROR: pygame is required for true multi-key teleop.", file=sys.stderr)
        print("Install with: sudo apt install python3-pygame  # or: python3 -m pip install pygame", file=sys.stderr)
        raise SystemExit(2) from exc

    rclpy.init(args=ros_args)
    node = ComboTeleop(
        parsed.topic,
        parsed.linear,
        parsed.angular,
        parsed.rate,
    )
    command_state = DifferentialKeyboardState(parsed.rise_time)

    pygame.init()
    pygame.display.set_caption(
        f"Differential keyboard teleop -> {parsed.topic}"
    )
    screen = pygame.display.set_mode((620, 260))
    font = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()
    print(HELP)

    running = True
    pressed = set()
    last_tuple = None
    last_debug_t = 0.0
    last_focus_warn_t = 0.0
    last_update_t = time.monotonic()
    try:
        while running and rclpy.ok():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    pressed.add(event.key)
                elif event.type == pygame.KEYUP:
                    pressed.discard(event.key)

            # Keep a get_pressed fallback because some SDL backends miss KEYUP
            # events when focus changes.  The event-set path is what enables
            # robust multi-key state rather than terminal character overwrite.
            keys = pygame.key.get_pressed()
            focused = bool(pygame.key.get_focused())
            def down(k):
                return focused and ((k in pressed) or bool(keys[k]))

            armed = down(pygame.K_SPACE)
            slow = down(pygame.K_LSHIFT) or down(pygame.K_RSHIFT)
            scale = parsed.slow_scale if slow else 1.0

            update_t = time.monotonic()
            linear_axis, angular_axis = command_state.update(
                dt=min(0.1, max(0.0, update_t - last_update_t)),
                armed=armed,
                focused=focused,
                forward=down(pygame.K_w),
                backward=down(pygame.K_s),
                left=down(pygame.K_a),
                right=down(pygame.K_d),
            )
            last_update_t = update_t
            if not focused:
                pressed.clear()
            vx = linear_axis * parsed.linear * scale
            wz = angular_axis * parsed.angular * scale

            node.publish_cmd(vx, wz)
            cmd_tuple = (round(vx, 4), round(wz, 4))
            now = time.time()
            if parsed.debug and (
                cmd_tuple != last_tuple
                or (cmd_tuple != (0.0, 0.0) and now - last_debug_t > 0.5)
            ):
                node.get_logger().info(
                    f"teleop cmd: vx={vx:+.3f}, wz={wz:+.3f} "
                    f"-> {parsed.topic}"
                )
                last_debug_t = now
            if not focused and now - last_focus_warn_t > 3.0:
                node.get_logger().warning("teleop window is not focused; keyboard input will stay zero until this window is active")
                last_focus_warn_t = now
            last_tuple = cmd_tuple

            screen.fill((20, 20, 20))
            lines = [
                "Differential keyboard teleop: focus this window",
                "Hold Space | W/S forward/back | A/D turn | Shift slow | Esc quit",
                f"vx={vx:+.3f} m/s   wz={wz:+.3f} rad/s",
                f"topic: {parsed.topic}   focused={focused}   armed={armed}   slow={slow}",
            ]
            for i, line in enumerate(lines):
                surf = font.render(line, True, (230, 230, 230))
                screen.blit(surf, (20, 25 + 42 * i))
            pygame.display.flip()

            rclpy.spin_once(node, timeout_sec=0.0)
            clock.tick(parsed.rate)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        pygame.quit()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
