#!/usr/bin/env python3
"""Pygame-based multi-key WASD teleop for mecanum /cmd_vel.

Unlike terminal character teleop, this node reads the current keyboard state each
frame, so combined commands such as W+Q (forward + rotate left) and W+A
(forward + strafe left) are represented in one Twist message.
"""
from __future__ import annotations

import argparse
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


HELP = """
WASD combo teleop window must be focused.

Keys:
  W/S       : forward / backward          -> linear.x
  A/D       : strafe left / strafe right  -> linear.y
  Q/E       : rotate left / rotate right  -> angular.z
  Space     : hold to force stop
  Shift     : hold for slow mode
  Esc/close : quit

Examples:
  W+Q = forward while rotating left
  W+A = forward while strafing left
  W+A+Q = forward + left strafe + left rotation
"""


class ComboTeleop(Node):
    def __init__(self, topic: str, linear: float, lateral: float, angular: float, rate_hz: float, slow_scale: float):
        super().__init__("wasd_combo_teleop")
        self.pub = self.create_publisher(Twist, topic, 10)
        self.topic = topic
        self.linear = float(linear)
        self.lateral = float(lateral)
        self.angular = float(angular)
        self.slow_scale = float(slow_scale)
        self.period = 1.0 / float(rate_hz)
        self.get_logger().info(
            f"Publishing combined WASD Twist to {topic}: "
            f"linear={self.linear:.3f}, lateral={self.lateral:.3f}, angular={self.angular:.3f}"
        )

    def publish_cmd(self, vx: float, vy: float, wz: float):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.angular.z = float(wz)
        self.pub.publish(msg)

    def publish_stop(self, n: int = 3):
        for _ in range(max(1, n)):
            self.publish_cmd(0.0, 0.0, 0.0)
            time.sleep(0.02)


def main(args=None):
    parser = argparse.ArgumentParser(description="Multi-key WASD teleop that publishes geometry_msgs/Twist.")
    parser.add_argument("--topic", default="/cmd_vel")
    parser.add_argument("--linear", type=float, default=0.20, help="Forward/backward speed in m/s.")
    parser.add_argument("--lateral", type=float, default=0.20, help="Left/right strafe speed in m/s.")
    parser.add_argument("--angular", type=float, default=0.50, help="Yaw speed in rad/s.")
    parser.add_argument("--rate", type=float, default=30.0, help="Publish rate in Hz.")
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
    node = ComboTeleop(parsed.topic, parsed.linear, parsed.lateral, parsed.angular, parsed.rate, parsed.slow_scale)

    pygame.init()
    pygame.display.set_caption("WASD combo teleop -> /cmd_vel")
    screen = pygame.display.set_mode((620, 260))
    font = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()
    print(HELP)

    running = True
    pressed = set()
    last_tuple = None
    last_debug_t = 0.0
    last_focus_warn_t = 0.0
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
                return (k in pressed) or bool(keys[k])

            stop = down(pygame.K_SPACE)
            slow = down(pygame.K_LSHIFT) or down(pygame.K_RSHIFT)
            scale = parsed.slow_scale if slow else 1.0

            if stop:
                vx = vy = wz = 0.0
            else:
                vx = (float(down(pygame.K_w)) - float(down(pygame.K_s))) * parsed.linear * scale
                # ROS base_link convention: +y is left. A = left, D = right.
                vy = (float(down(pygame.K_a)) - float(down(pygame.K_d))) * parsed.lateral * scale
                # ROS yaw convention: +z is counter-clockwise / left turn. Q = left, E = right.
                wz = (float(down(pygame.K_q)) - float(down(pygame.K_e))) * parsed.angular * scale

            node.publish_cmd(vx, vy, wz)
            cmd_tuple = (round(vx, 4), round(vy, 4), round(wz, 4))
            now = time.time()
            if parsed.debug and (cmd_tuple != last_tuple or (cmd_tuple != (0.0, 0.0, 0.0) and now - last_debug_t > 0.5)):
                node.get_logger().info(f"teleop cmd: vx={vx:+.3f}, vy={vy:+.3f}, wz={wz:+.3f} -> {parsed.topic}")
                last_debug_t = now
            if not focused and now - last_focus_warn_t > 3.0:
                node.get_logger().warning("teleop window is not focused; keyboard input will stay zero until this window is active")
                last_focus_warn_t = now
            last_tuple = cmd_tuple

            screen.fill((20, 20, 20))
            lines = [
                "WASD combo teleop: focus this window",
                "W/S forward/back | A/D strafe | Q/E rotate | Space stop | Shift slow | Esc quit",
                f"vx={vx:+.3f} m/s   vy={vy:+.3f} m/s   wz={wz:+.3f} rad/s",
                f"topic: {parsed.topic}   focused={focused}   pressed={len(pressed)}   debug={parsed.debug}",
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
