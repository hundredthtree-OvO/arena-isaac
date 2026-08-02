"""Simulator-independent command state for the interactive SMPL mirror."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InteractiveCommandController:
    """Translate discrete key events into a persistent locomotion command."""

    cruise_speed_mps: float = 0.95
    turn_rate_rps: float = 0.8
    moving: bool = False
    turning_left: bool = False
    turning_right: bool = False

    SPEED_PRESETS = {
        "slow": 0.75,
        "normal": 0.95,
        "fast": 1.20,
    }

    def handle(self, action: str, pressed: bool) -> None:
        if action == "move" and pressed:
            self.moving = True
        elif action == "stop" and pressed:
            self.moving = False
        elif action == "turn_left":
            self.turning_left = pressed
        elif action == "turn_right":
            self.turning_right = pressed
        elif action in self.SPEED_PRESETS and pressed:
            self.cruise_speed_mps = self.SPEED_PRESETS[action]

    def command(self) -> tuple[float, float, str]:
        speed = self.cruise_speed_mps if self.moving else 0.0
        yaw_rate = self.turn_rate_rps * (
            float(self.turning_left) - float(self.turning_right)
        )
        label = (
            f"interactive speed={speed:.2f}m/s "
            f"yaw_rate={yaw_rate:+.2f}rad/s"
        )
        return speed, yaw_rate, label
