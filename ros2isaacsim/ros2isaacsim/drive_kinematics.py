"""Pure wheel-speed mappings shared by the Isaac mobile-base controller."""

from __future__ import annotations

from typing import Iterable

import numpy as np


MECANUM_DRIVE = "mecanum"
DIFFERENTIAL_DRIVE = "differential"


def wheel_angular_speeds(
    *,
    vx: float,
    vy: float,
    wz: float,
    drive_mode: str,
    half_length: float,
    half_width: float,
    wheel_radius: float,
    wheel_signs: Iterable[float],
    max_wheel_speed: float,
) -> np.ndarray:
    """Map a base Twist to FL, FR, RL, RR joint angular velocities."""
    radius = float(wheel_radius)
    if radius <= 0.0:
        raise ValueError("wheel_radius must be positive")

    mode = str(drive_mode).strip().lower()
    if mode == DIFFERENTIAL_DRIVE:
        # Track width B is 2 * half_width, so B / 2 is half_width.
        left = (float(vx) - float(half_width) * float(wz)) / radius
        right = (float(vx) + float(half_width) * float(wz)) / radius
        omega = np.array([left, right, left, right], dtype=np.float32)
    elif mode == MECANUM_DRIVE:
        lever = float(half_length) + float(half_width)
        omega = np.array([
            (float(vx) - float(vy) - lever * float(wz)) / radius,
            (float(vx) + float(vy) + lever * float(wz)) / radius,
            (float(vx) + float(vy) - lever * float(wz)) / radius,
            (float(vx) - float(vy) + lever * float(wz)) / radius,
        ], dtype=np.float32)
    else:
        raise ValueError(f"unsupported drive mode: {drive_mode}")

    signs = np.asarray(list(wheel_signs), dtype=np.float32)
    if signs.shape != (4,):
        raise ValueError("wheel_signs must contain exactly four values")
    omega *= signs
    return np.clip(omega, -float(max_wheel_speed), float(max_wheel_speed))
