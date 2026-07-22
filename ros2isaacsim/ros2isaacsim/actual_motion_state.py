"""Actual chassis motion estimation independent from ROS publication."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class ActualMotionState:
    position: np.ndarray
    yaw: float
    vx: float
    vy: float
    wz: float


class ActualMotionStateEstimator:
    def __init__(self) -> None:
        self._position: Optional[np.ndarray] = None
        self._yaw: Optional[float] = None
        self._time: Optional[float] = None

    def reset(self) -> None:
        """Discard the previous sample after an intentional pose discontinuity."""
        self._position = None
        self._yaw = None
        self._time = None

    def sample(
        self,
        position: Sequence[float],
        quaternion_wxyz: Sequence[float],
        *,
        now: Optional[float] = None,
    ) -> ActualMotionState:
        position_array = np.asarray(position, dtype=np.float64)
        yaw = _quaternion_wxyz_to_yaw(quaternion_wxyz)
        sample_time = time.monotonic() if now is None else float(now)
        vx = vy = wz = 0.0
        if self._position is not None and self._yaw is not None and self._time is not None:
            dt = sample_time - self._time
            if dt > 0.0:
                delta_world = (position_array - self._position) / dt
                vx = math.cos(yaw) * delta_world[0] + math.sin(yaw) * delta_world[1]
                vy = -math.sin(yaw) * delta_world[0] + math.cos(yaw) * delta_world[1]
                yaw_delta = math.atan2(math.sin(yaw - self._yaw), math.cos(yaw - self._yaw))
                wz = yaw_delta / dt
        self._position = position_array.copy()
        self._yaw = yaw
        self._time = sample_time
        return ActualMotionState(position_array.copy(), yaw, float(vx), float(vy), float(wz))


def _quaternion_wxyz_to_yaw(quaternion_wxyz: Sequence[float]) -> float:
    w, x, y, z = (float(value) for value in quaternion_wxyz)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
