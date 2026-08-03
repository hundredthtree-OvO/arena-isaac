"""State and interpolation policy for externally driven pedestrians."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Sequence


class ExternalMotionMode(IntEnum):
    LOCOMOTION = 0
    FREEZE = 1
    TERMINAL_ALIGN = 2
    REPLAY_TRACK = 3


class ExternalMotionModeState:
    def __init__(self):
        self.mode = ExternalMotionMode.LOCOMOTION

    def transition(self, requested: int) -> tuple[ExternalMotionMode, ExternalMotionMode]:
        try:
            next_mode = ExternalMotionMode(int(requested))
        except ValueError as exc:
            raise ValueError(f"unsupported external motion mode: {requested}") from exc
        previous = self.mode
        self.mode = next_mode
        return previous, next_mode


@dataclass(frozen=True)
class ExternalMotionSample:
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    yaw: float
    speed: float
    expired: bool
    sequence: int | None = None
    age_sec: float = 0.0


class ExternalMotionState:
    def __init__(self, *, walk_speed_threshold: float = 0.05):
        self.walk_speed_threshold = max(0.0, float(walk_speed_threshold))
        self.enabled = False
        self._position = (0.0, 0.0, 0.0)
        self._velocity = (0.0, 0.0, 0.0)
        self._yaw = 0.0
        self._received_at = 0.0
        self._timeout_sec = 0.5
        self._sequence: int | None = None

    def set_command(
        self,
        *,
        position: Sequence[float],
        velocity: Sequence[float],
        yaw: float,
        received_at: float,
        timeout_sec: float,
        sequence: int | None = None,
    ) -> None:
        parsed_position = _vector3(position, "position")
        parsed_velocity = _vector3(velocity, "velocity")
        parsed_yaw = float(yaw)
        parsed_received_at = float(received_at)
        parsed_timeout = float(timeout_sec)
        parsed_sequence = None if sequence is None else int(sequence)
        if not all(
            math.isfinite(value)
            for value in (parsed_yaw, parsed_received_at, parsed_timeout)
        ):
            raise ValueError("external motion scalar contains a non-finite value")
        if parsed_timeout <= 0.0:
            raise ValueError("external motion timeout must be positive")
        self.enabled = True
        self._position = parsed_position
        self._velocity = parsed_velocity
        self._yaw = parsed_yaw
        self._received_at = parsed_received_at
        self._timeout_sec = parsed_timeout
        self._sequence = parsed_sequence

    def clear(self) -> None:
        self.enabled = False
        self._sequence = None

    def sample(self, now: float) -> ExternalMotionSample | None:
        if not self.enabled:
            return None
        age = max(0.0, float(now) - self._received_at)
        elapsed = min(age, self._timeout_sec)
        expired = age >= self._timeout_sec
        velocity = (0.0, 0.0, 0.0) if expired else self._velocity
        position = tuple(
            self._position[index] + self._velocity[index] * elapsed
            for index in range(3)
        )
        speed = math.hypot(velocity[0], velocity[1])
        return ExternalMotionSample(
            position=position,
            velocity=velocity,
            yaw=self._yaw,
            speed=speed,
            expired=expired,
            sequence=self._sequence,
            age_sec=age,
        )

    def should_walk(self, sample: ExternalMotionSample) -> bool:
        return not sample.expired and sample.speed >= self.walk_speed_threshold


def locomotion_path_points(
    sample: ExternalMotionSample,
    *,
    lookahead_m: float = 0.75,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Build the world-space path contract expected by Isaac's walk graph."""
    distance = max(0.05, float(lookahead_m))
    speed = max(float(sample.speed), 1e-6)
    direction_x = float(sample.velocity[0]) / speed
    direction_y = float(sample.velocity[1]) / speed
    start = tuple(float(value) for value in sample.position)
    return (
        start,
        (
            start[0] + direction_x * distance,
            start[1] + direction_y * distance,
            start[2],
        ),
    )


def animation_walk_blend(
    speed_mps: float,
    *,
    full_speed_mps: float,
    speed_exponent: float = 1.0,
) -> float:
    """Map metric target speed onto the nonlinear People Walk blend."""
    speed = max(0.0, float(speed_mps))
    full_speed = max(1e-6, float(full_speed_mps))
    exponent = max(1e-3, float(speed_exponent))
    normalized = min(1.0, speed / full_speed)
    return normalized ** (1.0 / exponent)


def animation_tracking_sample(
    reference: ExternalMotionSample,
    current_position: Sequence[float],
    *,
    tracking_gain: float = 1.5,
    max_speed_mps: float = 1.2,
) -> ExternalMotionSample:
    """Turn a planner reference into a bounded root-motion tracking command."""
    current = _vector3(current_position, "current_position")
    if reference.expired:
        return ExternalMotionSample(
            position=current,
            velocity=(0.0, 0.0, 0.0),
            yaw=float(reference.yaw),
            speed=0.0,
            expired=True,
            sequence=reference.sequence,
            age_sec=reference.age_sec,
        )

    gain = max(0.0, float(tracking_gain))
    limit = max(0.01, float(max_speed_mps))
    velocity_x = float(reference.velocity[0]) + gain * (
        float(reference.position[0]) - current[0]
    )
    velocity_y = float(reference.velocity[1]) + gain * (
        float(reference.position[1]) - current[1]
    )
    speed = math.hypot(velocity_x, velocity_y)
    if speed > limit:
        scale = limit / speed
        velocity_x *= scale
        velocity_y *= scale
        speed = limit
    yaw = float(reference.yaw)
    if speed > 1e-6:
        yaw = math.atan2(velocity_y, velocity_x)
    return ExternalMotionSample(
        position=current,
        velocity=(velocity_x, velocity_y, 0.0),
        yaw=yaw,
        speed=speed,
        expired=False,
        sequence=reference.sequence,
        age_sec=reference.age_sec,
    )


def turn_aware_animation_sample(
    reference: ExternalMotionSample,
    current_position: Sequence[float],
    current_yaw: float,
    *,
    tracking_gain: float = 1.5,
    max_speed_mps: float = 1.2,
    slow_angle_rad: float = math.radians(20.0),
    full_slow_angle_rad: float = math.radians(55.0),
    minimum_speed_scale: float = 0.25,
) -> ExternalMotionSample:
    """Slow forward-only locomotion through turns without rebasing its root."""
    sample = animation_tracking_sample(
        reference,
        current_position,
        tracking_gain=tracking_gain,
        max_speed_mps=max_speed_mps,
    )
    if sample.expired or sample.speed <= 1e-6:
        return sample

    target_yaw = math.atan2(sample.velocity[1], sample.velocity[0])
    error = math.atan2(
        math.sin(target_yaw - float(current_yaw)),
        math.cos(target_yaw - float(current_yaw)),
    )
    error_magnitude = abs(error)
    full_slow_angle = max(1e-3, float(full_slow_angle_rad))
    slow_angle = min(full_slow_angle, max(0.0, float(slow_angle_rad)))
    if error_magnitude <= slow_angle or full_slow_angle <= slow_angle + 1e-6:
        return sample
    fraction = min(
        1.0,
        (error_magnitude - slow_angle) / (full_slow_angle - slow_angle),
    )
    minimum_scale = min(1.0, max(0.0, float(minimum_speed_scale)))
    scale = max(minimum_scale, 1.0 - fraction)
    velocity = (
        sample.velocity[0] * scale,
        sample.velocity[1] * scale,
        sample.velocity[2],
    )
    return ExternalMotionSample(
        position=sample.position,
        velocity=velocity,
        yaw=target_yaw,
        speed=sample.speed * scale,
        expired=False,
        sequence=sample.sequence,
        age_sec=sample.age_sec,
    )


def bounded_yaw_step(
    current_yaw: float,
    target_yaw: float,
    *,
    max_rate_radps: float,
    dt: float,
) -> float:
    """Advance an angle toward its target without crossing the short arc."""
    current = float(current_yaw)
    target = float(target_yaw)
    rate = max(0.0, float(max_rate_radps))
    step_limit = rate * max(0.0, float(dt))
    error = math.atan2(math.sin(target - current), math.cos(target - current))
    if abs(error) <= step_limit:
        return target
    return current + math.copysign(step_limit, error)


def _vector3(value: Sequence[float], field_name: str) -> tuple[float, float, float]:
    try:
        parsed = (float(value[0]), float(value[1]), float(value[2]))
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"external motion {field_name} must contain three values") from exc
    if not all(math.isfinite(item) for item in parsed):
        raise ValueError(f"external motion {field_name} contains a non-finite value")
    return parsed
