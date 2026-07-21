"""Wheel-target-only backend for PhysX differential contact locomotion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

from ros2isaacsim.drive_kinematics import DIFFERENTIAL_DRIVE, wheel_angular_speeds


@dataclass(frozen=True)
class PhysxDiffContactConfig:
    wheel_radius: float
    track_width: float
    wheel_signs: tuple[float, float, float, float]
    max_wheel_speed: float
    linear_gain: float
    angular_gain: float


class PhysxDiffContactDrive:
    """Apply differential wheel targets without commanding the articulation root."""

    def __init__(
        self,
        config: PhysxDiffContactConfig,
        apply_wheel_targets: Callable[[np.ndarray], None],
    ) -> None:
        if config.wheel_radius <= 0.0:
            raise ValueError("wheel_radius must be positive")
        if config.track_width <= 0.0:
            raise ValueError("track_width must be positive")
        if config.max_wheel_speed <= 0.0:
            raise ValueError("max_wheel_speed must be positive")
        if config.linear_gain <= 0.0 or config.angular_gain <= 0.0:
            raise ValueError("linear_gain and angular_gain must be positive")
        self._config = config
        self._apply_wheel_targets = apply_wheel_targets
        self.last_limit_scale = 1.0
        self.last_effective_vx = 0.0
        self.last_effective_wz = 0.0

    def apply(self, vx: float, wz: float) -> np.ndarray:
        config = self._config
        self.last_effective_vx = float(vx) * config.linear_gain
        self.last_effective_wz = float(wz) * config.angular_gain
        speeds = wheel_angular_speeds(
            vx=self.last_effective_vx,
            vy=0.0,
            wz=self.last_effective_wz,
            drive_mode=DIFFERENTIAL_DRIVE,
            half_length=0.0,
            half_width=0.5 * config.track_width,
            wheel_radius=config.wheel_radius,
            wheel_signs=config.wheel_signs,
            max_wheel_speed=float("inf"),
        )
        peak = float(np.max(np.abs(speeds)))
        self.last_limit_scale = min(1.0, config.max_wheel_speed / peak) if peak > 0.0 else 1.0
        speeds *= self.last_limit_scale
        self._apply_wheel_targets(speeds)
        return speeds

    def stop(self) -> np.ndarray:
        return self.apply(0.0, 0.0)


def wheel_slip_diagnostics(
    *,
    joint_velocities: Sequence[float],
    wheel_radius: float,
    track_width: float,
    wheel_signs: Sequence[float],
    measured_vx: float,
    measured_vy: float,
    measured_wz: float,
    effective_wz: float,
    minimum_reference_speed: float = 0.05,
) -> dict:
    """Compare wheel tread speeds with the measured rigid-body hub speeds."""

    velocities = np.asarray(joint_velocities, dtype=np.float64).reshape(-1)
    signs = np.asarray(wheel_signs, dtype=np.float64).reshape(-1)
    if velocities.shape != (4,) or signs.shape != (4,):
        raise ValueError("joint_velocities and wheel_signs must contain four values")
    radius = float(wheel_radius)
    half_track = 0.5 * float(track_width)
    if radius <= 0.0 or half_track <= 0.0:
        raise ValueError("wheel_radius and track_width must be positive")

    tread_speeds = velocities * signs * radius
    hub_left = float(measured_vx) - half_track * float(measured_wz)
    hub_right = float(measured_vx) + half_track * float(measured_wz)
    hub_speeds = np.array([hub_left, hub_right, hub_left, hub_right], dtype=np.float64)
    reference = np.maximum.reduce(
        [
            np.abs(tread_speeds),
            np.abs(hub_speeds),
            np.full(4, max(float(minimum_reference_speed), 1.0e-6)),
        ]
    )
    slip_ratios = (tread_speeds - hub_speeds) / reference
    yaw_efficiency = (
        float(measured_wz) / float(effective_wz)
        if abs(float(effective_wz)) > 1.0e-6
        else 0.0
    )
    return {
        "tread_speeds_mps": tread_speeds.tolist(),
        "hub_speeds_mps": hub_speeds.tolist(),
        "slip_ratios": slip_ratios.tolist(),
        "mean_abs_slip_ratio": float(np.mean(np.abs(slip_ratios))),
        "yaw_efficiency": yaw_efficiency,
        "measured_vy": float(measured_vy),
    }


def separated_tire_wrench(
    *,
    joint_velocities: Sequence[float],
    wheel_radius: float,
    wheel_signs: Sequence[float],
    half_length: float,
    half_width: float,
    measured_vx: float,
    measured_vy: float,
    measured_wz: float,
    longitudinal_stiffness: float,
    lateral_stiffness: float,
    max_longitudinal_force: float,
    max_lateral_force: float,
    contact_mask: Sequence[bool] = (True, True, True, True),
) -> dict:
    """Resolve four link-local tire forces into a planar chassis wrench.

    The longitudinal channel follows wheel tread slip, while the lateral
    channel only damps lateral hub velocity. Keeping the channels independent
    avoids the isotropic Coulomb-friction tradeoff that prevents skid steering.
    """

    velocities = np.asarray(joint_velocities, dtype=np.float64).reshape(-1)
    signs = np.asarray(wheel_signs, dtype=np.float64).reshape(-1)
    contacts = np.asarray(contact_mask, dtype=bool).reshape(-1)
    if velocities.shape != (4,) or signs.shape != (4,) or contacts.shape != (4,):
        raise ValueError("joint_velocities, wheel_signs and contact_mask must contain four values")

    radius = float(wheel_radius)
    length = float(half_length)
    width = float(half_width)
    if radius <= 0.0 or length <= 0.0 or width <= 0.0:
        raise ValueError("wheel_radius, half_length and half_width must be positive")
    if longitudinal_stiffness < 0.0 or lateral_stiffness < 0.0:
        raise ValueError("tire stiffness values must be non-negative")
    if max_longitudinal_force < 0.0 or max_lateral_force < 0.0:
        raise ValueError("tire force limits must be non-negative")

    positions = np.array(
        [
            [length, width],
            [length, -width],
            [-length, width],
            [-length, -width],
        ],
        dtype=np.float64,
    )
    tread_speeds = velocities * signs * radius
    hub_longitudinal = float(measured_vx) - float(measured_wz) * positions[:, 1]
    hub_lateral = float(measured_vy) + float(measured_wz) * positions[:, 0]
    longitudinal_slip = tread_speeds - hub_longitudinal

    longitudinal_forces = np.clip(
        float(longitudinal_stiffness) * longitudinal_slip,
        -float(max_longitudinal_force),
        float(max_longitudinal_force),
    )
    lateral_forces = np.clip(
        -float(lateral_stiffness) * hub_lateral,
        -float(max_lateral_force),
        float(max_lateral_force),
    )
    longitudinal_forces *= contacts
    lateral_forces *= contacts

    yaw_torques = positions[:, 0] * lateral_forces - positions[:, 1] * longitudinal_forces
    return {
        "wheel_positions_m": positions.tolist(),
        "tread_speeds_mps": tread_speeds.tolist(),
        "hub_longitudinal_speeds_mps": hub_longitudinal.tolist(),
        "hub_lateral_speeds_mps": hub_lateral.tolist(),
        "longitudinal_slip_speeds_mps": longitudinal_slip.tolist(),
        "longitudinal_forces_n": longitudinal_forces.tolist(),
        "lateral_forces_n": lateral_forces.tolist(),
        "contact_mask": contacts.tolist(),
        "net_force_body_n": [
            float(np.sum(longitudinal_forces)),
            float(np.sum(lateral_forces)),
            0.0,
        ],
        "net_torque_body_nm": [0.0, 0.0, float(np.sum(yaw_torques))],
    }


def articulation_link_wrench_arrays(
    *,
    articulation_count: int,
    max_links: int,
    link_index: int,
    force: Sequence[float],
    torque: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build Isaac 4.5 articulation tensor inputs for one selected link."""

    count = int(articulation_count)
    links = int(max_links)
    selected = int(link_index)
    if count <= 0 or links <= 0:
        raise ValueError("articulation_count and max_links must be positive")
    if selected < 0 or selected >= links:
        raise ValueError("link_index is outside the articulation link buffer")
    force_vec = np.asarray(force, dtype=np.float32).reshape(-1)
    torque_vec = np.asarray(torque, dtype=np.float32).reshape(-1)
    if force_vec.shape != (3,) or torque_vec.shape != (3,):
        raise ValueError("force and torque must contain three values")

    forces = np.zeros((count, links, 3), dtype=np.float32)
    torques = np.zeros((count, links, 3), dtype=np.float32)
    forces[:, selected, :] = force_vec
    torques[:, selected, :] = torque_vec
    indices = np.arange(count, dtype=np.uint32)
    return forces, torques, indices


def single_articulation_targets(values: Iterable[float]) -> np.ndarray:
    """Return the `(M, K)` target shape required by Isaac 4.5 Articulation views."""
    targets = np.asarray(list(values), dtype=np.float32)
    if targets.ndim != 1:
        raise ValueError("single-articulation targets must be one-dimensional")
    return targets.reshape(1, -1)


def config_from_values(
    *,
    wheel_radius: float,
    track_width: float,
    wheel_signs: Iterable[float],
    max_wheel_speed: float,
    linear_gain: float = 1.0,
    angular_gain: float = 1.0,
) -> PhysxDiffContactConfig:
    signs = tuple(float(value) for value in wheel_signs)
    if len(signs) != 4:
        raise ValueError("wheel_signs must contain exactly four values")
    return PhysxDiffContactConfig(
        wheel_radius=float(wheel_radius),
        track_width=float(track_width),
        wheel_signs=signs,
        max_wheel_speed=float(max_wheel_speed),
        linear_gain=float(linear_gain),
        angular_gain=float(angular_gain),
    )
