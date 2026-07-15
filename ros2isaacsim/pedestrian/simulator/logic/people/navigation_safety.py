"""Pure helpers for bridge-side People navigation safety decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class LateralAvoidanceCandidate:
    """Candidate lateral avoidance target and whether it is safe to take."""

    label: str
    point_xyz: tuple[float, float, float]
    is_safe: bool


def path_target_progress_radius(
    *,
    target_count: int,
    constrain_to_path: bool,
    constrained_intermediate_radius: float,
    default_intermediate_radius: float,
    final_radius: float,
) -> float:
    if int(target_count) <= 1:
        return max(0.01, float(final_radius))
    if constrain_to_path:
        return max(0.01, float(constrained_intermediate_radius))
    return max(0.01, float(default_intermediate_radius))


def terminal_semantic_approach_radius(
    *,
    constrain_to_path: bool,
    default_radius: float,
    final_radius: float,
) -> float:
    """Return the root-pose acceptance radius around the final target.

    Constrained portal motion must not inherit the broad semantic stop radius,
    but it still needs a small terminal window so inflated voxel occupancy at a
    valid doorway target cannot leave the reported pose permanently behind.
    """

    radius = float(default_radius)
    if constrain_to_path:
        radius = min(radius, float(final_radius))
    return max(0.01, radius)


def select_safe_lateral_avoidance_choice(
    *,
    direction_of_collision: float,
    left: LateralAvoidanceCandidate,
    right: LateralAvoidanceCandidate,
    direction_threshold: float = 0.2,
) -> LateralAvoidanceCandidate | None:
    """Choose a safe lateral target or wait when both sides are unsafe.

    The helper keeps Isaac's directional preference when both candidates are
    safe, but it refuses to steer into a static obstacle. When only one side is
    safe, it is chosen even if the directional heuristic would prefer the other
    side.
    """

    left_safe = bool(left.is_safe)
    right_safe = bool(right.is_safe)
    if left_safe != right_safe:
        return left if left_safe else right
    if not left_safe:
        return None
    if direction_of_collision > float(direction_threshold):
        return left
    if direction_of_collision < -float(direction_threshold):
        return right
    # A deterministic side prevents two head-on agents from waiting forever.
    return left


def segment_is_safe(
    *,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    is_safe_point: Callable[[tuple[float, float]], bool],
    sample_step: float,
) -> bool:
    """Return True when every sampled point along the segment is safe.

    The helper is intentionally generic so tests can use a fake predicate while
    the Person wrapper feeds it the voxel guard check.
    """

    sx, sy = float(start_xy[0]), float(start_xy[1])
    ex, ey = float(end_xy[0]), float(end_xy[1])
    dx = ex - sx
    dy = ey - sy
    dist = math.hypot(dx, dy)
    step = max(float(sample_step), 1e-6)
    samples = max(1, int(math.ceil(dist / step)))
    for idx in range(samples + 1):
        t = idx / samples
        point = (sx + dx * t, sy + dy * t)
        if not bool(is_safe_point(point)):
            return False
    return True
