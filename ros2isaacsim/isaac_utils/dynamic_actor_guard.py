"""Shared geometry helpers for robot/pedestrian dynamic guard decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Collection, Sequence


@dataclass(frozen=True)
class RobotGuardState:
    name: str
    pos_xy: tuple[float, float]
    heading: float
    next_pos_xy: tuple[float, float]
    next_heading: float
    forward: float
    rear: float
    left: float
    right: float


@dataclass(frozen=True)
class PedestrianGuardState:
    name: str
    pos_xy: tuple[float, float]
    next_pos_xy: tuple[float, float]
    radius: float
    heading: float = 0.0
    next_heading: float = 0.0
    half_length: float = 0.0
    axis_sample_spacing: float = 0.05


@dataclass(frozen=True)
class HardGuardResult:
    vx: float
    vy: float
    wz: float
    scale: float
    mode: str
    blocked_by: str | None = None


def circle_intersects_grid_cells(
    *,
    center_xy: Sequence[float],
    radius: float,
    resolution: float,
    origin_xy: Sequence[float],
    occupied: Collection[tuple[int, int]],
) -> tuple[int, int] | None:
    """Return the first occupied grid square intersecting a circular footprint."""
    x, y = float(center_xy[0]), float(center_xy[1])
    radius = max(0.0, float(radius))
    resolution = float(resolution)
    origin_x, origin_y = float(origin_xy[0]), float(origin_xy[1])
    ix0 = math.floor((x - radius - origin_x) / resolution)
    ix1 = math.floor((x + radius - origin_x) / resolution)
    iy0 = math.floor((y - radius - origin_y) / resolution)
    iy1 = math.floor((y + radius - origin_y) / resolution)
    radius_sq = radius * radius
    for ix in range(ix0, ix1 + 1):
        for iy in range(iy0, iy1 + 1):
            if (ix, iy) not in occupied:
                continue
            cell_min_x = origin_x + ix * resolution
            cell_max_x = cell_min_x + resolution
            cell_min_y = origin_y + iy * resolution
            cell_max_y = cell_min_y + resolution
            nearest_x = min(max(x, cell_min_x), cell_max_x)
            nearest_y = min(max(y, cell_min_y), cell_max_y)
            if (x - nearest_x) ** 2 + (y - nearest_y) ** 2 <= radius_sq:
                return ix, iy
    return None


def footprint_extents(
    *,
    length: float,
    width: float,
    margin: float = 0.0,
    footprint_forward: float = -1.0,
    footprint_rear: float = -1.0,
    footprint_left: float = -1.0,
    footprint_right: float = -1.0,
) -> tuple[float, float, float, float]:
    forward = footprint_forward if footprint_forward > 0.0 else (0.5 * float(length) + float(margin))
    rear = footprint_rear if footprint_rear > 0.0 else (0.5 * float(length) + float(margin))
    left = footprint_left if footprint_left > 0.0 else (0.5 * float(width) + float(margin))
    right = footprint_right if footprint_right > 0.0 else (0.5 * float(width) + float(margin))
    return float(forward), float(rear), float(left), float(right)


def predict_robot_pose(
    *,
    pos_xy: Sequence[float],
    heading: float,
    vx: float,
    vy: float,
    wz: float,
    dt: float,
) -> tuple[tuple[float, float], float]:
    next_heading = float(heading) + float(wz) * float(dt)
    c = math.cos(next_heading)
    s = math.sin(next_heading)
    wx = c * float(vx) - s * float(vy)
    wy = s * float(vx) + c * float(vy)
    return (
        (float(pos_xy[0]) + wx * float(dt), float(pos_xy[1]) + wy * float(dt)),
        next_heading,
    )


def predict_pedestrian_step(
    *,
    pos_xy: Sequence[float],
    goal_xy: Sequence[float] | None,
    speed: float,
    dt: float,
) -> tuple[float, float]:
    if goal_xy is None or speed <= 0.0 or dt <= 0.0:
        return float(pos_xy[0]), float(pos_xy[1])
    dx = float(goal_xy[0]) - float(pos_xy[0])
    dy = float(goal_xy[1]) - float(pos_xy[1])
    dist = math.hypot(dx, dy)
    if dist <= 1e-6:
        return float(pos_xy[0]), float(pos_xy[1])
    step = min(float(speed) * float(dt), dist)
    scale = step / dist
    return float(pos_xy[0]) + dx * scale, float(pos_xy[1]) + dy * scale


def movement_allowed(current_score: float, next_score: float, *, escape_epsilon: float = 1e-4) -> bool:
    if current_score <= 0.0:
        return next_score <= 0.0
    return next_score + float(escape_epsilon) < current_score


def robot_pedestrian_scores(robot: RobotGuardState, ped: PedestrianGuardState) -> tuple[float, float]:
    current_score = _pedestrian_footprint_penetration(
        robot.pos_xy, robot.heading, robot, ped, ped.pos_xy, ped.heading
    )
    next_score = _swept_pair_penetration(robot, ped)
    return current_score, next_score


def pedestrian_robot_scores(ped: PedestrianGuardState, robot: RobotGuardState) -> tuple[float, float]:
    current_score = _pedestrian_footprint_penetration(
        robot.pos_xy, robot.heading, robot, ped, ped.pos_xy, ped.heading
    )
    next_score = _swept_pair_penetration(robot, ped)
    return current_score, next_score


def current_pedestrian_contacts(
    robot: RobotGuardState,
    pedestrians: Sequence[PedestrianGuardState],
) -> dict[str, float]:
    """Return only current zero-margin overlaps, without predicting or clipping."""

    contacts: dict[str, float] = {}
    for pedestrian in pedestrians:
        current_score, _ = robot_pedestrian_scores(robot, pedestrian)
        if current_score > 0.0:
            contacts[str(pedestrian.name)] = float(current_score)
    return contacts


def _swept_pair_penetration(
    robot: RobotGuardState,
    pedestrian: PedestrianGuardState,
) -> float:
    """Sample both trajectories so crossing paths cannot tunnel between endpoints."""
    robot_distance = math.hypot(
        float(robot.next_pos_xy[0]) - float(robot.pos_xy[0]),
        float(robot.next_pos_xy[1]) - float(robot.pos_xy[1]),
    )
    pedestrian_distance = math.hypot(
        float(pedestrian.next_pos_xy[0]) - float(pedestrian.pos_xy[0]),
        float(pedestrian.next_pos_xy[1]) - float(pedestrian.pos_xy[1]),
    )
    heading_delta = math.atan2(
        math.sin(float(robot.next_heading) - float(robot.heading)),
        math.cos(float(robot.next_heading) - float(robot.heading)),
    )
    pedestrian_heading_delta = math.atan2(
        math.sin(float(pedestrian.next_heading) - float(pedestrian.heading)),
        math.cos(float(pedestrian.next_heading) - float(pedestrian.heading)),
    )
    steps = max(
        1,
        int(math.ceil(max(robot_distance, pedestrian_distance) / 0.03)),
        int(math.ceil(abs(heading_delta) / math.radians(3.0))),
        int(math.ceil(abs(pedestrian_heading_delta) / math.radians(3.0))),
    )
    worst_score = 0.0
    for step in range(1, steps + 1):
        progress = step / steps
        robot_xy = (
            float(robot.pos_xy[0])
            + progress * (float(robot.next_pos_xy[0]) - float(robot.pos_xy[0])),
            float(robot.pos_xy[1])
            + progress * (float(robot.next_pos_xy[1]) - float(robot.pos_xy[1])),
        )
        pedestrian_xy = (
            float(pedestrian.pos_xy[0])
            + progress
            * (float(pedestrian.next_pos_xy[0]) - float(pedestrian.pos_xy[0])),
            float(pedestrian.pos_xy[1])
            + progress
            * (float(pedestrian.next_pos_xy[1]) - float(pedestrian.pos_xy[1])),
        )
        score = _pedestrian_footprint_penetration(
            robot_xy,
            float(robot.heading) + progress * heading_delta,
            robot,
            pedestrian,
            pedestrian_xy,
            float(pedestrian.heading) + progress * pedestrian_heading_delta,
        )
        worst_score = max(worst_score, score)
    return worst_score


def scale_robot_command_for_pedestrians(
    *,
    robot: RobotGuardState,
    pedestrians: Sequence[PedestrianGuardState],
    vx: float,
    vy: float,
    wz: float,
    horizon_sec: float,
    sample_dt_sec: float,
    margin_m: float = 0.0,
    binary_iterations: int = 10,
    escape_epsilon: float = 1e-4,
    overlap_escape_horizon_sec: float = 0.0,
    overlap_deadband_m: float = 0.0,
) -> HardGuardResult:
    """Scale a body-frame Twist only enough to avoid swept geometric overlap."""
    if not pedestrians:
        return HardGuardResult(float(vx), float(vy), float(wz), 1.0, "clear")
    physical_scores = [
        _pedestrian_footprint_penetration(
            robot.pos_xy,
            robot.heading,
            robot,
            pedestrian,
            pedestrian.pos_xy,
            pedestrian.heading,
        )
        for pedestrian in pedestrians
    ]
    max_current = max(physical_scores, default=0.0)
    if max_current > 0.0:
        escape_horizon = float(overlap_escape_horizon_sec)
        if escape_horizon <= 0.0:
            escape_horizon = min(
                max(float(sample_dt_sec), 0.05),
                max(float(horizon_sec), 0.05),
            )
        escape_candidates = [
            (float(vx), float(vy), float(wz)),
            (float(vx), float(vy), 0.0),
            (0.0, 0.0, float(wz)),
        ]
        blocked_by = None
        for candidate_vx, candidate_vy, candidate_wz in escape_candidates:
            if (
                abs(candidate_vx) <= 1e-9
                and abs(candidate_vy) <= 1e-9
                and abs(candidate_wz) <= 1e-9
            ):
                continue
            safe, candidate_blocked_by, final_score = _command_is_safe(
                robot=robot,
                pedestrians=pedestrians,
                vx=candidate_vx,
                vy=candidate_vy,
                wz=candidate_wz,
                horizon_sec=escape_horizon,
                sample_dt_sec=float(sample_dt_sec),
                margin_m=0.0,
                max_allowed_penetration=(
                    max_current + max(0.0, float(overlap_deadband_m))
                ),
            )
            blocked_by = candidate_blocked_by or blocked_by
            if safe and final_score + float(escape_epsilon) < max_current:
                return HardGuardResult(
                    candidate_vx,
                    candidate_vy,
                    candidate_wz,
                    1.0,
                    "escape",
                )
        return HardGuardResult(0.0, 0.0, 0.0, 0.0, "overlap_stop", blocked_by)

    inflated_scores = [
        _pedestrian_footprint_penetration(
            robot.pos_xy,
            robot.heading,
            robot,
            pedestrian,
            pedestrian.pos_xy,
            pedestrian.heading,
            radius=float(pedestrian.radius) + max(0.0, float(margin_m)),
        )
        for pedestrian in pedestrians
    ]
    max_inflated_current = max(inflated_scores, default=0.0)
    if max_inflated_current > 0.0:
        margin_candidates = [
            (float(vx), float(vy), float(wz)),
            (float(vx), float(vy), 0.0),
            (0.0, 0.0, float(wz)),
        ]
        blocked_by = None
        for candidate_vx, candidate_vy, candidate_wz in margin_candidates:
            if (
                abs(candidate_vx) <= 1e-9
                and abs(candidate_vy) <= 1e-9
                and abs(candidate_wz) <= 1e-9
            ):
                continue
            safe, candidate_blocked_by, final_score = _command_is_safe(
                robot=robot,
                pedestrians=pedestrians,
                vx=candidate_vx,
                vy=candidate_vy,
                wz=candidate_wz,
                horizon_sec=float(horizon_sec),
                sample_dt_sec=float(sample_dt_sec),
                margin_m=float(margin_m),
                max_allowed_penetration=(
                    max_inflated_current + max(0.0, float(overlap_deadband_m))
                ),
            )
            blocked_by = candidate_blocked_by or blocked_by
            if safe and final_score <= max_inflated_current + float(escape_epsilon):
                return HardGuardResult(
                    candidate_vx,
                    candidate_vy,
                    candidate_wz,
                    1.0,
                    "margin_escape",
                )
        return HardGuardResult(0.0, 0.0, 0.0, 0.0, "margin_stop", blocked_by)

    safe, blocked_by, _ = _command_is_safe(
        robot=robot,
        pedestrians=pedestrians,
        vx=float(vx),
        vy=float(vy),
        wz=float(wz),
        horizon_sec=float(horizon_sec),
        sample_dt_sec=float(sample_dt_sec),
        margin_m=float(margin_m),
    )
    if safe:
        return HardGuardResult(float(vx), float(vy), float(wz), 1.0, "clear")

    translation_safe, _, _ = _command_is_safe(
        robot=robot,
        pedestrians=pedestrians,
        vx=float(vx),
        vy=float(vy),
        wz=0.0,
        horizon_sec=float(horizon_sec),
        sample_dt_sec=float(sample_dt_sec),
        margin_m=float(margin_m),
    )
    if translation_safe and abs(float(wz)) > 1e-9:
        angular_scale = _largest_safe_component_scale(
            robot=robot,
            pedestrians=pedestrians,
            base_vx=float(vx),
            base_vy=float(vy),
            base_wz=0.0,
            scaled_vx=0.0,
            scaled_vy=0.0,
            scaled_wz=float(wz),
            horizon_sec=float(horizon_sec),
            sample_dt_sec=float(sample_dt_sec),
            margin_m=float(margin_m),
            binary_iterations=binary_iterations,
        )
        return HardGuardResult(
            float(vx),
            float(vy),
            float(wz) * angular_scale,
            angular_scale,
            "angular_scaled",
            blocked_by,
        )

    rotation_safe, _, _ = _command_is_safe(
        robot=robot,
        pedestrians=pedestrians,
        vx=0.0,
        vy=0.0,
        wz=float(wz),
        horizon_sec=float(horizon_sec),
        sample_dt_sec=float(sample_dt_sec),
        margin_m=float(margin_m),
    )
    if rotation_safe and math.hypot(float(vx), float(vy)) > 1e-9:
        linear_scale = _largest_safe_component_scale(
            robot=robot,
            pedestrians=pedestrians,
            base_vx=0.0,
            base_vy=0.0,
            base_wz=float(wz),
            scaled_vx=float(vx),
            scaled_vy=float(vy),
            scaled_wz=0.0,
            horizon_sec=float(horizon_sec),
            sample_dt_sec=float(sample_dt_sec),
            margin_m=float(margin_m),
            binary_iterations=binary_iterations,
        )
        return HardGuardResult(
            float(vx) * linear_scale,
            float(vy) * linear_scale,
            float(wz),
            linear_scale,
            "linear_scaled",
            blocked_by,
        )

    low = 0.0
    high = 1.0
    for _ in range(max(1, int(binary_iterations))):
        scale = 0.5 * (low + high)
        candidate_safe, _, _ = _command_is_safe(
            robot=robot,
            pedestrians=pedestrians,
            vx=float(vx) * scale,
            vy=float(vy) * scale,
            wz=float(wz) * scale,
            horizon_sec=float(horizon_sec),
            sample_dt_sec=float(sample_dt_sec),
            margin_m=float(margin_m),
        )
        if candidate_safe:
            low = scale
        else:
            high = scale
    if low <= 1e-3:
        return HardGuardResult(0.0, 0.0, 0.0, 0.0, "blocked", blocked_by)
    return HardGuardResult(
        float(vx) * low,
        float(vy) * low,
        float(wz) * low,
        low,
        "scaled",
        blocked_by,
    )


def _largest_safe_component_scale(
    *,
    robot: RobotGuardState,
    pedestrians: Sequence[PedestrianGuardState],
    base_vx: float,
    base_vy: float,
    base_wz: float,
    scaled_vx: float,
    scaled_vy: float,
    scaled_wz: float,
    horizon_sec: float,
    sample_dt_sec: float,
    margin_m: float,
    binary_iterations: int,
) -> float:
    low = 0.0
    high = 1.0
    for _ in range(max(1, int(binary_iterations))):
        scale = 0.5 * (low + high)
        safe, _, _ = _command_is_safe(
            robot=robot,
            pedestrians=pedestrians,
            vx=float(base_vx) + float(scaled_vx) * scale,
            vy=float(base_vy) + float(scaled_vy) * scale,
            wz=float(base_wz) + float(scaled_wz) * scale,
            horizon_sec=float(horizon_sec),
            sample_dt_sec=float(sample_dt_sec),
            margin_m=float(margin_m),
        )
        if safe:
            low = scale
        else:
            high = scale
    return low


def _command_is_safe(
    *,
    robot: RobotGuardState,
    pedestrians: Sequence[PedestrianGuardState],
    vx: float,
    vy: float,
    wz: float,
    horizon_sec: float,
    sample_dt_sec: float,
    margin_m: float,
    max_allowed_penetration: float = 0.0,
) -> tuple[bool, str | None, float]:
    horizon = max(0.0, float(horizon_sec))
    sample_dt = max(1e-3, float(sample_dt_sec))
    steps = max(1, int(math.ceil(horizon / sample_dt)))
    dt = horizon / steps if horizon > 0.0 else sample_dt
    x, y = float(robot.pos_xy[0]), float(robot.pos_xy[1])
    heading = float(robot.heading)
    worst_score = 0.0
    blocked_by = None
    for step in range(1, steps + 1):
        mid_heading = heading + 0.5 * float(wz) * dt
        c = math.cos(mid_heading)
        s = math.sin(mid_heading)
        x += (c * float(vx) - s * float(vy)) * dt
        y += (s * float(vx) + c * float(vy)) * dt
        heading += float(wz) * dt
        progress = step / steps
        for pedestrian in pedestrians:
            pedestrian_x = float(pedestrian.pos_xy[0]) + progress * (
                float(pedestrian.next_pos_xy[0]) - float(pedestrian.pos_xy[0])
            )
            pedestrian_y = float(pedestrian.pos_xy[1]) + progress * (
                float(pedestrian.next_pos_xy[1]) - float(pedestrian.pos_xy[1])
            )
            pedestrian_heading_delta = math.atan2(
                math.sin(float(pedestrian.next_heading) - float(pedestrian.heading)),
                math.cos(float(pedestrian.next_heading) - float(pedestrian.heading)),
            )
            score = _pedestrian_footprint_penetration(
                (x, y),
                heading,
                robot,
                pedestrian,
                (pedestrian_x, pedestrian_y),
                float(pedestrian.heading) + progress * pedestrian_heading_delta,
                radius=float(pedestrian.radius) + max(0.0, float(margin_m)),
            )
            if score > worst_score:
                worst_score = score
                blocked_by = pedestrian.name
            if score > float(max_allowed_penetration) + 1e-9:
                return False, blocked_by, worst_score
    final_score = max(
        _pedestrian_footprint_penetration(
            (x, y),
            heading,
            robot,
            pedestrian,
            pedestrian.next_pos_xy,
            pedestrian.next_heading,
            radius=float(pedestrian.radius) + max(0.0, float(margin_m)),
        )
        for pedestrian in pedestrians
    )
    return True, blocked_by, final_score


def _circle_footprint_penetration(
    robot_pos_xy: Sequence[float],
    robot_heading: float,
    robot: RobotGuardState,
    circle_xy: Sequence[float],
    radius: float,
) -> float:
    center_x_local = 0.5 * (float(robot.forward) - float(robot.rear))
    center_y_local = 0.5 * (float(robot.left) - float(robot.right))
    half_x = 0.5 * (float(robot.forward) + float(robot.rear))
    half_y = 0.5 * (float(robot.left) + float(robot.right))
    c = math.cos(float(robot_heading))
    s = math.sin(float(robot_heading))
    box_cx = float(robot_pos_xy[0]) + c * center_x_local - s * center_y_local
    box_cy = float(robot_pos_xy[1]) + s * center_x_local + c * center_y_local
    return circle_obb_penetration(
        circle_xy=circle_xy,
        radius=radius,
        box_center_xy=(box_cx, box_cy),
        box_yaw=float(robot_heading),
        box_half_x=half_x,
        box_half_y=half_y,
    )


def _pedestrian_footprint_penetration(
    robot_pos_xy: Sequence[float],
    robot_heading: float,
    robot: RobotGuardState,
    pedestrian: PedestrianGuardState,
    pedestrian_pos_xy: Sequence[float],
    pedestrian_heading: float,
    *,
    radius: float | None = None,
) -> float:
    """Return robot-box penetration against the pedestrian's oriented capsule."""
    half_length = max(0.0, float(pedestrian.half_length))
    spacing = max(0.01, float(pedestrian.axis_sample_spacing))
    if half_length <= 1e-9:
        offsets = (0.0,)
    else:
        steps = max(2, int(math.ceil((2.0 * half_length) / spacing)))
        offsets = tuple(
            -half_length + (2.0 * half_length * index / steps)
            for index in range(steps + 1)
        )
    c = math.cos(float(pedestrian_heading))
    s = math.sin(float(pedestrian_heading))
    effective_radius = float(pedestrian.radius if radius is None else radius)
    return max(
        _circle_footprint_penetration(
            robot_pos_xy,
            robot_heading,
            robot,
            (
                float(pedestrian_pos_xy[0]) + offset * c,
                float(pedestrian_pos_xy[1]) + offset * s,
            ),
            effective_radius,
        )
        for offset in offsets
    )


def circle_obb_penetration(
    *,
    circle_xy: Sequence[float],
    radius: float,
    box_center_xy: Sequence[float],
    box_yaw: float,
    box_half_x: float,
    box_half_y: float,
) -> float:
    dx = float(circle_xy[0]) - float(box_center_xy[0])
    dy = float(circle_xy[1]) - float(box_center_xy[1])
    c = math.cos(float(box_yaw))
    s = math.sin(float(box_yaw))
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    clamped_x = max(-float(box_half_x), min(float(box_half_x), local_x))
    clamped_y = max(-float(box_half_y), min(float(box_half_y), local_y))
    sep_x = local_x - clamped_x
    sep_y = local_y - clamped_y
    sep = math.hypot(sep_x, sep_y)
    if abs(local_x) <= float(box_half_x) and abs(local_y) <= float(box_half_y):
        inside_x = float(box_half_x) - abs(local_x)
        inside_y = float(box_half_y) - abs(local_y)
        return float(radius) + min(inside_x, inside_y)
    if sep >= float(radius):
        return 0.0
    return float(radius) - sep
