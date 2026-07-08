"""Shared geometry helpers for robot/pedestrian dynamic guard decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


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
    current_score = _circle_footprint_penetration(robot.pos_xy, robot.heading, robot, ped.pos_xy, ped.radius)
    next_score = max(
        _circle_footprint_penetration(robot.next_pos_xy, robot.next_heading, robot, ped.pos_xy, ped.radius),
        _circle_footprint_penetration(robot.next_pos_xy, robot.next_heading, robot, ped.next_pos_xy, ped.radius),
    )
    return current_score, next_score


def pedestrian_robot_scores(ped: PedestrianGuardState, robot: RobotGuardState) -> tuple[float, float]:
    current_score = _circle_footprint_penetration(robot.pos_xy, robot.heading, robot, ped.pos_xy, ped.radius)
    next_score = max(
        _circle_footprint_penetration(robot.pos_xy, robot.heading, robot, ped.next_pos_xy, ped.radius),
        _circle_footprint_penetration(robot.next_pos_xy, robot.next_heading, robot, ped.next_pos_xy, ped.radius),
    )
    return current_score, next_score


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
