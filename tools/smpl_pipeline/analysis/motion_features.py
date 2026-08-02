"""Motion-window scoring for selecting stable AMASS walking clips."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

try:
    from .rotations import planar_heading
except ImportError:
    from rotations import planar_heading


@dataclass(frozen=True)
class MotionWindow:
    start_frame: int
    end_frame_exclusive: int
    start_sec: float
    end_sec: float
    median_speed_mps: float
    moving_fraction: float
    displacement_m: float
    median_forward_alignment: float
    yaw_change_rad: float
    score: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class TurnWindow:
    start_frame: int
    end_frame_exclusive: int
    start_sec: float
    end_sec: float
    yaw_change_deg: float
    displacement_m: float
    path_length_m: float
    median_speed_mps: float
    median_forward_alignment: float
    score: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def rank_stable_walk_windows(
    trans: np.ndarray,
    poses: np.ndarray,
    fps: float,
    *,
    window_sec: float = 2.0,
    hop_sec: float = 0.25,
) -> list[MotionWindow]:
    """Rank straight walking windows using aggregate, not per-frame, gates."""
    trans = np.asarray(trans, dtype=np.float64)
    poses = np.asarray(poses, dtype=np.float64)
    frames = max(2, int(round(window_sec * fps)))
    hop = max(1, int(round(hop_sec * fps)))
    velocity = np.gradient(trans[:, :2], 1.0 / fps, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    heading = planar_heading(poses[:, :3])
    forward = np.column_stack((np.cos(heading), np.sin(heading)))
    direction = np.divide(
        velocity,
        speed[:, None],
        out=np.zeros_like(velocity),
        where=speed[:, None] > 0.15,
    )
    alignment = np.sum(forward * direction, axis=1)

    candidates: list[MotionWindow] = []
    for start in range(0, len(trans) - frames + 1, hop):
        end = start + frames
        local_speed = speed[start:end]
        moving = local_speed > 0.2
        moving_fraction = float(np.mean(moving))
        median_speed = float(np.median(local_speed))
        displacement = float(np.linalg.norm(trans[end - 1, :2] - trans[start, :2]))
        aligned = alignment[start:end][moving]
        median_alignment = float(np.median(aligned)) if aligned.size else -1.0
        yaw_change = float(abs(heading[end - 1] - heading[start]))

        if not (
            0.35 <= median_speed <= 1.6
            and moving_fraction >= 0.75
            and displacement >= 1.0
            and median_alignment >= 0.70
            and yaw_change <= 0.70
        ):
            continue
        score = (
            displacement
            + 0.5 * moving_fraction
            + 0.5 * median_alignment
            - 0.25 * yaw_change
            - 0.1 * abs(median_speed - 1.1)
        )
        candidates.append(
            MotionWindow(
                start_frame=start,
                end_frame_exclusive=end,
                start_sec=start / fps,
                end_sec=(end - 1) / fps,
                median_speed_mps=median_speed,
                moving_fraction=moving_fraction,
                displacement_m=displacement,
                median_forward_alignment=median_alignment,
                yaw_change_rad=yaw_change,
                score=float(score),
            )
        )
    return sorted(candidates, key=lambda item: item.score, reverse=True)


def rank_natural_turn_windows(
    trans: np.ndarray,
    poses: np.ndarray,
    fps: float,
    *,
    target_degrees: float = 90.0,
) -> list[TurnWindow]:
    """Rank moving turns that preserve alignment between body and trajectory."""
    trans = np.asarray(trans, dtype=np.float64)
    poses = np.asarray(poses, dtype=np.float64)
    velocity = np.gradient(trans[:, :2], 1.0 / fps, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    heading = planar_heading(poses[:, :3])
    forward = np.column_stack((np.cos(heading), np.sin(heading)))
    direction = np.divide(
        velocity,
        speed[:, None],
        out=np.zeros_like(velocity),
        where=speed[:, None] > 0.2,
    )
    alignment = np.sum(forward * direction, axis=1)

    candidates: list[TurnWindow] = []
    hop = max(1, int(round(0.05 * fps)))
    for duration_sec in np.arange(1.0, 3.01, 0.25):
        window = int(round(duration_sec * fps))
        for start in range(0, len(trans) - window, hop):
            end = start + window
            yaw_change = float(np.degrees(heading[end] - heading[start]))
            if not 75.0 <= abs(yaw_change) <= 105.0:
                continue
            displacement = float(
                np.linalg.norm(trans[end, :2] - trans[start, :2])
            )
            path_length = float(
                np.linalg.norm(
                    np.diff(trans[start : end + 1, :2], axis=0), axis=1
                ).sum()
            )
            moving = speed[start : end + 1] > 0.2
            aligned = alignment[start : end + 1][moving]
            median_alignment = float(np.median(aligned)) if aligned.size else -1.0
            median_speed = float(np.median(speed[start:end]))
            if not (
                0.5 <= displacement <= 2.0
                and path_length <= 2.2
                and median_alignment >= 0.75
                and 0.35 <= median_speed <= 1.8
            ):
                continue
            score = (
                abs(abs(yaw_change) - target_degrees) / 20.0
                + abs(displacement - 1.1) * 0.2
                + (1.0 - median_alignment)
                + max(0.0, path_length - 1.5) * 0.1
            )
            candidates.append(
                TurnWindow(
                    start_frame=start,
                    end_frame_exclusive=end + 1,
                    start_sec=start / fps,
                    end_sec=end / fps,
                    yaw_change_deg=yaw_change,
                    displacement_m=displacement,
                    path_length_m=path_length,
                    median_speed_mps=median_speed,
                    median_forward_alignment=median_alignment,
                    score=float(score),
                )
            )
    return sorted(candidates, key=lambda item: item.score)
