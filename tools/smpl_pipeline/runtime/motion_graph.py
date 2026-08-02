"""Contact-aware transition selection for the offline locomotion graph."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from ..analysis.rotations import axis_angle_to_matrix
except ImportError:
    from analysis.rotations import axis_angle_to_matrix


@dataclass(frozen=True)
class MotionClip:
    motion_id: str
    poses: np.ndarray
    root_trajectory: np.ndarray
    source_frames: np.ndarray
    foot_contacts: np.ndarray
    joint_positions: np.ndarray
    fps: float
    betas: np.ndarray
    gender: str
    event_start_sec: float = 0.0
    event_end_sec: float = float("inf")
    gait_phase: np.ndarray | None = None

    @classmethod
    def load(cls, motion_id: str, path: Path) -> "MotionClip":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                motion_id=motion_id,
                poses=np.asarray(data["poses"], dtype=np.float64),
                root_trajectory=np.asarray(
                    data["root_trajectory"], dtype=np.float64
                ),
                source_frames=np.asarray(data["source_frames"], dtype=np.int64),
                foot_contacts=np.asarray(data["foot_contacts"], dtype=bool),
                joint_positions=np.asarray(
                    data["joint_positions"], dtype=np.float64
                ),
                fps=float(np.asarray(data["fps"]).item()),
                betas=np.asarray(data["betas"], dtype=np.float64),
                gender=str(np.asarray(data["gender"]).item()),
                event_start_sec=float(
                    np.asarray(data["event_start_sec"]).item()
                )
                if "event_start_sec" in data
                else 0.0,
                event_end_sec=float(
                    np.asarray(data["event_end_sec"]).item()
                )
                if "event_end_sec" in data
                else float("inf"),
                gait_phase=np.asarray(data["gait_phase"], dtype=np.float64)
                if "gait_phase" in data
                else None,
            )

    def planar_speed(self) -> np.ndarray:
        return np.linalg.norm(
            np.gradient(self.root_trajectory[:, :2], 1.0 / self.fps, axis=0),
            axis=1,
        )


@dataclass(frozen=True)
class TransitionMatch:
    source_index: int
    target_index: int
    pose_cost: float
    speed_cost: float
    contact_cost: float
    total_cost: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "source_index": self.source_index,
            "target_index": self.target_index,
            "pose_cost": self.pose_cost,
            "speed_cost": self.speed_cost,
            "contact_cost": self.contact_cost,
            "total_cost": self.total_cost,
        }


def select_transition(
    source: MotionClip,
    target: MotionClip,
    *,
    source_start_index: int,
    source_end_index: int | None = None,
    target_end_index: int,
) -> TransitionMatch:
    """Select a compatible tail/head frame pair without changing action order."""
    source_rotations = axis_angle_to_matrix(source.poses[:, :66].reshape(-1, 22, 3))
    target_rotations = axis_angle_to_matrix(target.poses[:, :66].reshape(-1, 22, 3))
    source_speed = source.planar_speed()
    target_speed = target.planar_speed()
    best: TransitionMatch | None = None
    source_end = len(source.poses) if source_end_index is None else source_end_index
    for source_index in range(source_start_index, min(source_end, len(source.poses))):
        for target_index in range(0, min(target_end_index, len(target.poses))):
            relative = (
                source_rotations[source_index, 1:]
                - target_rotations[target_index, 1:]
            )
            pose_cost = float(np.sqrt(np.mean(relative * relative)))
            speed_cost = float(
                abs(source_speed[source_index] - target_speed[target_index])
            )
            contact_cost = float(
                np.mean(
                    source.foot_contacts[source_index]
                    != target.foot_contacts[target_index]
                )
            )
            total = pose_cost + 0.6 * speed_cost + 0.35 * contact_cost
            candidate = TransitionMatch(
                source_index,
                target_index,
                pose_cost,
                speed_cost,
                contact_cost,
                total,
            )
            if best is None or candidate.total_cost < best.total_cost:
                best = candidate
    if best is None:
        raise ValueError("transition search received an empty candidate range")
    return best


def find_shared_source_frame(source: MotionClip, target: MotionClip) -> tuple[int, int]:
    """Find the latest source frame represented by both clips."""
    target_lookup = {int(frame): index for index, frame in enumerate(target.source_frames)}
    for source_index in range(len(source.source_frames) - 1, -1, -1):
        frame = int(source.source_frames[source_index])
        if frame in target_lookup:
            return source_index, target_lookup[frame]
    raise ValueError(
        f"{source.motion_id} and {target.motion_id} have no shared source frame"
    )
