"""Foot-contact diagnostics shared by offline SMPL-H preview builders."""

from __future__ import annotations

import numpy as np


def foot_contact_diagnostics(
    joints: np.ndarray,
    fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate left/right contact from SMPL-H foot height and world speed."""
    feet = np.asarray(joints, dtype=np.float64)[:, [10, 11]]
    speed = np.linalg.norm(np.gradient(feet, 1.0 / fps, axis=0), axis=2)
    height = feet[..., 2] - feet[..., 2].min(axis=0)
    contacts = (height < 0.045) & (speed < 0.35)
    return contacts, speed


def stable_contact_frame(
    joints: np.ndarray,
    fps: float,
    *,
    start_frame: int = 0,
) -> int:
    contacts, speed = foot_contact_diagnostics(joints, fps)
    feet = np.asarray(joints, dtype=np.float64)[:, [10, 11]]
    height = feet[..., 2] - feet[..., 2].min(axis=0)
    score = height.sum(axis=1) + 0.08 * speed.sum(axis=1)
    contact_bonus = contacts.any(axis=1).astype(np.float64) * 0.1
    score -= contact_bonus
    return int(np.argmin(score[start_frame:]) + start_frame)


def upright_stance_frame(
    joints: np.ndarray,
    root_trajectory: np.ndarray,
    fps: float,
    *,
    start_frame: int = 0,
) -> tuple[int, dict[str, float | bool]]:
    """Select a quiet, upright frame instead of merely a planted foot frame."""
    joints = np.asarray(joints, dtype=np.float64)
    root_trajectory = np.asarray(root_trajectory, dtype=np.float64)
    contacts, foot_speed = foot_contact_diagnostics(joints, fps)
    root_speed = np.linalg.norm(
        np.gradient(root_trajectory[:, :2], 1.0 / fps, axis=0), axis=1
    )

    def knee_flexion(hip: int, knee: int, ankle: int) -> np.ndarray:
        upper = joints[:, hip] - joints[:, knee]
        lower = joints[:, ankle] - joints[:, knee]
        cosine = np.sum(upper * lower, axis=1) / np.maximum(
            np.linalg.norm(upper, axis=1) * np.linalg.norm(lower, axis=1),
            1.0e-8,
        )
        return np.pi - np.arccos(np.clip(cosine, -1.0, 1.0))

    knee_flex = 0.5 * (
        knee_flexion(1, 4, 7) + knee_flexion(2, 5, 8)
    )
    torso = joints[:, 12] - joints[:, 0]
    torso_tilt = np.arccos(
        np.clip(
            torso[:, 2] / np.maximum(np.linalg.norm(torso, axis=1), 1.0e-8),
            -1.0,
            1.0,
        )
    )
    both_feet_penalty = (~contacts.all(axis=1)).astype(np.float64)
    score = (
        2.0 * root_speed
        + 0.7 * foot_speed.sum(axis=1)
        + 0.8 * knee_flex
        + 0.6 * torso_tilt
        + 0.4 * both_feet_penalty
    )
    index = int(np.argmin(score[start_frame:]) + start_frame)
    return index, {
        "score": float(score[index]),
        "root_speed_mps": float(root_speed[index]),
        "mean_knee_flex_deg": float(np.degrees(knee_flex[index])),
        "torso_tilt_deg": float(np.degrees(torso_tilt[index])),
        "both_feet_contact": bool(contacts[index].all()),
    }
