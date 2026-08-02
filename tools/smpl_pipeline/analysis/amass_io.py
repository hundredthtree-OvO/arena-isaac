"""Small, simulator-independent helpers for inspecting AMASS motion files."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_KEYS = {
    "trans",
    "poses",
    "betas",
    "gender",
    "mocap_framerate",
}


@dataclass(frozen=True)
class MotionSummary:
    motion_id: str
    relative_path: str
    sha256: str
    gender: str
    fps: float
    frames: int
    duration_sec: float
    pose_dimensions: int
    beta_dimensions: int
    has_dmpls: bool
    path_length_m: float
    net_displacement_m: float
    average_speed_mps: float
    stationary_fraction: float
    vertical_min_m: float
    vertical_max_m: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar_string(value: np.ndarray) -> str:
    return str(value.item() if value.ndim == 0 else value.reshape(-1)[0])


def validate_motion_arrays(data: Any, path: Path) -> None:
    missing = REQUIRED_KEYS.difference(data.files)
    if missing:
        raise ValueError(f"{path}: missing AMASS keys: {sorted(missing)}")

    trans = np.asarray(data["trans"])
    poses = np.asarray(data["poses"])
    betas = np.asarray(data["betas"])
    fps = float(np.asarray(data["mocap_framerate"]).item())

    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"{path}: trans must have shape (frames, 3), got {trans.shape}")
    if poses.ndim != 2 or poses.shape[0] != trans.shape[0]:
        raise ValueError(
            f"{path}: poses must have the same frame count as trans, got {poses.shape}"
        )
    if betas.ndim != 1:
        raise ValueError(f"{path}: betas must be one-dimensional, got {betas.shape}")
    if trans.shape[0] < 2:
        raise ValueError(f"{path}: motion must contain at least two frames")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"{path}: invalid mocap_framerate {fps}")
    if not np.all(np.isfinite(trans)):
        raise ValueError(f"{path}: trans contains non-finite values")
    if not np.all(np.isfinite(poses)):
        raise ValueError(f"{path}: poses contains non-finite values")


def summarize_motion(
    path: Path,
    *,
    dataset_root: Path,
    stationary_speed_mps: float = 0.08,
    include_sha256: bool = True,
) -> MotionSummary:
    path = path.resolve()
    dataset_root = dataset_root.resolve()
    with np.load(path, allow_pickle=False) as data:
        validate_motion_arrays(data, path)
        trans = np.asarray(data["trans"], dtype=np.float64)
        poses = np.asarray(data["poses"])
        betas = np.asarray(data["betas"])
        fps = float(np.asarray(data["mocap_framerate"]).item())
        gender = _scalar_string(np.asarray(data["gender"]))

        planar_steps = np.linalg.norm(np.diff(trans[:, :2], axis=0), axis=1)
        frame_speeds = planar_steps * fps
        duration = (trans.shape[0] - 1) / fps
        path_length = float(planar_steps.sum())
        net_displacement = float(np.linalg.norm(trans[-1, :2] - trans[0, :2]))
        stationary_fraction = float(np.mean(frame_speeds < stationary_speed_mps))

        return MotionSummary(
            motion_id=path.stem.removesuffix("_poses"),
            relative_path=str(path.relative_to(dataset_root)),
            sha256=file_sha256(path) if include_sha256 else "",
            gender=gender,
            fps=fps,
            frames=int(trans.shape[0]),
            duration_sec=float(duration),
            pose_dimensions=int(poses.shape[1]),
            beta_dimensions=int(betas.shape[0]),
            has_dmpls="dmpls" in data.files,
            path_length_m=path_length,
            net_displacement_m=net_displacement,
            average_speed_mps=path_length / duration,
            stationary_fraction=stationary_fraction,
            vertical_min_m=float(trans[:, 2].min()),
            vertical_max_m=float(trans[:, 2].max()),
        )


def extract_clip_arrays(
    path: Path,
    *,
    start_sec: float,
    end_sec: float,
) -> dict[str, np.ndarray]:
    if start_sec < 0.0 or end_sec <= start_sec:
        raise ValueError("clip interval must satisfy 0 <= start_sec < end_sec")

    with np.load(path, allow_pickle=False) as data:
        validate_motion_arrays(data, path)
        fps = float(np.asarray(data["mocap_framerate"]).item())
        frame_count = int(data["trans"].shape[0])
        start_frame = max(0, int(round(start_sec * fps)))
        end_frame = min(frame_count, int(round(end_sec * fps)) + 1)
        if start_frame >= end_frame - 1:
            raise ValueError(
                f"clip interval selects fewer than two frames: {start_frame}:{end_frame}"
            )

        source_trans = np.asarray(data["trans"][start_frame:end_frame], dtype=np.float64)
        root_trajectory = source_trans - source_trans[0]
        local_trans = source_trans.copy()
        local_trans[:, :2] = source_trans[0, :2]

        result = {
            "poses": np.asarray(data["poses"][start_frame:end_frame]),
            "betas": np.asarray(data["betas"]),
            "gender": np.asarray(data["gender"]),
            "mocap_framerate": np.asarray(data["mocap_framerate"]),
            "source_trans": source_trans,
            "root_trajectory": root_trajectory,
            "local_trans": local_trans,
            "source_start_frame": np.asarray(start_frame, dtype=np.int64),
            "source_end_frame_exclusive": np.asarray(end_frame, dtype=np.int64),
        }
        if "dmpls" in data.files:
            result["dmpls"] = np.asarray(data["dmpls"][start_frame:end_frame])
        return result
