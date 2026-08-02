#!/usr/bin/env python3
"""Prepare a continuous AMASS navigation turn for UsdSkel comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from ..analysis.amass_io import file_sha256, validate_motion_arrays
    from ..analysis.gait import foot_contact_diagnostics
    from ..analysis.motion_features import rank_natural_turn_windows
    from ..analysis.rotations import axis_angle_to_matrix, planar_heading, z_rotation
    from .smplh_numpy import (
        load_model,
        pose_corrected_vertices,
        shaped_vertices_and_joints,
        skin_frame,
    )
except ImportError:
    from analysis.amass_io import file_sha256, validate_motion_arrays
    from analysis.gait import foot_contact_diagnostics
    from analysis.motion_features import rank_natural_turn_windows
    from analysis.rotations import axis_angle_to_matrix, planar_heading, z_rotation
    from builders.smplh_numpy import (
        load_model,
        pose_corrected_vertices,
        shaped_vertices_and_joints,
        skin_frame,
    )


DEFAULT_ROOT = Path("/home/stardust/resources/arena_ws/arena_assets/smpl")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", type=Path)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_ROOT / "smplh")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-fps", type=float, default=60.0)
    parser.add_argument("--context-sec", type=float, default=0.75)
    parser.add_argument("--clip-start-sec", type=float)
    parser.add_argument("--clip-end-sec", type=float)
    parser.add_argument("--turn-start-sec", type=float)
    parser.add_argument("--turn-end-sec", type=float)
    args = parser.parse_args()

    motion_path = args.motion.expanduser().resolve()
    with np.load(motion_path, allow_pickle=False) as data:
        validate_motion_arrays(data, motion_path)
        source_fps = float(np.asarray(data["mocap_framerate"]).item())
        poses_all = np.asarray(data["poses"], dtype=np.float64)
        trans_all = np.asarray(data["trans"], dtype=np.float64)
        betas = np.asarray(data["betas"], dtype=np.float64)
        gender = str(np.asarray(data["gender"]).item()).lower()

    explicit_values = (
        args.clip_start_sec,
        args.clip_end_sec,
        args.turn_start_sec,
        args.turn_end_sec,
    )
    explicit = all(value is not None for value in explicit_values)
    if any(value is not None for value in explicit_values) and not explicit:
        raise ValueError(
            "clip-start, clip-end, turn-start and turn-end must be provided together"
        )
    if explicit:
        if not (
            0.0 <= args.clip_start_sec
            <= args.turn_start_sec
            < args.turn_end_sec
            <= args.clip_end_sec
        ):
            raise ValueError("invalid explicit clip/turn interval")
        clip_start = int(round(args.clip_start_sec * source_fps))
        clip_end = min(
            len(poses_all), int(round(args.clip_end_sec * source_fps)) + 1
        )
        turn_start_source = int(round(args.turn_start_sec * source_fps))
        turn_end_source = int(round(args.turn_end_sec * source_fps)) + 1
        selection_metadata = {
            "method": "explicit_official_label",
            "start_sec": args.turn_start_sec,
            "end_sec": args.turn_end_sec,
        }
    else:
        ranked = rank_natural_turn_windows(trans_all, poses_all, source_fps)
        if not ranked:
            raise RuntimeError("no moving 90-degree turn found")
        selected = ranked[0]
        context = int(round(args.context_sec * source_fps))
        clip_start = max(0, selected.start_frame - context)
        clip_end = min(
            len(poses_all), selected.end_frame_exclusive + context
        )
        turn_start_source = selected.start_frame
        turn_end_source = selected.end_frame_exclusive
        selection_metadata = selected.to_dict()
    ratio = source_fps / args.target_fps
    stride = int(round(ratio))
    if stride < 1 or not np.isclose(ratio, stride, atol=1.0e-8):
        raise ValueError("source_fps / target_fps must be an integer")
    indices = np.arange(clip_start, clip_end, stride, dtype=np.int64)
    poses = poses_all[indices, :156]
    trans = trans_all[indices]
    rotations = axis_angle_to_matrix(poses.reshape(len(poses), 52, 3))

    initial_heading = float(planar_heading(poses[:1, :3])[0])
    normalize = z_rotation(-initial_heading)
    rotations[:, 0] = normalize @ rotations[:, 0]
    trajectory = (trans - trans[0]) @ normalize.T

    model_path = args.model_root.expanduser().resolve() / gender / "model.npz"
    model = load_model(model_path)
    shaped, rest_joints = shaped_vertices_and_joints(model, betas)
    posed_points = np.empty((len(rotations), len(shaped), 3), dtype=np.float32)
    joints = np.empty((len(rotations), len(rest_joints), 3), dtype=np.float64)
    minimum_z = np.inf
    for frame in range(len(rotations)):
        posed_points[frame] = pose_corrected_vertices(
            model, shaped, rotations[frame]
        )
        source_vertices, joints[frame] = skin_frame(
            model, shaped, rest_joints, poses[frame], trans[frame]
        )
        minimum_z = min(minimum_z, float(source_vertices[:, 2].min()))
    contacts, foot_speed = foot_contact_diagnostics(joints, args.target_fps)
    trajectory[:, 2] += -minimum_z

    local_translations = np.empty((len(rest_joints), 3), dtype=np.float32)
    local_translations[0] = rest_joints[0]
    for joint in range(1, len(rest_joints)):
        local_translations[joint] = (
            rest_joints[joint] - rest_joints[int(model.parents[joint])]
        )
    selected_start = (turn_start_source - clip_start) // stride
    selected_end = (turn_end_source - clip_start) // stride
    phase = np.full(len(indices), "walk_out")
    phase[:selected_start] = "walk_in"
    phase[selected_start:selected_end] = "natural_turn"

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        posed_points=posed_points,
        shaped_points=shaped.astype(np.float32),
        faces=model.faces,
        weights=model.weights.astype(np.float32),
        parents=model.parents,
        rest_joints=rest_joints.astype(np.float32),
        local_translations=local_translations,
        joint_rotations=rotations.astype(np.float32),
        root_trajectory=trajectory.astype(np.float32),
        phase=phase,
        source_frame=indices,
        source_foot_contacts=contacts.astype(np.uint8),
        source_foot_speed=foot_speed.astype(np.float32),
        fps=np.asarray(args.target_fps),
    )
    metadata = {
        "schema_version": 1,
        "source_motion": str(motion_path),
        "source_sha256": file_sha256(motion_path),
        "model_path": str(model_path),
        "selection": selection_metadata,
        "context_sec": 0.0 if explicit else args.context_sec,
        "clip_start_sec": clip_start / source_fps,
        "clip_end_sec": (clip_end - 1) / source_fps,
        "frames": len(indices),
        "fps": args.target_fps,
        "duration_sec": (len(indices) - 1) / args.target_fps,
        "turn_frame_range": [int(selected_start), int(selected_end - 1)],
        "left_contact_fraction": float(contacts[:, 0].mean()),
        "right_contact_fraction": float(contacts[:, 1].mean()),
        "output": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote natural turn preview data: {output}")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
