#!/usr/bin/env python3
"""Build a controlled SMPL-H root-motion sequence for UsdSkel export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from ..analysis.amass_io import file_sha256, validate_motion_arrays
    from ..analysis.gait import foot_contact_diagnostics, stable_contact_frame
    from ..analysis.motion_features import rank_stable_walk_windows
    from ..analysis.rotations import axis_angle_to_matrix, planar_heading, z_rotation
    from .smplh_numpy import (
        load_model,
        pose_corrected_vertices,
        shaped_vertices_and_joints,
        skin_frame,
    )
except ImportError:
    from analysis.amass_io import file_sha256, validate_motion_arrays
    from analysis.gait import foot_contact_diagnostics, stable_contact_frame
    from analysis.motion_features import rank_stable_walk_windows
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
    parser.add_argument("--hold-sec", type=float, default=1.0)
    parser.add_argument("--turn-sec", type=float, default=1.0)
    args = parser.parse_args()

    motion_path = args.motion.expanduser().resolve()
    with np.load(motion_path, allow_pickle=False) as data:
        validate_motion_arrays(data, motion_path)
        source_fps = float(np.asarray(data["mocap_framerate"]).item())
        poses_all = np.asarray(data["poses"], dtype=np.float64)
        trans_all = np.asarray(data["trans"], dtype=np.float64)
        betas = np.asarray(data["betas"], dtype=np.float64)
        gender = str(np.asarray(data["gender"]).item()).lower()

    ranked = rank_stable_walk_windows(trans_all, poses_all, source_fps)
    if not ranked:
        raise RuntimeError("no stable straight-walk window found")
    selected = ranked[0]
    ratio = source_fps / args.target_fps
    stride = int(round(ratio))
    if stride < 1 or not np.isclose(ratio, stride, atol=1.0e-8):
        raise ValueError("source_fps / target_fps must be an integer")
    indices = np.arange(
        selected.start_frame,
        selected.end_frame_exclusive + 1,
        stride,
        dtype=np.int64,
    )
    indices = indices[indices < len(poses_all)]
    poses = poses_all[indices, :156]
    trans = trans_all[indices]
    source_rotations = axis_angle_to_matrix(poses.reshape(len(poses), 52, 3))

    initial_heading = float(planar_heading(poses[:1, :3])[0])
    normalized_heading = planar_heading(poses[:, :3]) - initial_heading
    normalized_world_root = z_rotation(-initial_heading) @ source_rotations[:, 0]
    local_root = z_rotation(-normalized_heading) @ normalized_world_root
    trajectory = (trans - trans[0]) @ z_rotation(-initial_heading).T

    model_path = args.model_root.expanduser().resolve() / gender / "model.npz"
    model = load_model(model_path)
    shaped, rest_joints = shaped_vertices_and_joints(model, betas)
    source_joints = np.empty((len(poses), len(rest_joints), 3))
    for frame in range(len(poses)):
        # Joint contact is measured from the actual source pose.
        _, source_joints[frame] = skin_frame(
            model,
            shaped,
            rest_joints,
            poses[frame],
            trans[frame],
        )
    contacts, foot_speed = foot_contact_diagnostics(source_joints, args.target_fps)
    hold_source_frame = stable_contact_frame(
        source_joints,
        args.target_fps,
        start_frame=len(source_joints) // 2,
    )

    hold_frames = max(1, int(round(args.hold_sec * args.target_fps)))
    turn_frames = max(2, int(round(args.turn_sec * args.target_fps)))
    walk_count = len(poses)
    hold_index = hold_source_frame

    frame_sources = np.concatenate(
        (
            np.arange(walk_count),
            np.full(hold_frames, hold_index),
            np.full(turn_frames, hold_index),
            np.arange(1, walk_count),
        )
    )
    phase = np.concatenate(
        (
            np.full(walk_count, "walk_x"),
            np.full(hold_frames, "hold"),
            np.full(turn_frames, "turn"),
            np.full(walk_count - 1, "walk_y"),
        )
    )
    desired_yaw = np.concatenate(
        (
            normalized_heading,
            np.full(hold_frames, normalized_heading[-1]),
            np.linspace(normalized_heading[-1], np.pi / 2.0, turn_frames),
            np.pi / 2.0 + normalized_heading[1:] - normalized_heading[0],
        )
    )

    root_trajectory = np.empty((len(frame_sources), 3), dtype=np.float64)
    root_trajectory[:walk_count] = trajectory
    first_end = trajectory[-1]
    root_trajectory[walk_count : walk_count + hold_frames + turn_frames] = first_end
    second = trajectory[1:] @ z_rotation(np.pi / 2.0).T
    root_trajectory[walk_count + hold_frames + turn_frames :] = first_end + second

    rotations = source_rotations[frame_sources].copy()
    rotations[:, 0] = z_rotation(desired_yaw) @ local_root[frame_sources]
    posed_points = np.empty((len(rotations), len(shaped), 3), dtype=np.float32)
    for frame in range(len(rotations)):
        posed_points[frame] = pose_corrected_vertices(
            model, shaped, rotations[frame]
        )

    # Compute a conservative floor shift from full NumPy LBS at sampled frames.
    minimum_z = np.inf
    sample_step = max(1, len(rotations) // 30)
    for frame in range(0, len(rotations), sample_step):
        # skin_frame only needs matrices internally; floor from the earlier baked
        # source gives the same model-space root-to-foot offset.
        source_frame = int(frame_sources[frame])
        vertices, _ = skin_frame(
            model, shaped, rest_joints, poses[source_frame], np.zeros(3)
        )
        minimum_z = min(minimum_z, float(vertices[:, 2].min()))
    floor_shift = -minimum_z
    root_trajectory[:, 2] += floor_shift

    local_translations = np.empty((len(rest_joints), 3), dtype=np.float32)
    local_translations[0] = rest_joints[0]
    for joint in range(1, len(rest_joints)):
        local_translations[joint] = (
            rest_joints[joint] - rest_joints[int(model.parents[joint])]
        )

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
        root_trajectory=root_trajectory.astype(np.float32),
        phase=phase,
        source_frame=frame_sources,
        source_foot_contacts=contacts.astype(np.uint8),
        source_foot_speed=foot_speed.astype(np.float32),
        fps=np.asarray(args.target_fps),
    )
    metadata = {
        "schema_version": 1,
        "source_motion": str(motion_path),
        "source_sha256": file_sha256(motion_path),
        "model_path": str(model_path),
        "selection": selected.to_dict(),
        "frames": len(frame_sources),
        "fps": args.target_fps,
        "duration_sec": (len(frame_sources) - 1) / args.target_fps,
        "phase_ranges": {
            "walk_x": [0, walk_count - 1],
            "hold": [walk_count, walk_count + hold_frames - 1],
            "turn": [
                walk_count + hold_frames,
                walk_count + hold_frames + turn_frames - 1,
            ],
            "walk_y": [
                walk_count + hold_frames + turn_frames,
                len(frame_sources) - 1,
            ],
        },
        "hold_source_frame": hold_source_frame,
        "left_contact_fraction": float(contacts[:, 0].mean()),
        "right_contact_fraction": float(contacts[:, 1].mean()),
        "floor_shift_m": floor_shift,
        "output": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote controlled preview data: {output}")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
