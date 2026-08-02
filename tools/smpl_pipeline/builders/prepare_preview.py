#!/usr/bin/env python3
"""Prepare a compact, baked SMPL-H mesh animation for Isaac USD export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from ..analysis.amass_io import file_sha256, validate_motion_arrays
    from ..analysis.motion_features import rank_stable_walk_windows
    from ..analysis.rotations import planar_heading
    from .smplh_numpy import load_model, shaped_vertices_and_joints, skin_frame
except ImportError:
    from analysis.amass_io import file_sha256, validate_motion_arrays
    from analysis.motion_features import rank_stable_walk_windows
    from analysis.rotations import planar_heading
    from builders.smplh_numpy import load_model, shaped_vertices_and_joints, skin_frame


DEFAULT_ROOT = Path("/home/stardust/resources/arena_ws/arena_assets/smpl")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", type=Path)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_ROOT / "smplh")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-sec", type=float)
    parser.add_argument("--duration-sec", type=float, default=2.0)
    parser.add_argument("--target-fps", type=float, default=60.0)
    args = parser.parse_args()

    motion_path = args.motion.expanduser().resolve()
    with np.load(motion_path, allow_pickle=False) as data:
        validate_motion_arrays(data, motion_path)
        source_fps = float(np.asarray(data["mocap_framerate"]).item())
        poses_all = np.asarray(data["poses"], dtype=np.float64)
        trans_all = np.asarray(data["trans"], dtype=np.float64)
        betas = np.asarray(data["betas"], dtype=np.float64)
        gender = str(np.asarray(data["gender"]).item()).lower()

    if args.start_sec is None:
        ranked = rank_stable_walk_windows(trans_all, poses_all, source_fps)
        if not ranked:
            raise RuntimeError("no stable straight-walk window found")
        start_frame = ranked[0].start_frame
        selection = ranked[0].to_dict()
    else:
        start_frame = int(round(args.start_sec * source_fps))
        selection = {"method": "explicit", "start_sec": args.start_sec}
    source_count = int(round(args.duration_sec * source_fps)) + 1
    end_frame = min(len(poses_all), start_frame + source_count)
    if end_frame - start_frame < 2:
        raise ValueError("selected preview contains fewer than two frames")

    ratio = source_fps / args.target_fps
    stride = int(round(ratio))
    if stride < 1 or not np.isclose(ratio, stride, atol=1.0e-8):
        raise ValueError(
            "preview currently requires source_fps / target_fps to be an integer"
        )
    source_indices = np.arange(start_frame, end_frame, stride, dtype=np.int64)
    poses = poses_all[source_indices]
    trans = trans_all[source_indices].copy()

    model_path = args.model_root.expanduser().resolve() / gender / "model.npz"
    model = load_model(model_path)
    shaped, rest_joints = shaped_vertices_and_joints(model, betas)
    vertices = np.empty((len(poses), len(shaped), 3), dtype=np.float32)
    joints = np.empty((len(poses), len(rest_joints), 3), dtype=np.float32)
    for index, (pose, translation) in enumerate(zip(poses, trans)):
        vertices[index], joints[index] = skin_frame(
            model, shaped, rest_joints, pose, translation
        )

    # Normalize the clip to start at the origin and face +X.
    initial_heading = float(planar_heading(poses[:1, :3])[0])
    angle = -initial_heading
    rotate = np.asarray(
        [[np.cos(angle), -np.sin(angle), 0.0],
         [np.sin(angle), np.cos(angle), 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    origin = trans[0].copy()
    vertices = ((vertices - origin) @ rotate.T).astype(np.float32)
    joints = ((joints - origin) @ rotate.T).astype(np.float32)
    floor_z = float(vertices[..., 2].min())
    vertices[..., 2] -= floor_z
    joints[..., 2] -= floor_z
    root_trajectory = ((trans - origin) @ rotate.T).astype(np.float32)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        vertices=vertices,
        joints=joints,
        faces=model.faces,
        root_trajectory=root_trajectory,
        fps=np.asarray(args.target_fps, dtype=np.float64),
        source_indices=source_indices,
    )
    metadata = {
        "schema_version": 1,
        "source_motion": str(motion_path),
        "source_sha256": file_sha256(motion_path),
        "model_path": str(model_path),
        "gender": gender,
        "selection": selection,
        "start_frame": int(source_indices[0]),
        "end_frame": int(source_indices[-1]),
        "source_fps": source_fps,
        "target_fps": args.target_fps,
        "frames": len(source_indices),
        "initial_heading_rad": initial_heading,
        "floor_offset_m": floor_z,
        "net_root_displacement_m": float(
            np.linalg.norm(root_trajectory[-1, :2])
        ),
        "output": str(output),
        "output_sha256": file_sha256(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote baked preview data: {output}")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
