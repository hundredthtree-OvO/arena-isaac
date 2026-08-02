#!/usr/bin/env python3
"""Build normalized SMPL-H clips and a versioned motion-library manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

try:
    from ..analysis.amass_io import file_sha256, validate_motion_arrays
    from ..analysis.gait import foot_contact_diagnostics
    from ..analysis.rotations import planar_heading, z_rotation
    from ..runtime.motion_library import SCHEMA_VERSION, validate_manifest
    from .smplh_numpy import load_model, shaped_vertices_and_joints, skin_frame
except ImportError:
    from analysis.amass_io import file_sha256, validate_motion_arrays
    from analysis.gait import foot_contact_diagnostics
    from analysis.rotations import planar_heading, z_rotation
    from runtime.motion_library import SCHEMA_VERSION, validate_manifest
    from builders.smplh_numpy import load_model, shaped_vertices_and_joints, skin_frame


DEFAULT_ASSET_ROOT = Path("/home/stardust/resources/arena_ws/arena_assets/smpl")


def _median_speed(trans: np.ndarray, fps: float, head: bool) -> float:
    speed = np.linalg.norm(np.gradient(trans[:, :2], 1.0 / fps, axis=0), axis=1)
    count = max(2, min(len(speed), int(round(0.3 * fps))))
    values = speed[:count] if head else speed[-count:]
    return float(np.median(values))


def _build_clip(
    recipe: dict,
    *,
    dataset_root: Path,
    model_root: Path,
    clip_root: Path,
    asset_root: Path,
    target_fps: float,
) -> dict:
    source = (dataset_root / recipe["source_relative"]).resolve()
    with np.load(source, allow_pickle=False) as data:
        validate_motion_arrays(data, source)
        source_fps = float(np.asarray(data["mocap_framerate"]).item())
        ratio = source_fps / target_fps
        stride = int(round(ratio))
        if stride < 1 or not np.isclose(ratio, stride, atol=1.0e-8):
            raise ValueError(f"{source}: source_fps / target_fps must be an integer")
        start = int(round(recipe["start_sec"] * source_fps))
        end = min(
            len(data["poses"]), int(round(recipe["end_sec"] * source_fps)) + 1
        )
        indices = np.arange(start, end, stride, dtype=np.int64)
        poses = np.asarray(data["poses"][indices, :156], dtype=np.float32)
        trans = np.asarray(data["trans"][indices], dtype=np.float64)
        betas = np.asarray(data["betas"], dtype=np.float32)
        gender = str(np.asarray(data["gender"]).item()).lower()

    heading = planar_heading(poses[:, :3])
    initial_heading = float(heading[0])
    root_trajectory = (trans - trans[0]) @ z_rotation(-initial_heading).T
    model_path = model_root / gender / "model.npz"
    model = load_model(model_path)
    shaped, rest_joints = shaped_vertices_and_joints(model, betas)
    joints = np.empty((len(poses), len(rest_joints), 3), dtype=np.float64)
    for frame in range(len(poses)):
        _, joints[frame] = skin_frame(
            model, shaped, rest_joints, poses[frame], trans[frame]
        )
    contacts, foot_speed = foot_contact_diagnostics(joints, target_fps)
    gait_phase = np.arange(len(poses), dtype=np.float64) / max(1, len(poses) - 1)

    clip_path = clip_root / f"{recipe['id']}.npz"
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        clip_path,
        poses=poses,
        betas=betas,
        gender=np.asarray(gender),
        fps=np.asarray(target_fps),
        root_trajectory=root_trajectory.astype(np.float32),
        joint_positions=joints.astype(np.float32),
        source_frames=indices,
        foot_contacts=contacts.astype(np.uint8),
        foot_speed=foot_speed.astype(np.float32),
        event_start_sec=np.asarray(
            recipe["event_start_sec"] - recipe["start_sec"]
        ),
        event_end_sec=np.asarray(recipe["event_end_sec"] - recipe["start_sec"]),
        gait_phase=gait_phase.astype(np.float32),
    )
    duration = (len(poses) - 1) / target_fps
    result = {
        "id": recipe["id"],
        "category": recipe["category"],
        "direction": recipe["direction"],
        "description": recipe["official_description"],
        "source": {
            "dataset": "AMASS_CMU",
            "relative_path": recipe["source_relative"],
            "sha256": file_sha256(source),
            "start_sec": recipe["start_sec"],
            "end_sec": recipe["end_sec"],
        },
        "clip": {
            "relative_path": str(clip_path.relative_to(asset_root)),
            "sha256": file_sha256(clip_path),
        },
        "timing": {
            "fps": target_fps,
            "frames": len(poses),
            "duration_sec": duration,
            "event_start_sec": recipe["event_start_sec"] - recipe["start_sec"],
            "event_end_sec": recipe["event_end_sec"] - recipe["start_sec"],
        },
        "kinematics": {
            "initial_heading_rad": initial_heading,
            "yaw_change_deg": float(np.degrees(heading[-1] - heading[0])),
            "root_displacement_m": float(
                np.linalg.norm(root_trajectory[-1, :2])
            ),
            "start_speed_mps": _median_speed(root_trajectory, target_fps, True),
            "end_speed_mps": _median_speed(root_trajectory, target_fps, False),
        },
        "contacts": {
            "left_fraction": float(contacts[:, 0].mean()),
            "right_fraction": float(contacts[:, 1].mean()),
        },
    }
    if recipe["category"] == "walk":
        nominal_speed = float(np.median(np.linalg.norm(
            np.gradient(root_trajectory[:, :2], 1.0 / target_fps, axis=0),
            axis=1,
        )))
        result["gait"] = {
            "phase_encoding": "normalized_clip_frame",
            "nominal_speed_mps": nominal_speed,
            "cycle_duration_sec": duration,
            "stride_displacement_m": float(
                np.linalg.norm(root_trajectory[-1, :2])
            ),
            "validated_speed_scale_min": 0.85,
            "validated_speed_scale_max": 1.15,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    recipe_path = args.recipe.expanduser().resolve()
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    asset_root = args.asset_root.expanduser().resolve()
    motions = [
        _build_clip(
            item,
            dataset_root=asset_root / "CMU" / "CMU",
            model_root=asset_root / "smplh",
            clip_root=asset_root / "derived" / "clips",
            asset_root=asset_root,
            target_fps=float(recipe["target_fps"]),
        )
        for item in recipe["motions"]
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "library_id": recipe["library_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "recipe": str(recipe_path),
        "motions": motions,
    }
    validate_manifest(manifest)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(motions)} clips and manifest: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
