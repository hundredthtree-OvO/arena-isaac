#!/usr/bin/env python3
"""Compose a contact-aware turn, stop, hold and resume UsdSkel payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..analysis.gait import upright_stance_frame
from ..analysis.rotations import (
    axis_angle_to_matrix,
    matrix_to_quaternion,
    planar_heading,
    quaternion_slerp,
    quaternion_to_matrix,
    z_rotation,
)
from ..runtime.motion_graph import (
    MotionClip,
    select_transition,
)
from ..runtime.motion_library import load_manifest
from .smplh_numpy import (
    load_model,
    pose_corrected_vertices,
    shaped_vertices_and_joints,
    skin_frame_rotations,
)


DEFAULT_ASSET_ROOT = Path("/home/stardust/resources/arena_ws/arena_assets/smpl")


def _global_segment(
    clip: MotionClip,
    start: int,
    end: int,
    *,
    base_position: np.ndarray,
    base_yaw: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotations = axis_angle_to_matrix(clip.poses[start:end].reshape(-1, 52, 3))
    heading = planar_heading(clip.poses[:, :3])
    local_root = z_rotation(-heading[start:end]) @ rotations[:, 0]
    relative_heading = heading[start:end] - heading[start]
    rotations[:, 0] = z_rotation(base_yaw + relative_heading) @ local_root
    relative_path = clip.root_trajectory[start:end] - clip.root_trajectory[start]
    positions = base_position + relative_path @ z_rotation(base_yaw).T
    return rotations, positions, base_yaw + relative_heading


def _append_transition(
    rotations: list[np.ndarray],
    positions: list[np.ndarray],
    phases: list[str],
    *,
    target_rotation: np.ndarray,
    source_velocity: np.ndarray,
    target_velocity: np.ndarray,
    blend_frames: int,
    fps: float,
    phase: str,
    support_joint: int | None = None,
    skeleton: tuple | None = None,
) -> None:
    source_quaternion = matrix_to_quaternion(rotations[-1])
    target_quaternion = matrix_to_quaternion(target_rotation)
    position = positions[-1].copy()
    support_anchor = None
    if support_joint is not None and skeleton is not None:
        model, shaped, rest_joints = skeleton
        _, source_joints = skin_frame_rotations(
            model, shaped, rest_joints, rotations[-1], position
        )
        support_anchor = source_joints[support_joint].copy()
    for index in range(1, blend_frames + 1):
        fraction = index / blend_frames
        quaternion = quaternion_slerp(
            source_quaternion, target_quaternion, fraction
        )
        rotations.append(quaternion_to_matrix(quaternion))
        if support_anchor is None:
            velocity = (1.0 - fraction) * source_velocity + fraction * target_velocity
            position = position + velocity / fps
        else:
            _, local_joints = skin_frame_rotations(
                model,
                shaped,
                rest_joints,
                quaternion_to_matrix(quaternion),
                np.zeros(3),
            )
            position = support_anchor - local_joints[support_joint]
        positions.append(position.copy())
        phases.append(phase)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    graph = json.loads(args.graph.expanduser().read_text(encoding="utf-8"))
    manifest = load_manifest(args.manifest.expanduser().resolve())
    asset_root = args.asset_root.expanduser().resolve()
    entries = {entry["id"]: entry for entry in manifest["motions"]}

    def load(role: str) -> MotionClip:
        motion_id = graph["motion_ids"][role]
        entry = entries[motion_id]
        return MotionClip.load(
            motion_id, asset_root / entry["clip"]["relative_path"]
        )

    turn = load("turn")
    stop = load("stop")
    resume = load("resume")
    fps = turn.fps
    if not np.isclose([stop.fps, resume.fps], fps).all():
        raise ValueError("all motion graph clips must use one FPS")
    model_path = asset_root / "smplh" / turn.gender / "model.npz"
    model = load_model(model_path)
    shaped, rest_joints = shaped_vertices_and_joints(model, turn.betas)
    skeleton = (model, shaped, rest_joints)

    transition_cfg = graph["transition"]
    tail_frames = int(round(transition_cfg["turn_tail_search_sec"] * fps))
    head_frames = int(round(transition_cfg["stop_head_search_sec"] * fps))
    turn_stop = select_transition(
        turn,
        stop,
        source_start_index=max(0, len(turn.poses) - tail_frames),
        target_end_index=head_frames,
    )
    idle_tail_frames = max(
        1, int(round(transition_cfg["idle_tail_search_sec"] * fps))
    )
    idle_index, idle_diagnostics = upright_stance_frame(
        stop.joint_positions,
        stop.root_trajectory,
        fps,
        start_frame=max(turn_stop.target_index + 1, len(stop.poses) - idle_tail_frames),
    )
    resume_head_frames = max(
        1, int(round(transition_cfg["resume_head_search_sec"] * fps))
    )
    hold_resume = select_transition(
        stop,
        resume,
        source_start_index=idle_index,
        source_end_index=idle_index + 1,
        target_end_index=resume_head_frames,
    )

    turn_rot, turn_pos, turn_yaw = _global_segment(
        turn,
        0,
        turn_stop.source_index + 1,
        base_position=np.zeros(3),
        base_yaw=0.0,
    )
    rotations = list(turn_rot)
    positions = list(turn_pos)
    phases = ["turn_action"] * len(turn_rot)

    current_yaw = float(turn_yaw[-1])
    target_rot, _, _ = _global_segment(
        stop,
        turn_stop.target_index,
        turn_stop.target_index + 1,
        base_position=positions[-1],
        base_yaw=current_yaw,
    )
    turn_velocity = (
        positions[-1] - positions[-2]
    ) * fps
    stop_velocity_local = (
        stop.root_trajectory[turn_stop.target_index + 1]
        - stop.root_trajectory[turn_stop.target_index]
    ) * fps
    stop_velocity = stop_velocity_local @ z_rotation(current_yaw).T
    blend_frames = max(2, int(round(transition_cfg["blend_sec"] * fps)))
    turn_stop_contacts = (
        turn.foot_contacts[turn_stop.source_index]
        & stop.foot_contacts[turn_stop.target_index]
    )
    support_joint = (
        10 + int(np.flatnonzero(turn_stop_contacts)[0])
        if turn_stop_contacts.any()
        else None
    )
    _append_transition(
        rotations,
        positions,
        phases,
        target_rotation=target_rot[0],
        source_velocity=turn_velocity,
        target_velocity=stop_velocity,
        blend_frames=blend_frames,
        fps=fps,
        phase="blend_turn_to_stop",
        support_joint=support_joint,
        skeleton=skeleton,
    )

    stop_rot, stop_pos, stop_yaw = _global_segment(
        stop,
        turn_stop.target_index + 1,
        idle_index + 1,
        base_position=positions[-1],
        base_yaw=current_yaw,
    )
    # Remove the segment's first relative displacement after the blend.
    stop_pos += (
        stop.root_trajectory[turn_stop.target_index + 1]
        - stop.root_trajectory[turn_stop.target_index]
    ) @ z_rotation(current_yaw).T
    if support_joint is not None and len(stop_rot):
        _, blend_end_joints = skin_frame_rotations(
            model, shaped, rest_joints, rotations[-1], positions[-1]
        )
        _, stop_start_joints = skin_frame_rotations(
            model, shaped, rest_joints, stop_rot[0], stop_pos[0]
        )
        stop_pos += (
            blend_end_joints[support_joint] - stop_start_joints[support_joint]
        )
    rotations.extend(stop_rot)
    positions.extend(stop_pos)
    phases.extend(["stop_action"] * len(stop_rot))

    hold_frames = max(1, int(round(graph["hold_sec"] * fps)))
    for _ in range(hold_frames):
        rotations.append(rotations[-1].copy())
        positions.append(positions[-1].copy())
        phases.append("hold")

    resume_base_yaw = float(stop_yaw[-1])
    resume_target_rot, _, _ = _global_segment(
        resume,
        hold_resume.target_index,
        hold_resume.target_index + 1,
        base_position=positions[-1],
        base_yaw=resume_base_yaw,
    )
    resume_velocity_local = (
        resume.root_trajectory[hold_resume.target_index + 1]
        - resume.root_trajectory[hold_resume.target_index]
    ) * fps
    resume_velocity = resume_velocity_local @ z_rotation(resume_base_yaw).T
    resume_blend_frames = max(
        2, int(round(transition_cfg["resume_blend_sec"] * fps))
    )
    _append_transition(
        rotations,
        positions,
        phases,
        target_rotation=resume_target_rot[0],
        source_velocity=np.zeros(3),
        target_velocity=resume_velocity,
        blend_frames=resume_blend_frames,
        fps=fps,
        phase="blend_hold_to_resume",
        support_joint=10 + int(np.flatnonzero(
            stop.foot_contacts[hold_resume.source_index]
            & resume.foot_contacts[hold_resume.target_index]
        )[0]) if (
            stop.foot_contacts[hold_resume.source_index]
            & resume.foot_contacts[hold_resume.target_index]
        ).any() else None,
        skeleton=skeleton,
    )

    resume_rot, resume_pos, _ = _global_segment(
        resume,
        hold_resume.target_index + 1,
        len(resume.poses),
        base_position=positions[-1],
        base_yaw=resume_base_yaw,
    )
    resume_pos += (
        resume.root_trajectory[hold_resume.target_index + 1]
        - resume.root_trajectory[hold_resume.target_index]
    ) @ z_rotation(resume_base_yaw).T
    rotations.extend(resume_rot)
    positions.extend(resume_pos)
    phases.extend(["resume_action"] * len(resume_rot))

    rotations_array = np.asarray(rotations, dtype=np.float64)
    positions_array = np.asarray(positions, dtype=np.float64)
    posed_points = np.empty(
        (len(rotations_array), len(shaped), 3), dtype=np.float32
    )
    minimum_z = np.inf
    sample_step = max(1, len(rotations_array) // 40)
    for frame, frame_rotations in enumerate(rotations_array):
        posed_points[frame] = pose_corrected_vertices(
            model, shaped, frame_rotations
        )
        if frame % sample_step == 0 or frame == len(rotations_array) - 1:
            vertices, _ = skin_frame_rotations(
                model,
                shaped,
                rest_joints,
                frame_rotations,
                positions_array[frame],
            )
            minimum_z = min(minimum_z, float(vertices[:, 2].min()))
    positions_array[:, 2] -= minimum_z

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
        joint_rotations=rotations_array.astype(np.float32),
        root_trajectory=positions_array.astype(np.float32),
        phase=np.asarray(phases),
        fps=np.asarray(fps),
    )
    phase_ranges = {}
    for phase in dict.fromkeys(phases):
        indices = [index for index, value in enumerate(phases) if value == phase]
        phase_ranges[phase] = [indices[0], indices[-1]]
    report = {
        "schema_version": 1,
        "graph_id": graph["graph_id"],
        "frames": len(rotations_array),
        "fps": fps,
        "duration_sec": (len(rotations_array) - 1) / fps,
        "phase_ranges": phase_ranges,
        "turn_to_stop": turn_stop.to_dict(),
        "hold_frame": {
            "clip_index": idle_index,
            "source_frame": int(stop.source_frames[idle_index]),
            **idle_diagnostics,
        },
        "hold_to_resume": hold_resume.to_dict(),
        "floor_shift_m": float(-minimum_z),
        "output": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote motion graph preview: {output}")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
