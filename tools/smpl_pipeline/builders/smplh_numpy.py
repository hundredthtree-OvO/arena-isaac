"""Minimal NumPy SMPL-H linear blend skinning for offline previews."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from ..analysis.rotations import axis_angle_to_matrix
except ImportError:
    from analysis.rotations import axis_angle_to_matrix


@dataclass(frozen=True)
class SmplhModel:
    vertices_template: np.ndarray
    shapedirs: np.ndarray
    posedirs: np.ndarray
    joint_regressor: np.ndarray
    weights: np.ndarray
    parents: np.ndarray
    faces: np.ndarray


def load_model(path: Path) -> SmplhModel:
    with np.load(path, allow_pickle=False) as data:
        tree = np.asarray(data["kintree_table"], dtype=np.int64)
        parents = tree[0].copy()
        parents[0] = -1
        return SmplhModel(
            vertices_template=np.asarray(data["v_template"], dtype=np.float64),
            shapedirs=np.asarray(data["shapedirs"], dtype=np.float64),
            posedirs=np.asarray(data["posedirs"], dtype=np.float64),
            joint_regressor=np.asarray(data["J_regressor"], dtype=np.float64),
            weights=np.asarray(data["weights"], dtype=np.float64),
            parents=parents,
            faces=np.asarray(data["f"], dtype=np.int32),
        )


def _with_zeros(matrix: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :] = matrix
    return result


def _pack(vector: np.ndarray) -> np.ndarray:
    result = np.zeros((4, 4), dtype=np.float64)
    result[:, 3] = vector
    return result


def shaped_vertices_and_joints(
    model: SmplhModel, betas: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    beta_count = min(model.shapedirs.shape[-1], np.asarray(betas).size)
    shaped = model.vertices_template + np.tensordot(
        model.shapedirs[..., :beta_count],
        np.asarray(betas, dtype=np.float64)[:beta_count],
        axes=([2], [0]),
    )
    joints = model.joint_regressor @ shaped
    return shaped, joints


def pose_corrected_vertices(
    model: SmplhModel,
    shaped_vertices: np.ndarray,
    joint_rotations: np.ndarray,
) -> np.ndarray:
    pose_feature = (
        np.asarray(joint_rotations, dtype=np.float64)[1:] - np.eye(3)
    ).reshape(-1)
    return shaped_vertices + np.tensordot(
        model.posedirs,
        pose_feature,
        axes=([2], [0]),
    )


def skin_frame(
    model: SmplhModel,
    shaped_vertices: np.ndarray,
    rest_joints: np.ndarray,
    pose_axis_angle: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return posed vertices and joints for one 52-joint SMPL-H frame."""
    joint_count = model.parents.size
    pose = np.asarray(pose_axis_angle, dtype=np.float64).reshape(-1, 3)[:joint_count]
    rotations = axis_angle_to_matrix(pose)
    return skin_frame_rotations(
        model,
        shaped_vertices,
        rest_joints,
        rotations,
        translation,
    )


def skin_frame_rotations(
    model: SmplhModel,
    shaped_vertices: np.ndarray,
    rest_joints: np.ndarray,
    rotations: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return posed vertices and joints from explicit local rotation matrices."""
    joint_count = model.parents.size
    rotations = np.asarray(rotations, dtype=np.float64)[:joint_count]
    posed = pose_corrected_vertices(model, shaped_vertices, rotations)

    transforms = np.empty((joint_count, 4, 4), dtype=np.float64)
    transforms[0] = _with_zeros(
        np.column_stack((rotations[0], rest_joints[0]))
    )
    for index in range(1, joint_count):
        parent = int(model.parents[index])
        relative_joint = rest_joints[index] - rest_joints[parent]
        transforms[index] = transforms[parent] @ _with_zeros(
            np.column_stack((rotations[index], relative_joint))
        )

    rest_homogeneous = np.column_stack(
        (rest_joints, np.zeros(joint_count, dtype=np.float64))
    )
    corrected = np.stack(
        [
            transforms[index] - _pack(transforms[index] @ rest_homogeneous[index])
            for index in range(joint_count)
        ]
    )
    vertex_transforms = np.tensordot(model.weights, corrected, axes=([1], [0]))
    homogeneous = np.column_stack((posed, np.ones(len(posed))))
    vertices = np.einsum("nij,nj->ni", vertex_transforms, homogeneous)[:, :3]
    joints = transforms[:, :3, 3]
    offset = np.asarray(translation, dtype=np.float64)
    return vertices + offset, joints + offset
