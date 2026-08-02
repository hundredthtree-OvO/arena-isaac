"""Small NumPy rotation helpers used by the offline SMPL-H pipeline."""

from __future__ import annotations

import numpy as np


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Convert (..., 3) rotation vectors to (..., 3, 3) matrices."""
    vectors = np.asarray(axis_angle, dtype=np.float64)
    angles = np.linalg.norm(vectors, axis=-1, keepdims=True)
    axes = np.divide(
        vectors,
        angles,
        out=np.zeros_like(vectors),
        where=angles > 1.0e-12,
    )
    x, y, z = np.moveaxis(axes, -1, 0)
    zeros = np.zeros_like(x)
    skew = np.stack(
        (
            zeros, -z, y,
            z, zeros, -x,
            -y, x, zeros,
        ),
        axis=-1,
    ).reshape(vectors.shape[:-1] + (3, 3))
    identity = np.broadcast_to(np.eye(3), skew.shape)
    sin_angles = np.sin(angles)[..., None]
    cos_angles = np.cos(angles)[..., None]
    return identity + sin_angles * skew + (1.0 - cos_angles) * (skew @ skew)


def planar_heading(root_axis_angle: np.ndarray) -> np.ndarray:
    """Return unwrapped yaw for SMPL's local +Z forward axis."""
    rotations = axis_angle_to_matrix(root_axis_angle)
    forward = rotations @ np.asarray([0.0, 0.0, 1.0])
    return np.unwrap(np.arctan2(forward[..., 1], forward[..., 0]))


def z_rotation(angle: np.ndarray | float) -> np.ndarray:
    """Return (..., 3, 3) rotations about world Z."""
    angle = np.asarray(angle, dtype=np.float64)
    result = np.zeros(angle.shape + (3, 3), dtype=np.float64)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    result[..., 0, 0] = cosine
    result[..., 0, 1] = -sine
    result[..., 1, 0] = sine
    result[..., 1, 1] = cosine
    result[..., 2, 2] = 1.0
    return result


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert (..., 3, 3) matrices to scalar-first unit quaternions."""
    matrices = np.asarray(matrix, dtype=np.float64)
    flat = matrices.reshape(-1, 3, 3)
    result = np.empty((len(flat), 4), dtype=np.float64)
    for index, rotation in enumerate(flat):
        trace = float(np.trace(rotation))
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            quat = np.asarray(
                [
                    0.25 * scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                ]
            )
        else:
            diagonal = np.diag(rotation)
            axis = int(np.argmax(diagonal))
            if axis == 0:
                scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
                quat = np.asarray(
                    [
                        (rotation[2, 1] - rotation[1, 2]) / scale,
                        0.25 * scale,
                        (rotation[0, 1] + rotation[1, 0]) / scale,
                        (rotation[0, 2] + rotation[2, 0]) / scale,
                    ]
                )
            elif axis == 1:
                scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
                quat = np.asarray(
                    [
                        (rotation[0, 2] - rotation[2, 0]) / scale,
                        (rotation[0, 1] + rotation[1, 0]) / scale,
                        0.25 * scale,
                        (rotation[1, 2] + rotation[2, 1]) / scale,
                    ]
                )
            else:
                scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
                quat = np.asarray(
                    [
                        (rotation[1, 0] - rotation[0, 1]) / scale,
                        (rotation[0, 2] + rotation[2, 0]) / scale,
                        (rotation[1, 2] + rotation[2, 1]) / scale,
                        0.25 * scale,
                    ]
                )
        result[index] = quat / np.linalg.norm(quat)
    return result.reshape(matrices.shape[:-2] + (4,))


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert scalar-first unit quaternions to rotation matrices."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    result = np.empty(quaternion.shape[:-1] + (3, 3), dtype=np.float64)
    result[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    result[..., 0, 1] = 2.0 * (x * y - z * w)
    result[..., 0, 2] = 2.0 * (x * z + y * w)
    result[..., 1, 0] = 2.0 * (x * y + z * w)
    result[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    result[..., 1, 2] = 2.0 * (y * z - x * w)
    result[..., 2, 0] = 2.0 * (x * z - y * w)
    result[..., 2, 1] = 2.0 * (y * z + x * w)
    result[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return result


def quaternion_slerp(
    start: np.ndarray,
    end: np.ndarray,
    fraction: np.ndarray | float,
) -> np.ndarray:
    """Shortest-path spherical interpolation for matching quaternion arrays."""
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    fraction = np.asarray(fraction, dtype=np.float64)
    dot = np.sum(start * end, axis=-1, keepdims=True)
    end = np.where(dot < 0.0, -end, end)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    theta = np.arccos(dot)
    sine = np.sin(theta)
    expanded_fraction = fraction
    while expanded_fraction.ndim < start.ndim:
        expanded_fraction = expanded_fraction[..., None]
    linear = (1.0 - expanded_fraction) * start + expanded_fraction * end
    spherical = (
        np.sin((1.0 - expanded_fraction) * theta) / np.maximum(sine, 1.0e-8) * start
        + np.sin(expanded_fraction * theta) / np.maximum(sine, 1.0e-8) * end
    )
    result = np.where(sine < 1.0e-6, linear, spherical)
    return result / np.linalg.norm(result, axis=-1, keepdims=True)
