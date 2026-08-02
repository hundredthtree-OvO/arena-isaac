#!/usr/bin/env python3
"""Export controlled SMPL-H data as a skinned UsdSkel animation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdSkel, Vt

try:
    from ..analysis.rotations import matrix_to_quaternion
except ImportError:
    from analysis.rotations import matrix_to_quaternion


def _joint_paths(parents: np.ndarray) -> list[str]:
    paths: list[str] = []
    for index, parent in enumerate(parents):
        name = f"J{index:02d}"
        paths.append(name if parent < 0 else f"{paths[int(parent)]}/{name}")
    return paths


def _translation_matrix(vector: np.ndarray) -> Gf.Matrix4d:
    return Gf.Matrix4d(1.0).SetTranslate(
        Gf.Vec3d(float(vector[0]), float(vector[1]), float(vector[2]))
    )


def _quaternion(matrix: np.ndarray) -> Gf.Quatf:
    quaternion = matrix_to_quaternion(np.asarray(matrix))
    return Gf.Quatf(
        float(quaternion[0]),
        Gf.Vec3f(
            float(quaternion[1]),
            float(quaternion[2]),
            float(quaternion[3]),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preview", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    preview = args.preview.expanduser().resolve()
    with np.load(preview, allow_pickle=False) as data:
        points = np.asarray(data["posed_points"], dtype=np.float32)
        faces = np.asarray(data["faces"], dtype=np.int32)
        weights = np.asarray(data["weights"], dtype=np.float32)
        parents = np.asarray(data["parents"], dtype=np.int64)
        rest_joints = np.asarray(data["rest_joints"], dtype=np.float64)
        local_translations = np.asarray(data["local_translations"], dtype=np.float32)
        rotations = np.asarray(data["joint_rotations"], dtype=np.float64)
        trajectory = np.asarray(data["root_trajectory"], dtype=np.float32)
        phase = np.asarray(data["phase"])
        fps = float(np.asarray(data["fps"]).item())

    joint_paths = _joint_paths(parents)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(fps)
    stage.SetFramesPerSecond(fps)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(len(points) - 1)

    root = UsdSkel.Root.Define(stage, "/World/Avatar")
    skeleton = UsdSkel.Skeleton.Define(stage, "/World/Avatar/Skeleton")
    skeleton.CreateJointsAttr(joint_paths)
    rest_transforms = [_translation_matrix(value) for value in local_translations]
    bind_transforms = [_translation_matrix(value) for value in rest_joints]
    skeleton.CreateRestTransformsAttr(rest_transforms)
    skeleton.CreateBindTransformsAttr(bind_transforms)

    animation = UsdSkel.Animation.Define(stage, "/World/Avatar/Animation")
    animation.CreateJointsAttr(joint_paths)
    scale_values = Vt.Vec3hArray(
        [Gf.Vec3h(1.0, 1.0, 1.0)] * len(joint_paths)
    )
    animation.CreateScalesAttr(scale_values)
    for frame in range(len(points)):
        translations = local_translations.copy()
        translations[0] += trajectory[frame]
        animation.GetTranslationsAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(*value.tolist()) for value in translations]),
            Usd.TimeCode(frame),
        )
        animation.GetRotationsAttr().Set(
            Vt.QuatfArray([_quaternion(value) for value in rotations[frame]]),
            Usd.TimeCode(frame),
        )
    UsdSkel.BindingAPI.Apply(skeleton.GetPrim()).CreateAnimationSourceRel().SetTargets(
        [animation.GetPath()]
    )

    mesh = UsdGeom.Mesh.Define(stage, "/World/Avatar/Body")
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1))
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.62, 0.38, 0.27)])
    for frame, frame_points in enumerate(points):
        mesh.GetPointsAttr().Set(frame_points, Usd.TimeCode(frame))

    influence_count = min(8, weights.shape[1])
    indices = np.argpartition(weights, -influence_count, axis=1)[:, -influence_count:]
    selected = np.take_along_axis(weights, indices, axis=1)
    order = np.argsort(selected, axis=1)[:, ::-1]
    indices = np.take_along_axis(indices, order, axis=1).astype(np.int32)
    selected = np.take_along_axis(selected, order, axis=1)
    selected /= selected.sum(axis=1, keepdims=True)
    binding = UsdSkel.BindingAPI.Apply(mesh.GetPrim())
    binding.CreateSkeletonRel().SetTargets([skeleton.GetPath()])
    binding.CreateGeomBindTransformAttr(Gf.Matrix4d(1.0))
    binding.CreateJointIndicesPrimvar(False, influence_count).Set(indices.reshape(-1))
    binding.CreateJointWeightsPrimvar(False, influence_count).Set(selected.reshape(-1))

    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.CreateSizeAttr(1.0)
    ground.AddScaleOp().Set(Gf.Vec3d(8.0, 8.0, 0.02))
    ground.AddTranslateOp().Set(Gf.Vec3d(2.5, 2.5, -0.011))
    ground.CreateDisplayColorAttr([Gf.Vec3f(0.18, 0.20, 0.22)])

    curve = UsdGeom.BasisCurves.Define(stage, "/World/RootTrajectory")
    curve.CreateTypeAttr(UsdGeom.Tokens.linear)
    curve.CreateCurveVertexCountsAttr([len(trajectory)])
    curve.CreatePointsAttr(trajectory)
    curve.CreateWidthsAttr([0.015])
    curve.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.8, 0.2)])

    root.GetPrim().CreateAttribute(
        "smplPreview:source", Sdf.ValueTypeNames.String
    ).Set(str(preview))
    root.GetPrim().CreateAttribute(
        "smplPreview:phases", Sdf.ValueTypeNames.StringArray
    ).Set([str(item) for item in np.unique(phase)])
    stage.GetRootLayer().Save()
    print(f"Wrote controlled UsdSkel stage: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
