#!/usr/bin/env python3
"""Export prepared SMPL-H points to a standalone animated USD stage."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preview", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.preview.expanduser().resolve(), allow_pickle=False) as data:
        vertices = np.asarray(data["vertices"], dtype=np.float32)
        faces = np.asarray(data["faces"], dtype=np.int32)
        trajectory = np.asarray(data["root_trajectory"], dtype=np.float32)
        fps = float(np.asarray(data["fps"]).item())

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(fps)
    stage.SetFramesPerSecond(fps)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(len(vertices) - 1)

    world = UsdGeom.Xform.Define(stage, "/World")
    mesh = UsdGeom.Mesh.Define(stage, "/World/SMPLHPreview")
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1))
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.62, 0.38, 0.27)])
    for frame, points in enumerate(vertices):
        mesh.GetPointsAttr().Set(points, Usd.TimeCode(frame))

    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.CreateSizeAttr(1.0)
    ground.AddScaleOp().Set(Gf.Vec3d(8.0, 4.0, 0.02))
    ground.AddTranslateOp().Set(Gf.Vec3d(2.0, 0.0, -0.011))
    ground.CreateDisplayColorAttr([Gf.Vec3f(0.18, 0.20, 0.22)])

    curve = UsdGeom.BasisCurves.Define(stage, "/World/RootTrajectory")
    curve.CreateTypeAttr(UsdGeom.Tokens.linear)
    curve.CreateCurveVertexCountsAttr([len(trajectory)])
    curve.CreatePointsAttr(trajectory)
    curve.CreateWidthsAttr([0.015])
    curve.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.8, 0.2)])

    world.GetPrim().CreateAttribute(
        "smplPreview:source", Sdf.ValueTypeNames.String
    ).Set(str(args.preview.expanduser().resolve()))
    stage.GetRootLayer().Save()
    print(f"Wrote Isaac preview stage: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
