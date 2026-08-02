"""Isaac/UsdSkel adapter for one command-driven SMPL-H avatar."""

from __future__ import annotations

import numpy as np

from ..analysis.rotations import matrix_to_quaternion
from ..builders.smplh_numpy import pose_corrected_vertices
from .avatar_backend import AvatarFrame, SmplUsdAvatarBackend


def _joint_paths(parents: np.ndarray) -> list[str]:
    paths: list[str] = []
    for index, parent in enumerate(parents):
        name = f"J{index:02d}"
        paths.append(name if parent < 0 else f"{paths[int(parent)]}/{name}")
    return paths


class IsaacSmplAvatarAdapter:
    """Author one live UsdSkel and update it from an avatar backend frame."""

    def __init__(
        self,
        stage,
        backend: SmplUsdAvatarBackend,
        *,
        prim_path: str = "/World/Avatar",
    ) -> None:
        from pxr import Gf, UsdGeom, UsdSkel, Vt

        self.backend = backend
        self.Gf = Gf
        self.Vt = Vt
        model = backend.model
        rest_joints = backend.rest_joints
        local_translations = np.empty_like(rest_joints)
        local_translations[0] = rest_joints[0]
        for joint in range(1, len(rest_joints)):
            local_translations[joint] = (
                rest_joints[joint] - rest_joints[int(model.parents[joint])]
            )
        self.local_translations = local_translations

        root = UsdSkel.Root.Define(stage, prim_path)
        skeleton = UsdSkel.Skeleton.Define(stage, f"{prim_path}/Skeleton")
        joint_paths = _joint_paths(model.parents)
        skeleton.CreateJointsAttr(joint_paths)
        skeleton.CreateRestTransformsAttr(
            [self._translation_matrix(value) for value in local_translations]
        )
        skeleton.CreateBindTransformsAttr(
            [self._translation_matrix(value) for value in rest_joints]
        )

        self.animation = UsdSkel.Animation.Define(
            stage, f"{prim_path}/Animation"
        )
        self.animation.CreateJointsAttr(joint_paths)
        self.animation.CreateScalesAttr(
            Vt.Vec3hArray(
                [Gf.Vec3h(1.0, 1.0, 1.0)] * len(joint_paths)
            )
        )
        UsdSkel.BindingAPI.Apply(
            skeleton.GetPrim()
        ).CreateAnimationSourceRel().SetTargets([self.animation.GetPath()])

        self.mesh = UsdGeom.Mesh.Define(stage, f"{prim_path}/Body")
        self.mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        self.mesh.CreateFaceVertexCountsAttr([3] * len(model.faces))
        self.mesh.CreateFaceVertexIndicesAttr(model.faces.reshape(-1))
        self.mesh.CreateDisplayColorAttr([Gf.Vec3f(0.62, 0.38, 0.27)])

        influence_count = min(8, model.weights.shape[1])
        indices = np.argpartition(
            model.weights, -influence_count, axis=1
        )[:, -influence_count:]
        selected = np.take_along_axis(model.weights, indices, axis=1)
        order = np.argsort(selected, axis=1)[:, ::-1]
        indices = np.take_along_axis(indices, order, axis=1).astype(np.int32)
        selected = np.take_along_axis(selected, order, axis=1)
        selected /= selected.sum(axis=1, keepdims=True)
        binding = UsdSkel.BindingAPI.Apply(self.mesh.GetPrim())
        binding.CreateSkeletonRel().SetTargets([skeleton.GetPath()])
        binding.CreateGeomBindTransformAttr(Gf.Matrix4d(1.0))
        binding.CreateJointIndicesPrimvar(False, influence_count).Set(
            indices.reshape(-1)
        )
        binding.CreateJointWeightsPrimvar(False, influence_count).Set(
            selected.reshape(-1)
        )
        self.update(backend.frame)
        root.GetPrim().CreateAttribute(
            "smplRuntime:state", self._string_type()
        ).Set(backend.frame.state.value)
        self.state_attribute = root.GetPrim().GetAttribute("smplRuntime:state")

    def _string_type(self):
        from pxr import Sdf

        return Sdf.ValueTypeNames.String

    def _translation_matrix(self, vector: np.ndarray):
        return self.Gf.Matrix4d(1.0).SetTranslate(
            self.Gf.Vec3d(*[float(value) for value in vector])
        )

    def _quaternion(self, matrix: np.ndarray):
        quaternion = matrix_to_quaternion(matrix)
        return self.Gf.Quatf(
            float(quaternion[0]),
            self.Gf.Vec3f(*[float(value) for value in quaternion[1:]]),
        )

    def update(self, frame: AvatarFrame) -> None:
        translations = self.local_translations.copy()
        translations[0] += frame.root_position
        self.animation.GetTranslationsAttr().Set(
            self.Vt.Vec3fArray(
                [self.Gf.Vec3f(*value.tolist()) for value in translations]
            )
        )
        self.animation.GetRotationsAttr().Set(
            self.Vt.QuatfArray(
                [self._quaternion(value) for value in frame.rotations]
            )
        )
        corrected = pose_corrected_vertices(
            self.backend.model,
            self.backend.shaped,
            frame.rotations,
        )
        self.mesh.GetPointsAttr().Set(
            self.Vt.Vec3fArray(
                [self.Gf.Vec3f(*value.tolist()) for value in corrected]
            )
        )
        if hasattr(self, "state_attribute"):
            self.state_attribute.Set(frame.state.value)
