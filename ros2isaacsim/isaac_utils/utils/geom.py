import typing

import attrs
import geometry_msgs.msg
import numpy as np
from isaacsim.core.prims import Articulation
from isaacsim.core.prims import RigidPrim, XFormPrim
from omni.isaac.core.utils.rotations import (euler_angles_to_quat,
                                             quat_to_euler_angles)
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, UsdGeom, UsdPhysics


@attrs.define
class Translation:
    x: float
    y: float
    z: float

    def tuple(self) -> typing.Tuple[float, float, float]:
        return self.x, self.y, self.z

    def Vec3d(self) -> Gf.Vec3d:
        return Gf.Vec3d(self.x, self.y, self.z)

    @classmethod
    def parse(cls, values: geometry_msgs.msg.Point | typing.Collection[float]) -> "Translation":
        if isinstance(values, geometry_msgs.msg.Point):
            return cls(
                x=values.x,
                y=values.y,
                z=values.z,
            )

        if len(values) == 3:
            return cls(*values)

        if len(values) == 2:
            return cls(
                *values,
                0.0,
            )

        raise ValueError(
            f"Translation must be [x,y] or [x,y,z], got {values}"
        )


@attrs.define
class Rotation:
    w: float
    x: float
    y: float
    z: float

    def quat(self, convention: str = 'wxyz') -> typing.List[float]:
        return [float(getattr(self, axis)) for axis in convention if axis in 'wxyz']

    def euler(self, convention: str = 'xyz') -> typing.List[float]:
        x, y, z = quat_to_euler_angles(self.quat)
        axes: dict[str, float] = dict(
            x=x,
            y=y,
            z=z,
        )
        return [axes[axis] for axis in convention if axis in 'xyz']

    def Quatd(self) -> Gf.Quatd:
        return Gf.Quatd(self.w, self.x, self.y, self.z)

    @classmethod
    def parse(cls, values: geometry_msgs.msg.Quaternion | typing.Collection[float]) -> "Rotation":
        if isinstance(values, geometry_msgs.msg.Quaternion):
            return cls(
                x=values.x,
                y=values.y,
                z=values.z,
                w=values.w,
            )

        if len(values) == 4:
            return cls(
                *values
            )

        if len(values) == 3:
            return cls(
                *euler_angles_to_quat(values)
            )

        raise ValueError(
            f"Rotation must be [x,y,z] or [w,x,y,z], got {values}"
        )




def _set_pose_compat(prim_path: str, target, translation: Translation, rotation: Rotation):
    """Set pose across Isaac 4.5 wrapper variants.

    Some Isaac 4.5 XFormPrim objects expose get_world_pose but not
    set_world_pose. For the stable kinematic robot wrapper we can safely author
    translate+orient xform ops directly on the USD prim.
    """
    pos = np.array(translation.tuple(), dtype=np.float32)
    quat = np.array(rotation.quat(), dtype=np.float32)  # wxyz

    if hasattr(target, "set_world_pose"):
        try:
            target.set_world_pose(position=pos, orientation=quat)
            return
        except Exception:
            pass

    if hasattr(target, "set_local_pose"):
        for kwargs in (
            {"translation": pos, "orientation": quat},
            {"position": pos, "orientation": quat},
        ):
            try:
                target.set_local_pose(**kwargs)
                return
            except Exception:
                pass

    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return
    xformable = UsdGeom.Xformable(prim)
    ops = list(xformable.GetOrderedXformOps())
    translate_op = next((op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
    orient_op = next((op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeOrient), None)
    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    if orient_op is None:
        orient_op = xformable.AddOrientOp()
    xformable.SetXformOpOrder([translate_op, orient_op])
    translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    orient_op.Set(Gf.Quatd(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])))


def move(
    prim_path: str,
    translation: Translation,
    rotation: Rotation,
):
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)

    if not prim.IsValid():
        return

    target = None
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        target = Articulation(prim_path)

    if target is not None:  # i am an articulation
        dof = target.num_dof
        target.set_joint_positions(positions=np.zeros(dof, dtype="float32"))
        target.set_joint_velocities(velocities=np.zeros(dof, dtype="float32"))
        if hasattr(target, "wake_up"):
            target.wake_up()

    elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
        target = RigidPrim(prim_path)

    if target is not None:  # i am an articulation / rigidbody
        if hasattr(target, "set_linear_velocity"):
            target.set_linear_velocity([0, 0, 0])
        if hasattr(target, "set_angular_velocity"):
            target.set_angular_velocity([0, 0, 0])

    if target is None:
        target = XFormPrim(prim_path)

    _set_pose_compat(prim_path, target, translation, rotation)
