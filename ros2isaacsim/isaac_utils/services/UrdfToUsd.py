import os
import re
import sys
import tempfile
import typing
import shutil
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import isaac_utils.graphs.joint_states as joint_states
import isaac_utils.graphs.odom as odom
import isaac_utils.graphs.sensors.sensors as sensors
import isaac_utils.graphs.tf as tf
import isaac_utils.utils.paths as Paths
import omni.kit.commands as commands
import omni.usd
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from isaac_utils.graphs import control
from isaac_utils.mecanum_teleop import is_mecanum_model, mecanum_teleop_manager
from isaac_utils.dual_lidar import create_dual_lidar, create_dual_lidar_mount_frames
from ros2isaacsim.physx_diff_asset import (
    PHYSX_DIFF_CONTACT_BASE_COLLIDER_SPECS as _PHYSX_DIFF_CONTACT_BASE_COLLIDER_SPECS,
    PHYSX_DIFF_CONTACT_UPPER_BODY_COLLIDER_SPECS as _PHYSX_DIFF_CONTACT_UPPER_BODY_COLLIDER_SPECS,
)
try:
    from isaac_utils.robot_geometry import apply_auto_footprint_env
except Exception:  # pragma: no cover
    apply_auto_footprint_env = None  # type: ignore
from isaac_utils.utils import geom
from isaac_utils.utils.assets import get_package_asset_path
from isaac_utils.utils.prim import ensure_path
from rclpy.qos import QoSProfile

from isaacsim_msgs.srv import UrdfToUsd

from .utils import safe

profile = QoSProfile(depth=2000)

parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))


_MECANUM_ARM_LINK_NAMES = {
    "XMS5_R800_W4G3B4C_base",
    "XMS5_R800_W4G3B4C_link1",
    "XMS5_R800_W4G3B4C_link2",
    "XMS5_R800_W4G3B4C_link3",
    "XMS5_R800_W4G3B4C_link4",
    "XMS5_R800_W4G3B4C_link5",
    "link6",
    "tool_link",
    "gripper_link1",
    "gripper_link2",
    "tool_tip_link",
}

_NAV_GRIPPER_JOINT_NAMES = {
    "gripper_joint1",
    "gripper_joint2",
}

_NAV_ROLLER_JOINT_KEYWORDS = (
    "roller",
)

_PHYSX_DIFF_WHEEL_LINKS = (
    "wheel_fl_link",
    "wheel_fr_link",
    "wheel_rl_link",
    "wheel_rr_link",
)

_PHYSX_DIFF_WHEEL_JOINTS = tuple(name.replace("_link", "_joint") for name in _PHYSX_DIFF_WHEEL_LINKS)


def _safe_file_stem(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", name or "robot").strip("_")
    return stem or "robot"


def _usd_safe_identifier(name: str) -> str:
    """Return a USD/URDF-importer-safe identifier while preserving readability."""
    safe = re.sub(r"[^A-Za-z0-9_]", "_", name or "unnamed")
    if not safe or safe[0].isdigit():
        safe = f"_{safe}"
    return safe


def _ensure_stage_path(stage: Usd.Stage, path: str):
    if not path:
        return None
    prim = stage.GetPrimAtPath(path)
    if prim and prim.IsValid():
        return prim
    parent = os.path.dirname(path)
    if parent and parent != path:
        _ensure_stage_path(stage, parent)
    return UsdGeom.Xform.Define(stage, path).GetPrim()


def _choose_reference_prim_path(usd_path: str):
    """Return a prim path inside usd_path that can be referenced onto the stage."""
    temp_stage = Usd.Stage.Open(usd_path, Usd.Stage.LoadNone)
    if not temp_stage:
        return None
    default_prim = temp_stage.GetDefaultPrim()
    if default_prim and default_prim.IsValid():
        return default_prim.GetPath()
    for root_prim in temp_stage.GetPseudoRoot().GetChildren():
        if root_prim.IsA(UsdGeom.Xform):
            return root_prim.GetPath()
    return None


def _reference_usd_into_stage(usd_path: str, prim_path: str) -> bool:
    """Reference an imported robot USD file into the currently open Isaac stage."""
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return False
    _ensure_stage_path(stage, os.path.dirname(prim_path))
    prim = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
    if not prim or not prim.IsValid():
        return False
    prim.GetReferences().ClearReferences()
    ref_prim_path = _choose_reference_prim_path(usd_path)
    if ref_prim_path:
        prim.GetReferences().AddReference(usd_path, ref_prim_path)
    else:
        prim.GetReferences().AddReference(usd_path)
    try:
        prim.SetInstanceable(False)
    except Exception:
        pass
    return True


def _find_descendant_named(root_path: str, name: str) -> str:
    stage = omni.usd.get_context().get_stage()
    if stage is None or not name:
        return os.path.join(root_path, name)
    return _find_descendant_named_on_stage(stage, root_path, name)


def _find_descendant_named_on_stage(stage: Usd.Stage, root_path: str, name: str) -> str:
    if stage is None or not name:
        return os.path.join(root_path, name)
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return os.path.join(root_path, name)
    target_names = {name, name.replace('-', '_')}
    for prim in Usd.PrimRange(root):
        if prim.GetName() in target_names:
            return str(prim.GetPath())
    return os.path.join(root_path, name)


def _collision_paths_under(root_prim) -> list[str]:
    if not root_prim or not root_prim.IsValid():
        return []
    return [
        str(prim.GetPath())
        for prim in Usd.PrimRange(root_prim)
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]


def _collision_prototype_paths(stage: Usd.Stage) -> list[str]:
    paths = []
    for prototype in stage.GetPrototypes():
        for prim in Usd.PrimRange(prototype):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                paths.append(str(prim.GetPath()))
    return paths


def _author_collision_primitive(stage: Usd.Stage, parent_path: str, spec: dict[str, typing.Any]) -> str | None:
    link_prim = stage.GetPrimAtPath(parent_path)
    if not link_prim or not link_prim.IsValid():
        return None

    collider_path = f"{parent_path}/{spec['name']}"
    stage.RemovePrim(collider_path)

    kind = spec["kind"]
    if kind == "cube":
        prim = UsdGeom.Cube.Define(stage, collider_path).GetPrim()
        cube = UsdGeom.Cube(prim)
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(*spec["translate"]))
        xform.AddScaleOp().Set(Gf.Vec3f(*spec["size"]))
    elif kind == "capsule":
        prim = UsdGeom.Capsule.Define(stage, collider_path).GetPrim()
        capsule = UsdGeom.Capsule(prim)
        capsule.CreateAxisAttr().Set(getattr(UsdGeom.Tokens, spec["axis"]))
        capsule.CreateRadiusAttr(float(spec["radius"]))
        capsule.CreateHeightAttr(float(spec["height"]))
        xform = UsdGeom.Xformable(prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(*spec["translate"]))
    else:
        raise ValueError(f"unsupported PhysX collision primitive kind: {kind}")

    prim = stage.GetPrimAtPath(collider_path)
    if not prim or not prim.IsValid():
        return None
    UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    collision_api = UsdPhysics.CollisionAPI.Apply(prim)
    collision_api.CreateCollisionEnabledAttr(True)
    return collider_path


def _expected_physx_diff_contact_collision_paths(stage: Usd.Stage, root_path: str) -> tuple[str, ...]:
    paths = []
    for link_name in _PHYSX_DIFF_WHEEL_LINKS:
        link_path = _find_descendant_named_on_stage(stage, root_path, link_name)
        paths.append(f"{link_path}/physx_diff_tire_collision")
    for spec in _PHYSX_DIFF_CONTACT_BASE_COLLIDER_SPECS + _PHYSX_DIFF_CONTACT_UPPER_BODY_COLLIDER_SPECS:
        link_path = _find_descendant_named_on_stage(stage, root_path, spec["link"])
        paths.append(f"{link_path}/{spec['name']}")
    return tuple(paths)


def _find_articulation_root(root_path: str) -> str:
    """Find the best articulation/root path after referencing an imported robot."""
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return root_path
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return root_path
    for prim in Usd.PrimRange(root):
        applied = set(prim.GetAppliedSchemas())
        if "PhysicsArticulationRootAPI" in applied or "ArticulationRootAPI" in applied:
            return str(prim.GetPath())
    # Unified/imported mecanum robots often keep base_footprint as the logical
    # articulation root even when the schema is not exposed on the wrapper prim.
    guessed = _find_descendant_named(root_path, "base_footprint")
    if guessed and guessed != os.path.join(root_path, "base_footprint"):
        return guessed
    # URDF importer often composes the articulation on the referenced root prim.
    return root_path


def _author_physx_diff_contact_colliders(
    stage: Usd.Stage,
    root_path: str,
    robot_name: str,
) -> bool:
    """Author contact geometry explicitly after the Isaac 4.5 URDF import.

    Isaac 4.5 can preserve the collision Xform names while dropping their
    primitive geometry. These shapes are persisted into the imported asset
    before it is referenced so PhysX discovers them with the articulation.
    """
    if stage is None:
        return False

    material_name = _usd_safe_identifier(f"{robot_name}_physx_diff_tire")
    material = UsdShade.Material.Define(stage, f"{root_path}/PhysicsMaterials/{material_name}")
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    static_friction = float(os.environ.get("ARENA_ISAAC_DIFF_WHEEL_STATIC_FRICTION", "0.25"))
    dynamic_friction = float(os.environ.get("ARENA_ISAAC_DIFF_WHEEL_DYNAMIC_FRICTION", "0.20"))
    friction_combine_mode = os.environ.get(
        "ARENA_ISAAC_DIFF_WHEEL_FRICTION_COMBINE_MODE", "min"
    ).strip().lower()
    if friction_combine_mode not in {"average", "min", "multiply", "max"}:
        raise ValueError(f"invalid PhysX friction combine mode: {friction_combine_mode}")
    drive_damping = float(os.environ.get("ARENA_ISAAC_DIFF_WHEEL_DRIVE_DAMPING", "8.0"))
    drive_max_force = float(os.environ.get("ARENA_ISAAC_DIFF_WHEEL_DRIVE_MAX_FORCE", "35.0"))
    material_api.CreateStaticFrictionAttr().Set(static_friction)
    material_api.CreateDynamicFrictionAttr().Set(dynamic_friction)
    material_api.CreateRestitutionAttr().Set(0.0)
    physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx_material_api.CreateFrictionCombineModeAttr().Set(friction_combine_mode)

    tire_paths = []
    for link_name in _PHYSX_DIFF_WHEEL_LINKS:
        link_path = _find_descendant_named_on_stage(stage, root_path, link_name)
        link_prim = stage.GetPrimAtPath(link_path)
        if not link_prim or not link_prim.IsValid():
            print(f"[urdf_import] missing PhysX differential wheel link: {link_name}", file=sys.stderr)
            return False
        collider_path = f"{link_path}/physx_diff_tire_collision"
        stage.RemovePrim(collider_path)
        cylinder = UsdGeom.Cylinder.Define(stage, collider_path)
        cylinder.CreateRadiusAttr(0.08)
        cylinder.CreateHeightAttr(0.05)
        cylinder.CreateAxisAttr(UsdGeom.Tokens.x)
        cylinder.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        collision_api = UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
        collision_api.CreateCollisionEnabledAttr(True)
        binding = UsdShade.MaterialBindingAPI.Apply(cylinder.GetPrim())
        binding.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
        tire_paths.append(collider_path)

    base_link_path = _find_descendant_named_on_stage(stage, root_path, "base_link")
    base_link = stage.GetPrimAtPath(base_link_path)
    if not base_link or not base_link.IsValid():
        print("[urdf_import] missing base_link for PhysX differential chassis colliders", file=sys.stderr)
        return False
    chassis_paths = []
    for spec in _PHYSX_DIFF_CONTACT_BASE_COLLIDER_SPECS:
        collider_path = _author_collision_primitive(stage, _find_descendant_named_on_stage(stage, root_path, spec["link"]), spec)
        if not collider_path:
            print(f"[urdf_import] failed to author PhysX differential chassis collider: {spec['name']}", file=sys.stderr)
            return False
        chassis_paths.append(collider_path)

    upper_body_paths = []
    for spec in _PHYSX_DIFF_CONTACT_UPPER_BODY_COLLIDER_SPECS:
        link_path = _find_descendant_named_on_stage(stage, root_path, spec["link"])
        collider_path = _author_collision_primitive(stage, link_path, spec)
        if not collider_path:
            print(
                f"[urdf_import] failed to author PhysX differential upper-body collider: {spec['name']} on {spec['link']}",
                file=sys.stderr,
            )
            return False
        upper_body_paths.append(collider_path)

    valid_tires = sum(
        1
        for path in tire_paths
        if stage.GetPrimAtPath(path).HasAPI(UsdPhysics.CollisionAPI)
    )
    valid_chassis = sum(
        1
        for path in chassis_paths
        if stage.GetPrimAtPath(path).HasAPI(UsdPhysics.CollisionAPI)
    )
    valid_upper_body = sum(
        1
        for path in upper_body_paths
        if stage.GetPrimAtPath(path).HasAPI(UsdPhysics.CollisionAPI)
    )

    configured_drives = 0
    for joint_name in _PHYSX_DIFF_WHEEL_JOINTS:
        joint_path = _find_descendant_named_on_stage(stage, root_path, joint_name)
        joint_prim = stage.GetPrimAtPath(joint_path)
        if not joint_prim or not joint_prim.IsValid() or not joint_prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(0.0)
        drive.CreateDampingAttr().Set(drive_damping)
        drive.CreateMaxForceAttr().Set(drive_max_force)
        drive.CreateTargetVelocityAttr().Set(0.0)
        configured_drives += 1
    print(
        f"[urdf_import] authored PhysX differential contact geometry: "
        f"tires={valid_tires}/4, chassis={valid_chassis}/2, "
        f"upper_body={valid_upper_body}/{len(upper_body_paths)}, "
        f"drives={configured_drives}/4, friction=({static_friction:.3f}, {dynamic_friction:.3f}), "
        f"combine={friction_combine_mode}",
        file=sys.stderr,
    )
    return valid_tires == 4 and valid_chassis == 2 and valid_upper_body == len(upper_body_paths) and configured_drives == 4





def _mecanum_mode(robot_model: str) -> str:
    model = (robot_model or "").lower()
    if "physx_diff_contact" in model:
        return "physx_diff_contact"
    if "physx_root_velocity" in model or model.endswith("_physx"):
        return "physx_root_velocity"
    if model.endswith("_hybrid"):
        return "hybrid"
    if model.endswith("_kinematic"):
        return "kinematic"
    return "joint"


def _unified_mecanum_asset_paths(urdf_path: str) -> tuple[Path, Path] | None:
    path = Path(urdf_path).expanduser().resolve()
    if path.suffix.lower() != ".urdf":
        return None
    if "unified_v25730" not in str(path):
        return None
    return path, path.with_suffix(".usd")


def _cache_imported_usd(imported_usd_path: str, cached_usd_path: Path):
    src = Path(imported_usd_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"imported USD not found: {src}")
    cached_usd_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, cached_usd_path)
    src_dir = src.parent
    for extra_name in ("configuration",):
        extra_src = src_dir / extra_name
        extra_dst = cached_usd_path.parent / extra_name
        if extra_src.exists():
            if extra_dst.exists():
                shutil.rmtree(extra_dst)
            shutil.copytree(extra_src, extra_dst)
    for mesh_dir in src_dir.glob("*_meshes"):
        mesh_dst = cached_usd_path.parent / mesh_dir.name
        if mesh_dst.exists():
            shutil.rmtree(mesh_dst)
        shutil.copytree(mesh_dir, mesh_dst)
    return str(cached_usd_path)


def _is_complete_unified_usd_asset(cached_usd_path: Path) -> bool:
    if not cached_usd_path.exists():
        return False
    config_dir = cached_usd_path.parent / "configuration"
    if not config_dir.exists():
        return False
    if not any(config_dir.glob("*.usd")):
        return False
    return True


def _mecanum_dual_lidar_enabled(robot_model: str) -> bool:
    model = (robot_model or "").lower()
    return is_mecanum_model(model) and ("lidar" in model or "laser" in model)


def _disable_physics_tree(root_path: str):
    """Make a referenced robot visual/kinematic by removing common PhysX APIs.

    This is intentionally used for Milestone-B teleop modes where the imported
    URDF has incomplete wheel/roller collision/inertia and would otherwise fall,
    bounce, or fly when PhysX tries to solve a floating articulation.
    """
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return
    removed = {"articulation": 0, "rigid": 0, "collision": 0, "joint_prim_deactivated": 0}
    joints_root = stage.GetPrimAtPath(f"{root_path}/joints")
    if joints_root and joints_root.IsValid():
        try:
            joints_root.SetActive(False)
            removed["joint_prim_deactivated"] += 1
        except Exception:
            pass
    for prim in Usd.PrimRange(root):
        try:
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                removed["articulation"] += 1
        except Exception:
            pass
        try:
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                removed["rigid"] += 1
        except Exception:
            pass
        try:
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
                removed["collision"] += 1
        except Exception:
            pass
    print(f"[urdf_import] Disabled physics under {root_path}: {removed}", file=sys.stderr)


def _set_disable_gravity_for_matching_links(root_path: str, link_names: set[str], disable_gravity: bool):
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return
    updated = 0
    for prim in Usd.PrimRange(root):
        if prim.GetName() not in link_names:
            continue
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        try:
            attr = prim.GetAttribute("physxRigidBody:disableGravity")
            if not attr or not attr.IsValid():
                attr = prim.CreateAttribute("physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool, custom=False)
            attr.Set(bool(disable_gravity))
            updated += 1
        except Exception:
            continue
    print(f"[urdf_import] set disableGravity={disable_gravity} on {updated} matching rigid bodies under {root_path}", file=sys.stderr)


def _configure_navigation_joint_drives(root_path: str):
    """Bias imported joints toward navigation-stable behavior.

    The navigation asset should not let passive mecanum rollers or the gripper
    mimic pair inject large articulation motion during initial settling.  This
    helper configures imported joint drives by name after the USD stage exists.
    """
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return

    updated = {"roller": 0, "gripper": 0}

    def _set_drive_attrs(prim, drive_type: str, *, target_position=None, target_velocity=None, stiffness=None, damping=None, max_force=None):
        try:
            drive = UsdPhysics.DriveAPI.Get(prim, drive_type)
            if not drive:
                drive = UsdPhysics.DriveAPI.Apply(prim, drive_type)
            if target_position is not None:
                attr = drive.GetTargetPositionAttr()
                if not attr:
                    attr = drive.CreateTargetPositionAttr(float(target_position))
                else:
                    attr.Set(float(target_position))
            if target_velocity is not None:
                attr = drive.GetTargetVelocityAttr()
                if not attr:
                    attr = drive.CreateTargetVelocityAttr(float(target_velocity))
                else:
                    attr.Set(float(target_velocity))
            if stiffness is not None:
                attr = drive.GetStiffnessAttr()
                if not attr:
                    attr = drive.CreateStiffnessAttr(float(stiffness))
                else:
                    attr.Set(float(stiffness))
            if damping is not None:
                attr = drive.GetDampingAttr()
                if not attr:
                    attr = drive.CreateDampingAttr(float(damping))
                else:
                    attr.Set(float(damping))
            if max_force is not None:
                attr = drive.GetMaxForceAttr()
                if not attr:
                    attr = drive.CreateMaxForceAttr(float(max_force))
                else:
                    attr.Set(float(max_force))
            return True
        except Exception:
            return False

    for prim in Usd.PrimRange(root):
        name = prim.GetName().lower()
        drive_type = None
        if prim.IsA(UsdPhysics.RevoluteJoint):
            drive_type = "angular"
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            drive_type = "linear"
        if drive_type is None:
            continue

        if any(keyword in name for keyword in _NAV_ROLLER_JOINT_KEYWORDS):
            if _set_drive_attrs(
                prim,
                drive_type,
                target_velocity=0.0,
                stiffness=0.0,
                damping=5.0e4,
                max_force=1.0e5,
            ):
                updated["roller"] += 1
            continue

        if name in _NAV_GRIPPER_JOINT_NAMES:
            if _set_drive_attrs(
                prim,
                drive_type,
                target_position=0.0,
                target_velocity=0.0,
                stiffness=1.0e5,
                damping=1.0e4,
                max_force=1.0e5,
            ):
                updated["gripper"] += 1

    print(
        f"[urdf_import] navigation joint drive tuning under {root_path}: "
        f"rollers={updated['roller']}, grippers={updated['gripper']}",
        file=sys.stderr,
    )


def _is_empty_link(link_elem) -> bool:
    if link_elem is None:
        return False
    for child in list(link_elem):
        if child.tag in {"visual", "collision", "inertial"}:
            return False
    return True


def _make_mesh_paths_absolute(robot_elem, original_urdf_path: str, robot_name: str = "robot"):
    """Make mesh paths importer-safe for a sanitized URDF written to /tmp.

    Isaac's URDF importer may derive USD prim names from mesh basenames.  This
    robot has mesh files such as `XMS5-R800-W4G3B4C_base.stl`; even if link
    names are sanitized, those mesh basenames can still trigger:
        Invalid prim name 'XMS5-R800-W4G3B4C_base'
        RuntimeError: Used null prim

    For local meshes, copy/symlink them to a temp directory with USD-safe
    basenames and point the sanitized URDF at those copied files.  Do not alter
    remote/package URLs because resolving package:// requires ROS package index
    context and should be handled upstream.
    """
    urdf_dir = Path(original_urdf_path).resolve().parent
    mesh_out_dir = Path(tempfile.gettempdir()) / "arena_isaac_urdf_imports" / f"{_safe_file_stem(robot_name)}_meshes"
    mesh_out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for mesh in robot_elem.findall(".//mesh"):
        filename = mesh.attrib.get("filename", "")
        if not filename:
            continue
        if filename.startswith(("package://", "http://", "https://", "omniverse://")):
            continue

        src = Path(filename)
        if not src.is_absolute():
            src = (urdf_dir / filename).resolve()
        else:
            src = src.resolve()

        if not src.exists():
            # Keep an absolute path for better diagnostics from the importer.
            mesh.attrib["filename"] = str(src)
            continue

        safe_stem = _usd_safe_identifier(src.stem)
        # Avoid collisions when two folders contain meshes with the same basename.
        digest = hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:8]
        dst = mesh_out_dir / f"{safe_stem}_{digest}{src.suffix}"

        try:
            if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                shutil.copy2(src, dst)
                copied += 1
            mesh.attrib["filename"] = str(dst)
        except Exception as exc:
            print(f"[urdf_import] Failed to copy mesh {src} -> {dst}: {exc}", file=sys.stderr)
            mesh.attrib["filename"] = str(src)

    if copied:
        print(f"[urdf_import] Copied {copied} meshes to USD-safe paths under {mesh_out_dir}", file=sys.stderr)


def _sanitize_urdf_identifiers(robot_elem) -> bool:
    """Sanitize URDF identifiers that Isaac's URDF importer later maps to USD prim names.

    Isaac/USD prim names cannot contain characters such as '-'.  The importer can
    warn and rename some link names, but in this robot it still hits `Used null
    prim` while processing link/collision names like `XMS5-R800-W4G3B4C_base`.
    Sanitize all URDF `name` identifiers and fix references in parent/child,
    gazebo, mimic, and transmission tags before import.  Mesh filenames are left
    unchanged.
    """
    changed = False
    link_map = {}
    joint_map = {}

    for link in robot_elem.findall("link"):
        old_name = link.attrib.get("name")
        new_name = _usd_safe_identifier(old_name)
        if old_name and new_name != old_name:
            link.attrib["name"] = new_name
            link_map[old_name] = new_name
            changed = True

    for joint in robot_elem.findall("joint"):
        old_name = joint.attrib.get("name")
        new_name = _usd_safe_identifier(old_name)
        if old_name and new_name != old_name:
            joint.attrib["name"] = new_name
            joint_map[old_name] = new_name
            changed = True

    # Names of visual/collision/inertial/transmission/etc. can also become USD
    # child prim names.  Sanitize them broadly, but do not touch mesh filenames.
    for elem in robot_elem.iter():
        if "name" in elem.attrib and elem.tag not in {"link", "joint"}:
            old_name = elem.attrib.get("name")
            new_name = _usd_safe_identifier(old_name)
            if old_name and new_name != old_name:
                elem.attrib["name"] = new_name
                changed = True

    for parent in robot_elem.findall(".//parent"):
        ref = parent.attrib.get("link")
        if ref in link_map:
            parent.attrib["link"] = link_map[ref]
            changed = True

    for child in robot_elem.findall(".//child"):
        ref = child.attrib.get("link")
        if ref in link_map:
            child.attrib["link"] = link_map[ref]
            changed = True

    for gazebo in robot_elem.findall(".//gazebo"):
        ref = gazebo.attrib.get("reference")
        if ref in link_map:
            gazebo.attrib["reference"] = link_map[ref]
            changed = True
        elif ref in joint_map:
            gazebo.attrib["reference"] = joint_map[ref]
            changed = True

    for mimic in robot_elem.findall(".//mimic"):
        ref = mimic.attrib.get("joint")
        if ref in joint_map:
            mimic.attrib["joint"] = joint_map[ref]
            changed = True

    for hardware_joint in robot_elem.findall(".//joint"):
        # This covers ros2_control <joint name=...> children too; robot-level
        # joints were already changed above.
        ref = hardware_joint.attrib.get("name")
        if ref in joint_map:
            hardware_joint.attrib["name"] = joint_map[ref]
            changed = True

    if link_map or joint_map:
        print(f"[urdf_import] Sanitized link names: {link_map}", file=sys.stderr)
        print(f"[urdf_import] Sanitized joint names: {joint_map}", file=sys.stderr)

    return changed


def _prepare_urdf_for_isaac_import(urdf_path: str, robot_name: str) -> str:
    """Return an Isaac-importer-friendly URDF path.

    Isaac Sim 4.5's URDF importer may crash with RuntimeError("Used null prim")
    when the URDF root link is an empty frame link such as base_footprint, fixed
    to the real base link.  The mecanum730_xms5 URDF has exactly this shape:
        base_footprint --fixed--> base_link
    For import only, remove that empty root link and its fixed joint, then make
    relative mesh paths absolute because the sanitized URDF is written to /tmp.
    """
    src = Path(urdf_path).expanduser().resolve()
    try:
        tree = ET.parse(src)
        robot = tree.getroot()
    except Exception as exc:
        print(f"[urdf_import] Could not parse URDF for sanitation, using original: {exc}", file=sys.stderr)
        return str(src)

    links = {link.attrib.get("name"): link for link in robot.findall("link")}
    joints = list(robot.findall("joint"))
    children = set()
    parent_to_joints = {}
    for joint in joints:
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_name = parent.attrib.get("link")
        child_name = child.attrib.get("link")
        if child_name:
            children.add(child_name)
        parent_to_joints.setdefault(parent_name, []).append(joint)

    roots = [name for name in links if name not in children]
    changed = False

    if len(roots) == 1:
        root_name = roots[0]
        root_link = links.get(root_name)
        outgoing = parent_to_joints.get(root_name, [])
        if _is_empty_link(root_link) and len(outgoing) == 1:
            joint = outgoing[0]
            child_elem = joint.find("child")
            origin_elem = joint.find("origin")
            joint_type = joint.attrib.get("type", "")
            origin_is_identity = True
            if origin_elem is not None:
                xyz = origin_elem.attrib.get("xyz", "0 0 0").split()
                rpy = origin_elem.attrib.get("rpy", "0 0 0").split()
                try:
                    vals = [float(v) for v in xyz + rpy]
                    origin_is_identity = all(abs(v) < 1e-9 for v in vals)
                except Exception:
                    origin_is_identity = False
            if joint_type == "fixed" and child_elem is not None and origin_is_identity:
                robot.remove(root_link)
                robot.remove(joint)
                changed = True
                print(
                    f"[urdf_import] Removed empty fixed root link {root_name}; "
                    f"new root is {child_elem.attrib.get('link')}",
                    file=sys.stderr,
                )

    if _sanitize_urdf_identifiers(robot):
        changed = True

    _make_mesh_paths_absolute(robot, str(src), robot_name)

    out_dir = Path(tempfile.gettempdir()) / "arena_isaac_urdf_imports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_safe_file_stem(robot_name)}_isaac_import.urdf"
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    if changed:
        print(f"[urdf_import] Wrote sanitized URDF: {out_path}", file=sys.stderr)
    return str(out_path)


def import_urdf(
    urdf_path: str,
    prim_path: str,
    robot_name: str = "robot",
    author_physx_diff_contact: bool = False,
) -> typing.Optional[str]:
    """Import URDF to a real .usd file, then reference that file at prim_path."""

    status, import_config = commands.execute("URDFCreateImportConfig")
    if not status:
        return None

    import_config.set_merge_fixed_joints(False)
    import_config.set_convex_decomp(False)
    import_config.set_import_inertia_tensor(False)
    import_config.set_make_default_prim(True)
    import_config.set_distance_scale(1.0)
    import_config.set_fix_base(False)
    import_config.set_default_drive_type(2)
    import_config.set_self_collision(False)
    try:
        import_config.set_collision_from_visuals(False)
    except Exception:
        pass
    import_config.make_default_prim = True

    importer_urdf_path = _prepare_urdf_for_isaac_import(urdf_path, robot_name)

    out_dir = Path(tempfile.gettempdir()) / "arena_isaac_urdf_imports"
    out_dir.mkdir(parents=True, exist_ok=True)
    usd_dest_path = out_dir / f"{_safe_file_stem(robot_name)}.usd"
    if usd_dest_path.exists():
        try:
            usd_dest_path.unlink()
        except Exception:
            pass

    try:
        status, usd_path = commands.execute(
            "URDFParseAndImportFile",
            urdf_path=importer_urdf_path,
            import_config=import_config,
            dest_path=str(usd_dest_path),
        )
    except Exception as exc:
        print(f"[urdf_import] URDFParseAndImportFile failed: {exc}", file=sys.stderr)
        return None

    if not status:
        print("[urdf_import] URDFParseAndImportFile returned status=False", file=sys.stderr)
        return None

    # Isaac 4.5 URDFParseAndImportFile may return an internal/default-prim
    # identifier such as "/mecanum730_xms5_gripper" rather than the actual
    # layer file path. Always reference the destination .usd file we passed in.
    returned_usd_path = str(usd_path or "")
    usd_path = str(usd_dest_path)

    if not Path(usd_path).exists():
        print(
            f"[urdf_import] Expected USD file was not created: {usd_path}; "
            f"importer returned: {returned_usd_path}",
            file=sys.stderr,
        )
        return None

    if author_physx_diff_contact:
        asset_stage = Usd.Stage.Open(usd_path, Usd.Stage.LoadAll)
        asset_root = asset_stage.GetDefaultPrim() if asset_stage else None
        if not asset_root or not asset_root.IsValid():
            print("[urdf_import] imported PhysX differential asset has no default prim", file=sys.stderr)
            return None
        imported_colliders = _collision_paths_under(asset_root)
        imported_collision_prototypes = _collision_prototype_paths(asset_stage)
        if imported_colliders or imported_collision_prototypes:
            print(
                "[urdf_import] refusing PhysX differential asset with importer-authored "
                f"colliders={imported_colliders[:8]}, "
                f"collision_prototypes={imported_collision_prototypes[:8]}",
                file=sys.stderr,
            )
            return None
        if not _author_physx_diff_contact_colliders(asset_stage, str(asset_root.GetPath()), robot_name):
            print("[urdf_import] failed to author PhysX differential contact asset", file=sys.stderr)
            return None
        final_colliders = _collision_paths_under(asset_root)
        final_collision_prototypes = _collision_prototype_paths(asset_stage)
        expected_collision_paths = set(_expected_physx_diff_contact_collision_paths(asset_stage, str(asset_root.GetPath())))
        final_collision_paths = set(final_colliders)
        if final_collision_paths != expected_collision_paths or final_collision_prototypes:
            missing_colliders = sorted(expected_collision_paths - final_collision_paths)
            extra_colliders = sorted(final_collision_paths - expected_collision_paths)
            print(
                "[urdf_import] invalid PhysX differential contact asset after authoring: "
                f"colliders={len(final_colliders)}/{len(expected_collision_paths)} "
                f"missing={missing_colliders[:8]} extra={extra_colliders[:8]}, "
                f"collision_prototypes={final_collision_prototypes[:8]}",
                file=sys.stderr,
            )
            return None
        asset_stage.GetRootLayer().Save()
        print(
            f"[urdf_import] persisted PhysX differential contact geometry in {usd_path}; "
            f"colliders={len(final_colliders)}/{len(expected_collision_paths)}, collision_prototypes=0",
            file=sys.stderr,
        )

    print(
        f"[urdf_import] URDF importer returned {returned_usd_path}; "
        f"using layer file {usd_path}",
        file=sys.stderr,
    )

    if not _reference_usd_into_stage(usd_path, prim_path):
        print(f"[urdf_import] Failed to reference {usd_path} at {prim_path}", file=sys.stderr)
        return None

    print(f"imported URDF {importer_urdf_path} -> {usd_path}, referenced at {prim_path}", file=sys.stderr)
    return usd_path


@safe
def urdf_to_usd(request, response):
    name = request.name
    urdf_path = request.urdf_path
    robot_model = request.robot_model

    prim_path = Paths.scene.robot(name)
    mecanum_mode = _mecanum_mode(robot_model) if is_mecanum_model(robot_model) else ""
    physical_contact_mecanum = is_mecanum_model(robot_model) and mecanum_mode == "physx_diff_contact"
    unified_asset = _unified_mecanum_asset_paths(urdf_path) if is_mecanum_model(robot_model) else None
    unified_urdf_path = None
    unified_usd_path = None
    if unified_asset is not None:
        unified_urdf_path, unified_usd_path = unified_asset
    usd_path = import_urdf(
        urdf_path,
        prim_path,
        name,
        author_physx_diff_contact=physical_contact_mecanum,
    )
    if usd_path is None:
        return response
    if unified_usd_path is not None:
        try:
            cached_usd = _cache_imported_usd(usd_path, unified_usd_path)
            if _reference_usd_into_stage(cached_usd, prim_path):
                usd_path = cached_usd
            print(f"[urdf_import] cached unified mecanum asset {cached_usd} from {unified_urdf_path}", file=sys.stderr)
        except Exception as exc:
            print(f"[urdf_import] failed to cache unified mecanum asset: {exc}", file=sys.stderr)

    robot_stage_path = _find_articulation_root(prim_path)
    base_prim_path = _find_descendant_named(prim_path, request.base_frame)
    # For mecanum hybrid/kinematic teleop, do not let the malformed floating
    # articulation fall under PhysX.  Move the stable wrapper Xform instead of
    # the articulation root.  This is the mode to use for the provided URDF,
    # whose wheel and roller links have little/no collision geometry.
    if is_mecanum_model(robot_model) and mecanum_mode in ("hybrid", "kinematic"):
        _disable_physics_tree(prim_path)
        pose_target_path = prim_path
        controller_prim_path = prim_path
    else:
        pose_target_path = robot_stage_path
        controller_prim_path = robot_stage_path

    print(
        f"robot stage path={robot_stage_path}, base prim path={base_prim_path}, "
        f"pose_target={pose_target_path}, controller_prim={controller_prim_path}, mode={mecanum_mode}",
        file=sys.stderr,
    )

    # Apply the requested spawn pose before creating sensors/controllers.
    geom.move(
        prim_path=pose_target_path,
        translation=geom.Translation.parse(request.pose.position),
        rotation=geom.Rotation.parse(request.pose.orientation),
    )
    print(f"[urdf_import] Applied initial pose to {pose_target_path}", file=sys.stderr)

    # V16: derive the mobile-base footprint from the imported robot stage
    # geometry before the mecanum controller instantiates its guard.  This
    # replaces scattered empirical guard dimensions when enabled by profile/env.
    if is_mecanum_model(robot_model) and apply_auto_footprint_env is not None:
        try:
            apply_auto_footprint_env(prim_path, logger=None)
        except Exception as exc:
            print(f"[urdf_import] auto footprint estimation failed: {exc}", file=sys.stderr)

    dual_lidar_mecanum = _mecanum_dual_lidar_enabled(robot_model)
    lidar_backend = os.environ.get("ARENA_ISAAC_LIDAR_BACKEND", "rtx").strip().lower()
    if dual_lidar_mecanum:
        try:
            for lidar_root in [controller_prim_path]:
                if lidar_backend in {"synthetic", "synthetic_2d", "2d", "nav_2d"}:
                    create_dual_lidar_mount_frames(
                        robot_root_path=lidar_root,
                        robot_name=name,
                        logger=None,
                    )
                else:
                    create_dual_lidar(
                        robot_root_path=lidar_root,
                        robot_name=name,
                        logger=None,
                    )
            if lidar_backend in {"synthetic", "synthetic_2d", "2d", "nav_2d"}:
                print(
                    f"[urdf_import] Created calibration lidar mount frames only; RTX lidar disabled for backend={lidar_backend}",
                    file=sys.stderr,
                )
        except Exception as exc:
            print(f"[urdf_import] Dual lidar setup failed: {exc}", file=sys.stderr)

    stable_kinematic_mecanum = is_mecanum_model(robot_model) and mecanum_mode == "kinematic"
    single_source_mecanum_tf = stable_kinematic_mecanum or physical_contact_mecanum
    if single_source_mecanum_tf:
        # The controller is the single authoritative odom/tf source. Do not create Isaac ROS
        # bridge odom/tf/joint_state graphs that target a guessed base_link
        # path, because imported/reference assets can differ in hierarchy and
        # Isaac 4.5 may crash the graph when target prims are invalid.
        if stable_kinematic_mecanum:
            reason = "stable kinematic mecanum"
        elif physical_contact_mecanum:
            reason = "physical differential-contact mecanum"
        print(
            f"[urdf_import] Skipping Isaac ROS bridge odom/tf/joint_states for {reason} {name}",
            file=sys.stderr,
        )
    else:
        if not request.no_localization:
            odom.odom(
                os.path.join(prim_path, 'odom_publisher'),
                prim_path=base_prim_path,
                base_frame_id=os.path.join(name, request.base_frame),
                odom_frame_id=os.path.join(name, request.odom_frame),
            )

        tf.tf(
            os.path.join(prim_path, 'tf_publisher'),
            prim_path=base_prim_path,
            tf_prefix=name,
        )

        joint_states.joint_states(
            os.path.join(prim_path, 'joint_states_publisher'),
            prim_path=base_prim_path,
            joint_states_topic=f"/task_generator_node/{name}/joint_states",
        )

    if request.cmd_vel_topic:
        if is_mecanum_model(robot_model):
            if single_source_mecanum_tf:
                print(
                    f"[urdf_import] registering mecanum custom odom/tf publisher as the only TF source for {name}",
                    file=sys.stderr,
                )
            mecanum_teleop_manager.add_robot(
                name=name,
                prim_path=controller_prim_path,
                articulation_path=robot_stage_path,
                nav_base_path=controller_prim_path,
                cmd_vel_topic=request.cmd_vel_topic,
                robot_model=robot_model,
                odom_frame=request.odom_frame,
                base_frame=request.base_frame,
                asset_root_path=prim_path,
            )
        else:
            control.Control(
                prim_path=robot_stage_path,
                cmd_vel_topic=request.cmd_vel_topic,
            ).parse(
                robot_model=robot_model,
            )

    if single_source_mecanum_tf:
        if stable_kinematic_mecanum:
            reason = "stable kinematic mecanum"
        elif physical_contact_mecanum:
            reason = "physical differential-contact mecanum"
        else:
            reason = "prebuilt/unified physx mecanum"
        print(f"[urdf_import] Skipping optional sensor graph setup for {reason} {name}", file=sys.stderr)
    else:
        try:
            with open(request.urdf_path, 'r') as f:
                sensors.Sensors(
                    prim_path=prim_path,
                    base_topic=os.path.dirname(request.cmd_vel_topic)
                ).parse_gazebo(f.read())
        except Exception as exc:
            print(f"[urdf_import] Optional sensor graph setup skipped: {exc}", file=sys.stderr)

    response.usd_path = prim_path
    return response

# Urdf importer service callback.


def convert_urdf_to_usd(controller):
    service = controller.create_service(
        srv_type=UrdfToUsd,
        qos_profile=profile,
        srv_name='isaac/urdf_to_usd',
        callback=urdf_to_usd
    )
    return service
