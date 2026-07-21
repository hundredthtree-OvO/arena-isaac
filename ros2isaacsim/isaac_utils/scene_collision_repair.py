"""Editable scene collision proxy utilities for imported USD scenes.

V13.2 changes over v13.1:
- Recursive proxy generation: if a high-level Wall/Door/Partition object is too broad,
  the generator descends into child meshes and/or creates an editable fallback proxy.
- Oversized wall/door candidates are no longer silently skipped.  They are generated
  as editable boxes so the user can shrink/move them in Isaac Sim and export YAML.
- Door proxies remain disabled by default; wall proxies remain enabled by default.

Workflow:
  auto + visible + guard off -> edit proxy boxes in Isaac Sim -> export YAML -> config + guard on.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import omni.usd
import yaml
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

try:
    from pxr import PhysxSchema  # type: ignore
except Exception:  # pragma: no cover
    PhysxSchema = None  # type: ignore


@dataclass
class SceneRepairConfig:
    enabled: bool = True
    mode: str = "safe"  # off | report | safe | aggressive
    report_dir: str = "/tmp/arena_isaac_collision_reports"
    proxy_visible: bool = False
    proxy_root_name: str = "_collision_proxy"

    # V13.1+/V13.2 editable proxy workflow.
    proxy_source: str = "auto"  # auto | config | off
    proxy_config_path: str = "/home/stardust/resources/arena_ws/arena_assets/collision_configs/shenxinfu_841837.proxies.yaml"
    auto_write_config: bool = True

    thin_threshold_m: float = 0.03
    min_proxy_xy_m: float = 0.05
    min_proxy_z_m: float = 0.35
    proxy_margin_m: float = 0.02
    disable_original_for_proxy_roots: bool = True

    enable_wall_proxy: bool = True
    enable_door_proxy: bool = True
    enable_partition_proxy: bool = True
    enable_toilet_proxy: bool = True
    enable_urinal_proxy: bool = True
    enable_fixture_proxy: bool = True
    enable_cabinet_proxy: bool = True

    wall_default_enabled: bool = True
    door_default_enabled: bool = False
    # All doors retain the legacy disabled policy unless explicitly selected.
    door_collision_policy: str = "disabled"  # disabled | restore_selected_meshes
    door_collision_targets: Tuple[str, ...] = ()
    door_collision_meshes: Tuple[str, ...] = ()
    door_leaf_approximation: str = "convexHull"
    door_frame_approximation: str = "sdf"
    door_frame_sdf_resolution: int = 256
    partition_default_enabled: bool = True
    toilet_default_enabled: bool = True
    urinal_default_enabled: bool = True
    fixture_default_enabled: bool = True
    cabinet_default_enabled: bool = True

    max_proxy_count: int = 1024
    max_wall_length_m: float = 8.0
    max_wall_thickness_m: float = 0.80
    max_door_length_m: float = 1.50
    max_door_thickness_m: float = 0.35

    # V13.2: when a coarse prim is too broad, descend into child mesh/boundable prims.
    recursive_proxy_generation: bool = True
    # V13.2: if no usable child proxy is found, still create an editable fallback
    # proxy for the original coarse object instead of silently skipping it.
    create_oversized_editable_fallback: bool = True
    # For fallback proxy, keep wall enabled by default, but door disabled by default.
    # This preserves editability without forcing door openings closed.


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _env_csv(name: str) -> Tuple[str, ...]:
    return tuple(
        value.strip()
        for value in os.environ.get(name, "").split(",")
        if value.strip()
    )


def config_from_env() -> SceneRepairConfig:
    mode = os.environ.get("ARENA_ISAAC_SCENE_COLLISION_REPAIR_MODE", "safe").strip().lower()
    enabled = _env_bool("ARENA_ISAAC_ENABLE_SCENE_COLLISION_REPAIR", True)
    if mode == "off":
        enabled = False
    return SceneRepairConfig(
        enabled=enabled,
        mode=mode,
        report_dir=os.environ.get("ARENA_ISAAC_SCENE_COLLISION_REPORT_DIR", "/tmp/arena_isaac_collision_reports"),
        proxy_visible=_env_bool("ARENA_ISAAC_SCENE_COLLISION_PROXY_VISIBLE", False),
        proxy_source=os.environ.get("ARENA_ISAAC_SCENE_COLLISION_PROXY_SOURCE", "auto").strip().lower(),
        proxy_config_path=os.environ.get("ARENA_ISAAC_SCENE_COLLISION_PROXY_CONFIG", "/home/stardust/resources/arena_ws/arena_assets/collision_configs/shenxinfu_841837.proxies.yaml"),
        auto_write_config=_env_bool("ARENA_ISAAC_SCENE_COLLISION_AUTO_WRITE_CONFIG", True),
        thin_threshold_m=_env_float("ARENA_ISAAC_SCENE_COLLISION_THIN_THRESHOLD", 0.03),
        min_proxy_xy_m=_env_float("ARENA_ISAAC_SCENE_COLLISION_MIN_PROXY_XY", 0.05),
        min_proxy_z_m=_env_float("ARENA_ISAAC_SCENE_COLLISION_MIN_PROXY_Z", 0.35),
        proxy_margin_m=_env_float("ARENA_ISAAC_SCENE_COLLISION_PROXY_MARGIN", 0.02),
        disable_original_for_proxy_roots=_env_bool("ARENA_ISAAC_SCENE_COLLISION_DISABLE_ORIGINAL_FOR_PROXY_ROOTS", True),
        enable_wall_proxy=_env_bool("ARENA_ISAAC_SCENE_COLLISION_ENABLE_WALL_PROXY", True),
        enable_door_proxy=_env_bool("ARENA_ISAAC_SCENE_COLLISION_ENABLE_DOOR_PROXY", True),
        enable_partition_proxy=_env_bool("ARENA_ISAAC_SCENE_COLLISION_ENABLE_PARTITION_PROXY", True),
        enable_toilet_proxy=_env_bool("ARENA_ISAAC_SCENE_COLLISION_ENABLE_TOILET_PROXY", True),
        enable_urinal_proxy=_env_bool("ARENA_ISAAC_SCENE_COLLISION_ENABLE_URINAL_PROXY", True),
        enable_fixture_proxy=_env_bool("ARENA_ISAAC_SCENE_COLLISION_ENABLE_FIXTURE_PROXY", True),
        enable_cabinet_proxy=_env_bool("ARENA_ISAAC_SCENE_COLLISION_ENABLE_CABINET_PROXY", True),
        wall_default_enabled=_env_bool("ARENA_ISAAC_SCENE_COLLISION_WALL_DEFAULT_ENABLED", True),
        door_default_enabled=_env_bool("ARENA_ISAAC_SCENE_COLLISION_DOOR_DEFAULT_ENABLED", False),
        door_collision_policy=os.environ.get(
            "ARENA_ISAAC_SCENE_DOOR_COLLISION_POLICY", "disabled"
        ).strip(),
        door_collision_targets=_env_csv("ARENA_ISAAC_SCENE_DOOR_COLLISION_TARGETS"),
        door_collision_meshes=_env_csv("ARENA_ISAAC_SCENE_DOOR_COLLISION_MESHES"),
        door_leaf_approximation=os.environ.get(
            "ARENA_ISAAC_SCENE_DOOR_LEAF_APPROXIMATION", "convexHull"
        ).strip(),
        door_frame_approximation=os.environ.get(
            "ARENA_ISAAC_SCENE_DOOR_FRAME_APPROXIMATION", "sdf"
        ).strip(),
        door_frame_sdf_resolution=max(
            1,
            int(_env_float("ARENA_ISAAC_SCENE_DOOR_FRAME_SDF_RESOLUTION", 256)),
        ),
        partition_default_enabled=_env_bool("ARENA_ISAAC_SCENE_COLLISION_PARTITION_DEFAULT_ENABLED", True),
        toilet_default_enabled=_env_bool("ARENA_ISAAC_SCENE_COLLISION_TOILET_DEFAULT_ENABLED", True),
        urinal_default_enabled=_env_bool("ARENA_ISAAC_SCENE_COLLISION_URINAL_DEFAULT_ENABLED", True),
        fixture_default_enabled=_env_bool("ARENA_ISAAC_SCENE_COLLISION_FIXTURE_DEFAULT_ENABLED", True),
        cabinet_default_enabled=_env_bool("ARENA_ISAAC_SCENE_COLLISION_CABINET_DEFAULT_ENABLED", True),
        max_proxy_count=int(float(os.environ.get("ARENA_ISAAC_SCENE_COLLISION_MAX_PROXIES", "1024"))),
        max_wall_length_m=_env_float("ARENA_ISAAC_SCENE_COLLISION_MAX_WALL_LENGTH", 8.0),
        max_wall_thickness_m=_env_float("ARENA_ISAAC_SCENE_COLLISION_MAX_WALL_THICKNESS", 0.80),
        max_door_length_m=_env_float("ARENA_ISAAC_SCENE_COLLISION_MAX_DOOR_LENGTH", 1.50),
        max_door_thickness_m=_env_float("ARENA_ISAAC_SCENE_COLLISION_MAX_DOOR_THICKNESS", 0.35),
        recursive_proxy_generation=_env_bool("ARENA_ISAAC_SCENE_COLLISION_RECURSIVE_GENERATION", True),
        create_oversized_editable_fallback=_env_bool("ARENA_ISAAC_SCENE_COLLISION_OVERSIZED_FALLBACK", True),
    )


def _safe_token(text: str, max_len: int = 64) -> str:
    t = re.sub(r"[^A-Za-z0-9_]", "_", text)
    t = re.sub(r"_+", "_", t).strip("_") or "prim"
    if len(t) <= max_len:
        return t
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{t[:max_len-9]}_{h}"


def _path_hash(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:10]


def _is_under_proxy_root(path: str, proxy_root_name: str) -> bool:
    return f"/{proxy_root_name}/" in path or path.endswith(f"/{proxy_root_name}")


def _has_collision_like(prim: Usd.Prim) -> bool:
    try:
        schemas = list(prim.GetAppliedSchemas())
    except Exception:
        schemas = []
    if any("CollisionAPI" in str(s) for s in schemas):
        return True
    for attr_name in ("physics:collisionEnabled", "physics:approximation"):
        attr = prim.GetAttribute(attr_name)
        if attr and attr.IsValid():
            return True
    return False


def _set_collision_enabled(prim: Usd.Prim, enabled: bool) -> bool:
    try:
        api = UsdPhysics.CollisionAPI.Apply(prim) if not prim.HasAPI(UsdPhysics.CollisionAPI) else UsdPhysics.CollisionAPI(prim)
        api.CreateCollisionEnabledAttr(bool(enabled)).Set(bool(enabled))
        return True
    except Exception:
        try:
            prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(bool(enabled))
            return True
        except Exception:
            return False


def _disable_collision_recursive(root: Usd.Prim) -> int:
    count = 0
    for prim in Usd.PrimRange(root):
        if _has_collision_like(prim):
            if _set_collision_enabled(prim, False):
                count += 1
    return count


def _is_selected_door_mesh(prim: Usd.Prim, config: SceneRepairConfig) -> bool:
    path_parts = tuple(part for part in prim.GetPath().pathString.split("/") if part)
    return (
        prim.GetName() in config.door_collision_meshes
        and any(target in path_parts for target in config.door_collision_targets)
    )


def _door_mesh_approximation(prim: Usd.Prim, config: SceneRepairConfig) -> str:
    name = prim.GetName().lower()
    if "frame" in name or name == "static":
        return config.door_frame_approximation
    return config.door_leaf_approximation


def _restore_selected_door_mesh_collisions(root: Usd.Prim, config: SceneRepairConfig, report: Dict) -> int:
    restored = 0
    for prim in Usd.PrimRange(root):
        if not prim.IsValid() or not prim.IsActive() or not prim.IsA(UsdGeom.Mesh):
            continue
        if not _is_selected_door_mesh(prim, config) or not _has_collision_like(prim):
            continue
        approximation = _door_mesh_approximation(prim, config)
        try:
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True).Set(True)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(approximation)
            if approximation == "sdf" and PhysxSchema is not None:
                PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim).CreateSdfResolutionAttr().Set(
                    int(config.door_frame_sdf_resolution)
                )
            restored += 1
            report.setdefault("restored", []).append(
                {
                    "path": prim.GetPath().pathString,
                    "reason": "contact_mode_door_mesh_collision",
                    "approximation": approximation,
                    "sdf_resolution": (
                        int(config.door_frame_sdf_resolution)
                        if approximation == "sdf"
                        else None
                    ),
                }
            )
        except Exception as exc:
            report.setdefault("warnings", []).append(
                f"failed to restore door collision {prim.GetPath().pathString}: {exc}"
            )
    return restored


def _world_bbox(cache: UsdGeom.BBoxCache, prim: Usd.Prim) -> Optional[Tuple[Gf.Vec3d, Gf.Vec3d, Gf.Vec3d]]:
    try:
        box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        mn = box.GetMin()
        mx = box.GetMax()
        size = mx - mn
        vals = [mn[0], mn[1], mn[2], mx[0], mx[1], mx[2]]
        if not all(math.isfinite(float(v)) for v in vals):
            return None
        if size[0] <= 0.0 or size[1] <= 0.0 or size[2] <= 0.0:
            return None
        return mn, mx, size
    except Exception:
        return None


def _classify_proxy_candidate(prim: Usd.Prim, config: Optional[SceneRepairConfig] = None) -> Optional[str]:
    config = config or SceneRepairConfig()
    name = prim.GetName().lower()
    path = prim.GetPath().pathString.lower()
    if config.enable_wall_proxy and (name.startswith("wall_") or "/wall_" in path or "/wall/" in path):
        return "wall"
    if config.enable_door_proxy and (name.startswith("door_") or "/door_" in path or "/door/" in path):
        return "door"
    if config.enable_partition_proxy and (name.startswith("partition_") or "/partition_" in path):
        return "partition"
    if config.enable_toilet_proxy and (name.startswith("toilet_") or "/toilet_" in path):
        return "toilet"
    if config.enable_urinal_proxy and (name.startswith("edestalurinal_") or "/edestalurinal_" in path or "urinal" in name):
        return "urinal"
    if config.enable_fixture_proxy and (name.startswith("fpd_") or "/fpd_" in path or "fixture" in name):
        return "fixture"
    if config.enable_cabinet_proxy and (name.startswith("wardrobe_") or "/wardrobe_" in path or "cabinet" in name or "sink" in name or "basin" in name):
        return "cabinet"
    return None


def _kind_default_enabled(kind: str, config: SceneRepairConfig) -> bool:
    return bool({
        "wall": config.wall_default_enabled,
        "door": config.door_default_enabled,
        "partition": config.partition_default_enabled,
        "toilet": config.toilet_default_enabled,
        "urinal": config.urinal_default_enabled,
        "fixture": config.fixture_default_enabled,
        "cabinet": config.cabinet_default_enabled,
    }.get(kind, True))


def _classify_disable_candidate(prim: Usd.Prim, bbox: Optional[Tuple[Gf.Vec3d, Gf.Vec3d, Gf.Vec3d]], config: SceneRepairConfig) -> Optional[str]:
    name = prim.GetName().lower()
    path = prim.GetPath().pathString.lower()
    if any(k in path for k in ["ceiling", "lamp", "light", "vent", "air", "hvac"]):
        return "ceiling_light_vent_complex_mesh"
    if "/door_" in path or name.startswith("door_") or "/door/" in path:
        return "door_collision_disabled_in_safe_policy"
    if bbox is not None:
        mn, mx, size = bbox
        center_z = float((mn[2] + mx[2]) * 0.5)
        min_dim = min(float(size[0]), float(size[1]), float(size[2]))
        max_dim = max(float(size[0]), float(size[1]), float(size[2]))
        if min_dim < config.thin_threshold_m and center_z < 0.25 and max_dim < 1.80:
            return "low_thin_threshold_or_bottom_rail"
    return None


def _proxy_candidate_reason(kind: str, size: Gf.Vec3d, config: SceneRepairConfig) -> Tuple[bool, str]:
    sx, sy, sz = float(size[0]), float(size[1]), float(size[2])
    xy = sorted([sx, sy])
    thickness, length = xy[0], xy[1]
    if kind == "wall":
        if length > config.max_wall_length_m:
            return False, f"wall_candidate_too_long:{length:.3f}>{config.max_wall_length_m:.3f}"
        if thickness > config.max_wall_thickness_m:
            return False, f"wall_candidate_too_thick:{thickness:.3f}>{config.max_wall_thickness_m:.3f}"
    if kind == "door":
        if length > config.max_door_length_m:
            return False, f"door_candidate_too_long:{length:.3f}>{config.max_door_length_m:.3f}"
        if thickness > config.max_door_thickness_m:
            return False, f"door_candidate_too_thick:{thickness:.3f}>{config.max_door_thickness_m:.3f}"
    return True, "ok"


def _disable_collision_prim(prim: Usd.Prim, reason: str, report: Optional[Dict] = None) -> bool:
    if not prim or not prim.IsValid():
        return False
    if _set_collision_enabled(prim, False):
        if report is not None:
            report.setdefault("disabled", []).append({"path": prim.GetPath().pathString, "reason": reason})
        return True
    return False


def _apply_shenxinfu_841837_stall_fix(stage: Usd.Stage, scene_root_path: str, report: Dict) -> None:
    room_root = scene_root_path
    floor_geom_root = f"{room_root}/Meshes/Base/Floor_0000/Geom"
    collision_fix_root = f"{room_root}/CollisionFixes"

    vertical_names = {"A124", "A125"}
    floor_names = {"A67", "A70", "A123", "A150", "A164", "A240", "A262"}

    deactivated = 0
    collision_disabled = 0

    for name in sorted(vertical_names):
        prim = stage.GetPrimAtPath(f"{floor_geom_root}/{name}")
        if prim and prim.IsValid():
            prim.SetActive(False)
            deactivated += 1
            report.setdefault("disabled", []).append(
                {"path": prim.GetPath().pathString, "reason": "shenxinfu_841837_dirty_stall_vertical_entrance_mesh"}
            )

    for name in sorted(floor_names):
        prim = stage.GetPrimAtPath(f"{floor_geom_root}/{name}")
        if _disable_collision_prim(prim, "shenxinfu_841837_dirty_stall_floor_seam", report):
            collision_disabled += 1

    collider_path = f"{collision_fix_root}/dirty_stall_clean_floor"
    if not stage.GetPrimAtPath(collider_path).IsValid():
        UsdGeom.Xform.Define(stage, collision_fix_root)
        cube = UsdGeom.Cube.Define(stage, collider_path)
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(1.4705, 0.0275, -0.03))
        cube.AddScaleOp().Set(Gf.Vec3f(1.9535, 3.145, 0.06))
        cube.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True).Set(True)
        collision_disabled += 1
        report.setdefault("disabled", []).append(
            {"path": collider_path, "reason": "shenxinfu_841837_added_clean_floor_collider"}
        )

    report.setdefault("warnings", []).append(
        f"shenxinfu_841837_stall_fix applied: deactivated={deactivated} collision_changes={collision_disabled}"
    )


def _is_boundable_or_mesh(prim: Usd.Prim) -> bool:
    try:
        if prim.IsA(UsdGeom.Mesh):
            return True
        if prim.IsA(UsdGeom.Boundable):
            return True
    except Exception:
        pass
    # Some useful imported prims are Xform/Scope with authored extents under Static/Geom.
    name = prim.GetName().lower()
    return name in {"static", "geom", "mesh", "visuals", "collision"} or _has_collision_like(prim)


def _ensure_proxy_root(stage: Usd.Stage, proxy_root_path: str, visible: bool) -> Usd.Prim:
    root = UsdGeom.Xform.Define(stage, Sdf.Path(proxy_root_path)).GetPrim()
    if visible:
        UsdGeom.Imageable(root).MakeVisible()
    else:
        UsdGeom.Imageable(root).MakeInvisible()
    return root


def _remove_proxy_root(stage: Usd.Stage, scene_root_path: str, proxy_root_name: str) -> None:
    path = f"{scene_root_path}/{proxy_root_name}"
    try:
        if stage.GetPrimAtPath(path).IsValid():
            stage.RemovePrim(Sdf.Path(path))
    except Exception:
        pass


def _create_proxy_box(
    stage: Usd.Stage,
    proxy_root_path: str,
    *,
    name: str,
    kind: str,
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
    yaw: float = 0.0,
    enabled: bool = True,
    visible: bool = False,
    source: str = "",
    auto_reason: str = "",
) -> Optional[str]:
    try:
        _ensure_proxy_root(stage, proxy_root_path, visible)
        prim_path = f"{proxy_root_path}/{_safe_token(name)}_{_path_hash(name + source)}"
        cube = UsdGeom.Cube.Define(stage, Sdf.Path(prim_path))
        cube.CreateSizeAttr(1.0)
        prim = cube.GetPrim()
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(float(center[0]), float(center[1]), float(center[2])))
        if abs(float(yaw)) > 1e-9:
            xform.AddRotateZOp().Set(math.degrees(float(yaw)))
        xform.AddScaleOp().Set(Gf.Vec3d(float(size[0]), float(size[1]), float(size[2])))

        prim.CreateAttribute("arena:collisionProxy", Sdf.ValueTypeNames.Bool).Set(True)
        prim.CreateAttribute("arena:proxyEnabled", Sdf.ValueTypeNames.Bool).Set(bool(enabled))
        prim.CreateAttribute("arena:proxyKind", Sdf.ValueTypeNames.String).Set(str(kind))
        prim.CreateAttribute("arena:proxySource", Sdf.ValueTypeNames.String).Set(str(source))
        prim.CreateAttribute("arena:autoReason", Sdf.ValueTypeNames.String).Set(str(auto_reason))
        prim.CreateAttribute("arena:proxyCenter", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(*[float(v) for v in center]))
        prim.CreateAttribute("arena:proxySize", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(*[float(v) for v in size]))
        prim.CreateAttribute("arena:proxyYaw", Sdf.ValueTypeNames.Double).Set(float(yaw))

        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(bool(enabled)).Set(bool(enabled))
        if PhysxSchema is not None:
            try:
                PhysxSchema.PhysxCollisionAPI.Apply(prim)
            except Exception:
                pass
        try:
            # Cyan enabled, orange disabled/fallback-edit proxies.
            color = (0.1, 0.7, 1.0) if enabled else (1.0, 0.45, 0.05)
            cube.CreateDisplayColorAttr([color])
        except Exception:
            pass
        if visible:
            UsdGeom.Imageable(prim).MakeVisible()
        else:
            UsdGeom.Imageable(prim).MakeInvisible()
        return prim_path
    except Exception as exc:
        print(f"[scene_collision_repair] failed to create proxy {name}: {exc}")
        return None


def _proxy_dict_from_prim(prim: Usd.Prim, scene_root_path: str = "") -> Optional[Dict]:
    if not prim.IsValid() or not prim.IsActive():
        return None
    marker = prim.GetAttribute("arena:collisionProxy")
    if not (marker and marker.IsValid() and bool(marker.Get())):
        return None
    try:
        xf = UsdGeom.Xformable(prim)
        m = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        center_gf = m.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
        col0 = Gf.Vec3d(float(m[0][0]), float(m[1][0]), float(m[2][0]))
        col1 = Gf.Vec3d(float(m[0][1]), float(m[1][1]), float(m[2][1]))
        col2 = Gf.Vec3d(float(m[0][2]), float(m[1][2]), float(m[2][2]))
        sx = max(1e-6, col0.GetLength())
        sy = max(1e-6, col1.GetLength())
        sz = max(1e-6, col2.GetLength())
        yaw = math.atan2(float(col0[1]), float(col0[0]))
    except Exception:
        c = prim.GetAttribute("arena:proxyCenter").Get() or Gf.Vec3d(0, 0, 0)
        s = prim.GetAttribute("arena:proxySize").Get() or Gf.Vec3d(1, 1, 1)
        center_gf = c
        sx, sy, sz = float(s[0]), float(s[1]), float(s[2])
        yaw = float(prim.GetAttribute("arena:proxyYaw").Get() or 0.0)

    kind = prim.GetAttribute("arena:proxyKind").Get() or "proxy"
    source = prim.GetAttribute("arena:proxySource").Get() or ""
    reason = prim.GetAttribute("arena:autoReason").Get() if prim.GetAttribute("arena:autoReason").IsValid() else ""
    enabled_attr = prim.GetAttribute("arena:proxyEnabled")
    enabled = bool(enabled_attr.Get()) if enabled_attr and enabled_attr.IsValid() and enabled_attr.Get() is not None else True
    coll_attr = prim.GetAttribute("physics:collisionEnabled")
    if coll_attr and coll_attr.IsValid() and coll_attr.Get() is not None:
        enabled = bool(coll_attr.Get())
    return {
        "name": prim.GetName(),
        "enabled": enabled,
        "kind": str(kind),
        "prim_path": prim.GetPath().pathString,
        "source": str(source),
        "auto_reason": str(reason or ""),
        "center": [float(center_gf[0]), float(center_gf[1]), float(center_gf[2])],
        "size": [float(sx), float(sy), float(sz)],
        "yaw": float(yaw),
    }


def export_collision_proxies_to_yaml(
    scene_root_path: str,
    output_path: str,
    *,
    scene_name: Optional[str] = None,
    proxy_root_name: str = "_collision_proxy",
) -> Dict:
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(scene_root_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"scene root not found: {scene_root_path}")
    proxy_root_path = f"{scene_root_path}/{proxy_root_name}"
    proxy_root = stage.GetPrimAtPath(proxy_root_path)
    if not proxy_root or not proxy_root.IsValid():
        raise RuntimeError(f"proxy root not found: {proxy_root_path}")
    proxies: List[Dict] = []
    for prim in Usd.PrimRange(proxy_root):
        p = _proxy_dict_from_prim(prim, scene_root_path)
        if p:
            proxies.append(p)
    data = {
        "schema_version": 1,
        "scene": scene_name or scene_root_path.rstrip("/").split("/")[-1],
        "scene_root_path": scene_root_path,
        "proxy_root_name": proxy_root_name,
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "notes": "Editable collision proxies exported from Isaac Sim. center/size are meters; yaw is radians.",
        "proxies": proxies,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    print(f"[scene_collision_repair] exported {len(proxies)} collision proxies to {output_path}")
    return data


def _load_proxy_config(path: str) -> Dict:
    if not path:
        raise RuntimeError("empty collision proxy config path")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"invalid proxy config root: {path}")
    if "proxies" not in data or not isinstance(data["proxies"], list):
        raise RuntimeError(f"proxy config has no proxies list: {path}")
    return data


def _create_proxies_from_config(stage: Usd.Stage, scene_root_path: str, data: Dict, config: SceneRepairConfig) -> List[Dict]:
    proxy_root_name = str(data.get("proxy_root_name") or config.proxy_root_name)
    proxy_root_path = f"{scene_root_path}/{proxy_root_name}"
    _remove_proxy_root(stage, scene_root_path, proxy_root_name)
    out: List[Dict] = []
    for item in data.get("proxies", []):
        try:
            name = str(item.get("name") or f"{item.get('kind','proxy')}_{len(out)}")
            kind = str(item.get("kind") or "proxy")
            enabled = bool(item.get("enabled", True))
            center = tuple(float(v) for v in item.get("center", [0, 0, 0]))
            size = tuple(float(v) for v in item.get("size", [1, 1, 1]))
            yaw = float(item.get("yaw", 0.0))
            source = str(item.get("source") or item.get("prim_path") or "config")
            proxy_path = _create_proxy_box(
                stage, proxy_root_path,
                name=name, kind=kind, center=center, size=size, yaw=yaw,
                enabled=enabled, visible=config.proxy_visible, source=source,
                auto_reason=str(item.get("auto_reason", "config")),
            )
            if proxy_path:
                rec = dict(item)
                rec["proxy"] = proxy_path
                out.append(rec)
        except Exception as exc:
            print(f"[scene_collision_repair] failed to load proxy item from config: {exc}: {item}")
    return out


def _bbox_to_center_size(mn: Gf.Vec3d, mx: Gf.Vec3d, size: Gf.Vec3d, config: SceneRepairConfig) -> Tuple[Gf.Vec3d, Gf.Vec3d]:
    sx = max(float(size[0]) + 2.0 * config.proxy_margin_m, config.min_proxy_xy_m)
    sy = max(float(size[1]) + 2.0 * config.proxy_margin_m, config.min_proxy_xy_m)
    sz = max(float(size[2]) + 2.0 * config.proxy_margin_m, config.min_proxy_z_m)
    center = Gf.Vec3d((mn[0] + mx[0]) * 0.5, (mn[1] + mx[1]) * 0.5, (mn[2] + mx[2]) * 0.5)
    return center, Gf.Vec3d(sx, sy, sz)


def _descendant_proxy_specs(parent: Usd.Prim, inherited_kind: str, cache: UsdGeom.BBoxCache, config: SceneRepairConfig) -> List[Dict]:
    specs: List[Dict] = []
    parent_path = parent.GetPath().pathString
    selected: List[str] = []
    for prim in Usd.PrimRange(parent):
        if prim == parent:
            continue
        if not prim.IsValid() or not prim.IsActive():
            continue
        path = prim.GetPath().pathString
        if _is_under_proxy_root(path, config.proxy_root_name):
            continue
        if any(path.startswith(p + "/") for p in selected):
            continue
        if not _is_boundable_or_mesh(prim):
            continue
        bbox = _world_bbox(cache, prim)
        if bbox is None:
            continue
        mn, mx, size = bbox
        ok, reason = _proxy_candidate_reason(inherited_kind, size, config)
        # For recursive generation, keep only reasonable child pieces; otherwise
        # complex parent assets create nested broad boxes again.
        if not ok:
            continue
        center, proxy_size = _bbox_to_center_size(mn, mx, size, config)
        selected.append(path)
        specs.append({
            "source_prim": prim,
            "kind": inherited_kind,
            "center": center,
            "size": proxy_size,
            "enabled": _kind_default_enabled(inherited_kind, config),
            "reason": f"recursive_child_of:{parent_path}",
        })
    return specs


def _auto_proxy_specs(root: Usd.Prim, cache: UsdGeom.BBoxCache, config: SceneRepairConfig) -> Tuple[List[Dict], List[str]]:
    specs: List[Dict] = []
    warnings: List[str] = []
    blocked_prefixes: List[str] = []
    for prim in Usd.PrimRange(root):
        if not prim.IsValid() or not prim.IsActive():
            continue
        path = prim.GetPath().pathString
        if _is_under_proxy_root(path, config.proxy_root_name):
            continue
        if any(path.startswith(p + "/") for p in blocked_prefixes):
            continue
        kind = _classify_proxy_candidate(prim, config)
        if kind is None:
            continue
        bbox = _world_bbox(cache, prim)
        if bbox is None:
            warnings.append(f"no bbox for proxy candidate: {path}")
            continue
        mn, mx, size = bbox
        ok, reason = _proxy_candidate_reason(kind, size, config)
        if ok:
            center, proxy_size = _bbox_to_center_size(mn, mx, size, config)
            specs.append({
                "source_prim": prim,
                "kind": kind,
                "center": center,
                "size": proxy_size,
                "enabled": _kind_default_enabled(kind, config),
                "reason": "auto_object",
            })
            blocked_prefixes.append(path)
            continue

        child_specs: List[Dict] = []
        if config.recursive_proxy_generation:
            child_specs = _descendant_proxy_specs(prim, kind, cache, config)
        if child_specs:
            specs.extend(child_specs)
            blocked_prefixes.append(path)
            warnings.append(f"split broad {kind} candidate into {len(child_specs)} child proxies: {path}; reason={reason}")
            continue

        if config.create_oversized_editable_fallback:
            center, proxy_size = _bbox_to_center_size(mn, mx, size, config)
            # Keep door disabled by default.  Wall stays configured by wall_default_enabled.
            specs.append({
                "source_prim": prim,
                "kind": kind,
                "center": center,
                "size": proxy_size,
                "enabled": _kind_default_enabled(kind, config),
                "reason": f"oversized_editable_fallback:{reason}",
            })
            blocked_prefixes.append(path)
            warnings.append(f"created editable fallback for broad {kind} candidate: {path}; reason={reason}")
        else:
            warnings.append(f"skip {kind} proxy candidate {path}: {reason} size={[float(size[0]), float(size[1]), float(size[2])]}")
    return specs, warnings


def inspect_and_repair_scene(
    scene_root_path: str,
    scene_name: str = "scene",
    scene_usd_path: str = "",
    config: Optional[SceneRepairConfig] = None,
) -> Dict:
    config = config or config_from_env()
    report: Dict = {
        "scene_root_path": scene_root_path,
        "scene_name": scene_name,
        "scene_usd_path": scene_usd_path,
        "config": asdict(config),
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "collision_like_count": 0,
        "thin_collision_count": 0,
        "disabled": [],
        "restored": [],
        "proxies": [],
        "proxy_counts_by_kind": {},
        "warnings": [],
    }
    if not config.enabled or config.mode == "off":
        print(f"[scene_collision_repair] disabled for {scene_root_path}")
        return report

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(scene_root_path)
    if not root or not root.IsValid():
        report["warnings"].append(f"scene root not found: {scene_root_path}")
        print(f"[scene_collision_repair] scene root not found: {scene_root_path}")
        return report

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
    _remove_proxy_root(stage, scene_root_path, config.proxy_root_name)

    print(f"[scene_collision_repair] inspecting collision prims under {scene_root_path}")
    for prim in Usd.PrimRange(root):
        if not prim.IsValid() or not prim.IsActive():
            continue
        if _is_under_proxy_root(prim.GetPath().pathString, config.proxy_root_name):
            continue
        if not _has_collision_like(prim):
            continue
        report["collision_like_count"] += 1
        bbox = _world_bbox(bbox_cache, prim)
        if bbox is not None:
            _, _, size = bbox
            if min(float(size[0]), float(size[1]), float(size[2])) < config.thin_threshold_m:
                report["thin_collision_count"] += 1
        reason = _classify_disable_candidate(prim, bbox, config)
        if reason and config.mode in {"safe", "aggressive"}:
            if _set_collision_enabled(prim, False):
                entry = {"path": prim.GetPath().pathString, "reason": reason}
                if bbox is not None:
                    _, _, size = bbox
                    entry["size"] = [float(size[0]), float(size[1]), float(size[2])]
                report["disabled"].append(entry)

    scene_key = f"{scene_name} {scene_root_path} {scene_usd_path}".lower()
    if config.mode in {"safe", "aggressive"} and "shenxinfu_841837" in scene_key:
        try:
            _apply_shenxinfu_841837_stall_fix(stage, scene_root_path, report)
            print(f"[scene_collision_repair] applied shenxinfu_841837 stall fix under {scene_root_path}")
        except Exception as exc:
            report["warnings"].append(f"shenxinfu_841837_stall_fix failed: {exc}")
            print(f"[scene_collision_repair] shenxinfu_841837 stall fix failed: {exc}")

    if (
        config.mode in {"safe", "aggressive"}
        and config.door_collision_policy == "restore_selected_meshes"
    ):
        restored_count = _restore_selected_door_mesh_collisions(root, config, report)
        print(
            f"[scene_collision_repair] restored {restored_count} selected door mesh collisions "
            f"under {scene_root_path}"
        )

    if config.mode not in {"safe", "aggressive"}:
        _write_report(report, config)
        return report

    proxy_root_path = f"{scene_root_path}/{config.proxy_root_name}"
    if config.proxy_source == "off":
        pass
    elif config.proxy_source == "config":
        try:
            data = _load_proxy_config(config.proxy_config_path)
            loaded = _create_proxies_from_config(stage, scene_root_path, data, config)
            report["proxies"] = loaded
            for item in loaded:
                k = str(item.get("kind", "proxy"))
                report["proxy_counts_by_kind"][k] = int(report["proxy_counts_by_kind"].get(k, 0)) + 1
            print(f"[scene_collision_repair] loaded {len(loaded)} editable proxies from {config.proxy_config_path}")
        except Exception as exc:
            report["warnings"].append(f"failed to load proxy config {config.proxy_config_path}: {exc}")
            print(f"[scene_collision_repair] failed to load proxy config {config.proxy_config_path}: {exc}")
    else:
        specs, warnings = _auto_proxy_specs(root, bbox_cache, config)
        report["warnings"].extend(warnings)
        if len(specs) > config.max_proxy_count:
            report["warnings"].append(f"proxy spec count {len(specs)} exceeds max_proxy_count={config.max_proxy_count}; truncating")
            specs = specs[: config.max_proxy_count]
        print(f"[scene_collision_repair] proxy specs={len(specs)} under {scene_root_path}")
        for spec in specs:
            prim = spec["source_prim"]
            kind = spec["kind"]
            center: Gf.Vec3d = spec["center"]
            proxy_size: Gf.Vec3d = spec["size"]
            enabled = bool(spec["enabled"])
            reason = str(spec.get("reason", "auto"))
            src_path = prim.GetPath().pathString
            proxy_path = _create_proxy_box(
                stage,
                proxy_root_path,
                name=f"{kind}_{prim.GetName()}",
                kind=kind,
                center=(float(center[0]), float(center[1]), float(center[2])),
                size=(float(proxy_size[0]), float(proxy_size[1]), float(proxy_size[2])),
                yaw=0.0,
                enabled=enabled,
                visible=config.proxy_visible,
                source=src_path,
                auto_reason=reason,
            )
            if proxy_path:
                disabled_count = 0
                if config.disable_original_for_proxy_roots:
                    disabled_count = _disable_collision_recursive(prim)
                report["proxy_counts_by_kind"][kind] = int(report["proxy_counts_by_kind"].get(kind, 0)) + 1
                report["proxies"].append({
                    "name": proxy_path.rstrip("/").split("/")[-1],
                    "enabled": enabled,
                    "kind": kind,
                    "proxy": proxy_path,
                    "source": src_path,
                    "auto_reason": reason,
                    "center": [float(center[0]), float(center[1]), float(center[2])],
                    "size": [float(proxy_size[0]), float(proxy_size[1]), float(proxy_size[2])],
                    "yaw": 0.0,
                    "disabled_original_collision_prims": disabled_count,
                })
        if config.auto_write_config and config.proxy_config_path:
            try:
                export_collision_proxies_to_yaml(scene_root_path, config.proxy_config_path, scene_name=scene_name, proxy_root_name=config.proxy_root_name)
                report["auto_written_config"] = config.proxy_config_path
            except Exception as exc:
                report["warnings"].append(f"failed to auto-write proxy config: {exc}")

    _write_report(report, config)
    print(
        f"[scene_collision_repair] scene={scene_name} root={scene_root_path} "
        f"mode={config.mode} source={config.proxy_source} collision_like={report['collision_like_count']} "
        f"thin={report['thin_collision_count']} disabled={len(report['disabled'])} "
        f"restored={len(report['restored'])} proxies={len(report['proxies'])} "
        f"by_kind={report.get('proxy_counts_by_kind', {})}"
    )
    return report


def _write_report(report: Dict, config: SceneRepairConfig) -> None:
    try:
        os.makedirs(config.report_dir, exist_ok=True)
        safe_scene = _safe_token(report.get("scene_name", "scene"), 48)
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(config.report_dir, f"{safe_scene}_{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[scene_collision_repair] wrote report: {path}")
    except Exception as exc:
        print(f"[scene_collision_repair] failed to write report: {exc}")


__all__ = [
    "SceneRepairConfig",
    "config_from_env",
    "inspect_and_repair_scene",
    "export_collision_proxies_to_yaml",
]
