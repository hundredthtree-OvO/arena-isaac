"""Robot geometry helpers for Arena Isaac kinematic mobile-base validation.

V16 purpose
-----------
The previous lidar and guard values were fixed engineering guesses.  This module
estimates the mobile-base footprint from the robot already imported into the
Isaac USD stage, then exposes small utilities used by the lidar spawner and
kinematic controller.

The estimate is intentionally conservative and mobile-base oriented: it scans
boundable geometry below a configurable z band, excludes arms/grippers/sensors,
and computes an XY AABB in the robot root frame.  Users can still override the
result from the scene profile.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import omni.usd
from pxr import Gf, Usd, UsdGeom


@dataclass
class BodyFootprint:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    prim_count: int = 0

    @property
    def length(self) -> float:
        return max(0.0, float(self.x_max - self.x_min))

    @property
    def width(self) -> float:
        return max(0.0, float(self.y_max - self.y_min))

    @property
    def center_x(self) -> float:
        return 0.5 * float(self.x_min + self.x_max)

    @property
    def center_y(self) -> float:
        return 0.5 * float(self.y_min + self.y_max)


def _log(logger, level: str, text: str) -> None:
    if logger is None:
        print(text)
        return
    fn = getattr(logger, level, None) or getattr(logger, "info", None)
    if fn:
        fn(text)
    else:
        print(text)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _split_keywords(raw: Optional[str], default: Sequence[str]) -> List[str]:
    if raw is None or str(raw).strip() == "":
        return [str(x) for x in default]
    return [x.strip() for x in str(raw).replace(";", ",").split(",") if x.strip()]


def _contains_any(text: str, keywords: Sequence[str]) -> bool:
    lo = text.lower()
    return any(str(k).lower() in lo for k in keywords if str(k).strip())


def _corners(min_pt: Gf.Vec3d, max_pt: Gf.Vec3d) -> Iterable[Gf.Vec3d]:
    for x in (float(min_pt[0]), float(max_pt[0])):
        for y in (float(min_pt[1]), float(max_pt[1])):
            for z in (float(min_pt[2]), float(max_pt[2])):
                yield Gf.Vec3d(x, y, z)


def _bbox_valid(aabb) -> bool:
    try:
        mn = aabb.GetMin()
        mx = aabb.GetMax()
        vals = [float(mn[0]), float(mn[1]), float(mn[2]), float(mx[0]), float(mx[1]), float(mx[2])]
        if not all(math.isfinite(v) for v in vals):
            return False
        return float(mx[0]) > float(mn[0]) and float(mx[1]) > float(mn[1]) and float(mx[2]) >= float(mn[2])
    except Exception:
        return False


def estimate_body_footprint(
    robot_root_path: str,
    *,
    include_keywords: Optional[Sequence[str]] = None,
    exclude_keywords: Optional[Sequence[str]] = None,
    z_min: float = -0.10,
    z_max: float = 0.45,
    min_length: float = 0.20,
    min_width: float = 0.20,
    logger=None,
) -> Optional[BodyFootprint]:
    """Estimate mobile-base XY footprint in the robot root frame.

    ``z_min``/``z_max`` are in robot-root coordinates.  They intentionally clip
    out high arm/gripper geometry so the navigation footprint remains a base
    footprint, not a whole-arm envelope.
    """
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None
    root = stage.GetPrimAtPath(robot_root_path)
    if root is None or not root.IsValid():
        _log(logger, "warning", f"[robot_geometry] invalid robot root: {robot_root_path}")
        return None

    default_exclude = (
        "arm", "gripper", "finger", "camera", "lidar", "laser", "sensor", "marker",
        "debug", "_debug", "visual_sphere", "collision_proxy",
    )
    inc = [str(x) for x in include_keywords or [] if str(x).strip()]
    exc = [str(x) for x in exclude_keywords or default_exclude if str(x).strip()]

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=True)
    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())
    try:
        root_world = xcache.GetLocalToWorldTransform(root)
        root_inv = root_world.GetInverse()
    except Exception:
        root_inv = Gf.Matrix4d(1.0)

    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    used = 0
    root_path = str(root.GetPath())
    for prim in stage.Traverse():
        if not prim.IsValid() or not prim.IsActive():
            continue
        path = str(prim.GetPath())
        if path == root_path or not path.startswith(root_path + "/"):
            continue
        if _contains_any(path, exc):
            continue
        if inc and not _contains_any(path, inc):
            continue
        try:
            if not prim.IsA(UsdGeom.Boundable):
                continue
        except Exception:
            continue
        try:
            wb = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            if not _bbox_valid(wb):
                continue
            local_pts = [root_inv.Transform(p) for p in _corners(wb.GetMin(), wb.GetMax())]
            local_z_min = min(float(p[2]) for p in local_pts)
            local_z_max = max(float(p[2]) for p in local_pts)
            if local_z_max < z_min or local_z_min > z_max:
                continue
            xs.extend(float(p[0]) for p in local_pts)
            ys.extend(float(p[1]) for p in local_pts)
            zs.extend(float(p[2]) for p in local_pts)
            used += 1
        except Exception:
            continue

    if not xs or not ys:
        _log(logger, "warning", f"[robot_geometry] no body geometry matched under {robot_root_path}")
        return None
    fp = BodyFootprint(min(xs), max(xs), min(ys), max(ys), min(zs), max(zs), used)
    if fp.length < min_length or fp.width < min_width:
        _log(
            logger,
            "warning",
            f"[robot_geometry] ignored implausible footprint length={fp.length:.3f} width={fp.width:.3f} prims={used}",
        )
        return None
    _log(
        logger,
        "info",
        f"[robot_geometry] body footprint from stage: x=[{fp.x_min:+.3f},{fp.x_max:+.3f}] "
        f"y=[{fp.y_min:+.3f},{fp.y_max:+.3f}] length={fp.length:.3f} width={fp.width:.3f} prims={used}",
    )
    return fp


def estimate_body_footprint_from_env(robot_root_path: str, *, logger=None) -> Optional[BodyFootprint]:
    include = _split_keywords(os.environ.get("ARENA_ISAAC_ROBOT_GEOMETRY_INCLUDE_KEYWORDS"), [])
    exclude = _split_keywords(
        os.environ.get("ARENA_ISAAC_ROBOT_GEOMETRY_EXCLUDE_KEYWORDS"),
        ("arm", "gripper", "finger", "camera", "lidar", "laser", "sensor", "marker", "debug", "_debug"),
    )
    return estimate_body_footprint(
        robot_root_path,
        include_keywords=include,
        exclude_keywords=exclude,
        z_min=_env_float("ARENA_ISAAC_ROBOT_GEOMETRY_Z_MIN", -0.10),
        z_max=_env_float("ARENA_ISAAC_ROBOT_GEOMETRY_Z_MAX", 0.45),
        min_length=_env_float("ARENA_ISAAC_ROBOT_GEOMETRY_MIN_LENGTH", 0.20),
        min_width=_env_float("ARENA_ISAAC_ROBOT_GEOMETRY_MIN_WIDTH", 0.20),
        logger=logger,
    )


def apply_auto_footprint_env(robot_root_path: str, *, logger=None) -> Optional[BodyFootprint]:
    """If enabled, compute body footprint and update collision-guard env vars.

    The collision guard is instantiated later by the mecanum controller, so
    setting these environment variables here is enough for the current bridge
    process.
    """
    if not _env_bool("ARENA_ISAAC_ROBOT_AUTO_FOOTPRINT", False):
        return None
    fp = estimate_body_footprint_from_env(robot_root_path, logger=logger)
    if fp is None:
        return None
    os.environ["ARENA_ISAAC_COLLISION_GUARD_LENGTH"] = f"{fp.length:.6f}"
    os.environ["ARENA_ISAAC_COLLISION_GUARD_WIDTH"] = f"{fp.width:.6f}"
    _log(
        logger,
        "info",
        f"[robot_geometry] auto footprint applied to guard env: length={fp.length:.3f} width={fp.width:.3f}",
    )
    return fp


def resolve_lidar_mounts_from_env(
    robot_root_path: str,
    *,
    default_front_xyz: Tuple[float, float, float] = (0.42, 0.0, 0.20),
    default_rear_xyz: Tuple[float, float, float] = (-0.42, 0.0, 0.20),
    logger=None,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Return front/rear lidar mount positions in robot-root coordinates.

    Modes:
      - manual: use explicit ARENA_ISAAC_LIDAR_FRONT_X/REAR_X values.
      - auto_from_body_bbox: place lidar origins inside the body bbox by inset_x.
    """
    source = os.environ.get("ARENA_ISAAC_LIDAR_MOUNT_SOURCE", "manual").strip().lower()
    y = _env_float("ARENA_ISAAC_LIDAR_Y", default_front_xyz[1])
    z = _env_float("ARENA_ISAAC_LIDAR_Z", default_front_xyz[2])

    if source in {"auto", "auto_from_body", "auto_from_body_bbox", "body_bbox"}:
        fp = estimate_body_footprint_from_env(robot_root_path, logger=logger)
        if fp is not None:
            inset = max(0.0, _env_float("ARENA_ISAAC_LIDAR_INSET_X", 0.03))
            front_x = fp.x_max - inset
            rear_x = fp.x_min + inset
            # Keep origin inside the estimated body even if the footprint is small.
            if front_x <= fp.x_min:
                front_x = 0.5 * (fp.x_min + fp.x_max)
            if rear_x >= fp.x_max:
                rear_x = 0.5 * (fp.x_min + fp.x_max)
            front = (float(front_x), y, z)
            rear = (float(rear_x), y, z)
            _log(
                logger,
                "info",
                f"[lidar_mount] auto_from_body_bbox: front=({front[0]:+.3f},{front[1]:+.3f},{front[2]:+.3f}) "
                f"rear=({rear[0]:+.3f},{rear[1]:+.3f},{rear[2]:+.3f}) inset={inset:.3f}",
            )
            return front, rear
        _log(logger, "warning", "[lidar_mount] auto_from_body_bbox failed; falling back to manual/default lidar mounts")

    front = (
        _env_float("ARENA_ISAAC_LIDAR_FRONT_X", default_front_xyz[0]),
        _env_float("ARENA_ISAAC_LIDAR_FRONT_Y", y),
        _env_float("ARENA_ISAAC_LIDAR_FRONT_Z", z),
    )
    rear = (
        _env_float("ARENA_ISAAC_LIDAR_REAR_X", default_rear_xyz[0]),
        _env_float("ARENA_ISAAC_LIDAR_REAR_Y", y),
        _env_float("ARENA_ISAAC_LIDAR_REAR_Z", z),
    )
    _log(
        logger,
        "info",
        f"[lidar_mount] manual/default: front=({front[0]:+.3f},{front[1]:+.3f},{front[2]:+.3f}) "
        f"rear=({rear[0]:+.3f},{rear[1]:+.3f},{rear[2]:+.3f})",
    )
    return front, rear


__all__ = [
    "BodyFootprint",
    "estimate_body_footprint",
    "estimate_body_footprint_from_env",
    "apply_auto_footprint_env",
    "resolve_lidar_mounts_from_env",
]
