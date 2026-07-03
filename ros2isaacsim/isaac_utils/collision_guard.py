"""Editable-proxy kinematic collision guard for Isaac Sim navigation scenes.

V13.3 changes:
- Adds strict/escape overlap handling when the robot starts inside a proxy.
- ``overlap_policy=block``: current overlap blocks all motion.
- ``overlap_policy=escape``: current overlap allows only motion that reduces penetration.
- ``overlap_policy=allow``: legacy behavior; current overlap is always allowed.
- Keeps zero-velocity skip, warning throttling, oriented proxy boxes, and editable YAML proxies.

V14 changes:
- Supports ``ARENA_ISAAC_COLLISION_GUARD_BACKEND=2d``. In that mode this class delegates
  to ``Semantic2DCollisionGuard`` and ignores 3D proxy cubes at runtime.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, Usd, UsdGeom


@dataclass
class GuardConfig:
    enabled: bool = True
    length: float = 0.80
    width: float = 0.60
    margin: float = 0.05
    footprint_forward: float = -1.0
    footprint_rear: float = -1.0
    footprint_left: float = -1.0
    footprint_right: float = -1.0
    z_min: float = 0.00
    z_max: float = 1.00
    refresh_sec: float = 1.00
    log_sec: float = 5.00
    block_log_sec: float = 2.00
    component_fallback: bool = True
    zero_eps: float = 1e-4
    # V13.3: current-overlap handling.
    #   block  -> strict: if current pose overlaps a proxy, block motion.
    #   escape -> allow only candidate motion whose penetration score decreases.
    #   allow  -> legacy v13.1 behavior: allow all motion while already overlapping.
    overlap_policy: str = "escape"
    escape_epsilon: float = 1e-4
    backend: str = "proxy"  # proxy | 2d | semantic2d | map | voxel | octomap


@dataclass
class ProxyObstacle:
    path: str
    center_x: float
    center_y: float
    center_z: float
    size_x: float
    size_y: float
    size_z: float
    yaw: float = 0.0
    enabled: bool = True

    @property
    def min_z(self) -> float:
        return self.center_z - 0.5 * self.size_z

    @property
    def max_z(self) -> float:
        return self.center_z + 0.5 * self.size_z


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


def _env_policy() -> str:
    # Backward compatibility with v13.1 variable.
    explicit = os.environ.get("ARENA_ISAAC_COLLISION_GUARD_OVERLAP_POLICY", "").strip().lower()
    if explicit in {"block", "strict", "escape", "allow"}:
        return "block" if explicit == "strict" else explicit
    allow_escape = os.environ.get("ARENA_ISAAC_COLLISION_GUARD_ALLOW_ESCAPE", "").strip().lower()
    if allow_escape in {"0", "false", "no", "off"}:
        return "block"
    if allow_escape in {"1", "true", "yes", "on"}:
        return "escape"
    return "escape"


def config_from_env() -> GuardConfig:
    return GuardConfig(
        enabled=_env_bool("ARENA_ISAAC_ENABLE_KINEMATIC_COLLISION_GUARD", True),
        length=_env_float("ARENA_ISAAC_COLLISION_GUARD_LENGTH", 0.80),
        width=_env_float("ARENA_ISAAC_COLLISION_GUARD_WIDTH", 0.60),
        margin=_env_float("ARENA_ISAAC_COLLISION_GUARD_MARGIN", 0.05),
        footprint_forward=_env_float("ARENA_ISAAC_COLLISION_GUARD_FOOTPRINT_FORWARD", -1.0),
        footprint_rear=_env_float("ARENA_ISAAC_COLLISION_GUARD_FOOTPRINT_REAR", -1.0),
        footprint_left=_env_float("ARENA_ISAAC_COLLISION_GUARD_FOOTPRINT_LEFT", -1.0),
        footprint_right=_env_float("ARENA_ISAAC_COLLISION_GUARD_FOOTPRINT_RIGHT", -1.0),
        z_min=_env_float("ARENA_ISAAC_COLLISION_GUARD_Z_MIN", 0.00),
        z_max=_env_float("ARENA_ISAAC_COLLISION_GUARD_Z_MAX", 1.00),
        refresh_sec=_env_float("ARENA_ISAAC_COLLISION_GUARD_REFRESH_SEC", 1.00),
        log_sec=_env_float("ARENA_ISAAC_COLLISION_GUARD_LOG_SEC", 5.00),
        block_log_sec=_env_float("ARENA_ISAAC_COLLISION_GUARD_BLOCK_LOG_SEC", 2.00),
        component_fallback=_env_bool("ARENA_ISAAC_COLLISION_GUARD_COMPONENT_FALLBACK", True),
        zero_eps=_env_float("ARENA_ISAAC_COLLISION_GUARD_ZERO_EPS", 1e-4),
        overlap_policy=_env_policy(),
        escape_epsilon=_env_float("ARENA_ISAAC_COLLISION_GUARD_ESCAPE_EPS", 1e-4),
        backend=os.environ.get("ARENA_ISAAC_COLLISION_GUARD_BACKEND", "proxy").strip().lower(),
    )


def _has_collision_proxy_marker(prim: Usd.Prim) -> bool:
    path = prim.GetPath().pathString
    if "/_collision_proxy/" not in path:
        return False
    attr = prim.GetAttribute("arena:collisionProxy")
    if attr and attr.IsValid():
        try:
            return bool(attr.Get())
        except Exception:
            pass
    return prim.GetTypeName() == "Cube"


def _enabled_for_prim(prim: Usd.Prim) -> bool:
    # arena:proxyEnabled is the user-facing editable flag in YAML.
    # physics:collisionEnabled is also honored to allow UI-side disabling.
    for attr_name in ("arena:proxyEnabled", "physics:collisionEnabled"):
        attr = prim.GetAttribute(attr_name)
        if attr and attr.IsValid():
            v = attr.Get()
            if v is not None:
                return bool(v)
    return True


def _obstacle_from_matrix(prim: Usd.Prim) -> Optional[ProxyObstacle]:
    try:
        m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        center = m.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
        # USD matrix columns represent the transformed local basis vectors for the authored Cube.
        col0 = Gf.Vec3d(float(m[0][0]), float(m[1][0]), float(m[2][0]))
        col1 = Gf.Vec3d(float(m[0][1]), float(m[1][1]), float(m[2][1]))
        col2 = Gf.Vec3d(float(m[0][2]), float(m[1][2]), float(m[2][2]))
        sx = max(1e-6, col0.GetLength())
        sy = max(1e-6, col1.GetLength())
        sz = max(1e-6, col2.GetLength())
        yaw = math.atan2(float(col0[1]), float(col0[0]))
        return ProxyObstacle(
            path=prim.GetPath().pathString,
            center_x=float(center[0]), center_y=float(center[1]), center_z=float(center[2]),
            size_x=float(sx), size_y=float(sy), size_z=float(sz), yaw=float(yaw),
            enabled=_enabled_for_prim(prim),
        )
    except Exception:
        return None


def _obstacle_for_prim(prim: Usd.Prim) -> Optional[ProxyObstacle]:
    obs = _obstacle_from_matrix(prim)
    if obs is not None:
        return obs
    try:
        c_attr = prim.GetAttribute("arena:proxyCenter")
        s_attr = prim.GetAttribute("arena:proxySize")
        if c_attr and s_attr and c_attr.IsValid() and s_attr.IsValid():
            c = c_attr.Get()
            size = s_attr.Get()
            yaw_attr = prim.GetAttribute("arena:proxyYaw")
            yaw = float(yaw_attr.Get()) if yaw_attr and yaw_attr.IsValid() and yaw_attr.Get() is not None else 0.0
            return ProxyObstacle(
                path=prim.GetPath().pathString,
                center_x=float(c[0]), center_y=float(c[1]), center_z=float(c[2]),
                size_x=float(size[0]), size_y=float(size[1]), size_z=float(size[2]), yaw=yaw,
                enabled=_enabled_for_prim(prim),
            )
    except Exception:
        pass
    return None


def _footprint_extents(config: GuardConfig) -> Tuple[float, float, float, float, float, float]:
    forward = config.footprint_forward if config.footprint_forward > 0.0 else (0.5 * config.length + config.margin)
    rear = config.footprint_rear if config.footprint_rear > 0.0 else (0.5 * config.length + config.margin)
    left = config.footprint_left if config.footprint_left > 0.0 else (0.5 * config.width + config.margin)
    right = config.footprint_right if config.footprint_right > 0.0 else (0.5 * config.width + config.margin)
    center_x_local = 0.5 * (forward - rear)
    center_y_local = 0.5 * (left - right)
    half_x = 0.5 * (forward + rear)
    half_y = 0.5 * (left + right)
    return float(center_x_local), float(center_y_local), float(half_x), float(half_y), float(forward), float(left)


def _interval_overlap(a_min: float, a_max: float, b_min: float, b_max: float) -> bool:
    return not (a_max < b_min or b_max < a_min)


def _project_points(points: Sequence[Tuple[float, float]], axis: Tuple[float, float]) -> Tuple[float, float]:
    ax, ay = axis
    vals = [px * ax + py * ay for px, py in points]
    return min(vals), max(vals)


def _rect_corners(cx: float, cy: float, yaw: float, half_x: float, half_y: float):
    c = math.cos(yaw)
    s = math.sin(yaw)
    ux = (c, s)
    uy = (-s, c)
    return [
        (cx + dx * ux[0] + dy * uy[0], cy + dx * ux[1] + dy * uy[1])
        for dx in (-half_x, half_x)
        for dy in (-half_y, half_y)
    ], ux, uy


def _obb_penetration_2d(
    ax: float, ay: float, ayaw: float, ahx: float, ahy: float,
    bx: float, by: float, byaw: float, bhx: float, bhy: float,
) -> float:
    """Return 0 if separated, otherwise a positive conservative penetration score."""
    a_corners, a_ux, a_uy = _rect_corners(ax, ay, ayaw, ahx, ahy)
    b_corners, b_ux, b_uy = _rect_corners(bx, by, byaw, bhx, bhy)
    axes = [a_ux, a_uy, b_ux, b_uy]
    min_overlap = float("inf")
    for axis in axes:
        a0, a1 = _project_points(a_corners, axis)
        b0, b1 = _project_points(b_corners, axis)
        if not _interval_overlap(a0, a1, b0, b1):
            return 0.0
        overlap = min(a1, b1) - max(a0, b0)
        min_overlap = min(min_overlap, max(0.0, overlap))
    return 0.0 if min_overlap == float("inf") else float(min_overlap)


class KinematicCollisionGuard:
    def __init__(self, *, robot_name: str, logger=None, config: Optional[GuardConfig] = None):
        self.robot_name = robot_name
        self.logger = logger
        self.config = config or config_from_env()
        self._semantic2d = None
        self._voxel = None
        if self.config.backend in {"voxel", "octomap", "octomap_like", "3d"}:
            try:
                from isaac_utils.voxel_guard import VoxelCollisionGuard, config_from_env as voxel_config_from_env
                self._voxel = VoxelCollisionGuard(robot_name=robot_name, logger=logger, config=voxel_config_from_env())
                msg = f"[{self.robot_name}] collision_guard backend=voxel using {self._voxel.config.map_path}"
                logger.info(msg) if logger else print(msg)
            except Exception as exc:
                msg = f"[{self.robot_name}] failed to initialize voxel backend; falling back to proxy backend: {exc}"
                logger.warning(msg) if logger else print("WARNING:", msg)
                self.config.backend = "proxy"
        elif self.config.backend in {"2d", "semantic2d", "map", "occupancy"}:
            try:
                from isaac_utils.collision2d_guard import Semantic2DCollisionGuard, config_from_env as config2d_from_env
                self._semantic2d = Semantic2DCollisionGuard(robot_name=robot_name, logger=logger, config=config2d_from_env())
                msg = f"[{self.robot_name}] collision_guard backend=2d using {self._semantic2d.config.config_path}"
                logger.info(msg) if logger else print(msg)
            except Exception as exc:
                msg = f"[{self.robot_name}] failed to initialize collision2d backend; falling back to proxy backend: {exc}"
                logger.warning(msg) if logger else print("WARNING:", msg)
                self.config.backend = "proxy"
        self._obstacles: List[ProxyObstacle] = []
        self._last_refresh = 0.0
        self._last_log = 0.0
        self._last_block_log = 0.0
        self._last_inside_log = 0.0
        self.query_count = 0
        self.block_count = 0
        self.total_query_ms = 0.0
        self.max_query_ms = 0.0

    def _info(self, text: str):
        self.logger.info(text) if self.logger else print(text)

    def _warn(self, text: str):
        self.logger.warning(text) if self.logger else print("WARNING:", text)

    def refresh(self, force: bool = False) -> None:
        if self._voxel is not None:
            self._voxel.refresh(force=force)
            return
        if self._semantic2d is not None:
            self._semantic2d.refresh(force=force)
            return
        now = time.monotonic()
        if not force and now - self._last_refresh < self.config.refresh_sec:
            return
        self._last_refresh = now
        if not self.config.enabled:
            self._obstacles = []
            return
        stage = get_current_stage()
        if stage is None:
            self._obstacles = []
            return
        obstacles: List[ProxyObstacle] = []
        for prim in stage.Traverse():
            if not prim.IsValid() or not prim.IsActive():
                continue
            if not _has_collision_proxy_marker(prim):
                continue
            obs = _obstacle_for_prim(prim)
            if obs is None or not obs.enabled:
                continue
            if obs.max_z < self.config.z_min or obs.min_z > self.config.z_max:
                continue
            obstacles.append(obs)
        self._obstacles = obstacles
        self._maybe_log_stats(extra=f"refreshed_obstacles={len(obstacles)} policy={self.config.overlap_policy}")

    def _maybe_log_stats(self, extra: str = ""):
        now = time.monotonic()
        if self.config.log_sec <= 0 or now - self._last_log < self.config.log_sec:
            return
        self._last_log = now
        avg = self.total_query_ms / self.query_count if self.query_count else 0.0
        self._info(
            f"[{self.robot_name}] collision_guard: enabled={self.config.enabled} "
            f"obstacles={len(self._obstacles)} blocked={self.block_count}/{self.query_count} "
            f"avg_ms={avg:.3f} max_ms={self.max_query_ms:.3f} {extra}".strip()
        )

    def _collision_score(self, pos_xyz, heading: float) -> Tuple[float, Optional[str], float]:
        if not self.config.enabled:
            return 0.0, None, 0.0
        self.refresh()
        if not self._obstacles:
            return 0.0, None, 0.0
        t0 = time.perf_counter()
        center_x_local, center_y_local, half_len, half_wid, _forward, _left = _footprint_extents(self.config)
        x = float(pos_xyz[0])
        y = float(pos_xyz[1])
        c = math.cos(heading)
        s = math.sin(heading)
        center_x = x + c * center_x_local - s * center_y_local
        center_y = y + s * center_x_local + c * center_y_local
        best_score = 0.0
        best_path = None
        for obs in self._obstacles:
            score = _obb_penetration_2d(
                center_x, center_y, heading, half_len, half_wid,
                obs.center_x, obs.center_y, obs.yaw, 0.5 * obs.size_x, 0.5 * obs.size_y,
            )
            if score > best_score:
                best_score = score
                best_path = obs.path
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.query_count += 1
        self.total_query_ms += dt_ms
        self.max_query_ms = max(self.max_query_ms, dt_ms)
        if best_score > 0.0:
            self.block_count += 1
        self._maybe_log_stats()
        return best_score, best_path, dt_ms

    def pose_collides(self, pos_xyz, heading: float, *, log_block: bool = True) -> Tuple[bool, Optional[str], float]:
        if self._voxel is not None:
            return self._voxel.pose_collides(pos_xyz, heading, log_block=log_block)
        if self._semantic2d is not None:
            return self._semantic2d.pose_collides(pos_xyz, heading, log_block=log_block)
        score, path, dt_ms = self._collision_score(pos_xyz, heading)
        if score > 0.0:
            if log_block:
                self._maybe_log_block(path or "unknown", dt_ms, score=score)
            return True, path, dt_ms
        return False, None, dt_ms

    def _maybe_log_block(self, path: str, dt_ms: float, *, score: float = 0.0):
        now = time.monotonic()
        if now - self._last_block_log < self.config.block_log_sec:
            return
        self._last_block_log = now
        self._warn(f"[{self.robot_name}] collision_guard blocked kinematic step: obstacle={path}, penetration={score:.4f}, query_ms={dt_ms:.3f}")

    def _maybe_log_inside(self, path: str, *, current_score: float, next_score: float, action: str):
        now = time.monotonic()
        if now - self._last_inside_log < self.config.block_log_sec:
            return
        self._last_inside_log = now
        self._warn(
            f"[{self.robot_name}] collision_guard current pose overlaps {path}; "
            f"policy={self.config.overlap_policy} action={action} "
            f"current_penetration={current_score:.4f} next_penetration={next_score:.4f}"
        )

    def _predict(self, *, pos_xyz, heading: float, vx: float, vy: float, wz: float, dt: float):
        next_heading = heading + wz * dt
        c = math.cos(next_heading)
        s = math.sin(next_heading)
        wx = c * vx - s * vy
        wy = s * vx + c * vy
        next_pos = np.array([float(pos_xyz[0]) + wx * dt, float(pos_xyz[1]) + wy * dt, float(pos_xyz[2])], dtype=np.float32)
        return next_pos, next_heading

    def _candidate_is_allowed_from_overlap(self, *, pos_xyz, heading: float, vx: float, vy: float, wz: float, dt: float) -> Tuple[bool, str, float, float, Optional[str]]:
        current_score, current_path, _ = self._collision_score(pos_xyz, heading)
        if current_score <= 0.0:
            next_pos, next_heading = self._predict(pos_xyz=pos_xyz, heading=heading, vx=vx, vy=vy, wz=wz, dt=dt)
            next_score, next_path, next_ms = self._collision_score(next_pos, next_heading)
            if next_score <= 0.0:
                return True, "free", current_score, next_score, next_path
            self._maybe_log_block(next_path or "unknown", next_ms, score=next_score)
            return False, "blocked", current_score, next_score, next_path

        next_pos, next_heading = self._predict(pos_xyz=pos_xyz, heading=heading, vx=vx, vy=vy, wz=wz, dt=dt)
        next_score, next_path, _ = self._collision_score(next_pos, next_heading)
        path = next_path or current_path
        policy = self.config.overlap_policy
        if policy == "allow":
            self._maybe_log_inside(path or "unknown", current_score=current_score, next_score=next_score, action="allow_legacy")
            return True, "inside_allow_legacy", current_score, next_score, path
        if policy == "block":
            self._maybe_log_inside(path or "unknown", current_score=current_score, next_score=next_score, action="strict_block")
            return False, "inside_strict_block", current_score, next_score, path
        # escape mode: only motion that decreases penetration is allowed.
        if next_score + self.config.escape_epsilon < current_score:
            self._maybe_log_inside(path or "unknown", current_score=current_score, next_score=next_score, action="escape_allowed")
            return True, "inside_escape_reduce", current_score, next_score, path
        self._maybe_log_inside(path or "unknown", current_score=current_score, next_score=next_score, action="escape_block")
        return False, "inside_escape_block", current_score, next_score, path

    def filter_motion(self, *, pos_xyz, heading: float, vx: float, vy: float, wz: float, dt: float) -> Tuple[float, float, float, str]:
        if self._voxel is not None:
            return self._voxel.filter_motion(pos_xyz=pos_xyz, heading=heading, vx=vx, vy=vy, wz=wz, dt=dt)
        if self._semantic2d is not None:
            return self._semantic2d.filter_motion(pos_xyz=pos_xyz, heading=heading, vx=vx, vy=vy, wz=wz, dt=dt)
        if not self.config.enabled or dt <= 0.0:
            return vx, vy, wz, "disabled"
        if abs(vx) + abs(vy) + abs(wz) < self.config.zero_eps:
            return 0.0, 0.0, 0.0, "zero"

        ok, reason, _cur, _nxt, _path = self._candidate_is_allowed_from_overlap(pos_xyz=pos_xyz, heading=heading, vx=vx, vy=vy, wz=wz, dt=dt)
        if ok:
            return vx, vy, wz, "full" if reason == "free" else reason

        if not self.config.component_fallback:
            return 0.0, 0.0, 0.0, reason

        fallbacks = []
        if abs(wz) > self.config.zero_eps:
            fallbacks.append((0.0, 0.0, wz, "yaw_only"))
        if abs(vx) > self.config.zero_eps:
            fallbacks.append((vx, 0.0, 0.0, "vx_only"))
        if abs(vy) > self.config.zero_eps:
            fallbacks.append((0.0, vy, 0.0, "vy_only"))

        for vx_i, vy_i, wz_i, label in fallbacks:
            ok_i, reason_i, _cur_i, _nxt_i, _path_i = self._candidate_is_allowed_from_overlap(pos_xyz=pos_xyz, heading=heading, vx=vx_i, vy=vy_i, wz=wz_i, dt=dt)
            if ok_i:
                return vx_i, vy_i, wz_i, label if reason_i == "free" else f"{label}_{reason_i}"
        return 0.0, 0.0, 0.0, reason


__all__ = ["GuardConfig", "KinematicCollisionGuard", "config_from_env"]
