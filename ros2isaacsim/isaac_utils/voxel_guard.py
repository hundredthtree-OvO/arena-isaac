"""Voxel / OctoMap-like kinematic collision guard.

V15 goal
---------
This guard avoids using editable 3D proxy cubes as the runtime collision source.
It loads a compact voxel map generated from the current Isaac USD stage and
checks the robot's planar footprint against occupied voxel columns.

The map is OctoMap-like in spirit: it stores occupied 3D voxels.  For a mobile
base, runtime collision checks collapse occupied voxels in the configured height
band to XY columns and test the kinematic footprint before each USD transform
update.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


@dataclass
class VoxelGuardConfig:
    enabled: bool = True
    map_path: str = "/home/stardust/resources/arena_ws/arena_assets/collision_configs/shenxinfu_841837.voxel.json.gz"
    length: float = 0.80
    width: float = 0.60
    margin: float = 0.05
    z_min: float = 0.05
    z_max: float = 1.20
    refresh_sec: float = 1.0
    log_sec: float = 5.0
    block_log_sec: float = 2.0
    overlap_policy: str = "escape"  # escape | block | allow
    escape_epsilon: float = 1e-4
    zero_eps: float = 1e-4
    # Treat each occupied voxel as a small square on the ground.  This is added
    # to robot half extents so that cell centers near a boundary still count.
    cell_padding: float = -1.0  # <0 -> resolution * 0.5


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
    explicit = os.environ.get("ARENA_ISAAC_COLLISION_GUARD_OVERLAP_POLICY", "").strip().lower()
    if explicit in {"block", "strict", "escape", "allow"}:
        return "block" if explicit == "strict" else explicit
    allow_escape = os.environ.get("ARENA_ISAAC_COLLISION_GUARD_ALLOW_ESCAPE", "").strip().lower()
    if allow_escape in {"0", "false", "no", "off"}:
        return "block"
    if allow_escape in {"1", "true", "yes", "on"}:
        return "escape"
    return "escape"


def config_from_env() -> VoxelGuardConfig:
    return VoxelGuardConfig(
        enabled=_env_bool("ARENA_ISAAC_ENABLE_KINEMATIC_COLLISION_GUARD", True),
        map_path=os.environ.get(
            "ARENA_ISAAC_VOXEL_MAP_PATH",
            "/home/stardust/resources/arena_ws/arena_assets/collision_configs/shenxinfu_841837.voxel.json.gz",
        ),
        length=_env_float("ARENA_ISAAC_COLLISION_GUARD_LENGTH", 0.80),
        width=_env_float("ARENA_ISAAC_COLLISION_GUARD_WIDTH", 0.60),
        margin=_env_float("ARENA_ISAAC_COLLISION_GUARD_MARGIN", 0.05),
        z_min=_env_float("ARENA_ISAAC_VOXEL_GUARD_Z_MIN", _env_float("ARENA_ISAAC_COLLISION_GUARD_Z_MIN", 0.05)),
        z_max=_env_float("ARENA_ISAAC_VOXEL_GUARD_Z_MAX", _env_float("ARENA_ISAAC_COLLISION_GUARD_Z_MAX", 1.20)),
        refresh_sec=_env_float("ARENA_ISAAC_VOXEL_GUARD_REFRESH_SEC", 1.0),
        log_sec=_env_float("ARENA_ISAAC_COLLISION_GUARD_LOG_SEC", 5.0),
        block_log_sec=_env_float("ARENA_ISAAC_COLLISION_GUARD_BLOCK_LOG_SEC", 2.0),
        overlap_policy=_env_policy(),
        escape_epsilon=_env_float("ARENA_ISAAC_COLLISION_GUARD_ESCAPE_EPS", 1e-4),
        zero_eps=_env_float("ARENA_ISAAC_COLLISION_GUARD_ZERO_EPS", 1e-4),
        cell_padding=_env_float("ARENA_ISAAC_VOXEL_GUARD_CELL_PADDING", -1.0),
    )


def _open_json_maybe_gz(path: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


class VoxelCollisionGuard:
    """Collision guard backed by occupied voxel columns.

    Public API mirrors the proxy/2D guards enough for ``KinematicCollisionGuard``
    to delegate to it.
    """

    def __init__(self, *, robot_name: str, logger=None, config: Optional[VoxelGuardConfig] = None):
        self.robot_name = robot_name
        self.logger = logger
        self.config = config or config_from_env()
        self.resolution: float = 0.05
        self.origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.occupied_xy: Set[Tuple[int, int]] = set()
        self.columns_z: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.bounds_xy: Optional[Tuple[int, int, int, int]] = None
        self._mtime: Optional[float] = None
        self._last_refresh = 0.0
        self._last_log = 0.0
        self._last_block_log = 0.0
        self._last_inside_log = 0.0
        self.query_count = 0
        self.block_count = 0
        self.total_query_ms = 0.0
        self.max_query_ms = 0.0
        self.refresh(force=True)

    def _info(self, text: str):
        self.logger.info(text) if self.logger else print(text)

    def _warn(self, text: str):
        self.logger.warning(text) if self.logger else print("WARNING:", text)

    def refresh(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_refresh < self.config.refresh_sec:
            return
        self._last_refresh = now
        if not self.config.enabled:
            self.occupied_xy = set()
            self.columns_z = {}
            return
        path = self.config.map_path
        try:
            mtime = os.path.getmtime(path)
        except Exception:
            if force:
                self._warn(f"[{self.robot_name}] voxel_guard map not found: {path}")
            self.occupied_xy = set()
            self.columns_z = {}
            return
        if not force and self._mtime == mtime:
            return
        self._mtime = mtime
        try:
            with _open_json_maybe_gz(path) as f:
                data = json.load(f)
            self.resolution = float(data.get("resolution", data.get("voxel_size", 0.05)))
            origin = data.get("origin", [0.0, 0.0, 0.0])
            self.origin = (float(origin[0]), float(origin[1]), float(origin[2]))
            z_min_idx = math.floor((self.config.z_min - self.origin[2]) / self.resolution)
            z_max_idx = math.ceil((self.config.z_max - self.origin[2]) / self.resolution)
            columns: Dict[Tuple[int, int], Tuple[int, int]] = {}
            if data.get("columns"):
                for item in data.get("columns", []):
                    if len(item) < 4:
                        continue
                    ix, iy, iz0, iz1 = int(item[0]), int(item[1]), int(item[2]), int(item[3])
                    if iz1 < z_min_idx or iz0 > z_max_idx:
                        continue
                    columns[(ix, iy)] = (iz0, iz1)
            else:
                for item in data.get("voxels", []):
                    if len(item) < 3:
                        continue
                    ix, iy, iz = int(item[0]), int(item[1]), int(item[2])
                    if iz < z_min_idx or iz > z_max_idx:
                        continue
                    key = (ix, iy)
                    old = columns.get(key)
                    if old is None:
                        columns[key] = (iz, iz)
                    else:
                        columns[key] = (min(old[0], iz), max(old[1], iz))
            self.columns_z = columns
            self.occupied_xy = set(columns.keys())
            if self.occupied_xy:
                xs = [p[0] for p in self.occupied_xy]
                ys = [p[1] for p in self.occupied_xy]
                self.bounds_xy = (min(xs), max(xs), min(ys), max(ys))
            else:
                self.bounds_xy = None
            self._info(
                f"[{self.robot_name}] voxel_guard loaded: path={path} resolution={self.resolution:.3f} "
                f"columns={len(self.occupied_xy)} z_band=[{self.config.z_min:.2f},{self.config.z_max:.2f}] policy={self.config.overlap_policy}"
            )
        except Exception as exc:
            self._warn(f"[{self.robot_name}] failed to load voxel map {path}: {exc}")
            self.occupied_xy = set()
            self.columns_z = {}
            self.bounds_xy = None

    def _maybe_log_stats(self, extra: str = ""):
        now = time.monotonic()
        if self.config.log_sec <= 0 or now - self._last_log < self.config.log_sec:
            return
        self._last_log = now
        avg = self.total_query_ms / self.query_count if self.query_count else 0.0
        self._info(
            f"[{self.robot_name}] voxel_guard: enabled={self.config.enabled} columns={len(self.occupied_xy)} "
            f"blocked={self.block_count}/{self.query_count} avg_ms={avg:.3f} max_ms={self.max_query_ms:.3f} {extra}".strip()
        )

    def _candidate_indices(self, x: float, y: float, heading: float, half_len: float, half_wid: float) -> Iterable[Tuple[int, int]]:
        # Global AABB of the oriented footprint, expanded by one cell.
        c = math.cos(heading)
        s = math.sin(heading)
        ux = (c, s)
        uy = (-s, c)
        corners = [
            (x + dx * ux[0] + dy * uy[0], y + dx * ux[1] + dy * uy[1])
            for dx in (-half_len, half_len)
            for dy in (-half_wid, half_wid)
        ]
        pad = self.resolution
        min_x = min(p[0] for p in corners) - pad
        max_x = max(p[0] for p in corners) + pad
        min_y = min(p[1] for p in corners) - pad
        max_y = max(p[1] for p in corners) + pad
        ox, oy, _ = self.origin
        ix0 = math.floor((min_x - ox) / self.resolution)
        ix1 = math.floor((max_x - ox) / self.resolution)
        iy0 = math.floor((min_y - oy) / self.resolution)
        iy1 = math.floor((max_y - oy) / self.resolution)
        if self.bounds_xy is not None:
            bx0, bx1, by0, by1 = self.bounds_xy
            ix0, ix1 = max(ix0, bx0), min(ix1, bx1)
            iy0, iy1 = max(iy0, by0), min(iy1, by1)
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                yield ix, iy

    def _collision_score(self, pos_xyz, heading: float) -> Tuple[float, Optional[str], float]:
        if not self.config.enabled:
            return 0.0, None, 0.0
        self.refresh()
        if not self.occupied_xy:
            self._maybe_log_stats(extra="no_voxels")
            return 0.0, None, 0.0
        t0 = time.perf_counter()
        x = float(pos_xyz[0])
        y = float(pos_xyz[1])
        c = math.cos(-heading)
        s = math.sin(-heading)
        pad = self.config.cell_padding
        if pad < 0.0:
            pad = 0.5 * self.resolution
        half_len = max(0.01, 0.5 * self.config.length + self.config.margin + pad)
        half_wid = max(0.01, 0.5 * self.config.width + self.config.margin + pad)
        ox, oy, _ = self.origin
        best = 0.0
        best_key: Optional[Tuple[int, int]] = None
        for ix, iy in self._candidate_indices(x, y, heading, half_len, half_wid):
            if (ix, iy) not in self.occupied_xy:
                continue
            wx = ox + (ix + 0.5) * self.resolution
            wy = oy + (iy + 0.5) * self.resolution
            dx = wx - x
            dy = wy - y
            lx = c * dx - s * dy
            ly = s * dx + c * dy
            if abs(lx) <= half_len and abs(ly) <= half_wid:
                # Positive conservative penetration score: how deeply the cell center lies in the footprint.
                score = min(half_len - abs(lx), half_wid - abs(ly))
                if score > best:
                    best = float(score)
                    best_key = (ix, iy)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.query_count += 1
        self.total_query_ms += dt_ms
        self.max_query_ms = max(self.max_query_ms, dt_ms)
        if best > 0.0:
            self.block_count += 1
        self._maybe_log_stats()
        path = f"voxel:{best_key[0]},{best_key[1]}" if best_key is not None else None
        return best, path, dt_ms

    def pose_collides(self, pos_xyz, heading: float, *, log_block: bool = True) -> Tuple[bool, Optional[str], float]:
        score, path, dt_ms = self._collision_score(pos_xyz, heading)
        if score > 0.0:
            if log_block:
                self._maybe_log_block(path or "voxel", dt_ms, score=score)
            return True, path, dt_ms
        return False, None, dt_ms

    def _maybe_log_block(self, path: str, dt_ms: float, *, score: float = 0.0):
        now = time.monotonic()
        if now - self._last_block_log < self.config.block_log_sec:
            return
        self._last_block_log = now
        self._warn(f"[{self.robot_name}] voxel_guard blocked kinematic step: obstacle={path}, penetration={score:.4f}, query_ms={dt_ms:.3f}")

    def _maybe_log_inside(self, path: str, *, current_score: float, next_score: float, action: str):
        now = time.monotonic()
        if now - self._last_inside_log < self.config.block_log_sec:
            return
        self._last_inside_log = now
        self._warn(
            f"[{self.robot_name}] voxel_guard current pose overlaps {path}; policy={self.config.overlap_policy} "
            f"action={action} current_penetration={current_score:.4f} next_penetration={next_score:.4f}"
        )

    def _predict(self, *, pos_xyz, heading: float, vx: float, vy: float, wz: float, dt: float):
        next_heading = heading + wz * dt
        c = math.cos(next_heading)
        s = math.sin(next_heading)
        wx = c * vx - s * vy
        wy = s * vx + c * vy
        next_pos = np.array([float(pos_xyz[0]) + wx * dt, float(pos_xyz[1]) + wy * dt, float(pos_xyz[2])], dtype=np.float32)
        return next_pos, next_heading

    def _candidate_is_allowed_from_overlap(self, *, pos_xyz, heading: float, vx: float, vy: float, wz: float, dt: float):
        current_score, current_path, _ = self._collision_score(pos_xyz, heading)
        if current_score <= 0.0:
            next_pos, next_heading = self._predict(pos_xyz=pos_xyz, heading=heading, vx=vx, vy=vy, wz=wz, dt=dt)
            next_score, next_path, next_ms = self._collision_score(next_pos, next_heading)
            if next_score <= 0.0:
                return True, "free", current_score, next_score, next_path
            self._maybe_log_block(next_path or "voxel", next_ms, score=next_score)
            return False, "blocked", current_score, next_score, next_path

        next_pos, next_heading = self._predict(pos_xyz=pos_xyz, heading=heading, vx=vx, vy=vy, wz=wz, dt=dt)
        next_score, next_path, _ = self._collision_score(next_pos, next_heading)
        path = next_path or current_path or "voxel"
        policy = self.config.overlap_policy
        if policy == "allow":
            self._maybe_log_inside(path, current_score=current_score, next_score=next_score, action="allow_legacy")
            return True, "inside_allow_legacy", current_score, next_score, path
        if policy == "block":
            self._maybe_log_inside(path, current_score=current_score, next_score=next_score, action="strict_block")
            return False, "inside_strict_block", current_score, next_score, path
        if next_score + self.config.escape_epsilon < current_score:
            self._maybe_log_inside(path, current_score=current_score, next_score=next_score, action="escape_allowed")
            return True, "inside_escape_reduce", current_score, next_score, path
        self._maybe_log_inside(path, current_score=current_score, next_score=next_score, action="escape_block")
        return False, "inside_escape_block", current_score, next_score, path

    def filter_motion(self, *, pos_xyz, heading: float, vx: float, vy: float, wz: float, dt: float):
        if not self.config.enabled:
            return vx, vy, wz, "disabled"
        if abs(vx) < self.config.zero_eps and abs(vy) < self.config.zero_eps and abs(wz) < self.config.zero_eps:
            self._maybe_log_stats(extra="zero_velocity_skip")
            return vx, vy, wz, "zero"

        ok, mode, _, _, path = self._candidate_is_allowed_from_overlap(pos_xyz=pos_xyz, heading=heading, vx=vx, vy=vy, wz=wz, dt=dt)
        if ok:
            return vx, vy, wz, mode

        # Component fallback: if full twist collides, try dropping rotation and lateral/linear components.
        candidates = [
            (vx, vy, 0.0, "block_wz"),
            (vx, 0.0, wz, "block_vy"),
            (0.0, vy, wz, "block_vx"),
            (vx, 0.0, 0.0, "vx_only"),
            (0.0, vy, 0.0, "vy_only"),
            (0.0, 0.0, wz, "wz_only"),
        ]
        for cvx, cvy, cwz, cmode in candidates:
            if abs(cvx) < self.config.zero_eps and abs(cvy) < self.config.zero_eps and abs(cwz) < self.config.zero_eps:
                continue
            ok2, _, _, _, _ = self._candidate_is_allowed_from_overlap(pos_xyz=pos_xyz, heading=heading, vx=cvx, vy=cvy, wz=cwz, dt=dt)
            if ok2:
                return cvx, cvy, cwz, cmode
        return 0.0, 0.0, 0.0, "blocked"
