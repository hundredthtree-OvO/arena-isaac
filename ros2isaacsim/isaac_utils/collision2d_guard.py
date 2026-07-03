"""Semantic 2D collision guard for kinematic mobile-base motion.

V14 goal:
- Stop using large 3D room-level proxy boxes as the primary collision source.
- Represent walls/partitions/doors/fixtures in a small editable 2D YAML.
- Use 2D oriented rectangles/circles for pre-move collision checks.
- Optionally keep Isaac 3D proxies only as visualization/debug aids.

YAML schema, minimal:

scene: shenxinfu_841837
robot:
  footprint: {length: 0.80, width: 0.60, margin: 0.05}
walls:
  - {name: wall_01, from: [0,0], to: [2,0], thickness: 0.12, enabled: true}
objects:
  - {name: toilet_01, type: box, center: [1,1], size: [0.6,0.8], yaw: 0, enabled: true}
doors:
  - {name: stall_door_01, state: open, center: [1,1], width: 0.75, thickness: 0.05, yaw: 1.57}
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml


@dataclass
class Semantic2DGuardConfig:
    enabled: bool = True
    config_path: str = "/home/stardust/resources/arena_ws/arena_assets/collision_configs/shenxinfu_841837.collision2d.yaml"
    length: float = 0.80
    width: float = 0.60
    margin: float = 0.05
    footprint_forward: float = -1.0
    footprint_rear: float = -1.0
    footprint_left: float = -1.0
    footprint_right: float = -1.0
    refresh_sec: float = 1.00
    log_sec: float = 5.00
    block_log_sec: float = 2.00
    component_fallback: bool = True
    zero_eps: float = 1e-4
    overlap_policy: str = "escape"  # escape | block | allow
    escape_epsilon: float = 1e-4


@dataclass
class RectObstacle:
    name: str
    kind: str
    cx: float
    cy: float
    sx: float
    sy: float
    yaw: float
    enabled: bool = True


@dataclass
class CircleObstacle:
    name: str
    kind: str
    cx: float
    cy: float
    radius: float
    enabled: bool = True


@dataclass
class CollisionResult:
    score: float
    name: Optional[str]
    kind: Optional[str]
    query_ms: float


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
    p = os.environ.get("ARENA_ISAAC_COLLISION_GUARD_OVERLAP_POLICY", "").strip().lower()
    if p in {"escape", "block", "strict", "allow"}:
        return "block" if p == "strict" else p
    allow_escape = os.environ.get("ARENA_ISAAC_COLLISION_GUARD_ALLOW_ESCAPE", "").strip().lower()
    if allow_escape in {"0", "false", "no", "off"}:
        return "block"
    if allow_escape in {"1", "true", "yes", "on"}:
        return "escape"
    return "escape"


def config_from_env() -> Semantic2DGuardConfig:
    return Semantic2DGuardConfig(
        enabled=_env_bool("ARENA_ISAAC_ENABLE_KINEMATIC_COLLISION_GUARD", True),
        config_path=os.environ.get(
            "ARENA_ISAAC_COLLISION2D_CONFIG",
            "/home/stardust/resources/arena_ws/arena_assets/collision_configs/shenxinfu_841837.collision2d.yaml",
        ),
        length=_env_float("ARENA_ISAAC_COLLISION_GUARD_LENGTH", 0.80),
        width=_env_float("ARENA_ISAAC_COLLISION_GUARD_WIDTH", 0.60),
        margin=_env_float("ARENA_ISAAC_COLLISION_GUARD_MARGIN", 0.05),
        footprint_forward=_env_float("ARENA_ISAAC_COLLISION_GUARD_FOOTPRINT_FORWARD", -1.0),
        footprint_rear=_env_float("ARENA_ISAAC_COLLISION_GUARD_FOOTPRINT_REAR", -1.0),
        footprint_left=_env_float("ARENA_ISAAC_COLLISION_GUARD_FOOTPRINT_LEFT", -1.0),
        footprint_right=_env_float("ARENA_ISAAC_COLLISION_GUARD_FOOTPRINT_RIGHT", -1.0),
        refresh_sec=_env_float("ARENA_ISAAC_COLLISION2D_REFRESH_SEC", 1.00),
        log_sec=_env_float("ARENA_ISAAC_COLLISION_GUARD_LOG_SEC", 5.00),
        block_log_sec=_env_float("ARENA_ISAAC_COLLISION_GUARD_BLOCK_LOG_SEC", 2.00),
        component_fallback=_env_bool("ARENA_ISAAC_COLLISION_GUARD_COMPONENT_FALLBACK", True),
        zero_eps=_env_float("ARENA_ISAAC_COLLISION_GUARD_ZERO_EPS", 1e-4),
        overlap_policy=_env_policy(),
        escape_epsilon=_env_float("ARENA_ISAAC_COLLISION_GUARD_ESCAPE_EPS", 1e-4),
    )


def _as_bool(v: Any, default: bool = True) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _xy(v: Sequence[Any], *, default=(0.0, 0.0)) -> Tuple[float, float]:
    try:
        return float(v[0]), float(v[1])
    except Exception:
        return float(default[0]), float(default[1])


def _line_to_rect(name: str, kind: str, p0: Sequence[Any], p1: Sequence[Any], thickness: float, enabled: bool = True) -> RectObstacle:
    x0, y0 = _xy(p0)
    x1, y1 = _xy(p1)
    dx = x1 - x0
    dy = y1 - y0
    length = max(1e-6, math.hypot(dx, dy))
    yaw = math.atan2(dy, dx)
    return RectObstacle(
        name=name,
        kind=kind,
        cx=0.5 * (x0 + x1),
        cy=0.5 * (y0 + y1),
        sx=float(length),
        sy=max(1e-4, float(thickness)),
        yaw=float(yaw),
        enabled=bool(enabled),
    )


def _door_to_rect(item: Dict[str, Any]) -> Optional[RectObstacle]:
    state = str(item.get("state", "ignore")).strip().lower()
    enabled = _as_bool(item.get("enabled", True), True)
    if not enabled or state in {"open", "ignore", "disabled", "none", "free"}:
        return None
    name = str(item.get("name", "door"))
    thickness = float(item.get("thickness", 0.05))
    width = float(item.get("width", item.get("length", 0.75)))
    yaw = float(item.get("yaw", item.get("yaw_closed", 0.0)))
    if "center" in item:
        cx, cy = _xy(item.get("center", [0, 0]))
    elif "hinge" in item:
        hx, hy = _xy(item.get("hinge", [0, 0]))
        cx = hx + 0.5 * width * math.cos(yaw)
        cy = hy + 0.5 * width * math.sin(yaw)
    elif "from" in item and "to" in item:
        return _line_to_rect(name, "door", item["from"], item["to"], thickness, enabled=True)
    else:
        cx, cy = 0.0, 0.0
    return RectObstacle(name=name, kind="door", cx=cx, cy=cy, sx=width, sy=thickness, yaw=yaw, enabled=True)


def _rect_corners(cx: float, cy: float, yaw: float, half_x: float, half_y: float) -> Tuple[List[Tuple[float, float]], Tuple[float, float], Tuple[float, float]]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    ux = (c, s)
    uy = (-s, c)
    corners = [
        (cx + dx * ux[0] + dy * uy[0], cy + dx * ux[1] + dy * uy[1])
        for dx in (-half_x, half_x)
        for dy in (-half_y, half_y)
    ]
    return corners, ux, uy


def _project(points: Sequence[Tuple[float, float]], axis: Tuple[float, float]) -> Tuple[float, float]:
    ax, ay = axis
    vals = [x * ax + y * ay for x, y in points]
    return min(vals), max(vals)


def _interval_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return not (a1 < b0 or b1 < a0)


def _obb_penetration(
    ax: float, ay: float, ayaw: float, ahx: float, ahy: float,
    bx: float, by: float, byaw: float, bhx: float, bhy: float,
) -> float:
    a_pts, a_ux, a_uy = _rect_corners(ax, ay, ayaw, ahx, ahy)
    b_pts, b_ux, b_uy = _rect_corners(bx, by, byaw, bhx, bhy)
    min_overlap = float("inf")
    for axis in (a_ux, a_uy, b_ux, b_uy):
        a0, a1 = _project(a_pts, axis)
        b0, b1 = _project(b_pts, axis)
        if not _interval_overlap(a0, a1, b0, b1):
            return 0.0
        min_overlap = min(min_overlap, min(a1, b1) - max(a0, b0))
    return 0.0 if min_overlap == float("inf") else max(0.0, float(min_overlap))


def _footprint_extents(config: Semantic2DGuardConfig) -> Tuple[float, float, float, float]:
    forward = config.footprint_forward if config.footprint_forward > 0.0 else (0.5 * config.length + config.margin)
    rear = config.footprint_rear if config.footprint_rear > 0.0 else (0.5 * config.length + config.margin)
    left = config.footprint_left if config.footprint_left > 0.0 else (0.5 * config.width + config.margin)
    right = config.footprint_right if config.footprint_right > 0.0 else (0.5 * config.width + config.margin)
    center_x_local = 0.5 * (forward - rear)
    center_y_local = 0.5 * (left - right)
    half_x = 0.5 * (forward + rear)
    half_y = 0.5 * (left + right)
    return float(center_x_local), float(center_y_local), float(half_x), float(half_y)


def _point_to_obb_local(px: float, py: float, cx: float, cy: float, yaw: float) -> Tuple[float, float]:
    dx = px - cx
    dy = py - cy
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    return c * dx - s * dy, s * dx + c * dy


def _obb_circle_penetration(rx: float, ry: float, ryaw: float, rhx: float, rhy: float, cx: float, cy: float, radius: float) -> float:
    lx, ly = _point_to_obb_local(cx, cy, rx, ry, ryaw)
    qx = min(max(lx, -rhx), rhx)
    qy = min(max(ly, -rhy), rhy)
    dx = lx - qx
    dy = ly - qy
    dist = math.hypot(dx, dy)
    if dist >= radius:
        return 0.0
    return float(radius - dist)


class Semantic2DCollisionGuard:
    def __init__(self, *, robot_name: str, logger=None, config: Optional[Semantic2DGuardConfig] = None):
        self.robot_name = robot_name
        self.logger = logger
        self.config = config or config_from_env()
        self.rects: List[RectObstacle] = []
        self.circles: List[CircleObstacle] = []
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
        path = self.config.config_path
        try:
            mtime = os.path.getmtime(path)
        except Exception:
            if force or self._mtime is not None:
                self._warn(f"[{self.robot_name}] collision2d: config not found: {path}")
            self.rects = []
            self.circles = []
            self._mtime = None
            return
        if not force and self._mtime == mtime:
            self._maybe_log_stats(extra="cached")
            return
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        self._mtime = mtime
        self._load_data(data)
        self._maybe_log_stats(extra=f"loaded={path}")

    def _load_data(self, data: Dict[str, Any]) -> None:
        robot = data.get("robot", {}) or {}
        footprint = robot.get("footprint", {}) or {}
        if "length" in footprint:
            self.config.length = float(footprint.get("length", self.config.length))
        if "width" in footprint:
            self.config.width = float(footprint.get("width", self.config.width))
        if "margin" in footprint:
            self.config.margin = float(footprint.get("margin", self.config.margin))
        if "forward" in footprint:
            self.config.footprint_forward = float(footprint.get("forward", self.config.footprint_forward))
        if "rear" in footprint:
            self.config.footprint_rear = float(footprint.get("rear", self.config.footprint_rear))
        if "left" in footprint:
            self.config.footprint_left = float(footprint.get("left", self.config.footprint_left))
        if "right" in footprint:
            self.config.footprint_right = float(footprint.get("right", self.config.footprint_right))

        rects: List[RectObstacle] = []
        circles: List[CircleObstacle] = []

        for item in data.get("walls", []) or []:
            if not _as_bool(item.get("enabled", True), True):
                continue
            if "from" in item and "to" in item:
                rects.append(_line_to_rect(
                    name=str(item.get("name", "wall")),
                    kind=str(item.get("kind", "wall")),
                    p0=item.get("from", [0, 0]),
                    p1=item.get("to", [0, 0]),
                    thickness=float(item.get("thickness", 0.10)),
                    enabled=True,
                ))
            elif "center" in item and "size" in item:
                cx, cy = _xy(item.get("center", [0, 0]))
                sx, sy = _xy(item.get("size", [1, 0.1]), default=(1, 0.1))
                rects.append(RectObstacle(str(item.get("name", "wall")), "wall", cx, cy, sx, sy, float(item.get("yaw", 0.0)), True))

        for item in data.get("partitions", []) or []:
            if not _as_bool(item.get("enabled", True), True):
                continue
            if "from" in item and "to" in item:
                rects.append(_line_to_rect(str(item.get("name", "partition")), "partition", item.get("from", [0,0]), item.get("to", [0,0]), float(item.get("thickness", 0.08)), True))
            elif "center" in item and "size" in item:
                cx, cy = _xy(item.get("center", [0,0])); sx, sy = _xy(item.get("size", [1,0.1]), default=(1,0.1))
                rects.append(RectObstacle(str(item.get("name", "partition")), "partition", cx, cy, sx, sy, float(item.get("yaw", 0.0)), True))

        for item in data.get("objects", []) or []:
            if not _as_bool(item.get("enabled", True), True):
                continue
            typ = str(item.get("type", "box")).lower()
            cx, cy = _xy(item.get("center", [0, 0]))
            if typ in {"circle", "cylinder"}:
                circles.append(CircleObstacle(str(item.get("name", "object")), str(item.get("kind", "object")), cx, cy, float(item.get("radius", 0.25)), True))
            else:
                sx, sy = _xy(item.get("size", [0.5, 0.5]), default=(0.5, 0.5))
                rects.append(RectObstacle(str(item.get("name", "object")), str(item.get("kind", typ)), cx, cy, sx, sy, float(item.get("yaw", 0.0)), True))

        for item in data.get("doors", []) or []:
            rect = _door_to_rect(item)
            if rect is not None:
                rects.append(rect)

        self.rects = rects
        self.circles = circles

    def _maybe_log_stats(self, extra: str = ""):
        now = time.monotonic()
        if self.config.log_sec <= 0 or now - self._last_log < self.config.log_sec:
            return
        self._last_log = now
        avg = self.total_query_ms / self.query_count if self.query_count else 0.0
        self._info(
            f"[{self.robot_name}] collision2d_guard: enabled={self.config.enabled} "
            f"rects={len(self.rects)} circles={len(self.circles)} blocked={self.block_count}/{self.query_count} "
            f"avg_ms={avg:.3f} max_ms={self.max_query_ms:.3f} policy={self.config.overlap_policy} {extra}".strip()
        )

    def _collision_score(self, pos_xyz, heading: float) -> CollisionResult:
        if not self.config.enabled:
            return CollisionResult(0.0, None, None, 0.0)
        self.refresh()
        t0 = time.perf_counter()
        x = float(pos_xyz[0])
        y = float(pos_xyz[1])
        center_x_local, center_y_local, rhx, rhy = _footprint_extents(self.config)
        c_h = math.cos(heading)
        s_h = math.sin(heading)
        cx = x + c_h * center_x_local - s_h * center_y_local
        cy = y + s_h * center_x_local + c_h * center_y_local
        best = 0.0
        best_name = None
        best_kind = None
        for obs in self.rects:
            score = _obb_penetration(cx, cy, heading, rhx, rhy, obs.cx, obs.cy, obs.yaw, 0.5 * obs.sx, 0.5 * obs.sy)
            if score > best:
                best = score
                best_name = obs.name
                best_kind = obs.kind
        for obs in self.circles:
            score = _obb_circle_penetration(cx, cy, heading, rhx, rhy, obs.cx, obs.cy, obs.radius)
            if score > best:
                best = score
                best_name = obs.name
                best_kind = obs.kind
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.query_count += 1
        self.total_query_ms += dt_ms
        self.max_query_ms = max(self.max_query_ms, dt_ms)
        if best > 0.0:
            self.block_count += 1
        self._maybe_log_stats()
        return CollisionResult(best, best_name, best_kind, dt_ms)

    def _predict(self, *, pos_xyz, heading: float, vx: float, vy: float, wz: float, dt: float):
        next_heading = heading + wz * dt
        c = math.cos(next_heading)
        s = math.sin(next_heading)
        wx = c * vx - s * vy
        wy = s * vx + c * vy
        next_pos = np.array([float(pos_xyz[0]) + wx * dt, float(pos_xyz[1]) + wy * dt, float(pos_xyz[2])], dtype=np.float32)
        return next_pos, next_heading

    def _maybe_log_block(self, name: str, dt_ms: float, *, score: float):
        now = time.monotonic()
        if now - self._last_block_log < self.config.block_log_sec:
            return
        self._last_block_log = now
        self._warn(f"[{self.robot_name}] collision2d_guard blocked kinematic step: obstacle={name}, penetration={score:.4f}, query_ms={dt_ms:.3f}")

    def _maybe_log_inside(self, name: str, *, cur: float, nxt: float, action: str):
        now = time.monotonic()
        if now - self._last_inside_log < self.config.block_log_sec:
            return
        self._last_inside_log = now
        self._warn(f"[{self.robot_name}] collision2d_guard current pose overlaps {name}; policy={self.config.overlap_policy} action={action} current_penetration={cur:.4f} next_penetration={nxt:.4f}")

    def _candidate_allowed(self, *, pos_xyz, heading: float, vx: float, vy: float, wz: float, dt: float) -> Tuple[bool, str, float, float, Optional[str]]:
        cur = self._collision_score(pos_xyz, heading)
        next_pos, next_heading = self._predict(pos_xyz=pos_xyz, heading=heading, vx=vx, vy=vy, wz=wz, dt=dt)
        nxt = self._collision_score(next_pos, next_heading)
        name = nxt.name or cur.name
        if cur.score <= 0.0:
            if nxt.score <= 0.0:
                return True, "free", cur.score, nxt.score, name
            self._maybe_log_block(name or "unknown", nxt.query_ms, score=nxt.score)
            return False, "blocked", cur.score, nxt.score, name
        policy = self.config.overlap_policy
        if policy == "allow":
            self._maybe_log_inside(name or "unknown", cur=cur.score, nxt=nxt.score, action="allow_legacy")
            return True, "inside_allow_legacy", cur.score, nxt.score, name
        if policy == "block":
            self._maybe_log_inside(name or "unknown", cur=cur.score, nxt=nxt.score, action="strict_block")
            return False, "inside_strict_block", cur.score, nxt.score, name
        if nxt.score + self.config.escape_epsilon < cur.score:
            self._maybe_log_inside(name or "unknown", cur=cur.score, nxt=nxt.score, action="escape_allowed")
            return True, "inside_escape_reduce", cur.score, nxt.score, name
        self._maybe_log_inside(name or "unknown", cur=cur.score, nxt=nxt.score, action="escape_block")
        return False, "inside_escape_block", cur.score, nxt.score, name

    def filter_motion(self, *, pos_xyz, heading: float, vx: float, vy: float, wz: float, dt: float) -> Tuple[float, float, float, str]:
        if not self.config.enabled or dt <= 0.0:
            return vx, vy, wz, "2d_disabled"
        if abs(vx) + abs(vy) + abs(wz) < self.config.zero_eps:
            return 0.0, 0.0, 0.0, "2d_zero"
        ok, reason, _cur, _nxt, _name = self._candidate_allowed(pos_xyz=pos_xyz, heading=heading, vx=vx, vy=vy, wz=wz, dt=dt)
        if ok:
            return vx, vy, wz, "2d_full" if reason == "free" else f"2d_{reason}"
        if not self.config.component_fallback:
            return 0.0, 0.0, 0.0, f"2d_{reason}"
        fallbacks = []
        if abs(wz) > self.config.zero_eps:
            fallbacks.append((0.0, 0.0, wz, "yaw_only"))
        if abs(vx) > self.config.zero_eps:
            fallbacks.append((vx, 0.0, 0.0, "vx_only"))
        if abs(vy) > self.config.zero_eps:
            fallbacks.append((0.0, vy, 0.0, "vy_only"))
        for vx_i, vy_i, wz_i, label in fallbacks:
            ok_i, reason_i, _cur_i, _nxt_i, _name_i = self._candidate_allowed(pos_xyz=pos_xyz, heading=heading, vx=vx_i, vy=vy_i, wz=wz_i, dt=dt)
            if ok_i:
                return vx_i, vy_i, wz_i, f"2d_{label}" if reason_i == "free" else f"2d_{label}_{reason_i}"
        return 0.0, 0.0, 0.0, f"2d_{reason}"


__all__ = ["Semantic2DGuardConfig", "Semantic2DCollisionGuard", "config_from_env"]
