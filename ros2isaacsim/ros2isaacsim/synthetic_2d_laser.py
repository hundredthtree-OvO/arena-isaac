#!/usr/bin/env python3
"""Synthetic 2D LaserScan publisher for navigation-oriented map + people scans.

This node keeps the real calibration frames (for example base_scan_01 /
base_scan_02) but generates LaserScan data in 2D from a navigation map instead
of Isaac RTX lidar. Static obstacles come from either a live OccupancyGrid
topic, a ROS map YAML, or the legacy voxel JSON fallback. Dynamic pedestrians
are overlaid as circles using the task_generator people topics.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import rclpy
import yaml
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener

try:
    from nav_msgs.msg import OccupancyGrid
except Exception:  # pragma: no cover - depends on runtime environment
    OccupancyGrid = None

try:
    from people_msgs.msg import People
except Exception:  # pragma: no cover - depends on runtime environment
    People = None

try:
    from arena_people_msgs.msg import Pedestrians as ArenaPedestrians
except Exception:  # pragma: no cover - depends on runtime environment
    ArenaPedestrians = None


@dataclass(frozen=True)
class SideConfig:
    name: str
    frame_id: str
    topic: str
    enabled: bool
    angle_offset_rad: float
    reverse: bool


@dataclass
class StaticMapData:
    occupied_xy: Set[Tuple[int, int]]
    resolution: float
    origin: Tuple[float, float, float]
    bounds_xy: Optional[Tuple[int, int, int, int]]
    frame_id: str
    source: str
    mtime: Optional[float]


@dataclass
class DynamicCircle:
    x: float
    y: float
    radius: float
    stamp_monotonic: float


EMPTY_MAP = StaticMapData(set(), 0.05, (0.0, 0.0, 0.0), None, "odom", "empty", None)


def _open_json_maybe_gz(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _quat_rotate_xyzw(
    qx: float,
    qy: float,
    qz: float,
    qw: float,
    vx: float,
    vy: float,
    vz: float,
) -> Tuple[float, float, float]:
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    rx = vx + qw * tx + (qy * tz - qz * ty)
    ry = vy + qw * ty + (qz * tx - qx * tz)
    rz = vz + qw * tz + (qx * ty - qy * tx)
    return rx, ry, rz


def _make_bounds(occupied: Set[Tuple[int, int]]) -> Optional[Tuple[int, int, int, int]]:
    if not occupied:
        return None
    xs = [p[0] for p in occupied]
    ys = [p[1] for p in occupied]
    return min(xs), max(xs), min(ys), max(ys)


def _read_pgm_token(stream) -> Optional[bytes]:
    token = bytearray()
    while True:
        ch = stream.read(1)
        if not ch:
            break
        if ch == b"#":
            stream.readline()
            continue
        if ch.isspace():
            if token:
                break
            continue
        token.extend(ch)
    return bytes(token) if token else None


def _load_pgm_image(path: Path) -> Tuple[int, int, int, List[int]]:
    with path.open("rb") as f:
        magic = _read_pgm_token(f)
        if magic not in {b"P2", b"P5"}:
            raise ValueError(f"unsupported map image format for {path}; expected P2/P5 PGM")
        width_token = _read_pgm_token(f)
        height_token = _read_pgm_token(f)
        maxval_token = _read_pgm_token(f)
        if width_token is None or height_token is None or maxval_token is None:
            raise ValueError(f"incomplete PGM header: {path}")
        width = int(width_token)
        height = int(height_token)
        maxval = int(maxval_token)
        if width <= 0 or height <= 0 or maxval <= 0:
            raise ValueError(f"invalid PGM dimensions/maxval in {path}")

        total = width * height
        if magic == b"P2":
            values: List[int] = []
            while len(values) < total:
                token = _read_pgm_token(f)
                if token is None:
                    break
                values.append(int(token))
        else:
            bytes_per = 1 if maxval < 256 else 2
            raw = f.read(total * bytes_per)
            if len(raw) != total * bytes_per:
                raise ValueError(f"unexpected PGM payload length in {path}")
            if bytes_per == 1:
                values = list(raw)
            else:
                values = [
                    int.from_bytes(raw[i:i + 2], byteorder="big", signed=False)
                    for i in range(0, len(raw), 2)
                ]
        if len(values) != total:
            raise ValueError(f"unexpected PGM pixel count in {path}")
        return width, height, maxval, values


def _grid_from_voxel_map(
    data: Dict[str, Any],
    *,
    z_min: float,
    z_max: float,
    frame_id: str,
    source: str,
    mtime: Optional[float],
) -> StaticMapData:
    res = float(data.get("resolution", data.get("voxel_size", 0.05)))
    origin_raw = data.get("origin", [0.0, 0.0, 0.0])
    origin = (float(origin_raw[0]), float(origin_raw[1]), float(origin_raw[2]))
    z_min_idx = math.floor((float(z_min) - origin[2]) / res)
    z_max_idx = math.ceil((float(z_max) - origin[2]) / res)
    occupied: Set[Tuple[int, int]] = set()

    if data.get("columns"):
        for item in data.get("columns", []) or []:
            if len(item) < 4:
                continue
            ix, iy, iz0, iz1 = int(item[0]), int(item[1]), int(item[2]), int(item[3])
            if iz1 < z_min_idx or iz0 > z_max_idx:
                continue
            occupied.add((ix, iy))
    else:
        for item in data.get("voxels", []) or []:
            if len(item) < 3:
                continue
            ix, iy, iz = int(item[0]), int(item[1]), int(item[2])
            if iz < z_min_idx or iz > z_max_idx:
                continue
            occupied.add((ix, iy))

    return StaticMapData(occupied, res, origin, _make_bounds(occupied), frame_id, source, mtime)


def _grid_from_map_yaml(
    path: Path,
    *,
    frame_id: str,
    unknown_is_occupied: bool,
    mtime: Optional[float],
) -> StaticMapData:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    image_path = Path(str(data.get("image", "")).strip())
    if not image_path.is_absolute():
        image_path = (path.parent / image_path).resolve()
    width, height, maxval, pixels = _load_pgm_image(image_path)
    negate = bool(int(data.get("negate", 0)))
    occupied_thresh = float(data.get("occupied_thresh", 0.65))
    free_thresh = float(data.get("free_thresh", 0.196))
    resolution = float(data.get("resolution", 0.05))
    origin_raw = data.get("origin", [0.0, 0.0, 0.0])
    origin = (float(origin_raw[0]), float(origin_raw[1]), float(origin_raw[2]))
    occupied: Set[Tuple[int, int]] = set()

    for img_y in range(height):
        iy = height - 1 - img_y
        for ix in range(width):
            value = int(pixels[img_y * width + ix])
            occ = max(0.0, min(float(maxval), float(value))) / float(maxval)
            occ = occ if negate else 1.0 - occ
            if occ >= occupied_thresh or (unknown_is_occupied and occ > free_thresh):
                occupied.add((ix, iy))

    return StaticMapData(occupied, resolution, origin, _make_bounds(occupied), frame_id, f"map_yaml:{path}", mtime)


def _grid_from_occupancy_msg(
    msg: Any,
    *,
    threshold: int,
    unknown_is_occupied: bool,
    frame_id: str,
    source: str,
) -> StaticMapData:
    info = msg.info
    origin = (
        float(info.origin.position.x),
        float(info.origin.position.y),
        float(info.origin.position.z),
    )
    occupied: Set[Tuple[int, int]] = set()
    width = int(info.width)
    for idx, value in enumerate(msg.data):
        cell = int(value)
        if cell < 0 and not unknown_is_occupied:
            continue
        if cell >= threshold or (cell < 0 and unknown_is_occupied):
            ix = idx % width
            iy = idx // width
            occupied.add((ix, iy))
    return StaticMapData(
        occupied,
        float(info.resolution),
        origin,
        _make_bounds(occupied),
        frame_id,
        source,
        None,
    )


def _ray_grid_distance(
    ox: float,
    oy: float,
    dx: float,
    dy: float,
    data: StaticMapData,
    range_min: float,
    range_max: float,
) -> Optional[float]:
    occupied = data.occupied_xy
    if not occupied:
        return None
    res = float(data.resolution)
    gx0 = (float(ox) - data.origin[0]) / res
    gy0 = (float(oy) - data.origin[1]) / res
    ix = math.floor(gx0)
    iy = math.floor(gy0)
    start_cell = (int(ix), int(iy))

    step_x = 1 if dx > 0.0 else -1 if dx < 0.0 else 0
    step_y = 1 if dy > 0.0 else -1 if dy < 0.0 else 0
    inf = float("inf")

    if step_x == 0:
        t_max_x = inf
        t_delta_x = inf
    else:
        next_x = ix + 1 if step_x > 0 else ix
        world_next_x = data.origin[0] + next_x * res
        t_max_x = (world_next_x - ox) / dx
        t_delta_x = res / abs(dx)

    if step_y == 0:
        t_max_y = inf
        t_delta_y = inf
    else:
        next_y = iy + 1 if step_y > 0 else iy
        world_next_y = data.origin[1] + next_y * res
        t_max_y = (world_next_y - oy) / dy
        t_delta_y = res / abs(dy)

    max_steps = max(1, int(math.ceil(float(range_max) / max(res, 1e-6))) + 4)
    bx0 = by0 = bx1 = by1 = None
    if data.bounds_xy is not None:
        bx0, bx1, by0, by1 = data.bounds_xy

    for _ in range(max_steps):
        cell = (int(ix), int(iy))
        if cell != start_cell and cell in occupied:
            t_hit = min(t_max_x, t_max_y)
            if not math.isfinite(t_hit):
                t_hit = range_min
            t_hit = max(float(range_min), float(t_hit))
            return t_hit if t_hit <= range_max else None

        if t_max_x < t_max_y:
            t = t_max_x
            ix += step_x
            t_max_x += t_delta_x
        else:
            t = t_max_y
            iy += step_y
            t_max_y += t_delta_y

        if t > range_max:
            return None
        if bx0 is not None:
            if (ix < bx0 - 2 and step_x <= 0) or (ix > bx1 + 2 and step_x >= 0):
                return None
            if (iy < by0 - 2 and step_y <= 0) or (iy > by1 + 2 and step_y >= 0):
                return None
    return None


def _ray_circle_distance(
    ox: float,
    oy: float,
    dx: float,
    dy: float,
    circle: DynamicCircle,
    range_min: float,
    range_max: float,
) -> Optional[float]:
    rel_x = circle.x - ox
    rel_y = circle.y - oy
    proj = rel_x * dx + rel_y * dy
    if proj < range_min - circle.radius or proj > range_max + circle.radius:
        return None
    closest_sq = rel_x * rel_x + rel_y * rel_y - proj * proj
    radius_sq = circle.radius * circle.radius
    if closest_sq > radius_sq:
        return None
    half_chord = math.sqrt(max(0.0, radius_sq - closest_sq))
    for candidate in (proj - half_chord, proj + half_chord):
        if range_min <= candidate <= range_max:
            return candidate
    return None


class Synthetic2DLaser(Node):
    def __init__(self) -> None:
        super().__init__("synthetic_2d_laser")
        self.declare_parameter("map_path", "")
        self.declare_parameter("config_path", "")
        self.declare_parameter("map_yaml_path", "")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("auto_use_map_topic_frame", True)
        self.declare_parameter("map_occupied_threshold", 50)
        self.declare_parameter("map_unknown_is_occupied", False)
        self.declare_parameter("map_frame", "odom")
        self.declare_parameter("front_frame", "base_scan_01")
        self.declare_parameter("rear_frame", "base_scan_02")
        self.declare_parameter("front_topic", "/front_scan")
        self.declare_parameter("rear_topic", "/rear_scan")
        self.declare_parameter("front_enabled", True)
        self.declare_parameter("rear_enabled", True)
        self.declare_parameter("samples", 721)
        self.declare_parameter("fov_deg", 270.0)
        self.declare_parameter("range_min", 0.05)
        self.declare_parameter("range_max", 12.0)
        self.declare_parameter("update_rate", 10.0)
        self.declare_parameter("z_min", 0.05)
        self.declare_parameter("z_max", 1.20)
        self.declare_parameter("tf_timeout_sec", 0.05)
        self.declare_parameter("log_sec", 2.0)
        self.declare_parameter("front_angle_offset_deg", 0.0)
        self.declare_parameter("rear_angle_offset_deg", 0.0)
        self.declare_parameter("front_reverse", False)
        self.declare_parameter("rear_reverse", False)
        self.declare_parameter("dynamic_enabled", True)
        self.declare_parameter("people_frame", "")
        self.declare_parameter("people_topic", "/task_generator_node/people")
        self.declare_parameter("arena_people_topic", "/task_generator_node/arena_peds")
        self.declare_parameter("pedestrian_radius", 0.35)
        self.declare_parameter("pedestrian_timeout_sec", 1.0)

        map_path = str(self.get_parameter("map_path").value).strip()
        alias_path = str(self.get_parameter("config_path").value).strip()
        self.legacy_map_path = map_path or alias_path
        self.map_yaml_path = str(self.get_parameter("map_yaml_path").value).strip()
        self.map_topic = str(self.get_parameter("map_topic").value).strip()
        self.auto_use_map_topic_frame = bool(self.get_parameter("auto_use_map_topic_frame").value)
        self.map_occupied_threshold = int(self.get_parameter("map_occupied_threshold").value)
        self.map_unknown_is_occupied = bool(self.get_parameter("map_unknown_is_occupied").value)
        self.configured_map_frame = str(self.get_parameter("map_frame").value).strip() or "map"
        self.samples = max(2, int(self.get_parameter("samples").value))
        self.fov_deg = float(self.get_parameter("fov_deg").value)
        self.range_min = float(self.get_parameter("range_min").value)
        self.range_max = float(self.get_parameter("range_max").value)
        self.update_rate = max(0.1, float(self.get_parameter("update_rate").value))
        self.z_min = float(self.get_parameter("z_min").value)
        self.z_max = float(self.get_parameter("z_max").value)
        self.tf_timeout_sec = max(0.0, float(self.get_parameter("tf_timeout_sec").value))
        self.log_sec = float(self.get_parameter("log_sec").value)
        self.dynamic_enabled = bool(self.get_parameter("dynamic_enabled").value)
        self.people_frame = str(self.get_parameter("people_frame").value).strip()
        self.people_topic = str(self.get_parameter("people_topic").value).strip()
        self.arena_people_topic = str(self.get_parameter("arena_people_topic").value).strip()
        self.pedestrian_radius = max(0.01, float(self.get_parameter("pedestrian_radius").value))
        self.pedestrian_timeout_sec = max(0.0, float(self.get_parameter("pedestrian_timeout_sec").value))

        self._last_log = time.monotonic()
        self._scan_count = 0
        self._tf_failures = 0
        self._static_map = EMPTY_MAP
        self._topic_map_received = False
        self._dynamic_people: Dict[str, DynamicCircle] = {}

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.sides = [
            SideConfig(
                name="front",
                frame_id=str(self.get_parameter("front_frame").value),
                topic=str(self.get_parameter("front_topic").value),
                enabled=bool(self.get_parameter("front_enabled").value),
                angle_offset_rad=math.radians(float(self.get_parameter("front_angle_offset_deg").value)),
                reverse=bool(self.get_parameter("front_reverse").value),
            ),
            SideConfig(
                name="rear",
                frame_id=str(self.get_parameter("rear_frame").value),
                topic=str(self.get_parameter("rear_topic").value),
                enabled=bool(self.get_parameter("rear_enabled").value),
                angle_offset_rad=math.radians(float(self.get_parameter("rear_angle_offset_deg").value)),
                reverse=bool(self.get_parameter("rear_reverse").value),
            ),
        ]
        self.publishers_by_side = {
            side.name: self.create_publisher(LaserScan, side.topic, qos)
            for side in self.sides
            if side.enabled
        }

        if OccupancyGrid is not None and self.map_topic:
            self.create_subscription(OccupancyGrid, self.map_topic, self._map_topic_cb, qos)
            self.get_logger().info(f"synthetic_2d_laser listening for static map on {self.map_topic}")
        elif self.map_topic:
            self.get_logger().warning("nav_msgs/OccupancyGrid unavailable; map topic support disabled")

        if self.dynamic_enabled:
            if People is not None and self.people_topic:
                self.create_subscription(People, self.people_topic, self._people_cb, qos)
            if ArenaPedestrians is not None and self.arena_people_topic:
                self.create_subscription(ArenaPedestrians, self.arena_people_topic, self._people_cb, qos)
            if People is None and ArenaPedestrians is None:
                self.get_logger().warning("pedestrian message types unavailable; dynamic overlay disabled")

        self._refresh_file_map(force=True)
        self.timer = self.create_timer(1.0 / self.update_rate, self._tick)
        for side in self.sides:
            if side.enabled:
                self.get_logger().info(
                    f"[{side.name}] synthetic 2D LaserScan {side.frame_id} -> {side.topic}; "
                    f"map_frame={self._active_map_frame()}, samples={self.samples}, fov={self.fov_deg:.1f}deg, "
                    f"range=[{self.range_min:.2f},{self.range_max:.2f}], rate={self.update_rate:.1f}Hz"
                )

    def _active_map_frame(self) -> str:
        return self._static_map.frame_id or self.configured_map_frame

    def _load_file_map(self) -> StaticMapData:
        preferred_yaml = Path(os.path.expanduser(os.path.expandvars(self.map_yaml_path))) if self.map_yaml_path else None
        if preferred_yaml is not None and preferred_yaml.exists():
            try:
                return _grid_from_map_yaml(
                    preferred_yaml,
                    frame_id=self.configured_map_frame or "map",
                    unknown_is_occupied=self.map_unknown_is_occupied,
                    mtime=preferred_yaml.stat().st_mtime,
                )
            except Exception as exc:
                self.get_logger().warning(f"failed to load map yaml {preferred_yaml}: {exc}")

        legacy = Path(os.path.expanduser(os.path.expandvars(self.legacy_map_path))) if self.legacy_map_path else None
        if legacy is not None and legacy.exists():
            with _open_json_maybe_gz(legacy) as f:
                data = json.load(f) or {}
            return _grid_from_voxel_map(
                data,
                z_min=self.z_min,
                z_max=self.z_max,
                frame_id=self.configured_map_frame or "odom",
                source=f"voxel:{legacy}",
                mtime=legacy.stat().st_mtime,
            )

        if preferred_yaml is not None:
            raise FileNotFoundError(f"map yaml not found: {preferred_yaml}")
        if legacy is not None:
            raise FileNotFoundError(f"legacy voxel map not found: {legacy}")
        raise FileNotFoundError("no static map source configured")

    def _refresh_file_map(self, *, force: bool = False) -> None:
        if self._topic_map_received:
            return
        try:
            new_map = self._load_file_map()
        except Exception as exc:
            if force:
                self.get_logger().warning(f"static map unavailable; publishing inf ranges: {exc}")
            if not self._topic_map_received:
                self._static_map = EMPTY_MAP
            return
        if not force and self._static_map.mtime == new_map.mtime and self._static_map.source == new_map.source:
            return
        self._static_map = new_map
        self.get_logger().info(
            f"loaded synthetic laser static map: source={new_map.source} occupied={len(new_map.occupied_xy)} "
            f"resolution={new_map.resolution:.3f} frame={new_map.frame_id}"
        )

    def _map_topic_cb(self, msg: Any) -> None:
        frame_id = str(getattr(getattr(msg, "header", None), "frame_id", "")).strip()
        if not frame_id:
            frame_id = self.configured_map_frame or "map"
        elif not self.auto_use_map_topic_frame:
            frame_id = self.configured_map_frame or frame_id
        self._static_map = _grid_from_occupancy_msg(
            msg,
            threshold=self.map_occupied_threshold,
            unknown_is_occupied=self.map_unknown_is_occupied,
            frame_id=frame_id,
            source=f"topic:{self.map_topic}",
        )
        if not self._topic_map_received:
            self.get_logger().info(
                f"using OccupancyGrid static map from {self.map_topic}: occupied={len(self._static_map.occupied_xy)} "
                f"resolution={self._static_map.resolution:.3f} frame={self._static_map.frame_id}"
            )
        self._topic_map_received = True

    def _transform_point_to_map(self, x: float, y: float, z: float, source_frame: str) -> Optional[Tuple[float, float, float]]:
        source = str(source_frame).strip() or self._active_map_frame()
        target = self._active_map_frame()
        if source == target:
            return x, y, z
        try:
            tf = self.tf_buffer.lookup_transform(target, source, Time(), timeout=Duration(seconds=self.tf_timeout_sec))
        except TransformException:
            return None
        tx = float(tf.transform.translation.x)
        ty = float(tf.transform.translation.y)
        tz = float(tf.transform.translation.z)
        q = tf.transform.rotation
        rx, ry, rz = _quat_rotate_xyzw(q.x, q.y, q.z, q.w, x, y, z)
        return tx + rx, ty + ry, tz + rz

    def _extract_xyz_from_field(self, field: Any) -> Optional[Tuple[float, float, float]]:
        if field is None:
            return None
        if hasattr(field, "x") and hasattr(field, "y") and hasattr(field, "z"):
            return float(field.x), float(field.y), float(field.z)
        inner = getattr(field, "position", None)
        if inner is not None and hasattr(inner, "x"):
            return float(inner.x), float(inner.y), float(inner.z)
        pose = getattr(field, "pose", None)
        if pose is not None:
            inner2 = getattr(pose, "position", None)
            if inner2 is not None and hasattr(inner2, "x"):
                return float(inner2.x), float(inner2.y), float(inner2.z)
        return None

    def _people_cb(self, msg: Any) -> None:
        if not self.dynamic_enabled:
            return
        people = getattr(msg, "people", None) or getattr(msg, "pedestrians", None)
        if not people:
            return
        now = time.monotonic()
        source_frame = str(getattr(getattr(msg, "header", None), "frame_id", "")).strip() or self.people_frame or self._active_map_frame()
        updated: Dict[str, DynamicCircle] = {}

        for index, person in enumerate(people):
            pose_field = getattr(person, "pose", None) or getattr(person, "position", None)
            xyz = self._extract_xyz_from_field(pose_field)
            if xyz is None:
                xyz = self._extract_xyz_from_field(getattr(person, "position", None))
            if xyz is None:
                continue
            transformed = self._transform_point_to_map(xyz[0], xyz[1], xyz[2], source_frame)
            if transformed is None:
                continue
            key = (
                str(getattr(person, "stage_prefix", "")).strip()
                or str(getattr(person, "name", "")).strip()
                or str(getattr(person, "id", "")).strip()
                or f"person_{index}"
            )
            updated[key] = DynamicCircle(
                x=float(transformed[0]),
                y=float(transformed[1]),
                radius=self.pedestrian_radius,
                stamp_monotonic=now,
            )

        if updated:
            self._dynamic_people.update(updated)

    def _lookup_map_from_scan(self, frame_id: str):
        return self.tf_buffer.lookup_transform(
            self._active_map_frame(),
            frame_id,
            Time(),
            timeout=Duration(seconds=self.tf_timeout_sec),
        )

    def _beam_angle(self, side: SideConfig, index: int) -> float:
        angle_min = -0.5 * math.radians(self.fov_deg)
        inc = math.radians(self.fov_deg) / float(self.samples - 1)
        a = angle_min + float(index) * inc
        if side.reverse:
            a = -a
        return a + side.angle_offset_rad

    def _prune_dynamic_people(self) -> None:
        if self.pedestrian_timeout_sec <= 0.0:
            return
        cutoff = time.monotonic() - self.pedestrian_timeout_sec
        stale = [key for key, circle in self._dynamic_people.items() if circle.stamp_monotonic < cutoff]
        for key in stale:
            self._dynamic_people.pop(key, None)

    def _dynamic_hit(self, ox: float, oy: float, dx: float, dy: float) -> Optional[float]:
        if not self.dynamic_enabled or not self._dynamic_people:
            return None
        best: Optional[float] = None
        for circle in self._dynamic_people.values():
            hit = _ray_circle_distance(ox, oy, dx, dy, circle, self.range_min, self.range_max)
            if hit is None:
                continue
            if best is None or hit < best:
                best = hit
        return best

    def _make_scan(self, side: SideConfig) -> Optional[LaserScan]:
        try:
            tf = self._lookup_map_from_scan(side.frame_id)
        except TransformException as exc:
            self._tf_failures += 1
            now = time.monotonic()
            if self.log_sec > 0 and now - self._last_log >= self.log_sec:
                self._last_log = now
                self.get_logger().warning(
                    f"[{side.name}] TF unavailable {self._active_map_frame()} <- {side.frame_id}: {exc}"
                )
            return None

        tr = tf.transform.translation
        q = tf.transform.rotation
        ox = float(tr.x)
        oy = float(tr.y)
        ranges: List[float] = []

        for i in range(self.samples):
            angle = self._beam_angle(side, i)
            lx = math.cos(angle)
            ly = math.sin(angle)
            mx, my, _ = _quat_rotate_xyzw(q.x, q.y, q.z, q.w, lx, ly, 0.0)
            norm = math.hypot(mx, my)
            if norm < 1e-12:
                ranges.append(float("inf"))
                continue
            dx = mx / norm
            dy = my / norm

            static_hit = _ray_grid_distance(ox, oy, dx, dy, self._static_map, self.range_min, self.range_max)
            dynamic_hit = self._dynamic_hit(ox, oy, dx, dy)
            hit = static_hit
            if dynamic_hit is not None and (hit is None or dynamic_hit < hit):
                hit = dynamic_hit
            ranges.append(float(hit) if hit is not None else float("inf"))

        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = side.frame_id
        msg.angle_min = -0.5 * math.radians(self.fov_deg) + side.angle_offset_rad
        msg.angle_max = 0.5 * math.radians(self.fov_deg) + side.angle_offset_rad
        msg.angle_increment = math.radians(self.fov_deg) / float(self.samples - 1)
        msg.time_increment = (1.0 / self.update_rate) / float(self.samples - 1)
        msg.scan_time = 1.0 / self.update_rate
        msg.range_min = self.range_min
        msg.range_max = self.range_max
        msg.ranges = ranges
        msg.intensities = []
        return msg

    def _tick(self) -> None:
        self._refresh_file_map(force=False)
        self._prune_dynamic_people()
        published = 0
        finite_total = 0
        for side in self.sides:
            if not side.enabled:
                continue
            msg = self._make_scan(side)
            if msg is None:
                continue
            finite_total += sum(1 for r in msg.ranges if math.isfinite(r))
            self.publishers_by_side[side.name].publish(msg)
            published += 1
        self._scan_count += published
        now = time.monotonic()
        if self.log_sec > 0 and now - self._last_log >= self.log_sec:
            self._last_log = now
            self.get_logger().info(
                f"synthetic_2d_laser published={self._scan_count} finite_points_last_tick={finite_total} "
                f"static_source={self._static_map.source} static_cells={len(self._static_map.occupied_xy)} "
                f"dynamic_people={len(self._dynamic_people)} tf_failures={self._tf_failures}"
            )


def main(args: Optional[Iterable[str]] = None) -> None:
    rclpy.init(args=args)
    node = Synthetic2DLaser()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
