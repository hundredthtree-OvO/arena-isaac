"""Observe-only pedestrian visual envelope monitoring.

This module keeps Isaac-specific imports inside runtime helpers so the pure
geometry and aggregation helpers stay unit-testable under plain pytest.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


_SKELETON_RESOLVE_FAILURES: dict[str, str] = {}


@dataclass(frozen=True)
class _SkeletonTopologyCacheEntry:
    person_identity: int
    joint_order: tuple[str, ...]
    parent_indices: tuple[int, ...]


# Cache only immutable Python values. Holding UsdSkel.Cache or query objects
# here would keep native stage resources alive across pedestrian lifecycles.
_SKELETON_TOPOLOGY_CACHE: dict[str, _SkeletonTopologyCacheEntry] = {}


def _skeleton_failure(root_path: str, reason: str):
    _SKELETON_RESOLVE_FAILURES[str(root_path)] = str(reason)
    return None


def _cached_skeleton_topology(
    person,
    root_path: str,
) -> _SkeletonTopologyCacheEntry | None:
    entry = _SKELETON_TOPOLOGY_CACHE.get(str(root_path))
    if entry is None or entry.person_identity != id(person):
        return None
    return entry


def _prune_skeleton_topology_cache(people_by_alias: dict[str, object] | None) -> None:
    active = {
        str(root_path): id(person)
        for _agent_id, person, root_path in iter_active_people(people_by_alias)
    }
    for root_path, entry in tuple(_SKELETON_TOPOLOGY_CACHE.items()):
        if active.get(root_path) != entry.person_identity:
            _SKELETON_TOPOLOGY_CACHE.pop(root_path, None)
            _SKELETON_RESOLVE_FAILURES.pop(root_path, None)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _split_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def stable_person_name(person) -> str:
    requested = str(getattr(person, "_requested_stage_name", "") or "").strip()
    if requested:
        return requested
    stage_prefix = str(getattr(person, "_stage_prefix", "") or "").strip().strip("/")
    if stage_prefix:
        return stage_prefix.split("/")[-1]
    return "pedestrian"


@dataclass(frozen=True)
class PedestrianVisualEnvelopeConfig:
    enabled: bool = True
    publish_hz: float = 2.0
    sample_radius_m: float = 0.08
    segment_spacing_m: float = 0.05
    topic_name: str = "/isaac/pedestrian_visual_envelopes"
    log_path: str = ""
    exclude_prefixes: tuple[str, ...] = (
        "/World/Characters",
        "/World/CharactersCollision",
        "/World/xms_mecanum",
        "/World/robots/xms_mecanum",
        "/World/groundPlane",
        "/World/shenxinfu_841837/Meshes/Base/Floor_0000/Geom",
        "/World/shenxinfu_841837/CollisionFixes/dirty_stall_clean_floor",
    )


@dataclass(frozen=True)
class SkeletonVisualEnvelopeData:
    joint_world_positions: tuple[tuple[float, float, float], ...]
    parent_indices: tuple[int, ...]
    source: str = "anim_graph"


def pedestrian_visual_envelope_config_from_env() -> PedestrianVisualEnvelopeConfig:
    enabled = _env_bool("ARENA_ISAAC_ENABLE_PEDESTRIAN_VISUAL_ENVELOPE", True)
    publish_hz = max(
        1e-3,
        _env_float("ARENA_ISAAC_PEDESTRIAN_VISUAL_ENVELOPE_HZ", 2.0),
    )
    sample_radius_m = max(
        1e-6,
        _env_float("ARENA_ISAAC_PEDESTRIAN_VISUAL_ENVELOPE_RADIUS_M", 0.08),
    )
    segment_spacing_m = max(
        1e-6,
        _env_float("ARENA_ISAAC_PEDESTRIAN_VISUAL_ENVELOPE_SEGMENT_SPACING_M", 0.05),
    )
    exclude_prefixes = _split_csv(
        os.environ.get("ARENA_ISAAC_PEDESTRIAN_VISUAL_ENVELOPE_EXCLUDE_PREFIXES")
    )
    if not exclude_prefixes:
        exclude_prefixes = PedestrianVisualEnvelopeConfig.exclude_prefixes
    topic_name = os.environ.get(
        "ARENA_ISAAC_PEDESTRIAN_VISUAL_ENVELOPE_TOPIC",
        PedestrianVisualEnvelopeConfig.topic_name,
    )
    log_path = os.environ.get(
        "ARENA_ISAAC_PEDESTRIAN_VISUAL_ENVELOPE_LOG_PATH",
        "",
    )
    return PedestrianVisualEnvelopeConfig(
        enabled=enabled,
        publish_hz=publish_hz,
        sample_radius_m=sample_radius_m,
        segment_spacing_m=segment_spacing_m,
        topic_name=topic_name,
        log_path=str(log_path),
        exclude_prefixes=tuple(exclude_prefixes),
    )


def _to_xyz(point: Sequence[float]) -> tuple[float, float, float]:
    return (float(point[0]), float(point[1]), float(point[2]))


def _is_finite_xyz(point: Sequence[float]) -> bool:
    try:
        x, y, z = _to_xyz(point)
    except Exception:
        return False
    return all(math.isfinite(value) for value in (x, y, z))


def _dedupe_points(points: Iterable[Sequence[float]]) -> list[tuple[float, float, float]]:
    unique: list[tuple[float, float, float]] = []
    seen: set[tuple[float, float, float]] = set()
    for point in points:
        try:
            normalized = _to_xyz(point)
        except Exception:
            continue
        key = tuple(round(value, 6) for value in normalized)
        if key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
    return unique


def _sample_segment(
    start: Sequence[float],
    end: Sequence[float],
    spacing_m: float,
) -> list[tuple[float, float, float]]:
    start_xyz = _to_xyz(start)
    end_xyz = _to_xyz(end)
    if not _is_finite_xyz(start_xyz) or not _is_finite_xyz(end_xyz):
        return []
    spacing_m = max(float(spacing_m), 1e-6)
    delta = (
        end_xyz[0] - start_xyz[0],
        end_xyz[1] - start_xyz[1],
        end_xyz[2] - start_xyz[2],
    )
    length = math.sqrt(sum(component * component for component in delta))
    if length <= 1e-9:
        return [start_xyz]
    steps = max(1, int(math.ceil(length / spacing_m)))
    samples: list[tuple[float, float, float]] = []
    for index in range(steps + 1):
        t = float(index) / float(steps)
        samples.append(
            (
                start_xyz[0] + delta[0] * t,
                start_xyz[1] + delta[1] * t,
                start_xyz[2] + delta[2] * t,
            )
        )
    return samples


def sample_visual_envelope_points(
    joint_world_positions: Sequence[Sequence[float]],
    parent_indices: Sequence[int],
    segment_spacing_m: float,
) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    count = min(len(joint_world_positions), len(parent_indices))
    for index in range(count):
        parent_index = int(parent_indices[index])
        if parent_index < 0 or parent_index >= count:
            continue
        points.extend(
            _sample_segment(
                joint_world_positions[parent_index],
                joint_world_positions[index],
                segment_spacing_m,
            )
        )
    return _dedupe_points(points)


def xy_aabb_from_points(
    points: Sequence[Sequence[float]],
) -> dict[str, list[float]] | None:
    xy_points: list[tuple[float, float]] = []
    for point in points:
        try:
            x = float(point[0])
            y = float(point[1])
        except Exception:
            continue
        if not all(math.isfinite(value) for value in (x, y)):
            continue
        xy_points.append((x, y))
    if not xy_points:
        return None
    xs = [point[0] for point in xy_points]
    ys = [point[1] for point in xy_points]
    return {
        "min": [float(min(xs)), float(min(ys))],
        "max": [float(max(xs)), float(max(ys))],
    }


def convex_hull_xy(points: Sequence[Sequence[float]]) -> list[list[float]]:
    xy_points: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for point in points:
        try:
            x = float(point[0])
            y = float(point[1])
        except Exception:
            continue
        if not all(math.isfinite(value) for value in (x, y)):
            continue
        key = (round(x, 6), round(y, 6))
        if key in seen:
            continue
        seen.add(key)
        xy_points.append((x, y))
    if len(xy_points) <= 1:
        return [[float(point[0]), float(point[1])] for point in xy_points]

    def _cross(origin, left, right) -> float:
        return (
            (left[0] - origin[0]) * (right[1] - origin[1])
            - (left[1] - origin[1]) * (right[0] - origin[0])
        )

    sorted_points = sorted(xy_points)
    lower: list[tuple[float, float]] = []
    for point in sorted_points:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(sorted_points):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)

    hull = lower[:-1] + upper[:-1]
    if not hull:
        hull = sorted_points[:1]
    return [[float(point[0]), float(point[1])] for point in hull]


def _unique_strings(values: Iterable[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def _is_excluded_path(path: str, exclude_prefixes: Sequence[str]) -> bool:
    normalized = str(path).strip()
    if not normalized:
        return True
    return any(normalized == prefix or normalized.startswith(prefix + "/") or normalized.startswith(prefix) for prefix in exclude_prefixes)


def _max_consecutive_true(values: Sequence[bool]) -> int:
    best = 0
    current = 0
    for value in values:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def iter_active_people(people_by_alias: dict[str, object] | None):
    seen: set[int] = set()
    for alias, person in sorted((people_by_alias or {}).items(), key=lambda item: str(item[0])):
        if person is None:
            continue
        marker = id(person)
        if marker in seen:
            continue
        seen.add(marker)
        if not bool(getattr(person, "_active", False)):
            continue
        root_path = str(getattr(person, "character_skel_root_stage_path", "") or "").strip()
        if not root_path:
            continue
        yield stable_person_name(person) or str(alias) or "pedestrian", person, root_path


def _empty_visual_envelope_record(
    agent_id: str,
    root_path: str,
    joint_count: int = 0,
) -> dict[str, Any]:
    return {
        "agent_id": str(agent_id),
        "character_skel_root_stage_path": str(root_path),
        "available": False,
        "unavailable_reason": "skeleton_unavailable",
        "source": "",
        "joint_count": int(joint_count),
        "sample_count": 0,
        "xy_aabb": None,
        "hull": [],
        "overlap_sample_count": 0,
        "overlap_paths": [],
        "overlap_ratio": 0.0,
        "max_consecutive_overlap_samples": 0,
    }


def summarize_visual_envelope_record(
    agent_id: str,
    root_path: str,
    skeleton_data: SkeletonVisualEnvelopeData | None,
    overlap_query: Callable[[tuple[float, float, float], float], Iterable[str]],
    config: PedestrianVisualEnvelopeConfig,
) -> dict[str, Any]:
    if skeleton_data is None:
        return _empty_visual_envelope_record(agent_id, root_path)

    joint_positions = tuple(tuple(point) for point in skeleton_data.joint_world_positions)
    parent_indices = tuple(int(index) for index in skeleton_data.parent_indices)
    joint_count = len(joint_positions)
    if joint_count <= 0 or len(parent_indices) != joint_count:
        return _empty_visual_envelope_record(agent_id, root_path, joint_count=joint_count)
    if not all(_is_finite_xyz(point) for point in joint_positions):
        return _empty_visual_envelope_record(agent_id, root_path, joint_count=joint_count)

    sample_points = sample_visual_envelope_points(
        joint_positions,
        parent_indices,
        config.segment_spacing_m,
    )
    if not sample_points:
        return _empty_visual_envelope_record(agent_id, root_path, joint_count=joint_count)

    overlap_flags: list[bool] = []
    overlap_paths: list[str] = []
    overlap_sample_count = 0
    for point in sample_points:
        try:
            raw_paths = overlap_query(point, config.sample_radius_m)
        except Exception:
            raw_paths = ()
        filtered_paths = _unique_strings(
            path
            for path in raw_paths
            if not _is_excluded_path(path, config.exclude_prefixes)
        )
        has_overlap = bool(filtered_paths)
        overlap_flags.append(has_overlap)
        if has_overlap:
            overlap_sample_count += 1
            overlap_paths.extend(filtered_paths)

    unique_overlap_paths = _unique_strings(overlap_paths)
    xy_aabb = xy_aabb_from_points(sample_points)
    hull = convex_hull_xy(sample_points)
    sample_count = len(sample_points)
    overlap_ratio = float(overlap_sample_count) / float(sample_count) if sample_count else 0.0
    return {
        "agent_id": str(agent_id),
        "character_skel_root_stage_path": str(root_path),
        "available": True,
        "unavailable_reason": "",
        "source": str(skeleton_data.source),
        "joint_count": joint_count,
        "sample_count": sample_count,
        "xy_aabb": xy_aabb,
        "hull": hull,
        "overlap_sample_count": overlap_sample_count,
        "overlap_paths": unique_overlap_paths,
        "overlap_ratio": overlap_ratio,
        "max_consecutive_overlap_samples": _max_consecutive_true(overlap_flags),
    }


def collect_visual_envelope_payload(
    people_by_alias: dict[str, object] | None,
    *,
    skeleton_resolver: Callable[[object, str], SkeletonVisualEnvelopeData | None],
    overlap_query: Callable[[tuple[float, float, float], float], Iterable[str]],
    config: PedestrianVisualEnvelopeConfig,
    timestamp_iso: str | None = None,
) -> dict[str, Any]:
    agents = []
    for agent_id, person, root_path in iter_active_people(people_by_alias):
        skeleton_data = None
        try:
            skeleton_data = skeleton_resolver(person, root_path)
        except Exception:
            skeleton_data = None
        agents.append(
            summarize_visual_envelope_record(
                agent_id,
                root_path,
                skeleton_data,
                overlap_query,
                config,
            )
        )
    return {
        "timestamp": timestamp_iso or _dt.datetime.now().isoformat(timespec="milliseconds"),
        "topic": config.topic_name,
        "agents": agents,
    }


def _find_skeleton_prim(root_prim, UsdSkel, Usd=None):
    if root_prim is None or not root_prim.IsValid():
        return None
    if Usd is not None:
        try:
            for prim in Usd.PrimRange(
                root_prim,
                Usd.TraverseInstanceProxies(),
            ):
                if prim.IsA(UsdSkel.Skeleton):
                    return prim
        except Exception:
            pass
    try:
        if root_prim.IsA(UsdSkel.Skeleton):
            return root_prim
    except Exception:
        pass
    for child in root_prim.GetAllChildren():
        found = _find_skeleton_prim(child, UsdSkel)
        if found is not None:
            return found
    return None


def resolve_skeleton_data(person, root_path: str) -> SkeletonVisualEnvelopeData | None:
    """Resolve animated joint world positions for a single person."""

    try:
        import carb
        import omni.usd
        from pxr import Usd, UsdSkel
    except Exception:
        return _skeleton_failure(root_path, "isaac_import_failed")

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return _skeleton_failure(root_path, "stage_unavailable")
    root_prim = stage.GetPrimAtPath(root_path)
    if root_prim is None or not root_prim.IsValid():
        _SKELETON_TOPOLOGY_CACHE.pop(str(root_path), None)
        return _skeleton_failure(root_path, "skel_root_prim_invalid")

    topology_entry = _cached_skeleton_topology(person, root_path)
    if topology_entry is None:
        skel_root = UsdSkel.Root(root_prim)
        if not skel_root:
            return _skeleton_failure(
                root_path,
                f"not_skel_root:{root_prim.GetTypeName()}",
            )

        skeleton_prim = _find_skeleton_prim(root_prim, UsdSkel, Usd)
        if skeleton_prim is None:
            return _skeleton_failure(root_path, "skeleton_prim_not_found")

        # Building and populating this native cache for every 2 Hz sample leaks
        # Kit/USD allocations on Isaac Sim 4.5. Build it once per character
        # instance, copy out topology, and release all native handles here.
        cache = UsdSkel.Cache()
        try:
            cache.Populate(skel_root)
        except Exception:
            try:
                cache.Populate(skel_root, Usd.TraverseInstanceProxies())
            except Exception as exc:
                return _skeleton_failure(
                    root_path,
                    f"cache_populate_failed:{type(exc).__name__}",
                )

        skeleton = UsdSkel.Skeleton(skeleton_prim)
        query = cache.GetSkelQuery(skeleton)
        try:
            if not query:
                return _skeleton_failure(root_path, "skeleton_query_invalid")
        except Exception:
            return _skeleton_failure(root_path, "skeleton_query_validation_failed")

        try:
            topology = query.GetTopology()
            parent_indices = tuple(
                int(index) for index in topology.GetParentIndices()
            )
            joint_order = tuple(str(token) for token in query.GetJointOrder())
        except Exception as exc:
            return _skeleton_failure(
                root_path,
                f"topology_failed:{type(exc).__name__}",
            )
        if len(joint_order) != len(parent_indices):
            return _skeleton_failure(root_path, "joint_order_size_mismatch")
        topology_entry = _SkeletonTopologyCacheEntry(
            person_identity=id(person),
            joint_order=joint_order,
            parent_indices=parent_indices,
        )
        _SKELETON_TOPOLOGY_CACHE[str(root_path)] = topology_entry

    joint_order = topology_entry.joint_order
    parent_indices = topology_entry.parent_indices

    character_graph = getattr(person, "character_graph", None)
    if character_graph is None:
        return _skeleton_failure(root_path, "anim_graph_character_unavailable")
    joint_world_positions: list[tuple[float, float, float]] = []
    for joint_token in joint_order:
        # AnimGraph addresses joints by their leaf token (for example
        # "R_Ankle"), while UsdSkel joint order uses hierarchical paths.
        joint_name = joint_token.rsplit("/", 1)[-1]
        position = carb.Float3(0.0, 0.0, 0.0)
        rotation = carb.Float4(0.0, 0.0, 0.0, 0.0)
        try:
            character_graph.get_joint_transform(joint_name, position, rotation)
            joint_world_positions.append(
                (float(position[0]), float(position[1]), float(position[2]))
            )
        except Exception as exc:
            return _skeleton_failure(
                root_path,
                f"anim_graph_joint_failed:{joint_name}:{type(exc).__name__}",
            )

    if len(joint_world_positions) != len(parent_indices):
        return _skeleton_failure(root_path, "joint_topology_size_mismatch")
    _SKELETON_RESOLVE_FAILURES.pop(str(root_path), None)
    return SkeletonVisualEnvelopeData(
        joint_world_positions=tuple(joint_world_positions),
        parent_indices=parent_indices,
        source="anim_graph",
    )


@dataclass
class PedestrianVisualEnvelopeMonitor:
    controller: object
    topic_name: str = ""
    config: PedestrianVisualEnvelopeConfig = field(
        default_factory=pedestrian_visual_envelope_config_from_env
    )

    def __post_init__(self):
        self._publisher = None
        self._timer = None
        self._String = None
        self._warned_keys: set[str] = set()
        self._log_path = (
            Path(self.config.log_path).expanduser()
            if self.config.log_path
            else None
        )
        if not self.topic_name:
            self.topic_name = self.config.topic_name
        if not self.config.enabled:
            self._log_info(
                f"Skipping {self.topic_name}: ARENA_ISAAC_ENABLE_PEDESTRIAN_VISUAL_ENVELOPE is false"
            )
            return
        try:
            from std_msgs.msg import String
        except Exception as exc:
            self._log_warn(f"Skipping {self.topic_name}: std_msgs.String unavailable: {exc}")
            return
        self._String = String
        try:
            self._publisher = self.controller.create_publisher(String, self.topic_name, 10)
            period = 1.0 / max(float(self.config.publish_hz), 1e-3)
            self._timer = self.controller.create_timer(period, self._publish)
            self._log_info(
                f"Publishing pedestrian visual envelopes on {self.topic_name} at {float(self.config.publish_hz):.2f} Hz."
            )
        except Exception as exc:
            self._publisher = None
            self._timer = None
            self._log_warn(f"Failed to start {self.topic_name}: {exc}")

    def _people_by_alias(self) -> dict[str, object] | None:
        try:
            from pedestrian.simulator.logic.people_manager import PeopleManager
        except Exception as exc:
            self._log_warn_once(
                "people_manager_unavailable",
                f"Skipping {self.topic_name}: PeopleManager unavailable: {exc}",
            )
            return None
        return getattr(PeopleManager, "_people", None) or {}

    def _query_overlap_paths(self, point: tuple[float, float, float], radius_m: float) -> list[str]:
        try:
            import carb
            import omni.physx
        except Exception:
            return []

        hits: list[str] = []

        def _report(hit) -> bool:
            for attribute in ("rigid_body", "collision"):
                try:
                    value = getattr(hit, attribute)
                except Exception:
                    continue
                if value:
                    hits.append(str(value))
            return True

        try:
            omni.physx.get_physx_scene_query_interface().overlap_sphere(
                float(radius_m),
                carb.Float3(float(point[0]), float(point[1]), float(point[2])),
                _report,
                False,
            )
        except Exception as exc:
            self._log_warn_once(
                "physx_overlap_unavailable",
                f"[pedestrian_visual_envelope] overlap_sphere failed: {exc}",
            )
            return []
        return hits

    def _publish(self):
        if self._publisher is None or self._String is None:
            return
        people_by_alias = self._people_by_alias()
        if people_by_alias is None:
            return
        _prune_skeleton_topology_cache(people_by_alias)
        if self._log_path is None:
            try:
                if self._publisher.get_subscription_count() <= 0:
                    return
            except Exception:
                pass
        payload = collect_visual_envelope_payload(
            people_by_alias,
            skeleton_resolver=resolve_skeleton_data,
            overlap_query=self._query_overlap_paths,
            config=self.config,
        )
        for item in payload["agents"]:
            if not item.get("available", False):
                root_path = str(item.get("character_skel_root_stage_path", ""))
                reason = _SKELETON_RESOLVE_FAILURES.get(
                    root_path,
                    str(item.get("unavailable_reason", "skeleton_unavailable")),
                )
                item["unavailable_reason"] = reason
                self._log_warn_once(
                    f"skeleton_unavailable:{root_path}:{reason}",
                    f"Pedestrian visual envelope unavailable for "
                    f"{root_path}: {reason}",
                )
        serialized = json.dumps(payload, sort_keys=True)
        try:
            msg = self._String()
            msg.data = serialized
            self._publisher.publish(msg)
        except Exception as exc:
            self._log_warn(f"Failed to publish {self.topic_name}: {exc}")
        if self._log_path is not None:
            try:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                with self._log_path.open("a", encoding="utf-8") as handle:
                    handle.write(serialized + "\n")
            except OSError as exc:
                self._log_warn_once(
                    "visual_envelope_log_failed",
                    f"Failed to write pedestrian visual envelope log "
                    f"{self._log_path}: {exc}",
                )

    def _log_info(self, message: str):
        try:
            self.controller.get_logger().info(message)
        except Exception:
            print(message, flush=True)

    def _log_warn(self, message: str):
        try:
            self.controller.get_logger().warning(message)
        except Exception:
            print(message, flush=True)

    def _log_warn_once(self, key: str, message: str):
        if key in self._warned_keys:
            return
        self._warned_keys.add(key)
        self._log_warn(message)


def start_pedestrian_visual_envelope_monitor(controller):
    return PedestrianVisualEnvelopeMonitor(controller=controller)
