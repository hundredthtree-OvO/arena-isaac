"""Scene collision diagnostics for target prims and chassis-sized PhysX probes."""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


_CARDINAL_DIRECTIONS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("front", (1.0, 0.0, 0.0)),
    ("rear", (-1.0, 0.0, 0.0)),
    ("left", (0.0, 1.0, 0.0)),
    ("right", (0.0, -1.0, 0.0)),
)


@dataclass(frozen=True)
class SceneCollisionProbeConfig:
    """Environment-driven inputs for the scene collision probe."""

    output_path: str = "/tmp/arena_scene_collision_probe.json"
    target_hints: tuple[str, ...] = (
        "Edestalurinal_0000",
        "Edestalurinal_0001",
        "Edestalurinal_0002",
        "Edestalurinal_0003",
        "Edestalurinal_0004",
        "Toilet_0000",
        "Toilet_0001",
        "Door_0000",
    )
    discover_keywords: tuple[str, ...] = ("Partition", "Door")
    chassis_half_extents: tuple[float, float, float] = (0.20, 0.15, 0.20)
    probe_margin: float = 0.05
    probe_heights: tuple[float, ...] = (0.15, 0.55, 0.95)
    sweep_steps: int = 4


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _split_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_float_triplet(value: str | None, default: Iterable[float]) -> tuple[float, float, float]:
    if value:
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) == 3:
            try:
                return (float(parts[0]), float(parts[1]), float(parts[2]))
            except Exception:
                pass
    defaults = tuple(float(item) for item in default)
    return defaults[0], defaults[1], defaults[2]


def scene_collision_probe_config_from_env() -> SceneCollisionProbeConfig:
    """Build a probe config from the current process environment."""

    target_hints = _split_csv(
        os.environ.get("ARENA_ISAAC_SCENE_COLLISION_PROBE_TARGET_HINTS")
    )
    if not target_hints:
        target_hints = SceneCollisionProbeConfig.target_hints

    discover_keywords = _split_csv(
        os.environ.get("ARENA_ISAAC_SCENE_COLLISION_PROBE_DISCOVER_KEYWORDS")
    )
    if not discover_keywords:
        discover_keywords = SceneCollisionProbeConfig.discover_keywords

    height_samples = _split_csv(
        os.environ.get("ARENA_ISAAC_SCENE_COLLISION_PROBE_HEIGHTS")
    )
    if height_samples:
        probe_heights = tuple(
            float(value)
            for value in height_samples
            if value and _is_float(value)
        )
        if not probe_heights:
            probe_heights = SceneCollisionProbeConfig.probe_heights
    else:
        probe_heights = SceneCollisionProbeConfig.probe_heights

    return SceneCollisionProbeConfig(
        output_path=os.environ.get(
            "ARENA_ISAAC_SCENE_COLLISION_PROBE_OUTPUT",
            SceneCollisionProbeConfig.output_path,
        ),
        target_hints=tuple(target_hints),
        discover_keywords=tuple(discover_keywords),
        chassis_half_extents=_parse_float_triplet(
            os.environ.get("ARENA_ISAAC_SCENE_COLLISION_PROBE_CHASSIS_HALF_EXTENTS"),
            SceneCollisionProbeConfig.chassis_half_extents,
        ),
        probe_margin=_env_float(
            "ARENA_ISAAC_SCENE_COLLISION_PROBE_MARGIN",
            SceneCollisionProbeConfig.probe_margin,
        ),
        probe_heights=tuple(probe_heights),
        sweep_steps=max(1, int(_env_float("ARENA_ISAAC_SCENE_COLLISION_PROBE_SWEEP_STEPS", 4))),
    )


def _normalized_text(value: str) -> str:
    return str(value).strip().lower()


def _is_float(value: str) -> bool:
    try:
        float(value)
    except Exception:
        return False
    return True


def match_target_hints(path: str, name: str, config: SceneCollisionProbeConfig) -> tuple[str, ...]:
    """Return the configured hints that match a prim path/name pair."""

    path_norm = _normalized_text(path)
    name_norm = _normalized_text(name)
    matches = []
    for hint in config.target_hints:
        hint_norm = _normalized_text(hint)
        if not hint_norm:
            continue
        is_path_hint = "/" in hint_norm
        if (is_path_hint and path_norm == hint_norm) or (
            not is_path_hint and (name_norm == hint_norm or name_norm.startswith(hint_norm))
        ):
            matches.append(hint)
    for keyword in config.discover_keywords:
        keyword_norm = _normalized_text(keyword)
        if keyword_norm and keyword_norm in name_norm:
            matches.append(keyword)
    return tuple(dict.fromkeys(matches))


def _vector3(values: Iterable[float]) -> list[float]:
    return [float(values[index]) for index in range(3)]


def _finite_vector(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _scene_aabb_entry(prim, bbox_cache, xform_cache, UsdPhysics) -> dict[str, Any] | None:
    try:
        aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        minimum = _vector3(aligned.GetMin())
        maximum = _vector3(aligned.GetMax())
        if not _finite_vector(minimum + maximum):
            return None
        matrix = xform_cache.GetLocalToWorldTransform(prim)
        translation = _vector3(matrix.ExtractTranslation())
    except Exception:
        return None
    collision_attr = prim.GetAttribute("physics:collisionEnabled")
    enabled = True
    if collision_attr and collision_attr.IsValid() and collision_attr.Get() is not None:
        enabled = bool(collision_attr.Get())
    return {
        "path": str(prim.GetPath()),
        "name": str(prim.GetName()),
        "type": str(prim.GetTypeName()),
        "collision_api": bool(prim.HasAPI(UsdPhysics.CollisionAPI)),
        "collision_enabled": enabled if prim.HasAPI(UsdPhysics.CollisionAPI) else None,
        "world_translation": translation,
        "world_aabb_min": minimum,
        "world_aabb_max": maximum,
    }


def _unique_serializable_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        key = json.dumps(item, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _hit_paths(hit: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(hit[key])
        for key in ("collision", "rigid_body")
        if hit.get(key) not in (None, "", "None")
    )


def target_hits(hits: Iterable[dict[str, Any]], target_path: str) -> list[dict[str, Any]]:
    """Return PhysX hits whose collision/rigid-body path belongs to the target subtree."""

    prefix = str(target_path).rstrip("/") + "/"
    return [
        hit
        for hit in hits
        if any(path == target_path or path.startswith(prefix) for path in _hit_paths(hit))
    ]


def build_height_centers(
    minimum_z: float,
    maximum_z: float,
    chassis_half_z: float,
    probe_heights: Iterable[float],
) -> list[float]:
    """Choose sample heights that stay inside the target AABB when possible."""

    minimum_z = float(minimum_z)
    maximum_z = float(maximum_z)
    chassis_half_z = abs(float(chassis_half_z))
    if not math.isfinite(minimum_z) or not math.isfinite(maximum_z):
        return [0.0]
    if maximum_z <= minimum_z:
        return [0.5 * (minimum_z + maximum_z)]

    lower = minimum_z + chassis_half_z
    upper = maximum_z - chassis_half_z
    if upper < lower:
        return [0.5 * (minimum_z + maximum_z)]

    samples = [lower, 0.5 * (lower + upper), upper]
    for height in probe_heights:
        try:
            sample = float(height)
        except Exception:
            continue
        if minimum_z <= sample <= maximum_z:
            samples.append(sample)

    return [float(value) for value in dict.fromkeys(sorted(samples))]


def build_cardinal_sweep_offsets(
    target_half_extents_xy: Iterable[float],
    chassis_half_extents_xy: Iterable[float],
    *,
    probe_margin: float,
    sweep_steps: int,
) -> dict[str, list[float]]:
    """Return sweep distances for the four cardinal directions."""

    target_half_x, target_half_y = (abs(float(v)) for v in target_half_extents_xy)
    chassis_half_x, chassis_half_y = (abs(float(v)) for v in chassis_half_extents_xy)
    reach = max(target_half_x, target_half_y) + max(chassis_half_x, chassis_half_y) + abs(float(probe_margin))
    if sweep_steps <= 1:
        return {direction: [reach] for direction, _ in _CARDINAL_DIRECTIONS}
    step = reach / float(sweep_steps)
    distances = [round(step * index, 6) for index in range(sweep_steps + 1)]
    distances[0] = 0.0
    return {direction: distances[:] for direction, _ in _CARDINAL_DIRECTIONS}


def _query_overlap_box(physx_scene_query, center: tuple[float, float, float], half_extents: tuple[float, float, float]) -> tuple[list[dict[str, Any]], Any]:
    import carb

    hits: list[dict[str, Any]] = []

    def _on_hit(hit):
        item: dict[str, Any] = {}
        for attribute in ("rigid_body", "collision", "material"):
            try:
                value = getattr(hit, attribute)
            except Exception:
                continue
            item[attribute] = str(value)
        if not item:
            item["repr"] = repr(hit)
        hits.append(item)
        return True

    query_result = physx_scene_query.overlap_box(
        carb.Float3(*[float(value) for value in half_extents]),
        carb.Float3(*[float(value) for value in center]),
        carb.Float4(0.0, 0.0, 0.0, 1.0),
        _on_hit,
        False,
    )
    return _unique_serializable_items(hits), query_result


def _probe_sample_entry(
    physx_scene_query,
    center: tuple[float, float, float],
    half_extents: tuple[float, float, float],
    target_path: str,
) -> dict[str, Any]:
    hits, query_result = _query_overlap_box(physx_scene_query, center, half_extents)
    matched_target_hits = target_hits(hits, target_path)
    return {
        "center": [float(center[0]), float(center[1]), float(center[2])],
        "half_extents": [float(half_extents[0]), float(half_extents[1]), float(half_extents[2])],
        "hits": hits,
        "hit_count": len(hits),
        "target_hits": matched_target_hits,
        "target_hit_count": len(matched_target_hits),
        "query_result": (
            query_result
            if isinstance(query_result, (bool, int, float, str, type(None)))
            else str(query_result)
        ),
    }


def collect_scene_collision_report(*, config: SceneCollisionProbeConfig | None = None) -> dict[str, Any]:
    """Collect USD collision metadata and PhysX overlap samples for target scene prims."""

    import omni.physx
    import omni.timeline
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics

    config = config or scene_collision_probe_config_from_env()
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Isaac stage is not available")

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    physx_scene_query = omni.physx.get_physx_scene_query_interface()

    target_records: dict[str, dict[str, Any]] = {}
    all_collision_prims: list[dict[str, Any]] = []

    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        path = str(prim.GetPath())
        matches = match_target_hints(path, prim.GetName(), config)
        if not matches:
            continue
        entry = target_records.setdefault(
            path,
            {
                "path": path,
                "name": str(prim.GetName()),
                "type": str(prim.GetTypeName()),
                "matched_hints": [],
                "collision_api": bool(prim.HasAPI(UsdPhysics.CollisionAPI)),
                "collision_enabled": None,
                "world_translation": None,
                "world_aabb_min": None,
                "world_aabb_max": None,
                "subtree_collision_prims": [],
                "height_samples": [],
                "cardinal_sweeps": {},
            },
        )
        entry["matched_hints"] = list(dict.fromkeys(entry["matched_hints"] + list(matches)))
        scene_entry = _scene_aabb_entry(prim, bbox_cache, xform_cache, UsdPhysics)
        if scene_entry is not None:
            entry.update(
                {
                    "collision_api": scene_entry["collision_api"],
                    "collision_enabled": scene_entry["collision_enabled"],
                    "world_translation": scene_entry["world_translation"],
                    "world_aabb_min": scene_entry["world_aabb_min"],
                    "world_aabb_max": scene_entry["world_aabb_max"],
                }
            )

        for subtree_prim in Usd.PrimRange(prim):
            if not subtree_prim.IsValid() or not subtree_prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            subtree_entry = _scene_aabb_entry(subtree_prim, bbox_cache, xform_cache, UsdPhysics)
            if subtree_entry is None:
                continue
            all_collision_prims.append(subtree_entry)
            entry["subtree_collision_prims"].append(subtree_entry)

    for record in target_records.values():
        aabb_min = record["world_aabb_min"]
        aabb_max = record["world_aabb_max"]
        if aabb_min is None or aabb_max is None:
            continue
        center_x = 0.5 * (float(aabb_min[0]) + float(aabb_max[0]))
        center_y = 0.5 * (float(aabb_min[1]) + float(aabb_max[1]))
        heights = build_height_centers(
            float(aabb_min[2]),
            float(aabb_max[2]),
            float(config.chassis_half_extents[2]),
            config.probe_heights,
        )
        sweep_offsets = build_cardinal_sweep_offsets(
            (
                0.5 * (float(aabb_max[0]) - float(aabb_min[0])),
                0.5 * (float(aabb_max[1]) - float(aabb_min[1])),
            ),
            config.chassis_half_extents[:2],
            probe_margin=config.probe_margin,
            sweep_steps=config.sweep_steps,
        )
        base_half_extents = tuple(float(value) for value in config.chassis_half_extents)
        height_samples = []
        for height in heights:
            sample_center = (center_x, center_y, height)
            sample_entry = _probe_sample_entry(
                physx_scene_query,
                sample_center,
                base_half_extents,
                record["path"],
            )
            sample_entry["height"] = float(height)
            sample_entry["directional_sweeps"] = {}
            for direction_name, direction_vector in _CARDINAL_DIRECTIONS:
                distances = sweep_offsets[direction_name]
                direction_hits = []
                for distance in distances:
                    center = (
                        center_x + float(direction_vector[0]) * float(distance),
                        center_y + float(direction_vector[1]) * float(distance),
                        height,
                    )
                    direction_hits.append(
                        _probe_sample_entry(
                            physx_scene_query,
                            center,
                            base_half_extents,
                            record["path"],
                        )
                    )
                sample_entry["directional_sweeps"][direction_name] = {
                    "direction": [float(direction_vector[0]), float(direction_vector[1]), float(direction_vector[2])],
                    "distances": [float(distance) for distance in distances],
                    "samples": direction_hits,
                    "unique_hit_count": len(
                        _unique_serializable_items(
                            [hit for sample in direction_hits for hit in sample["hits"]]
                        )
                    ),
                }
            height_samples.append(sample_entry)
        record["height_samples"] = height_samples
        record["usd_enabled_collision_count"] = sum(
            1
            for item in record["subtree_collision_prims"]
            if item["collision_enabled"]
        )
        record["physx_target_hit_count"] = len(
            _unique_serializable_items(
                [
                    hit
                    for sample in height_samples
                    for hit in sample["target_hits"]
                ]
                + [
                    hit
                    for sample in height_samples
                    for sweep in sample["directional_sweeps"].values()
                    for sweep_sample in sweep["samples"]
                    for hit in sweep_sample["target_hits"]
                ]
            )
        )
        record["cardinal_sweeps"] = {
            direction: {
                "distances": [float(distance) for distance in sweep_offsets[direction]],
                "sample_count": len(sweep_offsets[direction]),
            }
            for direction, _ in _CARDINAL_DIRECTIONS
        }

    unique_targets = sorted(target_records.values(), key=lambda item: item["path"])
    unique_collision_prims = _unique_serializable_items(all_collision_prims)
    timeline = omni.timeline.get_timeline_interface()
    return {
        "timestamp": _dt.datetime.now().isoformat(timespec="milliseconds"),
        "stage_root_layer": stage.GetRootLayer().identifier,
        "service": "/isaac/debug_scene_collisions",
        "config": asdict(config),
        "timeline": {
            "playing": bool(timeline.is_playing()),
            "stopped": bool(timeline.is_stopped()),
            "current_time": float(timeline.get_current_time()),
        },
        "targets": unique_targets,
        "target_count": len(unique_targets),
        "collision_prim_count": len(unique_collision_prims),
        "collision_prims": unique_collision_prims,
    }


def write_scene_collision_report(report: dict[str, Any], output_path: str | None = None) -> str:
    path = Path(
        output_path
        or os.environ.get(
            "ARENA_ISAAC_SCENE_COLLISION_PROBE_OUTPUT",
            SceneCollisionProbeConfig.output_path,
        )
    ).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def scene_collision_probe_service(controller):
    """Register the scene collision probe Trigger service on the controller node."""

    from std_srvs.srv import Trigger

    def _callback(_request, response):
        try:
            config = scene_collision_probe_config_from_env()
            report = collect_scene_collision_report(config=config)
            output_path = write_scene_collision_report(report, config.output_path)
            response.success = True
            response.message = (
                f"report={output_path}; targets={report['target_count']}; "
                f"collision_prims={report['collision_prim_count']}"
            )
            controller.get_logger().info(f"Scene collision probe: {response.message}")
        except Exception as exc:
            response.success = False
            response.message = f"scene collision probe failed: {exc}"
            controller.get_logger().error(response.message)
        return response

    return controller.create_service(Trigger, "/isaac/debug_scene_collisions", _callback)
