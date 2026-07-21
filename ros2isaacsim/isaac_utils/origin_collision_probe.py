"""Live USD and PhysX diagnostics for unexplained collisions near world origin."""

from __future__ import annotations

import datetime as dt
import json
import math
import os
from pathlib import Path
from typing import Iterable


def _aabb_intersects(
    minimum: Iterable[float],
    maximum: Iterable[float],
    query_minimum: Iterable[float],
    query_maximum: Iterable[float],
) -> bool:
    minimum = tuple(float(value) for value in minimum)
    maximum = tuple(float(value) for value in maximum)
    query_minimum = tuple(float(value) for value in query_minimum)
    query_maximum = tuple(float(value) for value in query_maximum)
    return all(
        minimum[axis] <= query_maximum[axis] and maximum[axis] >= query_minimum[axis]
        for axis in range(3)
    )


def _hit_path(hit: dict) -> str:
    return str(hit.get("rigid_body", hit.get("collision", hit.get("repr", ""))))


def _diagnose_origin_hits(origin_candidates: list[dict], physx_hits: list[dict]) -> str:
    prototype_hits = [hit for hit in physx_hits if _hit_path(hit).startswith("/__Prototype_")]
    usd_robot_candidates = [
        entry for entry in origin_candidates if entry["path"].startswith("/World/robots/")
    ]
    physx_robot_hits = [
        hit for hit in physx_hits if _hit_path(hit).startswith("/World/robots/")
    ]
    if prototype_hits:
        return "physx_collision_prototype_at_origin"
    if physx_robot_hits and not usd_robot_candidates:
        return "stale_physx_robot_actor_suspected"
    if usd_robot_candidates:
        return "robot_usd_collider_is_at_origin"
    if physx_hits:
        return "non_robot_physx_shape_is_at_origin"
    return "no_physx_shape_detected_in_query_box"


def _vector(values) -> list[float]:
    return [float(values[index]) for index in range(3)]


def _nearest_rigid_body_path(prim, UsdPhysics) -> str | None:
    current = prim
    while current and current.IsValid() and not current.IsPseudoRoot():
        if current.HasAPI(UsdPhysics.RigidBodyAPI):
            return str(current.GetPath())
        current = current.GetParent()
    return None


def _collision_entry(prim, bbox_cache, xform_cache, UsdPhysics) -> dict | None:
    try:
        aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        minimum = _vector(aligned.GetMin())
        maximum = _vector(aligned.GetMax())
        if not all(math.isfinite(value) for value in minimum + maximum):
            return None
        matrix = xform_cache.GetLocalToWorldTransform(prim)
        translation = _vector(matrix.ExtractTranslation())
    except Exception:
        return None
    enabled_attr = prim.GetAttribute("physics:collisionEnabled")
    approximation_attr = prim.GetAttribute("physics:approximation")
    return {
        "path": str(prim.GetPath()),
        "type": prim.GetTypeName(),
        "enabled": True if not enabled_attr else bool(enabled_attr.Get()),
        "approximation": None if not approximation_attr else approximation_attr.Get(),
        "rigid_body": _nearest_rigid_body_path(prim, UsdPhysics),
        "world_translation": translation,
        "world_aabb_min": minimum,
        "world_aabb_max": maximum,
    }


def collect_origin_collision_report(
    *,
    center: tuple[float, float, float] = (0.0, 0.0, 0.35),
    half_extent: tuple[float, float, float] = (0.75, 0.75, 0.30),
) -> dict:
    """Collect matching USD bounds and live PhysX overlap hits."""
    import carb
    import omni.physx
    import omni.timeline
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Isaac stage is not available")

    query_minimum = [center[index] - half_extent[index] for index in range(3)]
    query_maximum = [center[index] + half_extent[index] for index in range(3)]
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    origin_candidates = []
    robot_colliders = []
    robot_rigid_bodies = []
    collision_count = 0

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and path.startswith("/World/robots/"):
            try:
                transform = xform_cache.GetLocalToWorldTransform(prim)
                robot_rigid_bodies.append(
                    {
                        "path": path,
                        "world_translation": _vector(transform.ExtractTranslation()),
                    }
                )
            except Exception:
                pass
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        collision_count += 1
        entry = _collision_entry(prim, bbox_cache, xform_cache, UsdPhysics)
        if entry is None:
            continue
        if path.startswith("/World/robots/"):
            robot_colliders.append(entry)
        if entry["enabled"] and _aabb_intersects(
            entry["world_aabb_min"],
            entry["world_aabb_max"],
            query_minimum,
            query_maximum,
        ):
            origin_candidates.append(entry)

    physx_hits = []

    def _on_hit(hit):
        item = {}
        for attribute in ("rigid_body", "collision", "material"):
            try:
                value = getattr(hit, attribute)
            except Exception:
                continue
            item[attribute] = str(value)
        if not item:
            item["repr"] = repr(hit)
        physx_hits.append(item)
        return True

    query_result = omni.physx.get_physx_scene_query_interface().overlap_box(
        carb.Float3(*half_extent),
        carb.Float3(*center),
        carb.Float4(0.0, 0.0, 0.0, 1.0),
        _on_hit,
        False,
    )
    unique_hits = []
    seen_hits = set()
    for hit in physx_hits:
        key = json.dumps(hit, sort_keys=True)
        if key in seen_hits:
            continue
        seen_hits.add(key)
        unique_hits.append(hit)

    prototype_hits = [hit for hit in unique_hits if _hit_path(hit).startswith("/__Prototype_")]
    diagnosis = _diagnose_origin_hits(origin_candidates, unique_hits)

    timeline = omni.timeline.get_timeline_interface()
    return {
        "timestamp": dt.datetime.now().isoformat(timespec="milliseconds"),
        "stage_root_layer": stage.GetRootLayer().identifier,
        "timeline": {
            "playing": bool(timeline.is_playing()),
            "stopped": bool(timeline.is_stopped()),
            "current_time": float(timeline.get_current_time()),
        },
        "query": {
            "center": list(center),
            "half_extent": list(half_extent),
            "aabb_min": query_minimum,
            "aabb_max": query_maximum,
        },
        "usd_collision_count": collision_count,
        "usd_origin_candidates": origin_candidates,
        "robot_colliders": robot_colliders,
        "robot_rigid_bodies": robot_rigid_bodies,
        "physx_overlap_result": (
            query_result
            if isinstance(query_result, (bool, int, float, str, type(None)))
            else str(query_result)
        ),
        "physx_hits": unique_hits,
        "physx_prototype_hits": prototype_hits,
        "diagnosis": diagnosis,
    }


def write_origin_collision_report(report: dict, output_path: str | None = None) -> str:
    path = Path(
        output_path
        or os.environ.get("ARENA_ISAAC_ORIGIN_PROBE_OUTPUT", "/tmp/arena_physx_origin_probe.json")
    ).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def origin_collision_probe_service(controller):
    from std_srvs.srv import Trigger

    def _callback(_request, response):
        try:
            center = (
                float(os.environ.get("ARENA_ISAAC_ORIGIN_PROBE_X", "0.0")),
                float(os.environ.get("ARENA_ISAAC_ORIGIN_PROBE_Y", "0.0")),
                float(os.environ.get("ARENA_ISAAC_ORIGIN_PROBE_Z", "0.35")),
            )
            half_extent = (
                float(os.environ.get("ARENA_ISAAC_ORIGIN_PROBE_HALF_X", "0.75")),
                float(os.environ.get("ARENA_ISAAC_ORIGIN_PROBE_HALF_Y", "0.75")),
                float(os.environ.get("ARENA_ISAAC_ORIGIN_PROBE_HALF_Z", "0.30")),
            )
            report = collect_origin_collision_report(center=center, half_extent=half_extent)
            output_path = write_origin_collision_report(report)
            usd_paths = [entry["path"] for entry in report["usd_origin_candidates"]]
            physx_paths = [
                hit.get("rigid_body", hit.get("collision", hit.get("repr", "unknown")))
                for hit in report["physx_hits"]
            ]
            response.success = True
            response.message = (
                f"report={output_path}; usd_origin_candidates={len(usd_paths)} {usd_paths[:6]}; "
                f"physx_hits={len(physx_paths)} {physx_paths[:6]}"
            )
            controller.get_logger().info(f"Origin collision probe: {response.message}")
        except Exception as exc:
            response.success = False
            response.message = f"origin collision probe failed: {exc}"
            controller.get_logger().error(response.message)
        return response

    return controller.create_service(Trigger, "/isaac/debug_origin_collision", _callback)
