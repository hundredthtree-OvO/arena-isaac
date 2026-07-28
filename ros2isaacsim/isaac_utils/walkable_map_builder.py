"""Export a static 2D occupancy asset from the loaded PhysX scene."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass(frozen=True)
class WalkableMapConfig:
    scene_root: str
    output_path: str
    resolution: float = 0.05
    origin: tuple[float, float, float] = (-2.24, -0.90, 0.75)
    world_bounds: tuple[float, float, float, float] = (-4.60, 3.60, -2.10, 1.80)
    exclude_path_keywords: tuple[str, ...] = (
        "/World/Characters",
        "/World/xms_mecanum",
        "_debug",
    )


def occupancy_values(
    raw_buffer: Iterable[float],
    *,
    occupied_value: float = 4.0,
    free_value: float = 5.0,
) -> list[int]:
    """Convert Isaac occupancy values to ROS OccupancyGrid values."""

    result: list[int] = []
    for value in raw_buffer:
        if abs(float(value) - occupied_value) < 1e-5:
            result.append(100)
        elif abs(float(value) - free_value) < 1e-5:
            result.append(0)
        else:
            result.append(-1)
    return result


def occupancy_grid_from_positions(
    *,
    width: int,
    height: int,
    resolution: float,
    origin_xy: tuple[float, float],
    free_positions: Iterable[Iterable[float]],
    occupied_positions: Iterable[Iterable[float]],
) -> list[int]:
    """Raster world-space omap positions into ROS bottom-left row order."""

    data = [-1] * (int(width) * int(height))
    ox, oy = float(origin_xy[0]), float(origin_xy[1])

    def index(position: Iterable[float]) -> int | None:
        values = tuple(float(value) for value in position)
        ix = math.floor((values[0] - ox) / float(resolution))
        iy = math.floor((values[1] - oy) / float(resolution))
        if ix < 0 or iy < 0 or ix >= width or iy >= height:
            return None
        return int(iy * width + ix)

    for position in free_positions:
        cell = index(position)
        if cell is not None:
            data[cell] = 0
    for position in occupied_positions:
        cell = index(position)
        if cell is not None:
            data[cell] = 100
    return data


def _collision_enabled(prim, UsdPhysics) -> bool:
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        return False
    attr = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
    value = attr.Get() if attr and attr.IsValid() else None
    return value is not False


def _scene_fingerprint(stage, scene_root: str, UsdPhysics) -> str:
    records: list[str] = []
    root = str(scene_root).rstrip("/")
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path != root and not path.startswith(root + "/"):
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        approximation = prim.GetAttribute("physics:approximation")
        records.append(
            "|".join(
                (
                    path,
                    str(prim.GetTypeName()),
                    str(_collision_enabled(prim, UsdPhysics)),
                    str(approximation.Get() if approximation and approximation.IsValid() else ""),
                )
            )
        )
    return hashlib.sha256("\n".join(sorted(records)).encode("utf-8")).hexdigest()


@contextmanager
def _static_scene_only(stage, config: WalkableMapConfig, UsdPhysics) -> Iterator[None]:
    """Temporarily disable collision shapes outside the selected static scene."""

    scene_root = str(config.scene_root).rstrip("/")
    excluded = tuple(str(value).lower() for value in config.exclude_path_keywords)
    changed: list[tuple[Any, Any, bool]] = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        path = str(prim.GetPath())
        path_lower = path.lower()
        in_scene = path == scene_root or path.startswith(scene_root + "/")
        if in_scene and not any(token in path_lower for token in excluded):
            continue
        api = UsdPhysics.CollisionAPI(prim)
        attr = api.GetCollisionEnabledAttr()
        was_authored = bool(attr and attr.IsValid() and attr.HasAuthoredValue())
        old_value = attr.Get() if attr and attr.IsValid() else None
        if old_value is False:
            continue
        if not attr or not attr.IsValid():
            attr = api.CreateCollisionEnabledAttr()
        attr.Set(False)
        changed.append((attr, old_value, was_authored))
    try:
        yield
    finally:
        for attr, old_value, was_authored in reversed(changed):
            if was_authored:
                attr.Set(True if old_value is None else bool(old_value))
            else:
                attr.Clear()


def _write_pgm(path: Path, data: list[int], width: int, height: int) -> None:
    pixels = bytearray()
    for y in range(height - 1, -1, -1):
        row = data[y * width : (y + 1) * width]
        pixels.extend(0 if value >= 50 else 254 if value == 0 else 205 for value in row)
    path.write_bytes(f"P5\n{width} {height}\n255\n".encode("ascii") + pixels)


def _write_map_yaml(path: Path, image_path: Path, resolution: float, origin: list[float]) -> None:
    path.write_text(
        "\n".join(
            (
                f"image: {image_path.name}",
                f"resolution: {resolution:.9g}",
                f"origin: [{origin[0]:.9g}, {origin[1]:.9g}, 0.0]",
                "negate: 0",
                "occupied_thresh: 0.65",
                "free_thresh: 0.196",
                "",
            )
        ),
        encoding="utf-8",
    )


def build_walkable_map(config: WalkableMapConfig, *, logger=None) -> dict[str, Any]:
    """Generate and persist an Isaac occupancy map from static collision geometry."""

    import omni.kit.app
    import omni.physx
    import omni.usd
    from pxr import UsdPhysics

    manager = omni.kit.app.get_app().get_extension_manager()
    manager.set_extension_enabled_immediate("isaacsim.asset.gen.omap", True)
    from isaacsim.asset.gen.omap.bindings import _omap

    context = omni.usd.get_context()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("no USD stage is loaded")
    if not stage.GetPrimAtPath(config.scene_root).IsValid():
        raise RuntimeError(f"scene root does not exist: {config.scene_root}")
    if config.resolution <= 0.0:
        raise ValueError("resolution must be positive")
    min_x, max_x, min_y, max_y = config.world_bounds
    if min_x >= max_x or min_y >= max_y:
        raise ValueError("world_bounds must be [min_x, max_x, min_y, max_y]")

    generator = _omap.Generator(
        omni.physx.acquire_physx_interface(),
        context.get_stage_id(),
    )
    generator.update_settings(float(config.resolution), 4.0, 5.0, 6.0)
    ox, oy, oz = config.origin
    lower = (min_x - ox, min_y - oy, 0.0)
    upper = (max_x - ox, max_y - oy, 0.0)
    with _static_scene_only(stage, config, UsdPhysics):
        generator.set_transform(config.origin, lower, upper)
        generator.generate2d()

    dimensions = generator.get_dimensions()
    width = int(dimensions[0])
    height = int(dimensions[1])
    raw = list(generator.get_buffer())
    if width <= 0 or height <= 0 or len(raw) != width * height:
        raise RuntimeError(
            f"invalid occupancy output: dimensions={width}x{height}, buffer={len(raw)}"
        )
    min_bound = generator.get_min_bound()
    data = occupancy_grid_from_positions(
        width=width,
        height=height,
        resolution=config.resolution,
        origin_xy=(float(min_bound[0]), float(min_bound[1])),
        free_positions=generator.get_free_positions(),
        occupied_positions=generator.get_occupied_positions(),
    )
    output = Path(config.output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    pgm_path = output.with_suffix(".pgm")
    yaml_path = output.with_suffix(".yaml")
    origin = [float(min_bound[0]), float(min_bound[1]), 0.0]
    payload = {
        "schema": "arena.walkable_map.v1",
        "source": "isaacsim.asset.gen.omap",
        "scene_root": config.scene_root,
        "scene_fingerprint": _scene_fingerprint(stage, config.scene_root, UsdPhysics),
        "resolution": float(config.resolution),
        "origin": origin,
        "sample_height_m": float(oz),
        "width": width,
        "height": height,
        "data": data,
        "counts": {
            "free": data.count(0),
            "occupied": data.count(100),
            "unknown": data.count(-1),
        },
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_pgm(pgm_path, data, width, height)
    _write_map_yaml(yaml_path, pgm_path, config.resolution, origin)
    if logger is not None:
        logger.info(
            f"Walkable map exported: output={output}, dimensions={width}x{height}, "
            f"counts={payload['counts']}"
        )
    return payload
