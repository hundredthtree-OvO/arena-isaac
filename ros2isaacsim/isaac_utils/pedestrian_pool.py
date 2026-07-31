from __future__ import annotations

import zlib


def parking_pose_for_name(
    name: str,
    *,
    origin: tuple[float, float, float] = (1000.0, 1000.0, 0.0),
    spacing_m: float = 2.0,
    columns: int = 32,
) -> list[float]:
    """Return a stable, separated parking slot for a pedestrian name."""
    key = str(name or "pedestrian").strip().encode("utf-8")
    slot = zlib.crc32(key) % (max(int(columns), 1) ** 2)
    column_count = max(int(columns), 1)
    return [
        float(origin[0]) + float(slot % column_count) * float(spacing_m),
        float(origin[1]) + float(slot // column_count) * float(spacing_m),
        float(origin[2]),
    ]


def park_all_people(manager) -> int:
    """Park unique managed people while preserving aliases for reactivation."""
    seen = set()
    parked = 0
    for person in list((getattr(manager, "people", {}) or {}).values()):
        if id(person) in seen:
            continue
        seen.add(id(person))
        stage_prefix = str(getattr(person, "_stage_prefix", "") or "")
        park = getattr(person, "park", None)
        if not stage_prefix or not callable(park):
            continue
        park(parking_pose_for_name(stage_prefix))
        parked += 1
    return parked
