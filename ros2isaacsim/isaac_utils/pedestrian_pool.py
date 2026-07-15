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
