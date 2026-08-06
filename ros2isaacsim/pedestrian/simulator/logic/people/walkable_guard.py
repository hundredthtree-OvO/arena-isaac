"""Read-only pedestrian collision queries against the exported walkable map."""

from __future__ import annotations

import json
from pathlib import Path


class StaticWalkableGuard:
    def __init__(self, map_path: str):
        payload = json.loads(Path(map_path).expanduser().read_text(encoding="utf-8"))
        if payload.get("schema") != "arena.walkable_map.v1":
            raise ValueError(f"unsupported walkable-map schema: {payload.get('schema')}")
        self.resolution = float(payload["resolution"])
        self.origin = tuple(float(value) for value in payload["origin"][:3])
        self.width = int(payload["width"])
        self.height = int(payload["height"])
        data = [int(value) for value in payload["data"]]
        if self.width <= 0 or self.height <= 0 or len(data) != self.width * self.height:
            raise ValueError("invalid walkable-map dimensions")
        # Treat unknown and out-of-map space as unavailable to pedestrians.
        self.occupied_xy = {
            (index % self.width, index // self.width)
            for index, value in enumerate(data)
            if value != 0
        }

    def refresh(self) -> None:
        """Keep the former guard call site stable for an immutable map."""

