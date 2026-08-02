"""Versioned motion-library manifest contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
MOTION_CATEGORIES = {"walk", "turn", "stop", "resume"}
MOTION_DIRECTIONS = {"left", "right", "forward", "stationary"}


def validate_manifest(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported motion-library schema {payload.get('schema_version')}"
        )
    if not isinstance(payload.get("library_id"), str) or not payload["library_id"]:
        raise ValueError("library_id must be a non-empty string")
    motions = payload.get("motions")
    if not isinstance(motions, list) or not motions:
        raise ValueError("motions must be a non-empty list")

    seen: set[str] = set()
    required = {
        "id",
        "category",
        "direction",
        "source",
        "clip",
        "timing",
        "kinematics",
        "contacts",
    }
    for motion in motions:
        missing = required.difference(motion)
        if missing:
            raise ValueError(f"motion entry missing keys: {sorted(missing)}")
        motion_id = motion["id"]
        if not isinstance(motion_id, str) or not motion_id:
            raise ValueError("motion id must be a non-empty string")
        if motion_id in seen:
            raise ValueError(f"duplicate motion id: {motion_id}")
        seen.add(motion_id)
        if motion["category"] not in MOTION_CATEGORIES:
            raise ValueError(f"{motion_id}: invalid category {motion['category']}")
        if motion["direction"] not in MOTION_DIRECTIONS:
            raise ValueError(f"{motion_id}: invalid direction {motion['direction']}")
        if motion["timing"]["frames"] < 2 or motion["timing"]["fps"] <= 0:
            raise ValueError(f"{motion_id}: invalid timing")
        for side in ("left_fraction", "right_fraction"):
            value = float(motion["contacts"][side])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{motion_id}: invalid contact fraction {side}")
        if motion["category"] == "walk":
            gait = motion.get("gait")
            if not isinstance(gait, dict):
                raise ValueError(f"{motion_id}: walk motion requires gait metadata")
            if float(gait["nominal_speed_mps"]) <= 0.0:
                raise ValueError(f"{motion_id}: invalid nominal gait speed")
            if not (
                0.0 < float(gait["validated_speed_scale_min"])
                <= 1.0
                <= float(gait["validated_speed_scale_max"])
            ):
                raise ValueError(f"{motion_id}: invalid validated speed scale")


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    validate_manifest(payload)
    return payload
