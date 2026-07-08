"""Service registration helpers for ros2isaacsim.

The character/pedestrian services import Isaac Sim Replicator Agent through
pedestrian.simulator.logic.people.person. For RTX LiDAR debugging we often want
ARENA_ISAAC_ENABLE_PEOPLE_STACK=true only to test AnimGraph schema/custom-kit
startup, not to import the full character backend. Therefore character services
are behind ARENA_ISAAC_ENABLE_CHARACTER_SERVICES.
"""
from __future__ import annotations

import os

from .DeletePrim import *
from .GetPrimAttributes import *
from .ImportObstacles import *
from .ImportUsds import *
from .ImportYaml import *
from .MovePrim import *
from .SpawnWall import *
from .SpawnDoor import *
from .UrdfToUsd import *
from .SpawnFloor import *


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _log_skip(controller, service_name: str, reason: str) -> None:
    try:
        controller.get_logger().info(f"Skipping {service_name}: {reason}")
    except Exception:
        print(f"Skipping {service_name}: {reason}", flush=True)


def _stub_spawn_ped(controller):
    _log_skip(
        controller,
        "/isaac/spawn_pedestrian",
        "ARENA_ISAAC_ENABLE_CHARACTER_SERVICES is false or character backend failed to import",
    )
    return None


def _stub_move_ped(controller):
    _log_skip(
        controller,
        "/isaac/move_pedestrians",
        "ARENA_ISAAC_ENABLE_CHARACTER_SERVICES is false or character backend failed to import",
    )
    return None


def _stub_delete_all_characters(controller):
    _log_skip(
        controller,
        "/isaac/delete_all_pedestrians",
        "ARENA_ISAAC_ENABLE_CHARACTER_SERVICES is false or character backend failed to import",
    )
    return None


_ENABLE_CHARACTER_SERVICES = _env_bool("ARENA_ISAAC_ENABLE_CHARACTER_SERVICES", False)

if _ENABLE_CHARACTER_SERVICES:
    try:
        # These imports require the actual Isaac People/Replicator Agent backend.
        from .SpawnCharacters import *
        from .MoveCharacters import *
        from .DeleteAllCharacters import *
    except Exception as exc:  # noqa: BLE001 - keep the bridge alive.
        print(
            f"[people_stack] character services disabled after import failure: {exc}",
            flush=True,
        )
        spawn_ped = _stub_spawn_ped
        move_ped = _stub_move_ped
        delete_all_characters = _stub_delete_all_characters
else:
    spawn_ped = _stub_spawn_ped
    move_ped = _stub_move_ped
    delete_all_characters = _stub_delete_all_characters
