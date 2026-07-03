"""Service registration helpers for ros2isaacsim.

The pedestrian services depend on Isaac AnimGraph/People extensions.  For the
light bridge memory test we intentionally do not load those extensions, so this
module must not import SpawnCharacters/MoveCharacters unconditionally.
"""

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


_ENABLE_PEOPLE_STACK = _env_bool("ARENA_ISAAC_ENABLE_PEOPLE_STACK", False)


def _log_skip(controller, service_name: str):
    try:
        controller.get_logger().info(
            f"Skipping {service_name}: ARENA_ISAAC_ENABLE_PEOPLE_STACK is false "
            "and AnimGraph/People extensions are not loaded."
        )
    except Exception:
        pass


if _ENABLE_PEOPLE_STACK:
    # These imports require omni.anim.graph / omni.anim.people.  Keep them behind
    # the flag so lightweight robot/lidar bridge startup does not crash.
    from .SpawnCharacters import *
    from .MoveCharacters import *
    from .DeleteAllCharacters import *
else:
    def spawn_ped(controller):
        _log_skip(controller, "/isaac/spawn_pedestrian")
        return None

    def move_ped(controller):
        _log_skip(controller, "/isaac/move_pedestrians")
        return None

    def delete_all_characters(controller):
        _log_skip(controller, "/isaac/delete_all_pedestrians")
        return None
