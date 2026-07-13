import asyncio

import isaac_utils.utils.paths as Paths
import carb
import omni.kit.app
import omni.kit.commands as commands
import omni.usd
from isaac_utils.managers.door_manager import door_manager
from pedestrian.simulator.logic.people_manager import PeopleManager
from rclpy.qos import QoSProfile

from isaacsim_msgs.srv import DeletePrim

from .utils import safe

profile = QoSProfile(depth=2000)
_retired_character_roots: set[str] = set()
_retired_cleanup_task = None


def _resolve_delete_path(name: str) -> str:
    text = str(name or "").strip()
    if text.startswith("/"):
        return text
    return Paths.scene.path(text)


def _remove_people_aliases_for_root(root_path: str):
    manager = PeopleManager.get_people_manager()
    people = getattr(manager, "people", {}) or {}
    matched_people = _people_for_root(root_path, people)
    carb.log_info(
        f"Delete pedestrian aliases for root={root_path}: "
        f"matched_people={len(matched_people)}, aliases={len(people)}"
    )
    matched_ids = {id(person) for person in matched_people}
    for person in matched_people:
        retire = getattr(person, "retire", None)
        if callable(retire):
            try:
                retire()
            except Exception:
                pass
    for key, person in list(people.items()):
        if id(person) in matched_ids:
            manager.remove_person(str(key))
    if matched_people:
        door_manager.remove_pedestrian(root_path)
    return bool(matched_people)


def _has_active_people() -> bool:
    manager = PeopleManager.get_people_manager()
    seen = set()
    for person in list((getattr(manager, "people", {}) or {}).values()):
        if id(person) in seen:
            continue
        seen.add(id(person))
        if bool(getattr(person, "_active", True)):
            return True
    return False


async def _destroy_retired_roots_when_quiescent():
    global _retired_cleanup_task
    try:
        # Let scripting, physics, and rendering observe every character as
        # inactive before invalidating any AnimGraph prim handles.
        await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()
        if _has_active_people():
            return
        stage = omni.usd.get_context().get_stage()
        for root_path in sorted(_retired_character_roots):
            prim = stage.GetPrimAtPath(root_path) if stage is not None else None
            if prim is not None and prim.IsValid():
                commands.execute("IsaacSimDestroyPrim", prim_path=root_path)
                carb.log_info(f"Destroyed retired pedestrian root after quiescence: {root_path}")
        _retired_character_roots.clear()
    finally:
        _retired_cleanup_task = None


def _schedule_retired_cleanup_if_quiescent():
    global _retired_cleanup_task
    if _has_active_people() or _retired_cleanup_task is not None:
        return
    _retired_cleanup_task = asyncio.ensure_future(_destroy_retired_roots_when_quiescent())


def _people_for_root(root_path: str, people: dict):
    matched_people = []
    matched_ids = set()
    for key, person in list(people.items()):
        key_text = str(key)
        stage_prefix = str(getattr(person, "_stage_prefix", "") or "")
        skel_root = str(getattr(person, "character_skel_root_stage_path", "") or "")
        if (
            key_text == root_path
            or key_text.startswith(root_path.rstrip("/") + "/")
            or stage_prefix == root_path
            or stage_prefix.startswith(root_path.rstrip("/") + "/")
            or skel_root.startswith(root_path.rstrip("/") + "/")
        ):
            if id(person) not in matched_ids:
                matched_people.append(person)
                matched_ids.add(id(person))
    return matched_people


@safe
def prim_deleter(request, response):
    prim_path = _resolve_delete_path(request.name)
    carb.log_info(f"DeletePrim request name={request.name}, resolved={prim_path}")
    if prim_path == Paths.scene.pedestrian():
        door_manager.reset()

    if prim_path.startswith("/World/Characters/") and _remove_people_aliases_for_root(prim_path):
        _retired_character_roots.add(prim_path)
        carb.log_info(f"Retired pedestrian without destroying live AnimGraph: {prim_path}")
        _schedule_retired_cleanup_if_quiescent()
    else:
        commands.execute(
            "IsaacSimDestroyPrim",
            prim_path=prim_path,
        )
    response.ret = True
    return response


def delete_prim(controller):
    service = controller.create_service(
        srv_type=DeletePrim,
        qos_profile=profile,
        srv_name='isaac/delete_prim',
        callback=prim_deleter
    )
    return service
