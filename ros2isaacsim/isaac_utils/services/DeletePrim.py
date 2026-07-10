import isaac_utils.utils.paths as Paths
import carb
import omni.kit.commands as commands
from isaac_utils.managers.door_manager import door_manager
from pedestrian.simulator.logic.people_manager import PeopleManager
from rclpy.qos import QoSProfile

from isaacsim_msgs.srv import DeletePrim

from .utils import safe

profile = QoSProfile(depth=2000)


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
        disposer = getattr(person, "dispose", None)
        if callable(disposer):
            try:
                disposer()
            except Exception:
                pass
    for key, person in list(people.items()):
        if id(person) in matched_ids:
            manager.remove_person(str(key))
    if matched_people:
        door_manager.remove_pedestrian(root_path)


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

    if prim_path.startswith("/World/Characters/"):
        _remove_people_aliases_for_root(prim_path)
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
