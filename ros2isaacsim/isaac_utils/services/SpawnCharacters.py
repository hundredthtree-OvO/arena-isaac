import carb
import omni.timeline
from isaacsim.core.api.world import World
from pedestrian.simulator.logic.people.person import Person
from pedestrian.simulator.logic.people_manager import PeopleManager
from rclpy.qos import QoSProfile

from isaacsim_msgs.srv import Pedestrian
from isaac_utils.animgraph_people import normalize_stage_name
from isaac_utils.managers.door_manager import door_manager

from .utils import safe

profile = QoSProfile(depth=2000)


try:
    from rclpy.logging import get_logger

    _LOGGER = get_logger("isaac_spawn_ped")
except Exception:
    _LOGGER = None


def _log_info(msg: str):
    try:
        if _LOGGER:
            _LOGGER.info(msg)
            return
    except Exception:
        pass
    print(msg)


def _log_warn(msg: str):
    try:
        if _LOGGER:
            _LOGGER.warn(msg)
            return
    except Exception:
        pass
    print(msg)


def _to_list3(value, default=(0.0, 0.0, 0.0)):
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except Exception:
        return [float(default[0]), float(default[1]), float(default[2])]


@safe
def pedestrian_spawn(request, response):
    """Spawn Isaac Sim 4.5 AnimGraph pedestrians.

    Important Isaac 4.5 behavior validated locally:
    - CharacterUtil expects a relative stage name, not /World/... paths.
    - ag.get_character only becomes valid after timeline.play(), but Person obtains
      it lazily inside its physics callbacks.
    - Store people by several keys so /isaac/move_pedestrians can find them using
      Arena's stage_prefix/path value.
    """

    world = World.instance()
    if world is None:
        world = World()

    manager = PeopleManager.get_people_manager()
    timeline = omni.timeline.get_timeline_interface()
    was_playing = bool(timeline.is_playing())
    spawned_count = 0

    # AnimGraph's variable synchronization ignores stage changes during play.
    # Pause the batch setup transaction without resetting simulation time, then
    # restore the caller's previous timeline state.
    if was_playing:
        timeline.pause()
    try:
        for person_msg in request.people:
            stage_name = normalize_stage_name(getattr(person_msg, "stage_prefix", None), default="Character")
            character_name = getattr(person_msg, "character_name", None) or None
            init_pos = _to_list3(getattr(person_msg, "initial_pose", None))
            init_yaw = float(getattr(person_msg, "orientation", 0.0) or 0.0)

            p = Person(world, stage_name, character_name, init_pos, init_yaw)

            # Register multiple aliases.  Arena messages often use only `character_0`,
            # while Isaac/AnimGraph uses /World/Characters/<name>/.../SkelRoot.
            keys = {
                stage_name,
                getattr(p, "_stage_prefix", None),
                getattr(p, "character_skel_root_stage_path", None),
                f"/World/Characters/{stage_name}",
            }
            for key in keys:
                if key:
                    manager.add_person(str(key), p)
                    door_manager.add_pedestrian(str(key))

            spawned_count += 1
            _log_info(
                f"Spawned pedestrian {stage_name}: root={getattr(p, '_stage_prefix', None)}, "
                f"skelroot={getattr(p, 'character_skel_root_stage_path', None)}, model={character_name}, "
                f"animgraph_setup={p.anim_graph_setup_ok}"
            )
    finally:
        if was_playing:
            timeline.play()

    response.ret = spawned_count == len(request.people)
    return response


def spawn_ped(controller):
    service = controller.create_service(
        srv_type=Pedestrian,
        qos_profile=profile,
        srv_name="isaac/spawn_pedestrian",
        callback=pedestrian_spawn,
    )
    return service
