import carb
import numpy as np
import omni.timeline
import omni.anim.navigation.core as nav
from pedestrian.simulator.logic.people.person import Person
from pedestrian.simulator.logic.people_manager import PeopleManager
from rclpy.qos import QoSProfile

from isaacsim_msgs.msg import NavPed
from isaacsim_msgs.srv import MovePed

from .utils import safe

profile = QoSProfile(depth=2000)


def _goal3(value):
    return [float(value[0]), float(value[1]), float(value[2])]


def _find_person(path: str):
    manager = PeopleManager.get_people_manager()
    candidates = [
        path,
        f"/World/Characters/{path}",
        f"/World/Characters/{path.strip('/')}",
    ]
    for key in candidates:
        person = manager.get_person(key)
        if person is not None:
            return person

    # Fallback: scan aliases and root/skelroot suffixes.
    for key, person in list(manager.people.items()):
        if key == path or key.endswith("/" + path.strip("/")):
            return person
        if getattr(person, "_stage_prefix", "").endswith("/" + path.strip("/")):
            return person
        if getattr(person, "character_skel_root_stage_path", "").endswith("/" + path.strip("/")):
            return person
    return None


@safe
def move_pedestrian(request: MovePed.Request, response: MovePed.Response):
    # AnimGraph characters only become controllable once the timeline is playing.
    try:
        omni.timeline.get_timeline_interface().play()
    except Exception:
        pass

    for nav_command in request.nav_list:
        nav_command: NavPed
        person = _find_person(nav_command.path)
        if not isinstance(person, Person):
            carb.log_error(f"Person not found for path/name: {nav_command.path}")
            continue

        goal = _goal3(nav_command.goal_pose)
        velocity = float(nav_command.velocity or 1.0)

        # Prefer navmesh paths when available; fall back to direct PathPoints.
        path_points = None
        try:
            inav = nav.acquire_interface()
            navmesh = inav.get_navmesh()
            if navmesh:
                start = [float(x) for x in person._state.position.tolist()]
                navmesh_path = navmesh.query_shortest_path(start, goal)
                if navmesh_path:
                    raw_points = navmesh_path.get_points()
                    path_points = [[float(p[0]), float(p[1]), float(p[2])] for p in raw_points]
        except Exception as exc:
            carb.log_warn(f"NavMesh path query failed; using direct PathPoints: {exc}")

        if not path_points:
            path_points = [goal]

        person.update_target_position(path_points, velocity)
        carb.log_info(
            f"Move pedestrian {nav_command.path}: goal={goal}, velocity={velocity}, points={len(path_points)}"
        )

    response.ret = True
    return response


def move_ped(controller):
    service = controller.create_service(
        srv_type=MovePed,
        qos_profile=profile,
        srv_name="isaac/move_pedestrians",
        callback=move_pedestrian,
    )
    return service
