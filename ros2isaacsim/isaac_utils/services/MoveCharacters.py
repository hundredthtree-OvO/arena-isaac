import carb
import omni.timeline
import omni.anim.navigation.core as nav
from pedestrian.simulator.logic.people.person import Person
from pedestrian.simulator.logic.people_manager import PeopleManager
from rclpy.qos import QoSProfile

from isaacsim_msgs.msg import NavPed
from isaacsim_msgs.srv import MovePed

from .move_command_utils import resolve_nav_velocity
from .utils import safe

profile = QoSProfile(depth=2000)


def _goal3(value):
    return [float(value[0]), float(value[1]), float(value[2])]


def _goal_or_none(value):
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except Exception:
        return None


def _flat_points_to_list3(value):
    try:
        vals = [float(x) for x in value]
    except Exception:
        return []
    if len(vals) < 3:
        return []
    points = []
    for i in range(0, len(vals) - 2, 3):
        points.append([vals[i], vals[i + 1], vals[i + 2]])
    return points


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

    accepted = True
    for nav_command in request.nav_list:
        nav_command: NavPed
        person = _find_person(nav_command.path)
        if not isinstance(person, Person):
            carb.log_error(f"Person not found for path/name: {nav_command.path}")
            accepted = False
            continue
        carb.log_info(
            f"Resolved pedestrian command {nav_command.path}: "
            f"root={getattr(person, '_stage_prefix', '')}, "
            f"skelroot={getattr(person, 'character_skel_root_stage_path', '')}, "
            f"animgraph_ready={person.anim_graph_ready}"
        )

        direct_pose = _goal_or_none(getattr(nav_command, "direct_pose", None))
        use_direct_pose = bool(getattr(nav_command, "use_direct_pose", False))
        use_external_motion = bool(getattr(nav_command, "use_external_motion", False))
        stop = bool(getattr(nav_command, "stop", False))
        velocity = resolve_nav_velocity(getattr(nav_command, "velocity", None), default=1.0)

        if use_external_motion and direct_pose is not None:
            external_velocity = _goal_or_none(
                getattr(nav_command, "external_velocity", None)
            )
            if external_velocity is None:
                carb.log_error(
                    f"External motion command has no valid world velocity: {nav_command.path}"
                )
                accepted = False
                continue
            try:
                person.set_external_motion(
                    direct_pose,
                    external_velocity,
                    yaw=float(getattr(nav_command, "orientation", 0.0) or 0.0),
                    timeout_sec=float(
                        getattr(nav_command, "external_timeout_sec", 0.5) or 0.5
                    ),
                    freeze_pose=bool(
                        getattr(nav_command, "external_freeze_pose", False)
                    ),
                    motion_mode=int(
                        getattr(nav_command, "external_motion_mode", 0)
                    ),
                )
            except (TypeError, ValueError) as exc:
                carb.log_error(
                    f"Invalid external motion command for {nav_command.path}: {exc}"
                )
                accepted = False
                continue
            continue

        if use_direct_pose and direct_pose is not None:
            person.set_direct_pose(
                direct_pose,
                yaw=float(getattr(nav_command, "orientation", 0.0) or 0.0),
                stop=stop,
            )
            carb.log_info(
                f"Direct pose pedestrian {nav_command.path}: pose={direct_pose}, "
                f"yaw={float(getattr(nav_command, 'orientation', 0.0) or 0.0):.3f}, stop={stop}"
            )
            continue

        # Prefer navmesh paths when available; fall back to direct PathPoints.
        path_points = _flat_points_to_list3(getattr(nav_command, "path_points_flat", None))
        goal = _goal3(nav_command.goal_pose)
        if stop and not path_points:
            person.stop_motion()
            carb.log_info(f"Stop pedestrian {nav_command.path}")
            continue

        navmesh_points = None
        try:
            inav = nav.acquire_interface()
            navmesh = inav.get_navmesh()
            if navmesh and not path_points:
                start = [float(x) for x in person._state.position.tolist()]
                navmesh_path = navmesh.query_shortest_path(start, goal)
                if navmesh_path:
                    raw_points = navmesh_path.get_points()
                    navmesh_points = [[float(p[0]), float(p[1]), float(p[2])] for p in raw_points]
        except Exception as exc:
            carb.log_warn(f"NavMesh path query failed; using direct PathPoints: {exc}")

        if not path_points:
            path_points = navmesh_points or [goal]

        generation = person.update_target_position(
            path_points,
            velocity,
            loop=bool(getattr(nav_command, "loop_path", False)),
            yaw=float(getattr(nav_command, "orientation", 0.0) or 0.0),
            constrain_to_path=bool(getattr(nav_command, "constrain_to_path", False)),
        )
        carb.log_info(
            f"Move pedestrian {nav_command.path}: goal={goal}, velocity={velocity}, "
            f"points={len(path_points)}, "
            f"constrain_to_path={bool(getattr(nav_command, 'constrain_to_path', False))}, "
            f"command_generation={generation}, "
            f"execution={'ready' if person.anim_graph_ready else 'pending_animgraph'}"
        )

    response.ret = bool(accepted)
    return response


def move_ped(controller):
    service = controller.create_service(
        srv_type=MovePed,
        qos_profile=profile,
        srv_name="isaac/move_pedestrians",
        callback=move_pedestrian,
    )
    return service
