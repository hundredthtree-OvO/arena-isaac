from __future__ import annotations

import math
from dataclasses import dataclass

from pedestrian.simulator.logic.people_manager import PeopleManager

try:
    from geometry_msgs.msg import Point
    from people_msgs.msg import People, Person as PeoplePerson
except Exception:  # pragma: no cover - optional runtime dependency
    Point = None
    People = None
    PeoplePerson = None

from .pedestrian_state_utils import iter_unique_people


def _valid_position(position) -> bool:
    try:
        values = [float(position[0]), float(position[1]), float(position[2])]
    except Exception:
        return False
    return all(math.isfinite(value) for value in values)


@dataclass
class PedestrianStatePublisher:
    controller: object
    topic_name: str = "/isaac/pedestrian_states"
    publish_hz: float = 10.0

    def __post_init__(self):
        self._publisher = None
        self._timer = None
        if People is None or PeoplePerson is None or Point is None:
            self._log_warn("people_msgs is unavailable; /isaac/pedestrian_states will not be published.")
            return
        self._publisher = self.controller.create_publisher(People, self.topic_name, 10)
        period = 1.0 / max(float(self.publish_hz), 1e-3)
        self._timer = self.controller.create_timer(period, self._publish)
        self._log_info(
            f"Publishing live pedestrian states on {self.topic_name} at {float(self.publish_hz):.1f} Hz."
        )

    def _publish(self):
        if self._publisher is None:
            return
        msg = People()
        try:
            msg.header.stamp = self.controller.get_clock().now().to_msg()
            msg.header.frame_id = "map"
        except Exception:
            pass

        manager = PeopleManager.get_people_manager()
        for name, person in iter_unique_people(getattr(manager, "people", {}) or {}):
            if not bool(getattr(person, "_active", True)):
                continue
            state = getattr(person, "_state", None)
            position = getattr(state, "position", None)
            if position is None or len(position) < 3 or not _valid_position(position):
                continue
            entry = PeoplePerson()
            entry.name = str(name)
            entry.position = Point(x=float(position[0]), y=float(position[1]), z=float(position[2]))
            entry.velocity = Point(x=0.0, y=0.0, z=0.0)
            entry.reliability = 1.0
            entry.tagnames = ["isaac", "pedestrian_state"]
            entry.tags = [
                str(getattr(person, "_stage_prefix", "") or ""),
                str(getattr(person, "character_skel_root_stage_path", "") or ""),
            ]
            msg.people.append(entry)

        self._publisher.publish(msg)

    def _log_info(self, message: str):
        try:
            self.controller.get_logger().info(message)
        except Exception:
            print(message, flush=True)

    def _log_warn(self, message: str):
        try:
            self.controller.get_logger().warning(message)
        except Exception:
            print(message, flush=True)


def start_pedestrian_state_publisher(controller):
    return PedestrianStatePublisher(controller=controller)
