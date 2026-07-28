from __future__ import annotations

import math
from dataclasses import dataclass
import time

from pedestrian.simulator.logic.people_manager import PeopleManager

try:
    from geometry_msgs.msg import Point
    from people_msgs.msg import People, Person as PeoplePerson
except Exception:  # pragma: no cover - optional runtime dependency
    Point = None
    People = None
    PeoplePerson = None

from .pedestrian_state_utils import (
    estimate_pedestrian_velocity,
    iter_unique_people,
    pedestrian_state_publishable,
    pedestrian_state_tags,
    pedestrian_state_tagnames,
)


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
        self._previous_states = {}
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
        observed_at_sec = time.monotonic()
        try:
            now = self.controller.get_clock().now()
            msg.header.stamp = now.to_msg()
            observed_at_sec = float(now.nanoseconds) * 1e-9
            msg.header.frame_id = "map"
        except Exception:
            pass

        manager = PeopleManager.get_people_manager()
        current_names = set()
        for name, person in iter_unique_people(getattr(manager, "people", {}) or {}):
            if not pedestrian_state_publishable(person):
                continue
            current_names.add(str(name))
            state = getattr(person, "_state", None)
            position = getattr(state, "position", None)
            if position is None or len(position) < 3 or not _valid_position(position):
                continue
            entry = PeoplePerson()
            entry.name = str(name)
            entry.position = Point(x=float(position[0]), y=float(position[1]), z=float(position[2]))
            pose_valid = bool(getattr(person, "_pose_valid", False))
            motion_state = str(
                getattr(person, "_motion_state", "unknown")
            ).strip().lower()
            velocity = (0.0, 0.0, 0.0)
            if pose_valid and motion_state in {"accepted", "executing"}:
                velocity = estimate_pedestrian_velocity(
                    self._previous_states.get(str(name)),
                    position,
                    observed_at_sec,
                )
            if pose_valid:
                self._previous_states[str(name)] = (
                    observed_at_sec,
                    tuple(float(position[index]) for index in range(3)),
                )
            entry.velocity = Point(x=velocity[0], y=velocity[1], z=velocity[2])
            entry.reliability = 1.0 if pose_valid else 0.0
            entry.tagnames = list(pedestrian_state_tagnames())
            entry.tags = pedestrian_state_tags(person)
            msg.people.append(entry)

        self._previous_states = {
            name: state
            for name, state in self._previous_states.items()
            if name in current_names
        }
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
