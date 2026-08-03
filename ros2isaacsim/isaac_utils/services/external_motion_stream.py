from __future__ import annotations

import json
import math
import os
import time
import threading
from dataclasses import dataclass, field

try:
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
except Exception:  # pragma: no cover - keep the bridge importable in pure pytest.
    class HistoryPolicy:  # type: ignore[override]
        KEEP_LAST = "KEEP_LAST"

    class ReliabilityPolicy:  # type: ignore[override]
        BEST_EFFORT = "BEST_EFFORT"

    class DurabilityPolicy:  # type: ignore[override]
        VOLATILE = "VOLATILE"

    class QoSProfile:  # type: ignore[override]
        def __init__(
            self,
            *,
            history=None,
            depth=1,
            reliability=None,
            durability=None,
        ):
            self.history = history
            self.depth = depth
            self.reliability = reliability
            self.durability = durability

try:
    from std_msgs.msg import String
except Exception:  # pragma: no cover - keep the bridge importable in pure pytest.
    class String:  # type: ignore[override]
        def __init__(self, data: str = ""):
            self.data = data

try:
    from .MoveCharacters import _find_person as _move_find_person
except Exception:  # pragma: no cover - MoveCharacters depends on Isaac runtime.
    _move_find_person = None


DEFAULT_EXTERNAL_MOTION_STREAM_TOPIC = os.environ.get(
    "ARENA_ISAAC_EXTERNAL_MOTION_STREAM_TOPIC",
    "/isaac/pedestrian_external_motion",
)


def _latest_only_qos_profile() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def _find_person(agent_id: str):
    if callable(_move_find_person):
        return _move_find_person(agent_id)
    return None


def _as_vec3(value):
    try:
        vector = [float(value[0]), float(value[1]), float(value[2])]
    except Exception:
        return None
    if not all(math.isfinite(component) for component in vector):
        return None
    return vector


def _as_int(value, default: int | None = None):
    try:
        return int(value)
    except Exception:
        return default


def _as_float(value, default: float | None = None):
    try:
        return float(value)
    except Exception:
        return default


def _external_motion_flush_hz() -> float:
    return max(1e-3, _as_float(os.environ.get("ARENA_ISAAC_EXTERNAL_MOTION_FLUSH_HZ", "60.0"), 60.0) or 60.0)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _controller_logger(controller):
    try:
        return controller.get_logger()
    except Exception:
        return None


@dataclass(frozen=True)
class CachedExternalMotionCommand:
    stream_id: str
    agent_id: str
    sequence: int
    received_at: float
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    yaw: float
    timeout_sec: float
    freeze_pose: bool
    motion_mode: int

    def age_sec(self, now: float) -> float:
        return max(0.0, float(now) - float(self.received_at))

    def expired(self, now: float) -> bool:
        return self.age_sec(now) >= float(self.timeout_sec)

    def remaining_timeout_sec(self, now: float) -> float:
        return max(0.0, float(self.timeout_sec) - self.age_sec(now))


@dataclass
class ExternalMotionStreamBridge:
    controller: object
    topic_name: str = DEFAULT_EXTERNAL_MOTION_STREAM_TOPIC
    qos_profile: QoSProfile = field(default_factory=_latest_only_qos_profile)

    def __post_init__(self):
        self._subscription = None
        self._flush_timer = None
        self._latest_command_by_agent: dict[str, CachedExternalMotionCommand] = {}
        self._latest_sequence_by_stream_agent: dict[tuple[str, str], int] = {}
        self._last_dispatched_sequence_by_key: dict[tuple[str, str], int] = {}
        self._expired_logged_by_key: dict[tuple[str, str], int] = {}
        self._last_dispatch_log_at_by_agent: dict[str, float] = {}
        self._lock = threading.Lock()

        create_subscription = getattr(self.controller, "create_subscription", None)
        if not callable(create_subscription):
            self._log_warn(
                "Controller has no create_subscription; external motion stream disabled."
            )
            return

        self._subscription = create_subscription(
            String,
            self.topic_name,
            self._on_message,
            self.qos_profile,
        )
        self._log_info(
            f"Subscribed to latest-only external motion stream on {self.topic_name}."
        )

        create_timer = getattr(self.controller, "create_timer", None)
        if callable(create_timer):
            try:
                flush_hz = _external_motion_flush_hz()
                period_sec = 1.0 / flush_hz
                self._flush_timer = create_timer(period_sec, self._flush_pending_commands)
                self._log_info(
                    "External motion stream cache flush timer enabled at "
                    f"{flush_hz:.1f} Hz."
                )
            except Exception as exc:
                self._log_warn(
                    f"Failed to create external motion cache flush timer; "
                    f"falling back to inline flushes: {exc}"
                )
        else:
            self._log_warn(
                "Controller has no create_timer; external motion cache will flush inline."
            )

    def _on_message(self, msg):
        payload = self._decode_payload(msg)
        if payload is None:
            return

        version = _as_int(payload.get("version", 1))
        if version != 1:
            self._log_warn(f"Ignoring unsupported external motion stream version: {version}")
            return

        stream_id = str(payload.get("stream_id", "")).strip()
        if not stream_id:
            self._log_warn("Ignoring external motion stream payload without stream_id.")
            return

        sequence = _as_int(payload.get("sequence"))
        if sequence is None:
            self._log_warn(
                f"Ignoring external motion stream payload for {stream_id} without sequence."
            )
            return

        commands = payload.get("commands")
        if not isinstance(commands, list):
            self._log_warn(
                f"Ignoring external motion stream payload for {stream_id} with non-list commands."
            )
            return

        for command in commands:
            self._cache_command(stream_id, sequence, command)

        if self._flush_timer is None:
            self._flush_pending_commands()

    def _decode_payload(self, msg):
        raw = getattr(msg, "data", None)
        if raw is None:
            self._log_warn("Ignoring external motion stream message without data.")
            return None
        if not isinstance(raw, str):
            raw = str(raw)
        try:
            payload = json.loads(raw)
        except Exception as exc:
            self._log_warn(f"Failed to decode external motion stream JSON: {exc}")
            return None
        if not isinstance(payload, dict):
            self._log_warn("Ignoring external motion stream payload that is not an object.")
            return None
        return payload

    def _cache_command(self, stream_id: str, sequence: int, command):
        if not isinstance(command, dict):
            self._log_warn(
                f"Ignoring external motion command with non-object payload: stream_id={stream_id} "
                f"sequence={sequence}"
            )
            return

        agent_id = str(command.get("agent_id", "")).strip()
        if not agent_id:
            self._log_warn(
                f"Ignoring external motion command without agent_id: stream_id={stream_id} "
                f"sequence={sequence}"
            )
            return

        cache_key = (stream_id, agent_id)
        command_sequence = _as_int(command.get("sequence"), sequence)
        if command_sequence is None:
            self._log_error(
                f"Invalid external motion sequence for agent_id={agent_id} "
                f"stream_id={stream_id} sequence={sequence}"
            )
            return

        with self._lock:
            latest_sequence = self._latest_sequence_by_stream_agent.get(cache_key)
            if latest_sequence is not None and command_sequence <= latest_sequence:
                self._log_info(
                    f"Dropping stale external motion cache stream_id={stream_id} "
                    f"agent_id={agent_id} sequence={command_sequence}; "
                    f"latest={latest_sequence}"
                )
                return
            self._latest_sequence_by_stream_agent[cache_key] = command_sequence
            if _as_bool(command.get("cancel", False)):
                self._latest_command_by_agent.pop(agent_id, None)
                self._last_dispatched_sequence_by_key.pop(cache_key, None)
                self._expired_logged_by_key.pop(cache_key, None)
                return

        position = _as_vec3(command.get("position"))
        velocity = _as_vec3(command.get("velocity"))
        yaw = _as_float(command.get("yaw"))
        timeout_sec = _as_float(command.get("timeout_sec"), 0.5)
        motion_mode = _as_int(command.get("motion_mode"), 0)
        freeze_pose = _as_bool(command.get("freeze_pose", False))

        if (
            position is None
            or velocity is None
            or yaw is None
            or timeout_sec is None
        ):
            self._log_error(
                f"Invalid external motion command for agent_id={agent_id} "
                f"stream_id={stream_id} sequence={sequence}"
            )
            return

        received_at = time.monotonic()
        cached_command = CachedExternalMotionCommand(
            stream_id=stream_id,
            agent_id=agent_id,
            sequence=command_sequence,
            received_at=received_at,
            position=position,
            velocity=velocity,
            yaw=yaw,
            timeout_sec=timeout_sec,
            freeze_pose=freeze_pose,
            motion_mode=motion_mode if motion_mode is not None else 0,
        )

        with self._lock:
            # A restarted director has a new stream id. New receipt wins so an
            # old stream can never continue driving the same character.
            self._latest_command_by_agent[agent_id] = cached_command

    def _flush_pending_commands(self, *_args):
        now = time.monotonic()
        with self._lock:
            pending = [
                command
                for command in self._latest_command_by_agent.values()
                if command.sequence
                > self._last_dispatched_sequence_by_key.get(
                    (command.stream_id, command.agent_id),
                    -1,
                )
            ]

        for command in sorted(
            pending,
            key=lambda item: (item.received_at, item.agent_id, item.stream_id),
        ):
            age_sec = command.age_sec(now)
            expired = command.expired(now)
            command_key = (command.stream_id, command.agent_id)
            if expired:
                with self._lock:
                    already_logged = (
                        self._expired_logged_by_key.get(command_key)
                        == command.sequence
                    )
                    self._expired_logged_by_key[command_key] = command.sequence
                if not already_logged:
                    self._log_info(
                        f"Skipping expired external motion command agent_id={command.agent_id} "
                        f"stream_id={command.stream_id} sequence={command.sequence} "
                        f"age_sec={age_sec:.3f} expired={expired}"
                    )
                continue

            person = _find_person(command.agent_id)
            if person is None or not hasattr(person, "set_external_motion"):
                self._log_error(
                    f"Person not found for external motion stream agent_id={command.agent_id} "
                    f"stream_id={command.stream_id} sequence={command.sequence} "
                    f"age_sec={age_sec:.3f} expired={expired}"
                )
                continue

            remaining_timeout_sec = command.remaining_timeout_sec(now)
            if remaining_timeout_sec <= 0.0:
                self._log_info(
                    f"Skipping stale external motion command agent_id={command.agent_id} "
                    f"stream_id={command.stream_id} sequence={command.sequence} "
                    f"age_sec={age_sec:.3f} expired=True"
                )
                with self._lock:
                    self._last_dispatched_sequence_by_key[
                        (command.stream_id, command.agent_id)
                    ] = command.sequence
                continue

            try:
                position = tuple(
                    float(command.position[index])
                    + float(command.velocity[index]) * age_sec
                    for index in range(3)
                )
                person.set_external_motion(
                    position,
                    command.velocity,
                    command.yaw,
                    timeout_sec=remaining_timeout_sec,
                    freeze_pose=command.freeze_pose,
                    motion_mode=command.motion_mode,
                )
                external_motion = getattr(person, "_external_motion", None)
                if external_motion is not None and hasattr(external_motion, "set_command"):
                    try:
                        external_motion.set_command(
                            position=position,
                            velocity=command.velocity,
                            yaw=command.yaw,
                            received_at=now,
                            timeout_sec=remaining_timeout_sec,
                            sequence=command.sequence,
                        )
                    except Exception as exc:
                        self._log_warn(
                            f"Failed to sync external motion state for agent_id={command.agent_id} "
                            f"stream_id={command.stream_id} sequence={command.sequence}: {exc}"
                        )
            except Exception as exc:
                self._log_error(
                    f"Failed to dispatch external motion command for agent_id={command.agent_id} "
                    f"stream_id={command.stream_id} sequence={command.sequence} "
                    f"age_sec={age_sec:.3f} expired={expired}: {exc}"
                )
                continue

            with self._lock:
                first_dispatch = (
                    self._last_dispatched_sequence_by_key.get(command_key)
                    != command.sequence
                )
                self._last_dispatched_sequence_by_key[command_key] = command.sequence
            last_log_at = self._last_dispatch_log_at_by_agent.get(
                command.agent_id,
                -math.inf,
            )
            if first_dispatch and now - last_log_at >= 2.0:
                self._last_dispatch_log_at_by_agent[command.agent_id] = now
                self._log_info(
                    f"Dispatched external motion command agent_id={command.agent_id} "
                    f"stream_id={command.stream_id} sequence={command.sequence} "
                    f"age_sec={age_sec:.3f} expired={expired} "
                    f"timeout_sec={remaining_timeout_sec:.3f}"
                )

    def _log_info(self, message: str):
        logger = _controller_logger(self.controller)
        if logger is not None:
            try:
                logger.info(message)
                return
            except Exception:
                pass
        print(message, flush=True)

    def _log_warn(self, message: str):
        logger = _controller_logger(self.controller)
        if logger is not None:
            try:
                logger.warning(message)
                return
            except Exception:
                pass
        print(message, flush=True)

    def _log_error(self, message: str):
        logger = _controller_logger(self.controller)
        if logger is not None:
            try:
                logger.error(message)
                return
            except Exception:
                pass
        print(message, flush=True)


def start_external_motion_stream(controller):
    return ExternalMotionStreamBridge(controller=controller)


__all__ = [
    "DEFAULT_EXTERNAL_MOTION_STREAM_TOPIC",
    "DurabilityPolicy",
    "ExternalMotionStreamBridge",
    "HistoryPolicy",
    "QoSProfile",
    "ReliabilityPolicy",
    "String",
    "start_external_motion_stream",
]
