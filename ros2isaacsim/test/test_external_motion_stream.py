import json
import sys
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "isaac_utils" / "services" / "external_motion_stream.py"
SPEC = spec_from_file_location("external_motion_stream", MODULE_PATH)
EXTERNAL_MOTION_STREAM = module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = EXTERNAL_MOTION_STREAM
SPEC.loader.exec_module(EXTERNAL_MOTION_STREAM)


class _FakeLogger:
    def __init__(self):
        self.info_messages = []
        self.warning_messages = []
        self.error_messages = []

    def info(self, *_args, **_kwargs):
        self.info_messages.append(_args[0] if _args else "")
        return None

    def warning(self, *_args, **_kwargs):
        self.warning_messages.append(_args[0] if _args else "")
        return None

    def error(self, *_args, **_kwargs):
        self.error_messages.append(_args[0] if _args else "")
        return None


class _FakeController:
    def __init__(self):
        self.subscriptions = []
        self.timers = []
        self.logger = _FakeLogger()

    def create_subscription(self, msg_type, topic_name, callback, qos_profile):
        self.subscriptions.append((msg_type, topic_name, callback, qos_profile))
        return object()

    def create_timer(self, period_sec, callback):
        timer = object()
        self.timers.append((period_sec, callback, timer))
        return timer

    def get_logger(self):
        return self.logger


class _FakeExternalMotion:
    def __init__(self):
        self.calls = []

    def set_command(
        self,
        *,
        position,
        velocity,
        yaw,
        received_at,
        timeout_sec,
        sequence=None,
    ):
        self.calls.append(
            {
                "position": tuple(position),
                "velocity": tuple(velocity),
                "yaw": yaw,
                "received_at": received_at,
                "timeout_sec": timeout_sec,
                "sequence": sequence,
            }
        )


class _FakePerson:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.calls = []
        self._external_motion = _FakeExternalMotion()

    def set_external_motion(
        self,
        position,
        velocity,
        yaw,
        *,
        timeout_sec=0.5,
        freeze_pose=False,
        motion_mode=0,
    ):
        self.calls.append(
            {
                "position": tuple(position),
                "velocity": tuple(velocity),
                "yaw": yaw,
                "timeout_sec": timeout_sec,
                "freeze_pose": freeze_pose,
                "motion_mode": motion_mode,
            }
        )


class _Msg:
    def __init__(self, data):
        self.data = data


class TestExternalMotionStream(unittest.TestCase):
    def test_subscription_uses_latest_only_best_effort_qos(self):
        controller = _FakeController()

        bridge = EXTERNAL_MOTION_STREAM.start_external_motion_stream(controller)

        self.assertIsNotNone(bridge)
        self.assertEqual(len(controller.subscriptions), 1)
        self.assertEqual(len(controller.timers), 1)
        msg_type, topic_name, callback, qos_profile = controller.subscriptions[0]
        self.assertEqual(msg_type, EXTERNAL_MOTION_STREAM.String)
        self.assertEqual(
            topic_name,
            EXTERNAL_MOTION_STREAM.DEFAULT_EXTERNAL_MOTION_STREAM_TOPIC,
        )
        self.assertTrue(callable(callback))
        self.assertEqual(qos_profile.depth, 1)
        self.assertEqual(
            qos_profile.history,
            EXTERNAL_MOTION_STREAM.HistoryPolicy.KEEP_LAST,
        )
        self.assertEqual(
            qos_profile.reliability,
            EXTERNAL_MOTION_STREAM.ReliabilityPolicy.BEST_EFFORT,
        )
        self.assertGreater(controller.timers[0][0], 0.0)

    def test_latest_cache_keeps_fresh_late_agent_updates_per_agent(self):
        controller = _FakeController()
        persons = {
            "alice": _FakePerson("alice"),
            "bob": _FakePerson("bob"),
        }

        def _lookup(agent_id):
            return persons.get(agent_id)

        with mock.patch.object(
            EXTERNAL_MOTION_STREAM,
            "_move_find_person",
            side_effect=_lookup,
        ), mock.patch.object(
            EXTERNAL_MOTION_STREAM.time,
            "monotonic",
            side_effect=[10.0, 10.1, 10.6, 10.7, 10.8],
        ):
            bridge = EXTERNAL_MOTION_STREAM.start_external_motion_stream(controller)
            callback = controller.subscriptions[0][2]
            flush = controller.timers[0][1]

            callback(
                _Msg(
                    json.dumps(
                        {
                            "version": 1,
                            "stream_id": "stream-a",
                            "sequence": 7,
                            "commands": [
                                {
                                    "agent_id": "alice",
                                    "position": [1.0, 2.0, 3.0],
                                    "velocity": [0.25, -0.5, 0.0],
                                    "yaw": 0.4,
                                    "timeout_sec": 1.25,
                                    "freeze_pose": True,
                                    "motion_mode": 3,
                                }
                            ],
                        },
                        sort_keys=True,
                    )
                )
            )
            flush()

            callback(
                _Msg(
                    json.dumps(
                        {
                            "version": 1,
                            "stream_id": "stream-a",
                            "sequence": 6,
                            "commands": [
                                {
                                    "agent_id": "alice",
                                    "position": [9.0, 9.0, 9.0],
                                    "velocity": [9.0, 0.0, 0.0],
                                    "yaw": 9.0,
                                    "timeout_sec": 0.5,
                                    "freeze_pose": False,
                                    "motion_mode": 9,
                                },
                                {
                                    "agent_id": "bob",
                                    "position": [4.5, 5.5, 6.5],
                                    "velocity": [0.0, 0.5, 0.0],
                                    "yaw": -0.2,
                                    "timeout_sec": 0.75,
                                    "freeze_pose": False,
                                    "motion_mode": 2,
                                },
                            ],
                        },
                        sort_keys=True,
                    )
                )
            )
            flush()

        self.assertIsNotNone(bridge)
        self.assertEqual(len(persons["alice"].calls), 1)
        self.assertEqual(len(persons["bob"].calls), 1)
        self.assertAlmostEqual(persons["alice"].calls[0]["position"][0], 1.025)
        self.assertAlmostEqual(persons["alice"].calls[0]["position"][1], 1.95)
        self.assertAlmostEqual(persons["alice"].calls[0]["position"][2], 3.0)
        self.assertEqual(persons["alice"].calls[0]["velocity"], (0.25, -0.5, 0.0))
        self.assertAlmostEqual(persons["alice"].calls[0]["yaw"], 0.4)
        self.assertAlmostEqual(persons["alice"].calls[0]["timeout_sec"], 1.15, places=3)
        self.assertTrue(persons["alice"].calls[0]["freeze_pose"])
        self.assertEqual(persons["alice"].calls[0]["motion_mode"], 3)
        self.assertAlmostEqual(persons["bob"].calls[0]["position"][0], 4.5)
        self.assertAlmostEqual(persons["bob"].calls[0]["position"][1], 5.55)
        self.assertAlmostEqual(persons["bob"].calls[0]["position"][2], 6.5)
        self.assertEqual(persons["bob"].calls[0]["velocity"], (0.0, 0.5, 0.0))
        self.assertAlmostEqual(persons["bob"].calls[0]["yaw"], -0.2)
        self.assertAlmostEqual(persons["bob"].calls[0]["timeout_sec"], 0.65, places=3)
        self.assertFalse(persons["bob"].calls[0]["freeze_pose"])
        self.assertEqual(persons["bob"].calls[0]["motion_mode"], 2)
        self.assertEqual(len(persons["alice"]._external_motion.calls), 1)
        self.assertAlmostEqual(persons["alice"]._external_motion.calls[0]["position"][0], 1.025)
        self.assertAlmostEqual(persons["alice"]._external_motion.calls[0]["position"][1], 1.95)
        self.assertAlmostEqual(persons["alice"]._external_motion.calls[0]["position"][2], 3.0)
        self.assertEqual(persons["alice"]._external_motion.calls[0]["velocity"], (0.25, -0.5, 0.0))
        self.assertAlmostEqual(persons["alice"]._external_motion.calls[0]["yaw"], 0.4)
        self.assertAlmostEqual(persons["alice"]._external_motion.calls[0]["received_at"], 10.1)
        self.assertAlmostEqual(persons["alice"]._external_motion.calls[0]["timeout_sec"], 1.15, places=6)
        self.assertEqual(persons["alice"]._external_motion.calls[0]["sequence"], 7)

        self.assertEqual(len(persons["bob"]._external_motion.calls), 1)
        self.assertAlmostEqual(persons["bob"]._external_motion.calls[0]["position"][0], 4.5)
        self.assertAlmostEqual(persons["bob"]._external_motion.calls[0]["position"][1], 5.55)
        self.assertAlmostEqual(persons["bob"]._external_motion.calls[0]["position"][2], 6.5)
        self.assertEqual(persons["bob"]._external_motion.calls[0]["velocity"], (0.0, 0.5, 0.0))
        self.assertAlmostEqual(persons["bob"]._external_motion.calls[0]["yaw"], -0.2)
        self.assertAlmostEqual(persons["bob"]._external_motion.calls[0]["received_at"], 10.7)
        self.assertAlmostEqual(persons["bob"]._external_motion.calls[0]["timeout_sec"], 0.65, places=6)
        self.assertEqual(persons["bob"]._external_motion.calls[0]["sequence"], 6)
        self.assertTrue(
            any(
                "Dropping stale external motion cache stream_id=stream-a agent_id=alice sequence=6"
                in message
                and "latest=7" in message
                for message in controller.logger.info_messages
            )
        )
        self.assertTrue(
            any(
                "Dispatched external motion command agent_id=alice stream_id=stream-a sequence=7"
                in message
                and "expired=False" in message
                for message in controller.logger.info_messages
            )
        )
        self.assertTrue(
            any(
                "Dispatched external motion command agent_id=bob stream_id=stream-a sequence=6"
                in message
                and "expired=False" in message
                for message in controller.logger.info_messages
            )
        )

    def test_invalid_version_and_payloads_are_ignored(self):
        controller = _FakeController()
        person = _FakePerson("alice")

        with mock.patch.object(
            EXTERNAL_MOTION_STREAM,
            "_move_find_person",
            return_value=person,
        ):
            bridge = EXTERNAL_MOTION_STREAM.start_external_motion_stream(controller)
            callback = controller.subscriptions[0][2]
            flush = controller.timers[0][1]
            callback(_Msg("{not json}"))
            callback(
                _Msg(
                    json.dumps(
                        {
                            "version": 2,
                            "stream_id": "stream-a",
                            "sequence": 1,
                            "commands": [
                                {
                                    "agent_id": "alice",
                                    "position": [1.0, 2.0, 3.0],
                                    "velocity": [0.0, 0.0, 0.0],
                                    "yaw": 0.0,
                                    "timeout_sec": 0.5,
                                    "freeze_pose": False,
                                    "motion_mode": 0,
                                }
                            ],
                        },
                        sort_keys=True,
                    )
                )
            )
            callback(
                _Msg(
                    json.dumps(
                        {
                            "version": 1,
                            "stream_id": "stream-a",
                            "sequence": 1,
                            "commands": "not-a-list",
                        },
                        sort_keys=True,
                    )
                )
            )
            flush()

        self.assertIsNotNone(bridge)
        self.assertEqual(person.calls, [])

    def test_expired_cached_command_is_skipped_with_age_diagnostics(self):
        controller = _FakeController()
        person = _FakePerson("alice")

        with mock.patch.object(
            EXTERNAL_MOTION_STREAM,
            "_move_find_person",
            return_value=person,
        ), mock.patch.object(
            EXTERNAL_MOTION_STREAM.time,
            "monotonic",
            side_effect=[100.0, 100.25],
        ):
            bridge = EXTERNAL_MOTION_STREAM.start_external_motion_stream(controller)
            callback = controller.subscriptions[0][2]
            flush = controller.timers[0][1]
            callback(
                _Msg(
                    json.dumps(
                        {
                            "version": 1,
                            "stream_id": "stream-expired",
                            "sequence": 10,
                            "commands": [
                                {
                                    "agent_id": "alice",
                                    "position": [1.0, 2.0, 3.0],
                                    "velocity": [0.1, 0.0, 0.0],
                                    "yaw": 0.4,
                                    "timeout_sec": 0.1,
                                    "freeze_pose": False,
                                    "motion_mode": 0,
                                }
                            ],
                        },
                        sort_keys=True,
                    )
                )
            )

            flush()

        self.assertIsNotNone(bridge)
        self.assertEqual(person.calls, [])
        self.assertTrue(
            any(
                "sequence=10" in message
                and "age_sec=0.250" in message
                and "expired=True" in message
                for message in controller.logger.info_messages
            )
        )

    def test_cancel_removes_cached_command_before_timer_dispatch(self):
        controller = _FakeController()
        person = _FakePerson("alice")

        with mock.patch.object(
            EXTERNAL_MOTION_STREAM,
            "_move_find_person",
            return_value=person,
        ):
            bridge = EXTERNAL_MOTION_STREAM.start_external_motion_stream(controller)
            callback = controller.subscriptions[0][2]
            flush = controller.timers[0][1]
            callback(
                _Msg(
                    json.dumps(
                        {
                            "version": 1,
                            "stream_id": "stream-a",
                            "sequence": 1,
                            "commands": [
                                {
                                    "agent_id": "alice",
                                    "position": [1.0, 2.0, 0.0],
                                    "velocity": [0.2, 0.0, 0.0],
                                    "yaw": 0.0,
                                    "timeout_sec": 1.0,
                                }
                            ],
                        }
                    )
                )
            )
            callback(
                _Msg(
                    json.dumps(
                        {
                            "version": 1,
                            "stream_id": "stream-a",
                            "sequence": 2,
                            "commands": [{"agent_id": "alice", "cancel": True}],
                        }
                    )
                )
            )
            flush()

        self.assertIsNotNone(bridge)
        self.assertEqual(person.calls, [])
        self.assertNotIn("alice", bridge._latest_command_by_agent)

    def test_new_stream_replaces_previous_stream_for_same_agent(self):
        controller = _FakeController()
        person = _FakePerson("alice")

        with mock.patch.object(
            EXTERNAL_MOTION_STREAM,
            "_move_find_person",
            return_value=person,
        ):
            bridge = EXTERNAL_MOTION_STREAM.start_external_motion_stream(controller)
            callback = controller.subscriptions[0][2]
            flush = controller.timers[0][1]
            for stream_id, position in (("old", 1.0), ("new", 2.0)):
                callback(
                    _Msg(
                        json.dumps(
                            {
                                "version": 1,
                                "stream_id": stream_id,
                                "sequence": 1,
                                "commands": [
                                    {
                                        "agent_id": "alice",
                                        "position": [position, 0.0, 0.0],
                                        "velocity": [0.0, 0.0, 0.0],
                                        "yaw": 0.0,
                                        "timeout_sec": 1.0,
                                    }
                                ],
                            }
                        )
                    )
                )
            flush()

        self.assertIsNotNone(bridge)
        self.assertEqual(len(person.calls), 1)
        self.assertEqual(person.calls[0]["position"], (2.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
