import unittest
import math
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "isaac_utils"
    / "services"
    / "pedestrian_state_utils.py"
)
SPEC = spec_from_file_location("pedestrian_state_utils", MODULE_PATH)
PEDESTRIAN_STATE_UTILS = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PEDESTRIAN_STATE_UTILS)
iter_unique_people = PEDESTRIAN_STATE_UTILS.iter_unique_people
estimate_pedestrian_velocity = PEDESTRIAN_STATE_UTILS.estimate_pedestrian_velocity
pedestrian_state_publishable = PEDESTRIAN_STATE_UTILS.pedestrian_state_publishable
pedestrian_state_reliable = PEDESTRIAN_STATE_UTILS.pedestrian_state_reliable
pedestrian_state_tags = PEDESTRIAN_STATE_UTILS.pedestrian_state_tags
pedestrian_state_tagnames = PEDESTRIAN_STATE_UTILS.pedestrian_state_tagnames
pedestrian_state_yaw = PEDESTRIAN_STATE_UTILS.pedestrian_state_yaw
stable_person_name = PEDESTRIAN_STATE_UTILS.stable_person_name


class _DummyPerson:
    def __init__(
        self,
        requested="",
        stage_prefix="",
        skelroot="",
        pose_valid=False,
        motion_state="unknown",
        motion_generation=0,
        guard_blocked=False,
        guard_block_generation=0,
        guard_block_count=0,
        guard_block_reason="",
        active=True,
        parked=False,
        embodiment_generation=0,
        reactivation_ready=False,
        orientation=(0.0, 0.0, 0.0, 1.0),
    ):
        self._requested_stage_name = requested
        self._stage_prefix = stage_prefix
        self.character_skel_root_stage_path = skelroot
        self._pose_valid = pose_valid
        self._motion_state = motion_state
        self._motion_command_generation = motion_generation
        self._guard_blocked = guard_blocked
        self._guard_block_generation = guard_block_generation
        self._guard_block_count = guard_block_count
        self._guard_block_reason = guard_block_reason
        self._active = active
        self.is_parked = parked
        self.embodiment_generation = embodiment_generation
        self.reactivation_ready = reactivation_ready
        self._state = type("_State", (), {"orientation": orientation})()


class TestPedestrianStateUtils(unittest.TestCase):
    def test_stable_person_name_prefers_requested_stage_name(self):
        person = _DummyPerson(requested="toilet_agent_01", stage_prefix="/World/Characters/ignored")
        self.assertEqual(stable_person_name(person), "toilet_agent_01")

    def test_stable_person_name_falls_back_to_stage_leaf(self):
        person = _DummyPerson(stage_prefix="/World/Characters/toilet_agent_02")
        self.assertEqual(stable_person_name(person), "toilet_agent_02")

    def test_iter_unique_people_deduplicates_aliases(self):
        person = _DummyPerson(requested="toilet_agent_03")
        aliases = {
            "toilet_agent_03": person,
            "/World/Characters/toilet_agent_03": person,
        }
        unique = list(iter_unique_people(aliases))
        self.assertEqual(len(unique), 1)
        self.assertEqual(unique[0][0], "toilet_agent_03")

    def test_parked_people_are_not_publishable(self):
        self.assertTrue(pedestrian_state_publishable(_DummyPerson()))
        self.assertFalse(pedestrian_state_publishable(_DummyPerson(parked=True)))
        self.assertFalse(pedestrian_state_publishable(_DummyPerson(active=False)))

    def test_idle_person_keeps_reliable_last_pose_during_read_gap(self):
        self.assertTrue(
            pedestrian_state_reliable(
                _DummyPerson(pose_valid=False, motion_state="idle")
            )
        )
        self.assertFalse(
            pedestrian_state_reliable(
                _DummyPerson(pose_valid=False, motion_state="executing")
            )
        )

    def test_pedestrian_state_tags_append_guard_fields_without_removing_existing_ones(self):
        person = _DummyPerson(
            stage_prefix="/World/Characters/toilet_agent_04",
            skelroot="/World/Characters/toilet_agent_04/SkelRoot",
            pose_valid=True,
            motion_state="executing",
            motion_generation=7,
            guard_blocked=True,
            guard_block_generation=7,
            guard_block_count=3,
            guard_block_reason="robot",
            embodiment_generation=4,
            reactivation_ready=True,
        )
        self.assertEqual(
            pedestrian_state_tagnames(),
            (
                "isaac",
                "pedestrian_state",
                "pose_valid",
                "motion_state",
                "command_generation",
                "guard_blocked",
                "guard_block_generation",
                "guard_block_count",
                "guard_block_reason",
                "yaw_rad",
                "yaw_valid",
                "embodiment_generation",
                "reactivation_ready",
            ),
        )
        self.assertEqual(
            pedestrian_state_tags(person),
            [
                "/World/Characters/toilet_agent_04",
                "/World/Characters/toilet_agent_04/SkelRoot",
                "true",
                "executing",
                "7",
                "true",
                "7",
                "3",
                "robot",
                "0.000000000",
                "true",
                "4",
                "true",
            ],
        )

    def test_yaw_is_extracted_from_xyzw_orientation(self):
        half_yaw = math.pi / 4.0
        person = _DummyPerson(
            orientation=(0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))
        )

        self.assertAlmostEqual(pedestrian_state_yaw(person), math.pi / 2.0)

    def test_velocity_is_estimated_from_consecutive_valid_positions(self):
        velocity = estimate_pedestrian_velocity(
            (1.0, (1.0, 2.0, 0.0)),
            (1.2, 1.8, 0.0),
            1.2,
        )

        self.assertAlmostEqual(velocity[0], 1.0)
        self.assertAlmostEqual(velocity[1], -1.0)
        self.assertAlmostEqual(velocity[2], 0.0)

    def test_velocity_resets_after_large_sample_gap(self):
        self.assertEqual(
            estimate_pedestrian_velocity(
                (1.0, (1.0, 2.0, 0.0)),
                (2.0, 3.0, 0.0),
                3.0,
            ),
            (0.0, 0.0, 0.0),
        )

    def test_velocity_rejects_pool_reactivation_teleport(self):
        self.assertEqual(
            estimate_pedestrian_velocity(
                (1.0, (1000.0, 1000.0, 0.0)),
                (-3.8, -0.9, 0.0),
                1.1,
            ),
            (0.0, 0.0, 0.0),
        )


if __name__ == "__main__":
    unittest.main()
