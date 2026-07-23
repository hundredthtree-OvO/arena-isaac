import unittest
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
pedestrian_state_publishable = PEDESTRIAN_STATE_UTILS.pedestrian_state_publishable
pedestrian_state_tags = PEDESTRIAN_STATE_UTILS.pedestrian_state_tags
pedestrian_state_tagnames = PEDESTRIAN_STATE_UTILS.pedestrian_state_tagnames
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
            ],
        )


if __name__ == "__main__":
    unittest.main()
