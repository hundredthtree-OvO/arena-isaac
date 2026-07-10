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
stable_person_name = PEDESTRIAN_STATE_UTILS.stable_person_name


class _DummyPerson:
    def __init__(self, requested="", stage_prefix=""):
        self._requested_stage_name = requested
        self._stage_prefix = stage_prefix


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


if __name__ == "__main__":
    unittest.main()
