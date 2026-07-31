import unittest

from isaac_utils.pedestrian_pool import park_all_people


class _Person:
    def __init__(self, stage_prefix):
        self._stage_prefix = stage_prefix
        self.parked_at = None

    def park(self, pose):
        self.parked_at = list(pose)


class _Manager:
    def __init__(self, people):
        self.people = people


class DeleteAllCharactersTests(unittest.TestCase):
    def test_parks_unique_people_without_clearing_aliases(self):
        first = _Person("/World/Characters/toilet_agent_01")
        second = _Person("/World/Characters/toilet_agent_02")
        aliases = {
            "toilet_agent_01": first,
            first._stage_prefix: first,
            "toilet_agent_02": second,
        }
        manager = _Manager(aliases)

        parked = park_all_people(manager)

        self.assertEqual(parked, 2)
        self.assertIsNotNone(first.parked_at)
        self.assertIsNotNone(second.parked_at)
        self.assertEqual(manager.people, aliases)


if __name__ == "__main__":
    unittest.main()
