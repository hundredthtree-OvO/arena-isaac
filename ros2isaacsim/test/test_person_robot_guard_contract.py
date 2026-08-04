import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestPersonRobotGuardContract(unittest.TestCase):
    @staticmethod
    def _person_source() -> str:
        return (
            ROOT
            / "pedestrian"
            / "simulator"
            / "logic"
            / "people"
            / "person.py"
        ).read_text(encoding="utf-8")

    def test_detect_only_disables_predictive_and_root_motion_robot_blockers(self):
        source = self._person_source()
        blocker_source = source[
            source.index("    def _candidate_robot_pose_blocker") :
            source.index("    def _restore_last_safe_pose")
        ]

        self.assertIn("not self._runtime_robot_interaction_enabled()", blocker_source)
        self.assertIn("if not self._runtime_robot_interaction_enabled():", source)

if __name__ == "__main__":
    unittest.main()
