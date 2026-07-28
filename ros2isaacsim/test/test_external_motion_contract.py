import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestExternalMotionContract(unittest.TestCase):
    def test_move_service_routes_external_commands_before_direct_pose(self):
        source = (
            ROOT / "isaac_utils" / "services" / "MoveCharacters.py"
        ).read_text(encoding="utf-8")

        external_branch = source.index("if use_external_motion and direct_pose is not None:")
        direct_branch = source.index("if use_direct_pose and direct_pose is not None:")

        self.assertLess(external_branch, direct_branch)
        self.assertIn("person.set_external_motion(", source)
        self.assertIn("external_freeze_pose", source)

    def test_person_update_checks_external_authority_before_behavior_script(self):
        source = (
            ROOT
            / "pedestrian"
            / "simulator"
            / "logic"
            / "people"
            / "person.py"
        ).read_text(encoding="utf-8")
        update_source = source[source.index("    def update(self, dt: float):") :]

        external_branch = update_source.index(
            "external_sample = self._external_motion.sample"
        )
        behavior_branch = update_source.index("if self._behavior_script_enabled:")

        self.assertLess(external_branch, behavior_branch)
        self.assertIn("self._external_motion.clear()", source)
        self.assertIn('"PathPoints"', source)
        self.assertIn('"Action", "None"', source)
        self.assertIn("[carb.Float3(*point) for point in points]", source)
        self.assertIn("animation_tracking_sample", source)
        self.assertIn("_external_motion_hard_sync_distance_m", source)
        self.assertIn("self._external_hold_position", source)
        self.assertIn("self._apply_external_hold_pose()", update_source)
        self.assertIn("if freeze_pose:", source)

    def test_stationary_external_hold_has_one_transform_owner(self):
        source = (
            ROOT
            / "pedestrian"
            / "simulator"
            / "logic"
            / "people"
            / "person.py"
        ).read_text(encoding="utf-8")
        hold_source = source[
            source.index("    def _apply_external_hold_pose(self)") :
            source.index("    def _external_animation_sample", source.index("    def _apply_external_hold_pose(self)"))
        ]

        self.assertIn("self.character_graph.set_world_transform(", hold_source)
        self.assertNotIn("self._set_stage_root_pose(", hold_source)
        self.assertIn("self._state.orientation = orientation", hold_source)


if __name__ == "__main__":
    unittest.main()
