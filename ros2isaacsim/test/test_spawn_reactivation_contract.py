import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestSpawnReactivationContract(unittest.TestCase):
    def test_only_new_characters_pause_the_timeline(self):
        source = (
            ROOT / "isaac_utils" / "services" / "SpawnCharacters.py"
        ).read_text(encoding="utf-8")

        reactivate_index = source.index("existing.reactivate(init_pos, init_yaw)")
        pause_index = source.index("timeline.pause()")
        resume_index = source.index("timeline.play()", reactivate_index)

        self.assertLess(reactivate_index, pause_index)
        self.assertLess(reactivate_index, resume_index)
        self.assertIn("if new_people and was_playing:", source)
        self.assertNotIn("if request.people and was_playing:", source)

    def test_person_rebuilds_runtime_before_dispatching_cached_motion(self):
        source = (
            ROOT
            / "pedestrian"
            / "simulator"
            / "logic"
            / "people"
            / "person.py"
        ).read_text(encoding="utf-8")
        update_start = source.index("    def update(self, dt: float):")
        dispatch_index = source.index("self._dispatch_behavior_commands_if_ready()", update_start)
        latch_index = source.index("if self._reactivation_stabilization_frames > 0:", update_start)

        self.assertLess(latch_index, dispatch_index)
        reactivate_start = source.index("    def reactivate(")
        reactivate_end = source.index("    def dispose(", reactivate_start)
        reactivate_source = source[reactivate_start:reactivate_end]

        self.assertIn("self._reset_people_runtime_for_reactivation()", reactivate_source)
        self.assertIn("self._embodiment_generation += 1", reactivate_source)
        self.assertIn("self._reactivation_stabilization_frames =", reactivate_source)
        self.assertIn("self._hold_reactivation_runtime_pose()", source[latch_index:dispatch_index])

    def test_reactivation_readiness_requires_completed_warmup(self):
        source = (
            ROOT
            / "pedestrian"
            / "simulator"
            / "logic"
            / "people"
            / "person.py"
        ).read_text(encoding="utf-8")
        ready_start = source.index("    def reactivation_ready(self)")
        ready_end = source.index("    @property", ready_start + 10)
        ready_source = source[ready_start:ready_end]

        self.assertIn("self._reactivation_stabilization_frames == 0", ready_source)
        self.assertIn("self._anim_graph_ready", ready_source)
        self.assertIn("self._pose_valid", ready_source)

    def test_reactivation_does_not_seed_a_degenerate_stationary_path(self):
        source = (
            ROOT
            / "pedestrian"
            / "simulator"
            / "logic"
            / "people"
            / "person.py"
        ).read_text(encoding="utf-8")

        reset_start = source.index("    def _reset_behavior_runtime_at_pose")
        reset_end = source.index("    def update(self, dt: float):", reset_start)
        reset_source = source[reset_start:reset_end]

        self.assertNotIn("stationary_path", reset_source)
        self.assertIn("navigation_manager.positions_over_time = []", reset_source)
        self.assertIn("navigation_manager.delta_time_list = []", reset_source)
        self.assertNotIn('set_variable("PathPoints"', reset_source)

    def test_runtime_reset_keeps_animgraph_and_navigation_objects(self):
        source = (
            ROOT
            / "pedestrian"
            / "simulator"
            / "logic"
            / "people"
            / "person.py"
        ).read_text(encoding="utf-8")

        discard_start = source.index("    def _discard_official_people_command")
        reset_start = source.index("    def _reset_people_runtime_for_reactivation")
        reset_end = source.index("    def _reset_behavior_runtime_at_pose", reset_start)
        discard_source = source[discard_start:reset_start]
        reset_source = source[reset_start:reset_end]

        self.assertIn("agent.current_command = None", discard_source)
        self.assertIn("agent.commands = []", discard_source)
        self.assertNotIn("navigation_manager.destroy()", reset_source)
        self.assertNotIn("agent.renew_character_state()", reset_source)
        self.assertNotIn("self.add_animation_graph_to_agent()", reset_source)


if __name__ == "__main__":
    unittest.main()
