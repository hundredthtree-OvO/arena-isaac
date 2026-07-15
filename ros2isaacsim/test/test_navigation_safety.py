import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "pedestrian"
    / "simulator"
    / "logic"
    / "people"
    / "navigation_safety.py"
)
SPEC = spec_from_file_location("navigation_safety", MODULE_PATH)
NAVIGATION_SAFETY = module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = NAVIGATION_SAFETY
SPEC.loader.exec_module(NAVIGATION_SAFETY)
LateralAvoidanceCandidate = NAVIGATION_SAFETY.LateralAvoidanceCandidate
path_target_progress_radius = NAVIGATION_SAFETY.path_target_progress_radius
segment_is_safe = NAVIGATION_SAFETY.segment_is_safe
select_safe_lateral_avoidance_choice = NAVIGATION_SAFETY.select_safe_lateral_avoidance_choice
terminal_semantic_approach_radius = NAVIGATION_SAFETY.terminal_semantic_approach_radius


class TestNavigationSafety(unittest.TestCase):
    def test_constrained_path_uses_tight_intermediate_but_normal_final_radius(self):
        common = {
            "constrain_to_path": True,
            "constrained_intermediate_radius": 0.08,
            "default_intermediate_radius": 0.25,
            "final_radius": 0.15,
        }

        self.assertEqual(path_target_progress_radius(target_count=3, **common), 0.08)
        self.assertEqual(path_target_progress_radius(target_count=1, **common), 0.15)

    def test_constrained_terminal_pose_uses_tight_final_target_radius(self):
        self.assertEqual(
            terminal_semantic_approach_radius(
                constrain_to_path=True,
                default_radius=0.5,
                final_radius=0.15,
            ),
            0.15,
        )
        self.assertEqual(
            terminal_semantic_approach_radius(
                constrain_to_path=False,
                default_radius=0.5,
                final_radius=0.15,
            ),
            0.5,
        )

    def test_chooses_preferred_safe_candidate(self):
        left = LateralAvoidanceCandidate("left", (1.0, 0.0, 0.0), True)
        right = LateralAvoidanceCandidate("right", (-1.0, 0.0, 0.0), True)

        selected = select_safe_lateral_avoidance_choice(
            direction_of_collision=0.5,
            left=left,
            right=right,
        )

        self.assertIs(selected, left)

    def test_waits_when_both_candidates_are_unsafe(self):
        left = LateralAvoidanceCandidate("left", (1.0, 0.0, 0.0), False)
        right = LateralAvoidanceCandidate("right", (-1.0, 0.0, 0.0), False)

        selected = select_safe_lateral_avoidance_choice(
            direction_of_collision=-0.7,
            left=left,
            right=right,
        )

        self.assertIsNone(selected)

    def test_falls_back_to_the_only_safe_candidate(self):
        left = LateralAvoidanceCandidate("left", (1.0, 0.0, 0.0), False)
        right = LateralAvoidanceCandidate("right", (-1.0, 0.0, 0.0), True)

        selected = select_safe_lateral_avoidance_choice(
            direction_of_collision=0.7,
            left=left,
            right=right,
        )

        self.assertIs(selected, right)

    def test_chooses_left_when_both_candidates_are_safe_but_direction_is_ambiguous(self):
        left = LateralAvoidanceCandidate("left", (1.0, 0.0, 0.0), True)
        right = LateralAvoidanceCandidate("right", (-1.0, 0.0, 0.0), True)

        selected = select_safe_lateral_avoidance_choice(
            direction_of_collision=0.05,
            left=left,
            right=right,
        )

        self.assertIs(selected, left)

    def test_segment_sampler_rejects_a_thin_wall_between_safe_endpoints(self):
        def is_safe_point(point_xy):
            x, y = point_xy
            return not (0.45 <= x <= 0.55 and abs(y) < 0.05)

        self.assertFalse(
            segment_is_safe(
                start_xy=(0.0, 0.0),
                end_xy=(1.0, 0.0),
                is_safe_point=is_safe_point,
                sample_step=0.25,
            )
        )


if __name__ == "__main__":
    unittest.main()
