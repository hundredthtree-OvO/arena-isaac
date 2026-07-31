import unittest

from pedestrian.simulator.logic.people.external_motion import (
    ExternalMotionMode,
    ExternalMotionModeState,
    ExternalMotionSample,
    ExternalMotionState,
    animation_walk_blend,
    animation_tracking_sample,
    bounded_yaw_step,
    locomotion_path_points,
    turn_aware_animation_sample,
)


class TestExternalMotionState(unittest.TestCase):
    def test_animation_walk_blend_compensates_nonlinear_root_motion(self):
        blend = animation_walk_blend(
            0.4,
            full_speed_mps=0.8,
            speed_exponent=3.0,
        )

        self.assertAlmostEqual(blend, 0.5 ** (1.0 / 3.0))

    def test_animation_walk_blend_clamps_at_full_speed(self):
        self.assertEqual(
            animation_walk_blend(
                1.2,
                full_speed_mps=0.8,
                speed_exponent=3.0,
            ),
            1.0,
        )

    def test_animation_walk_blend_preserves_idle(self):
        self.assertEqual(
            animation_walk_blend(
                0.0,
                full_speed_mps=0.8,
                speed_exponent=3.0,
            ),
            0.0,
        )

    def test_motion_mode_state_reports_explicit_transitions(self):
        state = ExternalMotionModeState()

        previous, current = state.transition(ExternalMotionMode.FREEZE)
        self.assertEqual(previous, ExternalMotionMode.LOCOMOTION)
        self.assertEqual(current, ExternalMotionMode.FREEZE)

        previous, current = state.transition(ExternalMotionMode.TERMINAL_ALIGN)
        self.assertEqual(previous, ExternalMotionMode.FREEZE)
        self.assertEqual(current, ExternalMotionMode.TERMINAL_ALIGN)

        previous, current = state.transition(ExternalMotionMode.REPLAY_TRACK)
        self.assertEqual(previous, ExternalMotionMode.TERMINAL_ALIGN)
        self.assertEqual(current, ExternalMotionMode.REPLAY_TRACK)

    def test_motion_mode_state_rejects_unknown_modes(self):
        state = ExternalMotionModeState()

        with self.assertRaises(ValueError):
            state.transition(99)

    def test_interpolates_world_pose_from_latest_velocity(self):
        state = ExternalMotionState()
        state.set_command(
            position=[1.0, 2.0, 0.0],
            velocity=[0.5, -0.25, 0.0],
            yaw=0.4,
            received_at=10.0,
            timeout_sec=0.5,
        )

        sample = state.sample(10.2)

        self.assertAlmostEqual(sample.position[0], 1.1)
        self.assertAlmostEqual(sample.position[1], 1.95)
        self.assertEqual(sample.velocity, (0.5, -0.25, 0.0))
        self.assertAlmostEqual(sample.yaw, 0.4)
        self.assertFalse(sample.expired)
        self.assertTrue(state.should_walk(sample))

    def test_timeout_holds_last_safe_extrapolated_pose_and_idles(self):
        state = ExternalMotionState()
        state.set_command(
            position=[0.0, 0.0, 0.0],
            velocity=[1.0, 0.0, 0.0],
            yaw=0.0,
            received_at=4.0,
            timeout_sec=0.3,
        )

        sample = state.sample(5.0)

        self.assertAlmostEqual(sample.position[0], 0.3)
        self.assertEqual(sample.velocity, (0.0, 0.0, 0.0))
        self.assertTrue(sample.expired)
        self.assertFalse(state.should_walk(sample))

    def test_clear_releases_external_authority(self):
        state = ExternalMotionState()
        state.set_command(
            position=[0.0, 0.0, 0.0],
            velocity=[0.0, 0.0, 0.0],
            yaw=0.0,
            received_at=1.0,
            timeout_sec=0.5,
        )

        state.clear()

        self.assertIsNone(state.sample(1.1))

    def test_rejects_non_positive_timeout(self):
        state = ExternalMotionState()

        with self.assertRaises(ValueError):
            state.set_command(
                position=[0.0, 0.0, 0.0],
                velocity=[0.0, 0.0, 0.0],
                yaw=0.0,
                received_at=1.0,
                timeout_sec=0.0,
            )

    def test_locomotion_path_points_follow_external_world_velocity(self):
        sample = ExternalMotionSample(
            position=(1.0, 2.0, 0.3),
            velocity=(0.0, -0.5, 0.0),
            yaw=-1.57,
            speed=0.5,
            expired=False,
        )

        start, lookahead = locomotion_path_points(sample, lookahead_m=0.6)

        self.assertEqual(start, (1.0, 2.0, 0.3))
        self.assertAlmostEqual(lookahead[0], 1.0)
        self.assertAlmostEqual(lookahead[1], 1.4)
        self.assertAlmostEqual(lookahead[2], 0.3)

    def test_tracking_sample_uses_reference_velocity_and_bounded_position_error(self):
        reference = ExternalMotionSample(
            position=(2.0, 0.0, 0.0),
            velocity=(0.5, 0.0, 0.0),
            yaw=0.0,
            speed=0.5,
            expired=False,
        )

        sample = animation_tracking_sample(
            reference,
            current_position=(1.8, 0.0, 0.0),
            tracking_gain=1.5,
            max_speed_mps=1.2,
        )

        self.assertEqual(sample.position, (1.8, 0.0, 0.0))
        self.assertAlmostEqual(sample.velocity[0], 0.8)
        self.assertAlmostEqual(sample.speed, 0.8)

    def test_tracking_sample_idles_when_reference_expires(self):
        reference = ExternalMotionSample(
            position=(2.0, 0.0, 0.0),
            velocity=(0.5, 0.0, 0.0),
            yaw=0.0,
            speed=0.5,
            expired=True,
        )

        sample = animation_tracking_sample(reference, current_position=(1.7, 0.0, 0.0))

        self.assertTrue(sample.expired)
        self.assertEqual(sample.position, (1.7, 0.0, 0.0))
        self.assertEqual(sample.speed, 0.0)

    def test_turn_aware_tracking_keeps_animgraph_walking_through_large_turn(self):
        reference = ExternalMotionSample(
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.6, 0.0),
            yaw=1.57,
            speed=0.6,
            expired=False,
        )

        sample = turn_aware_animation_sample(
            reference,
            current_position=(0.0, 0.0, 0.0),
            current_yaw=0.0,
            full_slow_angle_rad=1.0,
            minimum_speed_scale=0.25,
        )

        self.assertGreater(sample.speed, 0.0)
        self.assertAlmostEqual(sample.speed, reference.speed * 0.25)
        self.assertAlmostEqual(sample.yaw, 1.57079632679)

    def test_turn_aware_tracking_slows_a_moderate_heading_change(self):
        reference = ExternalMotionSample(
            position=(0.0, 0.0, 0.0),
            velocity=(0.4, 0.4, 0.0),
            yaw=0.785,
            speed=0.566,
            expired=False,
        )

        sample = turn_aware_animation_sample(
            reference,
            current_position=(0.0, 0.0, 0.0),
            current_yaw=0.0,
            slow_angle_rad=0.3,
            full_slow_angle_rad=1.0,
            minimum_speed_scale=0.25,
        )

        self.assertGreater(sample.speed, 0.0)
        self.assertLess(sample.speed, reference.speed)

    def test_turn_aware_tracking_preserves_small_heading_change(self):
        reference = ExternalMotionSample(
            position=(0.0, 0.0, 0.0),
            velocity=(1.0, 0.0, 0.0),
            yaw=0.0,
            speed=1.0,
            expired=False,
        )

        sample = turn_aware_animation_sample(
            reference,
            current_position=(0.0, 0.0, 0.0),
            current_yaw=0.1,
            slow_angle_rad=0.3,
            full_slow_angle_rad=1.0,
        )

        self.assertAlmostEqual(sample.speed, reference.speed)

    def test_bounded_yaw_step_uses_shortest_arc(self):
        step = bounded_yaw_step(
            3.0,
            -3.0,
            max_rate_radps=0.5,
            dt=0.2,
        )

        self.assertGreater(step, 3.0)
        self.assertLess(step, 3.2)

    def test_bounded_yaw_step_snaps_when_target_is_within_step_limit(self):
        self.assertAlmostEqual(
            bounded_yaw_step(
                0.0,
                0.05,
                max_rate_radps=1.0,
                dt=0.1,
            ),
            0.05,
        )


if __name__ == "__main__":
    unittest.main()
