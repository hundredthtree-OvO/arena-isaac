import unittest

from pedestrian.simulator.logic.people.external_motion import (
    ExternalMotionSample,
    ExternalMotionState,
    animation_tracking_sample,
    bounded_yaw_step,
    locomotion_path_points,
)


class TestExternalMotionState(unittest.TestCase):
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
