import math
import unittest

import numpy as np

from ros2isaacsim.actual_motion_state import ActualMotionStateEstimator
from ros2isaacsim.motion_backends import PhysxDiffContactBackend, motion_backend_for_mode


class TestMotionBackendBoundaries(unittest.TestCase):
    def test_contact_backend_recovers_articulation_before_wheel_commands(self):
        calls = []

        class Drive:
            last_effective_wz = 0.0

            def apply(self, vx, wz):
                calls.append("wheel_targets")
                return np.zeros(4, dtype=np.float32)

        class Config:
            separated_tire_force_enabled = True

        class Robot:
            _articulation = object()
            _joint_indices = [0, 1, 2, 3]
            _physx_diff_drive = Drive()
            config = Config()

            def _ensure_diff_articulation_force_view(self):
                calls.append("recover_view")
                return True

            def _publish_actual_odom_tf(self):
                pass

            def _publish_applied_cmd_vel(self):
                pass

            def _read_joint_velocities(self, _indices):
                return np.zeros(4, dtype=np.float32)

            def _current_physx_diff_contacts(self):
                return []

            def _apply_separated_tire_forces(self, _actual, _contacts):
                return None

        backend = PhysxDiffContactBackend()
        backend._log_diagnostics = lambda *args: None
        backend.apply(Robot(), 0.2, 0.0, 0.1, 0.02, "differential")

        self.assertEqual(calls[:2], ["recover_view", "wheel_targets"])

    def test_modes_own_distinct_guard_and_odom_policies(self):
        wheels = motion_backend_for_mode("physx_wheels")
        contact = motion_backend_for_mode("physx_diff_contact")

        self.assertTrue(wheels.uses_collision_guard)
        self.assertFalse(wheels.uses_measured_odom_twist)
        self.assertFalse(contact.uses_collision_guard)
        self.assertTrue(contact.uses_measured_odom_twist)
        self.assertFalse(contact.applies_navigation_hold)

    def test_actual_state_estimator_reports_body_frame_velocity(self):
        estimator = ActualMotionStateEstimator()
        yaw_90 = [math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0)]

        initial = estimator.sample([0.0, 0.0, 0.0], yaw_90, now=10.0)
        moved = estimator.sample([0.0, 2.0, 0.0], yaw_90, now=12.0)

        self.assertEqual((initial.vx, initial.vy, initial.wz), (0.0, 0.0, 0.0))
        self.assertAlmostEqual(moved.vx, 1.0, places=6)
        self.assertAlmostEqual(moved.vy, 0.0, places=6)
        self.assertAlmostEqual(moved.wz, 0.0, places=6)
        np.testing.assert_allclose(moved.position, [0.0, 2.0, 0.0])

    def test_actual_state_estimator_reset_hides_episode_teleport_velocity(self):
        estimator = ActualMotionStateEstimator()
        identity = [1.0, 0.0, 0.0, 0.0]

        estimator.sample([0.0, 0.0, 0.0], identity, now=10.0)
        estimator.reset()
        reset_pose = estimator.sample([4.0, -1.0, 0.03], identity, now=10.1)

        self.assertEqual((reset_pose.vx, reset_pose.vy, reset_pose.wz), (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
