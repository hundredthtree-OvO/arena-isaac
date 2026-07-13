import unittest

import numpy as np

from ros2isaacsim.drive_kinematics import (
    DIFFERENTIAL_DRIVE,
    MECANUM_DRIVE,
    wheel_angular_speeds,
)
from ros2isaacsim.gamepad_diff_teleop import apply_deadzone


class TestDriveKinematics(unittest.TestCase):
    def test_differential_uses_measured_track_width(self):
        speeds = wheel_angular_speeds(
            vx=0.2,
            vy=0.4,
            wz=0.5,
            drive_mode=DIFFERENTIAL_DRIVE,
            half_length=0.1575,
            half_width=0.1725,
            wheel_radius=0.060,
            wheel_signs=[1.0, 1.0, 1.0, 1.0],
            max_wheel_speed=20.0,
        )
        expected_left = (0.2 - 0.1725 * 0.5) / 0.060
        expected_right = (0.2 + 0.1725 * 0.5) / 0.060
        np.testing.assert_allclose(
            speeds,
            [expected_left, expected_right, expected_left, expected_right],
        )

    def test_mecanum_mapping_is_unchanged(self):
        speeds = wheel_angular_speeds(
            vx=0.2,
            vy=0.1,
            wz=0.5,
            drive_mode=MECANUM_DRIVE,
            half_length=0.1575,
            half_width=0.1725,
            wheel_radius=0.060,
            wheel_signs=[1.0, 1.0, 1.0, 1.0],
            max_wheel_speed=20.0,
        )
        lever = 0.1575 + 0.1725
        np.testing.assert_allclose(
            speeds,
            [
                (0.2 - 0.1 - lever * 0.5) / 0.060,
                (0.2 + 0.1 + lever * 0.5) / 0.060,
                (0.2 + 0.1 - lever * 0.5) / 0.060,
                (0.2 - 0.1 + lever * 0.5) / 0.060,
            ],
        )

    def test_deadzone_is_continuous_and_symmetric(self):
        self.assertEqual(apply_deadzone(0.05, 0.1), 0.0)
        self.assertAlmostEqual(apply_deadzone(0.55, 0.1), 0.5)
        self.assertAlmostEqual(apply_deadzone(-0.55, 0.1), -0.5)


if __name__ == "__main__":
    unittest.main()
