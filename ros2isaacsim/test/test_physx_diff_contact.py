import unittest

import numpy as np

from ros2isaacsim.physx_diff_contact import (
    PhysxDiffContactDrive,
    articulation_link_wrench_arrays,
    config_from_values,
    separated_tire_wrench,
    single_articulation_targets,
    wheel_slip_diagnostics,
)


class TestPhysxDiffContactDrive(unittest.TestCase):
    def setUp(self):
        self.applied = []
        config = config_from_values(
            wheel_radius=0.08,
            track_width=0.345,
            wheel_signs=[1.0, 1.0, 1.0, 1.0],
            max_wheel_speed=20.0,
        )
        self.drive = PhysxDiffContactDrive(config, self.applied.append)

    def test_applies_differential_wheel_targets_only(self):
        speeds = self.drive.apply(vx=0.4, wz=0.5)
        half_track = 0.345 / 2.0
        expected = [
            (0.4 - half_track * 0.5) / 0.08,
            (0.4 + half_track * 0.5) / 0.08,
            (0.4 - half_track * 0.5) / 0.08,
            (0.4 + half_track * 0.5) / 0.08,
        ]
        np.testing.assert_allclose(speeds, expected)
        np.testing.assert_allclose(self.applied[-1], expected)

    def test_stop_applies_zero_targets(self):
        np.testing.assert_array_equal(self.drive.stop(), np.zeros(4))
        np.testing.assert_array_equal(self.applied[-1], np.zeros(4))

    def test_preserves_curvature_when_wheel_speed_is_limited(self):
        applied = []
        drive = PhysxDiffContactDrive(
            config_from_values(
                wheel_radius=0.08,
                track_width=0.345,
                wheel_signs=[1.0, 1.0, 1.0, 1.0],
                max_wheel_speed=4.0,
                linear_gain=1.35,
                angular_gain=1.8,
            ),
            applied.append,
        )
        speeds = drive.apply(vx=0.4, wz=0.8)
        raw_left = (0.4 * 1.35 - 0.5 * 0.345 * 0.8 * 1.8) / 0.08
        raw_right = (0.4 * 1.35 + 0.5 * 0.345 * 0.8 * 1.8) / 0.08
        scale = 4.0 / max(abs(raw_left), abs(raw_right))
        np.testing.assert_allclose(
            speeds,
            [raw_left * scale, raw_right * scale, raw_left * scale, raw_right * scale],
        )
        self.assertAlmostEqual(drive.last_effective_vx, 0.54)
        self.assertAlmostEqual(drive.last_effective_wz, 1.44)
        self.assertAlmostEqual(drive.last_limit_scale, scale)

    def test_rejects_invalid_geometry(self):
        with self.assertRaises(ValueError):
            config_from_values(
                wheel_radius=0.08,
                track_width=0.345,
                wheel_signs=[1.0, 1.0],
                max_wheel_speed=20.0,
            )

    def test_isaac_articulation_targets_are_batched(self):
        targets = single_articulation_targets([-1.0, 1.0, -1.0, 1.0])
        self.assertEqual(targets.shape, (1, 4))
        np.testing.assert_array_equal(targets[0], [-1.0, 1.0, -1.0, 1.0])

    def test_reports_wheel_slip_against_measured_chassis_motion(self):
        diagnostics = wheel_slip_diagnostics(
            joint_velocities=[3.105, -3.105, 3.105, -3.105],
            wheel_radius=0.08,
            track_width=0.345,
            wheel_signs=[1.0, 1.0, 1.0, 1.0],
            measured_vx=0.0,
            measured_vy=0.02,
            measured_wz=-0.144,
            effective_wz=-1.44,
        )
        np.testing.assert_allclose(
            diagnostics["tread_speeds_mps"],
            [0.2484, -0.2484, 0.2484, -0.2484],
        )
        np.testing.assert_allclose(
            diagnostics["hub_speeds_mps"],
            [0.02484, -0.02484, 0.02484, -0.02484],
        )
        np.testing.assert_allclose(
            diagnostics["slip_ratios"],
            [0.9, -0.9, 0.9, -0.9],
        )
        self.assertAlmostEqual(diagnostics["yaw_efficiency"], 0.1)
        self.assertAlmostEqual(diagnostics["measured_vy"], 0.02)

    def test_separated_tire_force_drives_forward_without_yaw(self):
        wrench = separated_tire_wrench(
            joint_velocities=[2.0, 2.0, 2.0, 2.0],
            wheel_radius=0.08,
            wheel_signs=[1.0, 1.0, 1.0, 1.0],
            half_length=0.1575,
            half_width=0.1725,
            measured_vx=0.0,
            measured_vy=0.0,
            measured_wz=0.0,
            longitudinal_stiffness=60.0,
            lateral_stiffness=20.0,
            max_longitudinal_force=12.0,
            max_lateral_force=4.0,
        )
        np.testing.assert_allclose(wrench["longitudinal_forces_n"], [9.6] * 4)
        np.testing.assert_allclose(wrench["net_force_body_n"], [38.4, 0.0, 0.0])
        np.testing.assert_allclose(wrench["net_torque_body_nm"], [0.0, 0.0, 0.0])

    def test_separated_tire_force_generates_skid_steer_yaw(self):
        wrench = separated_tire_wrench(
            joint_velocities=[2.0, -2.0, 2.0, -2.0],
            wheel_radius=0.08,
            wheel_signs=[1.0, 1.0, 1.0, 1.0],
            half_length=0.1575,
            half_width=0.1725,
            measured_vx=0.0,
            measured_vy=0.0,
            measured_wz=0.0,
            longitudinal_stiffness=100.0,
            lateral_stiffness=20.0,
            max_longitudinal_force=12.0,
            max_lateral_force=4.0,
        )
        np.testing.assert_allclose(wrench["net_force_body_n"], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(wrench["net_torque_body_nm"][2], -8.28)

    def test_separated_tire_force_damps_lateral_motion_only(self):
        wrench = separated_tire_wrench(
            joint_velocities=[0.0] * 4,
            wheel_radius=0.08,
            wheel_signs=[1.0] * 4,
            half_length=0.1575,
            half_width=0.1725,
            measured_vx=0.0,
            measured_vy=0.1,
            measured_wz=0.0,
            longitudinal_stiffness=60.0,
            lateral_stiffness=20.0,
            max_longitudinal_force=12.0,
            max_lateral_force=4.0,
        )
        np.testing.assert_allclose(wrench["net_force_body_n"], [0.0, -8.0, 0.0])
        np.testing.assert_allclose(wrench["net_torque_body_nm"], [0.0, 0.0, 0.0])

    def test_separated_tire_force_ignores_airborne_wheels(self):
        wrench = separated_tire_wrench(
            joint_velocities=[2.0] * 4,
            wheel_radius=0.08,
            wheel_signs=[1.0] * 4,
            half_length=0.1575,
            half_width=0.1725,
            measured_vx=0.0,
            measured_vy=0.0,
            measured_wz=0.0,
            longitudinal_stiffness=60.0,
            lateral_stiffness=20.0,
            max_longitudinal_force=12.0,
            max_lateral_force=4.0,
            contact_mask=[True, False, True, False],
        )
        np.testing.assert_allclose(wrench["longitudinal_forces_n"], [9.6, 0.0, 9.6, 0.0])
        self.assertAlmostEqual(wrench["net_force_body_n"][0], 19.2)

    def test_builds_articulation_wrench_for_only_the_selected_link(self):
        forces, torques, indices = articulation_link_wrench_arrays(
            articulation_count=1,
            max_links=6,
            link_index=2,
            force=[1.0, 2.0, 0.0],
            torque=[0.0, 0.0, -3.0],
        )
        self.assertEqual(forces.shape, (1, 6, 3))
        self.assertEqual(torques.shape, (1, 6, 3))
        np.testing.assert_array_equal(indices, np.array([0], dtype=np.uint32))
        np.testing.assert_allclose(forces[0, 2], [1.0, 2.0, 0.0])
        np.testing.assert_allclose(torques[0, 2], [0.0, 0.0, -3.0])
        self.assertEqual(np.count_nonzero(forces), 2)
        self.assertEqual(np.count_nonzero(torques), 1)


if __name__ == "__main__":
    unittest.main()
