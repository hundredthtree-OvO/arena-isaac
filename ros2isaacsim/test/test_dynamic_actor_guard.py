import unittest

try:
    from isaac_utils.dynamic_actor_guard import (
        PedestrianGuardState,
        RobotGuardState,
        circle_intersects_grid_cells,
        movement_allowed,
        pedestrian_robot_scores,
        predict_pedestrian_step,
        predict_robot_pose,
        robot_pedestrian_scores,
        scale_robot_command_for_pedestrians,
    )
except ImportError:
    from ros2isaacsim.isaac_utils.dynamic_actor_guard import (
        PedestrianGuardState,
        RobotGuardState,
        circle_intersects_grid_cells,
        movement_allowed,
        pedestrian_robot_scores,
        predict_pedestrian_step,
        predict_robot_pose,
        robot_pedestrian_scores,
        scale_robot_command_for_pedestrians,
    )


class TestDynamicActorGuard(unittest.TestCase):
    @staticmethod
    def _robot(pos_xy=(0.0, 0.0)):
        return RobotGuardState(
            name="robot",
            pos_xy=pos_xy,
            heading=0.0,
            next_pos_xy=pos_xy,
            next_heading=0.0,
            forward=0.4,
            rear=0.4,
            left=0.3,
            right=0.3,
        )

    @staticmethod
    def _ped(name="ped", pos_xy=(1.0, 0.0), next_pos_xy=None):
        return PedestrianGuardState(
            name=name,
            pos_xy=pos_xy,
            next_pos_xy=pos_xy if next_pos_xy is None else next_pos_xy,
            radius=0.22,
        )

    def test_hard_guard_leaves_clear_command_unchanged(self):
        result = scale_robot_command_for_pedestrians(
            robot=self._robot(),
            pedestrians=[self._ped(pos_xy=(0.0, 2.0))],
            vx=0.5,
            vy=0.0,
            wz=0.0,
            horizon_sec=1.0,
            sample_dt_sec=0.05,
            margin_m=0.03,
        )
        self.assertEqual(result.mode, "clear")
        self.assertEqual(result.scale, 1.0)

    def test_hard_guard_scales_command_before_swept_collision(self):
        result = scale_robot_command_for_pedestrians(
            robot=self._robot(),
            pedestrians=[self._ped(pos_xy=(0.9, 0.0))],
            vx=0.8,
            vy=0.0,
            wz=0.0,
            horizon_sec=1.0,
            sample_dt_sec=0.05,
            margin_m=0.03,
        )
        self.assertIn(result.mode, {"scaled", "blocked"})
        self.assertLess(result.scale, 1.0)
        self.assertEqual(result.blocked_by, "ped")

    def test_overlap_only_allows_strict_escape(self):
        robot = self._robot()
        pedestrian = self._ped(pos_xy=(0.5, 0.0))
        deeper = scale_robot_command_for_pedestrians(
            robot=robot,
            pedestrians=[pedestrian],
            vx=0.2,
            vy=0.0,
            wz=0.0,
            horizon_sec=0.5,
            sample_dt_sec=0.05,
        )
        tangent = scale_robot_command_for_pedestrians(
            robot=robot,
            pedestrians=[pedestrian],
            vx=0.0,
            vy=0.2,
            wz=0.0,
            horizon_sec=0.5,
            sample_dt_sec=0.05,
        )
        escape = scale_robot_command_for_pedestrians(
            robot=robot,
            pedestrians=[pedestrian],
            vx=-0.2,
            vy=0.0,
            wz=0.0,
            horizon_sec=0.5,
            sample_dt_sec=0.05,
        )
        self.assertEqual(deeper.mode, "overlap_stop")
        self.assertEqual(tangent.mode, "overlap_stop")
        self.assertEqual(escape.mode, "escape")

    def test_overlap_escape_can_use_a_longer_projection_window(self):
        robot = self._robot()
        pedestrian = self._ped(pos_xy=(0.38, 0.42))

        result = scale_robot_command_for_pedestrians(
            robot=robot,
            pedestrians=[pedestrian],
            vx=-0.08,
            vy=0.0,
            wz=-0.8,
            horizon_sec=0.8,
            sample_dt_sec=0.04,
            overlap_escape_horizon_sec=0.35,
            overlap_deadband_m=0.015,
        )

        self.assertEqual(result.mode, "escape")

    def test_overlap_escape_still_rejects_a_command_that_gets_deeper(self):
        result = scale_robot_command_for_pedestrians(
            robot=self._robot(),
            pedestrians=[self._ped(pos_xy=(0.5, 0.0))],
            vx=0.2,
            vy=0.0,
            wz=0.0,
            horizon_sec=0.8,
            sample_dt_sec=0.04,
            overlap_escape_horizon_sec=0.35,
            overlap_deadband_m=0.015,
        )

        self.assertEqual(result.mode, "overlap_stop")

    def test_circle_grid_collision_is_not_orientation_dependent(self):
        occupied = {(0, 0)}

        self.assertEqual(
            circle_intersects_grid_cells(
                center_xy=(1.2, 0.5), radius=0.21, resolution=1.0, origin_xy=(0.0, 0.0), occupied=occupied
            ),
            (0, 0),
        )
        self.assertIsNone(
            circle_intersects_grid_cells(
                center_xy=(1.3, 0.5), radius=0.21, resolution=1.0, origin_xy=(0.0, 0.0), occupied=occupied
            )
        )

    def test_predict_pedestrian_step_moves_toward_goal(self):
        next_xy = predict_pedestrian_step(
            pos_xy=(0.0, 0.0),
            goal_xy=(1.0, 0.0),
            speed=0.5,
            dt=1.0,
        )
        self.assertEqual(next_xy, (0.5, 0.0))

    def test_robot_blocks_when_next_step_hits_pedestrian(self):
        next_pos_xy, next_heading = predict_robot_pose(
            pos_xy=(0.0, 0.0),
            heading=0.0,
            vx=0.5,
            vy=0.0,
            wz=0.0,
            dt=1.0,
        )
        robot = RobotGuardState(
            name="robot",
            pos_xy=(0.0, 0.0),
            heading=0.0,
            next_pos_xy=next_pos_xy,
            next_heading=next_heading,
            forward=0.4,
            rear=0.4,
            left=0.3,
            right=0.3,
        )
        ped = PedestrianGuardState(
            name="ped",
            pos_xy=(0.8, 0.0),
            next_pos_xy=(0.8, 0.0),
            radius=0.22,
        )
        current_score, next_score = robot_pedestrian_scores(robot, ped)
        self.assertEqual(current_score, 0.0)
        self.assertGreater(next_score, 0.0)
        self.assertFalse(movement_allowed(current_score, next_score))

    def test_escape_motion_is_allowed_when_overlap_shrinks(self):
        robot = RobotGuardState(
            name="robot",
            pos_xy=(0.0, 0.0),
            heading=0.0,
            next_pos_xy=(-0.1, 0.0),
            next_heading=0.0,
            forward=0.4,
            rear=0.4,
            left=0.3,
            right=0.3,
        )
        ped = PedestrianGuardState(
            name="ped",
            pos_xy=(0.35, 0.0),
            next_pos_xy=(0.35, 0.0),
            radius=0.22,
        )
        current_score, next_score = robot_pedestrian_scores(robot, ped)
        self.assertGreater(current_score, 0.0)
        self.assertLess(next_score, current_score)
        self.assertTrue(movement_allowed(current_score, next_score))

    def test_pedestrian_stops_when_robot_will_enter_next_pose(self):
        robot = RobotGuardState(
            name="robot",
            pos_xy=(0.0, 0.0),
            heading=0.0,
            next_pos_xy=(0.5, 0.0),
            next_heading=0.0,
            forward=0.4,
            rear=0.4,
            left=0.3,
            right=0.3,
        )
        ped = PedestrianGuardState(
            name="ped",
            pos_xy=(1.0, 0.0),
            next_pos_xy=(0.7, 0.0),
            radius=0.22,
        )
        current_score, next_score = pedestrian_robot_scores(ped, robot)
        self.assertEqual(current_score, 0.0)
        self.assertGreater(next_score, 0.0)
        self.assertFalse(movement_allowed(current_score, next_score))


if __name__ == "__main__":
    unittest.main()
