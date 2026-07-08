import unittest

from ros2isaacsim.isaac_utils.dynamic_actor_guard import (
    PedestrianGuardState,
    RobotGuardState,
    movement_allowed,
    pedestrian_robot_scores,
    predict_pedestrian_step,
    predict_robot_pose,
    robot_pedestrian_scores,
)


class TestDynamicActorGuard(unittest.TestCase):
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
