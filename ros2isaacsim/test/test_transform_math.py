import math
import unittest

import numpy as np

from ros2isaacsim.transform_math import child_relative_to_parent


class _RowMatrix:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)

    def __mul__(self, other):
        return _RowMatrix(self.values @ other.values)

    def GetInverse(self):
        return _RowMatrix(np.linalg.inv(self.values))


def _row_transform(x, y, z, yaw):
    c = math.cos(yaw)
    s = math.sin(yaw)
    return _RowMatrix(
        [
            [c, s, 0.0, 0.0],
            [-s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [x, y, z, 1.0],
        ]
    )


class TestTransformMath(unittest.TestCase):
    def test_recovers_rear_lidar_transform_at_nonzero_spawn_pose(self):
        parent_world = _row_transform(-4.4, -1.0, 0.03, 0.4)
        expected_local = _row_transform(-0.2, -0.13, 0.208, math.pi)
        child_world = expected_local * parent_world

        actual_local = child_relative_to_parent(parent_world, child_world)

        np.testing.assert_allclose(actual_local.values, expected_local.values, atol=1.0e-9)

    def test_inverse_parent_must_be_on_the_right(self):
        parent_world = _row_transform(-4.4, -1.0, 0.03, 0.4)
        child_local = _row_transform(-0.2, -0.13, 0.208, math.pi)
        child_world = child_local * parent_world

        wrong_order = parent_world.GetInverse() * child_world
        correct = child_relative_to_parent(parent_world, child_world)

        self.assertGreater(
            np.linalg.norm(wrong_order.values[3, :3] - child_local.values[3, :3]),
            1.0,
        )
        np.testing.assert_allclose(correct.values[3, :3], [-0.2, -0.13, 0.208], atol=1.0e-9)


if __name__ == "__main__":
    unittest.main()
