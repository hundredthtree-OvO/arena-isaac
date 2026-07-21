import unittest

from isaac_utils.origin_collision_probe import _aabb_intersects, _diagnose_origin_hits


class TestOriginCollisionProbe(unittest.TestCase):
    def test_detects_overlapping_bounds(self):
        self.assertTrue(
            _aabb_intersects(
                (-0.2, -0.2, 0.1),
                (0.2, 0.2, 0.4),
                (-0.75, -0.75, 0.05),
                (0.75, 0.75, 0.65),
            )
        )

    def test_rejects_bounds_outside_query(self):
        self.assertFalse(
            _aabb_intersects(
                (2.0, 2.0, 0.1),
                (2.2, 2.2, 0.4),
                (-0.75, -0.75, 0.05),
                (0.75, 0.75, 0.65),
            )
        )

    def test_identifies_physx_collision_prototype_at_origin(self):
        hits = [
            {
                "rigid_body": "/__Prototype_42/base_link",
                "collision": "/__Prototype_42/base_link/collisions/chassis",
            }
        ]
        self.assertEqual(
            _diagnose_origin_hits([], hits),
            "physx_collision_prototype_at_origin",
        )

    def test_prototype_diagnosis_wins_when_robot_is_inside_query(self):
        candidates = [{"path": "/World/robots/xms_mecanum/base_link/chassis"}]
        hits = [
            {"collision": "/World/robots/xms_mecanum/base_link/chassis"},
            {"collision": "/__Prototype_7/base_link/collisions/chassis"},
        ]
        self.assertEqual(
            _diagnose_origin_hits(candidates, hits),
            "physx_collision_prototype_at_origin",
        )


if __name__ == "__main__":
    unittest.main()
