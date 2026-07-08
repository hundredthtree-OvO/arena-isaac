import json
import os
import tempfile
import unittest

from ros2isaacsim.isaac_utils.dynamic_actor_guard import PedestrianGuardState
from ros2isaacsim.isaac_utils.voxel_guard import VoxelCollisionGuard, VoxelGuardConfig


class TestVoxelGuardDynamic(unittest.TestCase):
    def _make_empty_map(self) -> str:
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(
            {
                "resolution": 0.1,
                "origin": [0.0, 0.0, 0.0],
                "voxels": [],
            },
            handle,
        )
        handle.flush()
        handle.close()
        self.addCleanup(lambda: os.unlink(handle.name))
        return handle.name

    def test_dynamic_pedestrian_blocks_without_static_voxels(self):
        cfg = VoxelGuardConfig(
            enabled=True,
            map_path=self._make_empty_map(),
            length=0.8,
            width=0.6,
            margin=0.0,
        )
        guard = VoxelCollisionGuard(robot_name="robot", config=cfg)
        guard._dynamic_pedestrian_states = lambda dt: [
            PedestrianGuardState(
                name="ped",
                pos_xy=(0.8, 0.0),
                next_pos_xy=(0.8, 0.0),
                radius=0.22,
            )
        ]
        vx, vy, wz, mode = guard.filter_motion(
            pos_xyz=(0.0, 0.0, 0.0),
            heading=0.0,
            vx=0.5,
            vy=0.0,
            wz=0.0,
            dt=1.0,
        )
        self.assertEqual((vx, vy, wz), (0.0, 0.0, 0.0))
        self.assertEqual(mode, "blocked")

    def test_dynamic_overlap_allows_escape_direction(self):
        cfg = VoxelGuardConfig(
            enabled=True,
            map_path=self._make_empty_map(),
            length=0.8,
            width=0.6,
            margin=0.0,
        )
        guard = VoxelCollisionGuard(robot_name="robot", config=cfg)
        guard._dynamic_pedestrian_states = lambda dt: [
            PedestrianGuardState(
                name="ped",
                pos_xy=(0.35, 0.0),
                next_pos_xy=(0.35, 0.0),
                radius=0.22,
            )
        ]
        vx, vy, wz, mode = guard.filter_motion(
            pos_xyz=(0.0, 0.0, 0.0),
            heading=0.0,
            vx=-0.1,
            vy=0.0,
            wz=0.0,
            dt=1.0,
        )
        self.assertEqual((vx, vy, wz), (-0.1, 0.0, 0.0))
        self.assertIn(mode, {"inside_escape_reduce", "inside_escape_to_free", "inside_escape_tolerated"})


if __name__ == "__main__":
    unittest.main()
