import json
from pathlib import Path
import tempfile
import unittest

from pedestrian.simulator.logic.people.walkable_guard import StaticWalkableGuard


class TestStaticWalkableGuard(unittest.TestCase):
    def test_loads_non_free_cells_as_static_obstacles(self):
        payload = {
            "schema": "arena.walkable_map.v1",
            "resolution": 0.1,
            "origin": [-1.0, -2.0, 0.0],
            "width": 3,
            "height": 2,
            "data": [0, 100, -1, 0, 0, 100],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "walkable.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            guard = StaticWalkableGuard(str(path))

        self.assertEqual(guard.resolution, 0.1)
        self.assertEqual(guard.origin, (-1.0, -2.0, 0.0))
        self.assertEqual(guard.occupied_xy, {(1, 0), (2, 0), (2, 1)})


if __name__ == "__main__":
    unittest.main()
