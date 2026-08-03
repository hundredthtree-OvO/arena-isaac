from pathlib import Path
import tempfile
import unittest

from isaac_utils.walkable_map_builder import (
    _write_map_yaml,
    _write_pgm,
    merge_height_slices,
    occupancy_grid_from_positions,
    occupancy_values,
)


class TestWalkableMapBuilder(unittest.TestCase):
    def test_occupancy_values_convert_isaac_labels(self):
        self.assertEqual(occupancy_values([4, 5, 6, 9]), [100, 0, -1, -1])

    def test_world_positions_use_ros_bottom_left_row_order(self):
        data = occupancy_grid_from_positions(
            width=3,
            height=2,
            resolution=0.5,
            origin_xy=(-1.0, -1.0),
            free_positions=[(-0.75, -0.75, 0.0), (-0.75, -0.25, 0.0)],
            occupied_positions=[(-0.25, -0.75, 0.0)],
        )

        self.assertEqual(data, [0, 100, -1, 0, -1, -1])

    def test_pgm_flips_ros_rows_for_image_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.pgm"
            _write_pgm(path, [100, 0, -1, 100], width=2, height=2)
            payload = path.read_bytes()

        self.assertTrue(payload.startswith(b"P5\n2 2\n255\n"))
        self.assertEqual(payload[-4:], bytes([205, 0, 0, 254]))

    def test_map_yaml_references_sibling_image(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.yaml"
            _write_map_yaml(path, Path(directory) / "map.pgm", 0.05, [-4.6, -2.1, 0.0])
            text = path.read_text(encoding="utf-8")

        self.assertIn("image: map.pgm", text)
        self.assertIn("origin: [-4.6, -2.1, 0.0]", text)

    def test_height_slices_use_occupied_union(self):
        merged = merge_height_slices(
            [
                [0, 0, -1, -1],
                [100, 0, 0, -1],
                [0, 100, -1, 0],
            ]
        )

        self.assertEqual(merged, [100, 100, 0, 0])

    def test_height_slice_merge_rejects_mismatched_shapes(self):
        with self.assertRaisesRegex(ValueError, "same number of cells"):
            merge_height_slices([[0, 0], [0]])


if __name__ == "__main__":
    unittest.main()
