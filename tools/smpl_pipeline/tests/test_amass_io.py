from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

from analysis.amass_io import extract_clip_arrays, summarize_motion  # noqa: E402


class AmassIoTest(unittest.TestCase):
    def _write_motion(self, root: Path) -> Path:
        path = root / "01" / "01_01_poses.npz"
        path.parent.mkdir(parents=True)
        fps = 10.0
        trans = np.zeros((11, 3), dtype=np.float64)
        trans[:, 0] = np.linspace(0.0, 1.0, 11)
        trans[:, 2] = 0.9
        np.savez(
            path,
            trans=trans,
            poses=np.zeros((11, 156), dtype=np.float64),
            betas=np.zeros(16, dtype=np.float64),
            gender=np.asarray("female"),
            mocap_framerate=np.asarray(fps),
            dmpls=np.zeros((11, 8), dtype=np.float64),
        )
        return path

    def test_summarize_motion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._write_motion(root)
            result = summarize_motion(path, dataset_root=root)

        self.assertEqual(result.motion_id, "01_01")
        self.assertEqual(result.relative_path, "01/01_01_poses.npz")
        self.assertEqual(result.frames, 11)
        self.assertEqual(result.pose_dimensions, 156)
        self.assertEqual(result.beta_dimensions, 16)
        self.assertAlmostEqual(result.duration_sec, 1.0)
        self.assertAlmostEqual(result.path_length_m, 1.0)
        self.assertAlmostEqual(result.net_displacement_m, 1.0)
        self.assertAlmostEqual(result.average_speed_mps, 1.0)
        self.assertEqual(result.stationary_fraction, 0.0)

    def test_extract_clip_separates_planar_root_translation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._write_motion(root)
            result = extract_clip_arrays(path, start_sec=0.2, end_sec=0.6)

        self.assertEqual(result["poses"].shape, (5, 156))
        np.testing.assert_allclose(result["root_trajectory"][0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(result["root_trajectory"][-1], [0.4, 0.0, 0.0])
        np.testing.assert_allclose(result["local_trans"][:, 0], 0.2)
        np.testing.assert_allclose(result["local_trans"][:, 2], 0.9)

    def test_rejects_invalid_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._write_motion(root)
            with self.assertRaises(ValueError):
                extract_clip_arrays(path, start_sec=1.0, end_sec=0.5)


if __name__ == "__main__":
    unittest.main()
