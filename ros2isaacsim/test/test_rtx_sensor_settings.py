import unittest

from ros2isaacsim.isaac_utils.rtx_sensor_settings import (
    RTX_SENSOR_COORD_FRAME_DEFAULT,
    RTX_SENSOR_COORD_FRAME_REPLICATOR_AGENT,
    desired_rtx_sensor_coord_frame,
)


class TestRtxSensorSettings(unittest.TestCase):
    def test_auto_restores_isaac_default_for_rtx_backend(self):
        self.assertEqual(
            desired_rtx_sensor_coord_frame(lidar_backend="rtx", mode="auto"),
            RTX_SENSOR_COORD_FRAME_DEFAULT,
        )

    def test_auto_skips_synthetic_backends(self):
        self.assertIsNone(
            desired_rtx_sensor_coord_frame(lidar_backend="synthetic_2d", mode="auto")
        )

    def test_keep_mode_preserves_current_setting(self):
        self.assertIsNone(
            desired_rtx_sensor_coord_frame(lidar_backend="rtx", mode="keep")
        )

    def test_replicator_mode_uses_agent_frame(self):
        self.assertEqual(
            desired_rtx_sensor_coord_frame(lidar_backend="rtx", mode="replicator_agent_core"),
            RTX_SENSOR_COORD_FRAME_REPLICATOR_AGENT,
        )


if __name__ == "__main__":
    unittest.main()
