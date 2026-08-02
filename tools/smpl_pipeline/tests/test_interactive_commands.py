import unittest

from tools.smpl_pipeline.runtime.interactive_commands import (
    InteractiveCommandController,
)


class InteractiveCommandControllerTest(unittest.TestCase):
    def test_move_stop_and_speed_presets_are_persistent(self):
        controller = InteractiveCommandController()
        self.assertEqual(controller.command()[:2], (0.0, 0.0))

        controller.handle("move", True)
        self.assertEqual(controller.command()[:2], (0.95, 0.0))
        controller.handle("fast", True)
        self.assertEqual(controller.command()[:2], (1.20, 0.0))
        controller.handle("stop", True)
        self.assertEqual(controller.command()[:2], (0.0, 0.0))

    def test_turn_command_tracks_key_press_and_release(self):
        controller = InteractiveCommandController(moving=True)
        controller.handle("turn_left", True)
        self.assertEqual(controller.command()[:2], (0.95, 0.8))
        controller.handle("turn_left", False)
        controller.handle("turn_right", True)
        self.assertEqual(controller.command()[:2], (0.95, -0.8))
        controller.handle("turn_right", False)
        self.assertEqual(controller.command()[:2], (0.95, 0.0))


if __name__ == "__main__":
    unittest.main()
