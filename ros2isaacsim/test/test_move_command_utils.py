import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "isaac_utils"
    / "services"
    / "move_command_utils.py"
)
SPEC = spec_from_file_location("move_command_utils", MODULE_PATH)
MOVE_COMMAND_UTILS = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOVE_COMMAND_UTILS)
resolve_nav_velocity = MOVE_COMMAND_UTILS.resolve_nav_velocity


class TestMoveCommandUtils(unittest.TestCase):
    def test_zero_velocity_is_preserved(self):
        self.assertEqual(resolve_nav_velocity(0.0), 0.0)

    def test_none_velocity_uses_default(self):
        self.assertEqual(resolve_nav_velocity(None, default=1.25), 1.25)

    def test_numeric_strings_are_accepted(self):
        self.assertEqual(resolve_nav_velocity("0.5"), 0.5)


if __name__ == "__main__":
    unittest.main()
