import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from ros2isaacsim.physx_diff_asset import (
    ARM_FOLDED_POSITIONS,
    LOCAL_MESH_DIR,
    PHYSX_DIFF_CONTACT_BASE_COLLIDER_SPECS,
    PHYSX_DIFF_CONTACT_UPPER_BODY_COLLIDER_SPECS,
    ROLLER_JOINT_TOKEN,
    SOURCE_URDF,
    WHEEL_JOINTS,
    WHEEL_LINKS,
    build_physx_diff_urdf,
)


class TestPhysxDiffAsset(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output = Path(self.temp_dir.name) / "physx_diff.urdf"
        build_physx_diff_urdf(SOURCE_URDF, self.output, LOCAL_MESH_DIR)
        self.robot = ET.parse(self.output).getroot()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_removes_all_urdf_collisions_before_isaac_import(self):
        self.assertEqual(self.robot.findall(".//collision"), [])

    def test_contact_lod_uses_a_small_link_local_primitive_set(self):
        specs = (
            PHYSX_DIFF_CONTACT_BASE_COLLIDER_SPECS
            + PHYSX_DIFF_CONTACT_UPPER_BODY_COLLIDER_SPECS
        )
        self.assertEqual(len(specs), 11)
        self.assertEqual(len({spec["name"] for spec in specs}), len(specs))
        normalized_link_names = {
            link.attrib["name"].replace("-", "_") for link in self.robot.findall("link")
        }
        for spec in specs:
            self.assertIn(spec["link"], normalized_link_names)
            self.assertIn(spec["kind"], {"cube", "capsule"})
            self.assertEqual(len(spec["translate"]), 3)
            if spec["kind"] == "cube":
                self.assertTrue(all(value > 0.0 for value in spec["size"]))
            else:
                self.assertIn(spec["axis"], {"x", "y", "z"})
                self.assertGreater(spec["radius"], 0.0)
                self.assertGreater(spec["height"], 0.0)

    def test_base_collision_lod_has_no_vertical_gap(self):
        base_cubes = [
            spec
            for spec in (
                PHYSX_DIFF_CONTACT_BASE_COLLIDER_SPECS
                + PHYSX_DIFF_CONTACT_UPPER_BODY_COLLIDER_SPECS
            )
            if spec["link"] == "base_link" and spec["kind"] == "cube"
        ]
        intervals = sorted(
            (
                spec["translate"][2] - 0.5 * spec["size"][2],
                spec["translate"][2] + 0.5 * spec["size"][2],
            )
            for spec in base_cubes
        )
        self.assertEqual(len(intervals), 3)
        self.assertLessEqual(intervals[0][0], 0.10)
        self.assertGreaterEqual(intervals[-1][1], 1.09)
        for current, following in zip(intervals, intervals[1:]):
            self.assertLessEqual(following[0], current[1] + 1.0e-6)

    def test_sets_explicit_wheel_inertia(self):
        for link_name in WHEEL_LINKS:
            link = self.robot.find(f"./link[@name='{link_name}']")
            self.assertIsNotNone(link, link_name)
            self.assertEqual(link.find("inertial/mass").attrib["value"], "2")
            self.assertEqual(
                link.find("inertial/inertia").attrib,
                {
                    "ixx": "0.0064",
                    "ixy": "0",
                    "ixz": "0",
                    "iyy": "0.00361667",
                    "iyz": "0",
                    "izz": "0.00361667",
                },
            )

    def test_wheel_joints_remain_continuous_with_contact_damping(self):
        for joint_name in WHEEL_JOINTS:
            joint = self.robot.find(f"./joint[@name='{joint_name}']")
            self.assertIsNotNone(joint, joint_name)
            self.assertEqual(joint.attrib["type"], "continuous")
            self.assertEqual(joint.find("dynamics").attrib["damping"], "0.05")

    def test_mecanum_rollers_are_fixed_visual_geometry(self):
        roller_joints = [
            joint
            for joint in self.robot.findall("joint")
            if ROLLER_JOINT_TOKEN in joint.attrib.get("name", "").lower()
        ]
        self.assertEqual(len(roller_joints), 28)
        for joint in roller_joints:
            self.assertEqual(joint.attrib["type"], "fixed")
            self.assertIsNone(joint.find("axis"))
            self.assertIsNone(joint.find("limit"))

    def test_preserves_manipulator_and_uses_only_local_meshes(self):
        self.assertIsNotNone(self.robot.find("./joint[@name='joint1']"))
        self.assertIsNotNone(self.robot.find("./link[@name='base_scan_01']"))
        self.assertIsNotNone(self.robot.find("./link[@name='base_scan_02']"))
        for mesh in self.robot.findall(".//mesh"):
            filename = mesh.attrib["filename"]
            self.assertFalse(Path(filename).is_absolute(), filename)
            resolved = (self.output.parent / filename).resolve()
            expected = LOCAL_MESH_DIR / resolved.name
            self.assertTrue(expected.exists(), filename)

    def test_bakes_folded_manipulator_as_fixed_joints_without_collision_prototypes(self):
        for joint_name in ARM_FOLDED_POSITIONS:
            fixed_joint = self.robot.find(f"./joint[@name='{joint_name}']")
            self.assertEqual(fixed_joint.attrib["type"], "fixed")
            self.assertIsNone(fixed_joint.find("axis"))
            self.assertIsNone(fixed_joint.find("limit"))

        transmitted_joints = {
            joint.attrib.get("name")
            for transmission in self.robot.findall("transmission")
            for joint in transmission.findall("joint")
        }
        self.assertTrue(set(ARM_FOLDED_POSITIONS).isdisjoint(transmitted_joints))


if __name__ == "__main__":
    unittest.main()
