import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_SCRIPT = REPO_ROOT / "scripts" / "arena_scene_profile.py"
PROFILE_PATH = REPO_ROOT / "scripts" / "profiles" / "shenxinfu_841837.yaml"
MODE_DIR = PROFILE_PATH.parent / "modes"

spec = importlib.util.spec_from_file_location("arena_scene_profile", PROFILE_SCRIPT)
profile_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(profile_module)


class TestPhysxDiffProfile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = profile_module.load_profile(str(PROFILE_PATH))

    def test_spawn_phase_selects_contact_asset_without_changing_default(self):
        default_cmd = profile_module.spawn_cmd(self.profile)
        contact_cmd = profile_module.spawn_cmd(self.profile, "physx_diff_contact")

        self.assertIn("mecanum730_xms5_lidar_physx_wheels", default_cmd)
        self.assertIn("mecanum730_xms5_lidar_physx_diff_contact", contact_cmd)
        urdf = contact_cmd[contact_cmd.index("--urdf-path") + 1]
        self.assertTrue(urdf.endswith("mecanum730_xms5_physx_diff_contact.urdf"))
        self.assertEqual(contact_cmd[contact_cmd.index("--x") + 1], "0.0")
        self.assertEqual(contact_cmd[contact_cmd.index("--y") + 1], "0.0")
        self.assertEqual(contact_cmd[contact_cmd.index("--z") + 1], "0.03")

    def test_profile_composes_explicit_mode_files(self):
        self.assertEqual(self.profile["default_mode"], "physx_wheels")
        self.assertEqual(
            set(self.profile["modes"]),
            {"physx_wheels", "physx_diff_contact"},
        )
        self.assertTrue((MODE_DIR / "physx_wheels.yaml").is_file())
        self.assertTrue((MODE_DIR / "physx_diff_contact.yaml").is_file())
        self.assertEqual(
            self.profile["phases"]["physx_diff_contact"]["mode"],
            "physx_diff_contact",
        )

    def test_bridge_phase_disables_guard_handoff_and_hard_sync(self):
        cmd, env = profile_module.bridge_cmd(self.profile, "physx_diff_contact")
        joined = " ".join(cmd)
        self.assertIn("enable_kinematic_collision_guard:=false", joined)
        self.assertIn("enable_people_stack:=true", joined)
        self.assertIn("enable_character_services:=true", joined)
        self.assertIn("people_extension_mode:=replicator_agent_core", joined)
        self.assertIn("enable_navmesh:=false", joined)
        self.assertEqual(env["ARENA_ISAAC_COLLISION_GUARD_BACKEND"], "none")
        self.assertEqual(env["ARENA_ISAAC_PEDESTRIAN_ROBOT_POLICY"], "stop")
        self.assertEqual(env["ARENA_ISAAC_PEDESTRIAN_PHYSICS_PROXY_ENABLED"], "false")
        self.assertEqual(env["ARENA_ISAAC_PEDESTRIAN_HARD_GUARD_ENABLED"], "true")
        self.assertEqual(env["ARENA_ISAAC_PEDESTRIAN_HARD_GUARD_MARGIN_M"], "0.04")
        self.assertEqual(env["ARENA_ISAAC_PEDESTRIAN_HARD_GUARD_LATENCY_SEC"], "0.1")
        self.assertEqual(env["ARENA_ISAAC_PEDESTRIAN_HARD_GUARD_SAMPLE_DT_SEC"], "0.04")
        self.assertEqual(env["ARENA_ISAAC_PEDESTRIAN_HARD_GUARD_HORIZON_SEC"], "0.35")
        self.assertEqual(env["ARENA_ISAAC_PEDESTRIAN_HARD_GUARD_RELEASE_MARGIN_M"], "0.08")
        self.assertEqual(env["ARENA_ISAAC_PEDESTRIAN_HARD_GUARD_RELEASE_HOLD_SEC"], "0.25")
        self.assertEqual(env["ARENA_ISAAC_PEDESTRIAN_HARD_GUARD_ESCAPE_HORIZON_SEC"], "0.35")
        self.assertEqual(env["ARENA_ISAAC_PEDESTRIAN_HARD_GUARD_OVERLAP_DEADBAND_M"], "0.015")
        self.assertEqual(env["ARENA_ISAAC_SCENE_DOOR_COLLISION_POLICY"], "restore_selected_meshes")
        self.assertEqual(
            env["ARENA_ISAAC_SCENE_DOOR_COLLISION_TARGETS"],
            "Door_YS3UEUZVAJ2V4PTUKU888888_mdynP,"
            "Door_YS2HVXZVALGD2PTUJQ888888_7qcYe,Door_0000",
        )
        self.assertEqual(
            env["ARENA_ISAAC_SCENE_DOOR_COLLISION_MESHES"],
            "Door,DoorFrame,Door_Frame,Static,A06865e7a93064541911395c3271e1c34",
        )
        self.assertEqual(env["ARENA_ISAAC_SCENE_DOOR_LEAF_APPROXIMATION"], "convexHull")
        self.assertEqual(env["ARENA_ISAAC_SCENE_DOOR_FRAME_APPROXIMATION"], "sdf")
        self.assertEqual(env["ARENA_ISAAC_SCENE_DOOR_FRAME_SDF_RESOLUTION"], "512")
        self.assertEqual(env["ARENA_ISAAC_FULL_ASSET_HANDOFF_ENABLED"], "false")
        self.assertEqual(env["ARENA_ISAAC_ARM_HOLD_HARD_SYNC"], "false")
        self.assertEqual(env["ARENA_ISAAC_DIFF_WHEEL_DRIVE_DAMPING"], "30.0")
        self.assertEqual(env["ARENA_ISAAC_DIFF_WHEEL_DRIVE_MAX_FORCE"], "300.0")
        self.assertEqual(env["ARENA_ISAAC_DIFF_WHEEL_STATIC_FRICTION"], "0.02")
        self.assertEqual(env["ARENA_ISAAC_DIFF_WHEEL_DYNAMIC_FRICTION"], "0.01")
        self.assertEqual(env["ARENA_ISAAC_DIFF_WHEEL_FRICTION_COMBINE_MODE"], "min")
        self.assertEqual(env["ARENA_ISAAC_DIFF_WHEEL_RADIUS"], "0.08")
        self.assertEqual(env["ARENA_ISAAC_DIFF_WHEEL_SIGNS"], "1.0,1.0,1.0,1.0")
        self.assertEqual(env["ARENA_ISAAC_DIFF_MAX_WHEEL_SPEED"], "20.0")
        self.assertEqual(env["ARENA_ISAAC_DIFF_LINEAR_GAIN"], "1.0")
        self.assertEqual(env["ARENA_ISAAC_DIFF_ANGULAR_GAIN"], "1.0")
        self.assertEqual(env["ARENA_ISAAC_DIFF_TIRE_FORCE_ENABLED"], "true")
        self.assertEqual(env["ARENA_ISAAC_DIFF_TIRE_LONGITUDINAL_STIFFNESS"], "60.0")
        self.assertEqual(env["ARENA_ISAAC_DIFF_TIRE_LATERAL_STIFFNESS"], "20.0")
        self.assertEqual(env["ARENA_ISAAC_DIFF_TIRE_MAX_LONGITUDINAL_FORCE"], "12.0")
        self.assertEqual(env["ARENA_ISAAC_DIFF_TIRE_MAX_LATERAL_FORCE"], "4.0")
        self.assertEqual(env["ARENA_ISAAC_DIFF_TIRE_CONTACT_REFRESH_SEC"], "0.1")
        self.assertEqual(
            env["ARENA_ISAAC_DIFF_DIAGNOSTICS_OUTPUT"],
            "/tmp/arena_physx_diff_diagnostics.jsonl",
        )
        self.assertEqual(
            env["ARENA_ISAAC_DIFF_DIAGNOSTICS_RUN_LABEL"],
            "separated_tire_v1_k60_20_f12_4",
        )

    def test_physx_wheels_does_not_inherit_contact_wheel_parameters(self):
        contact_keys = {
            "ARENA_ISAAC_DIFF_WHEEL_DRIVE_DAMPING",
            "ARENA_ISAAC_DIFF_WHEEL_DRIVE_MAX_FORCE",
            "ARENA_ISAAC_DIFF_WHEEL_STATIC_FRICTION",
            "ARENA_ISAAC_DIFF_WHEEL_DYNAMIC_FRICTION",
            "ARENA_ISAAC_DIFF_WHEEL_FRICTION_COMBINE_MODE",
            "ARENA_ISAAC_DIFF_WHEEL_RADIUS",
            "ARENA_ISAAC_DIFF_MAX_WHEEL_SPEED",
            "ARENA_ISAAC_DIFF_WHEEL_SIGNS",
            "ARENA_ISAAC_DIFF_LINEAR_GAIN",
            "ARENA_ISAAC_DIFF_ANGULAR_GAIN",
            "ARENA_ISAAC_DIFF_TIRE_FORCE_ENABLED",
            "ARENA_ISAAC_DIFF_TIRE_LONGITUDINAL_STIFFNESS",
            "ARENA_ISAAC_DIFF_TIRE_LATERAL_STIFFNESS",
            "ARENA_ISAAC_DIFF_TIRE_MAX_LONGITUDINAL_FORCE",
            "ARENA_ISAAC_DIFF_TIRE_MAX_LATERAL_FORCE",
            "ARENA_ISAAC_DIFF_TIRE_CONTACT_REFRESH_SEC",
            "ARENA_ISAAC_DIFF_DIAGNOSTICS_OUTPUT",
            "ARENA_ISAAC_DIFF_DIAGNOSTICS_RUN_LABEL",
            "ARENA_ISAAC_PEDESTRIAN_HARD_GUARD_RELEASE_MARGIN_M",
            "ARENA_ISAAC_PEDESTRIAN_HARD_GUARD_RELEASE_HOLD_SEC",
            "ARENA_ISAAC_PEDESTRIAN_HARD_GUARD_ESCAPE_HORIZON_SEC",
            "ARENA_ISAAC_PEDESTRIAN_HARD_GUARD_OVERLAP_DEADBAND_M",
        }
        with mock.patch.dict(os.environ, {key: "stale" for key in contact_keys}):
            command, env = profile_module.bridge_cmd(self.profile, "rtx_scan")
        self.assertTrue(contact_keys.isdisjoint(env))
        self.assertIn("enable_kinematic_collision_guard:=true", " ".join(command))
        self.assertEqual(env["ARENA_ISAAC_COLLISION_GUARD_BACKEND"], "voxel")
        self.assertEqual(env["ARENA_ISAAC_MAX_LINEAR_SPEED"], "0.8")
        self.assertEqual(env["ARENA_ISAAC_SCENE_DOOR_COLLISION_POLICY"], "disabled")
        self.assertEqual(env["ARENA_ISAAC_PEDESTRIAN_ROBOT_POLICY"], "avoid")
        self.assertEqual(env["ARENA_ISAAC_PEDESTRIAN_PHYSICS_PROXY_ENABLED"], "true")
        self.assertEqual(env["ARENA_ISAAC_PEDESTRIAN_HARD_GUARD_ENABLED"], "false")

    def test_contact_gamepad_uses_explicit_faster_limits(self):
        command = profile_module.gamepad_cmd(self.profile, "physx_diff_contact")
        self.assertIn("linear_scale:=0.4", command)
        self.assertIn("angular_scale:=0.8", command)


if __name__ == "__main__":
    unittest.main()
