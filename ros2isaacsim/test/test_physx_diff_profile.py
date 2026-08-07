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

    def test_spawn_phase_selects_contact_asset_by_default_and_for_rtx_scan(self):
        default_cmd = profile_module.spawn_cmd(self.profile)
        rtx_cmd = profile_module.spawn_cmd(self.profile, "rtx_scan")
        contact_cmd = profile_module.spawn_cmd(self.profile, "physx_diff_contact")

        self.assertIn("mecanum730_xms5_lidar_physx_diff_contact", default_cmd)
        self.assertIn("mecanum730_xms5_lidar_physx_diff_contact", rtx_cmd)
        self.assertIn("mecanum730_xms5_lidar_physx_diff_contact", contact_cmd)
        urdf = contact_cmd[contact_cmd.index("--urdf-path") + 1]
        self.assertTrue(urdf.endswith("mecanum730_xms5_physx_diff_contact.urdf"))
        self.assertEqual(contact_cmd[contact_cmd.index("--x") + 1], "0.0")
        self.assertEqual(contact_cmd[contact_cmd.index("--y") + 1], "0.0")
        self.assertEqual(contact_cmd[contact_cmd.index("--z") + 1], "0.03")

    def test_profile_composes_explicit_mode_files(self):
        self.assertEqual(self.profile["default_mode"], "physx_diff_contact")
        self.assertEqual(set(self.profile["modes"]), {"physx_diff_contact"})
        mode_files = sorted(path.name for path in MODE_DIR.glob("*.yaml"))
        self.assertEqual(mode_files, ["physx_diff_contact.yaml"])
        self.assertEqual(
            self.profile["phases"]["physx_diff_contact"]["mode"],
            "physx_diff_contact",
        )
        self.assertEqual(self.profile["phases"]["rtx_scan"]["mode"], "physx_diff_contact")

    def test_bridge_phase_disables_guard_handoff_and_hard_sync(self):
        cmd, env = profile_module.bridge_cmd(self.profile, "physx_diff_contact")
        joined = " ".join(cmd)
        self.assertNotIn("kinematic_collision_guard", joined)
        self.assertIn("enable_people_stack:=true", joined)
        self.assertIn("enable_character_services:=true", joined)
        self.assertIn("people_extension_mode:=replicator_agent_core", joined)
        self.assertIn("enable_navmesh:=false", joined)
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
        self.assertEqual(env["ARENA_ISAAC_PEDESTRIAN_HARD_BODY_RADIUS_M"], "0.26")
        self.assertEqual(env["ARENA_ISAAC_PEDESTRIAN_HARD_BODY_HALF_LENGTH_M"], "0.16")
        self.assertEqual(
            env["ARENA_ISAAC_PEDESTRIAN_HARD_BODY_AXIS_SAMPLE_SPACING_M"],
            "0.05",
        )
        self.assertEqual(env["ARENA_ISAAC_ROBOT_FOOTPRINT_LENGTH"], "0.7")
        self.assertEqual(env["ARENA_ISAAC_ROBOT_FOOTPRINT_WIDTH"], "0.42")
        self.assertEqual(env["ARENA_ISAAC_ROBOT_FOOTPRINT_MARGIN"], "0.02")
        self.assertTrue(
            env["ARENA_ISAAC_PEDESTRIAN_WALKABLE_MAP_PATH"].endswith(
                "shenxinfu_841837.walkable.json"
            )
        )
        self.assertFalse(any(key.startswith("ARENA_ISAAC_COLLISION_GUARD_") for key in env))
        self.assertEqual(
            env["ARENA_ISAAC_EXTERNAL_MOTION_ANIMATION_FULL_SPEED_MPS"],
            "0.3082",
        )
        self.assertEqual(
            env["ARENA_ISAAC_EXTERNAL_MOTION_ANIMATION_SPEED_EXPONENT"],
            "1.767",
        )
        self.assertEqual(env["ARENA_ISAAC_EXTERNAL_TURN_SLOW_ANGLE_DEG"], "20.0")
        self.assertEqual(
            env["ARENA_ISAAC_EXTERNAL_TURN_FULL_SLOW_ANGLE_DEG"],
            "55.0",
        )
        self.assertEqual(env["ARENA_ISAAC_EXTERNAL_TURN_MIN_SPEED_SCALE"], "0.25")
        self.assertEqual(env["ARENA_ISAAC_EXTERNAL_MOTION_YAW_RATE_RADPS"], "2.8")
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
        self.assertEqual(env["ARENA_ISAAC_LIDAR_UPDATE_RATE"], "15.0")
        self.assertEqual(env["ARENA_ISAAC_LIDAR_DEDUPLICATE"], "true")
        self.assertEqual(env["ARENA_ISAAC_LIDAR_FRONT_TOPIC"], "/front_scan")
        self.assertEqual(env["ARENA_ISAAC_LIDAR_REAR_TOPIC"], "/rear_scan")

    def test_rtx_scan_uses_contact_mode_without_collision_guard_envs(self):
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
        }
        with mock.patch.dict(os.environ, {key: "stale" for key in contact_keys}):
            command, env = profile_module.bridge_cmd(self.profile, "rtx_scan")
        joined = " ".join(command)
        self.assertNotIn("kinematic_collision_guard", joined)
        self.assertIn("enable_scene_collision_repair:=true", joined)
        self.assertIn("scene_collision_proxy_source:=off", joined)
        self.assertEqual(env["ARENA_ISAAC_ROBOT_FOOTPRINT_LENGTH"], "0.7")
        self.assertEqual(env["ARENA_ISAAC_ROBOT_FOOTPRINT_WIDTH"], "0.42")
        self.assertEqual(env["ARENA_ISAAC_ROBOT_FOOTPRINT_MARGIN"], "0.02")
        self.assertFalse(any(key.startswith("ARENA_ISAAC_COLLISION_GUARD_") for key in env))
        self.assertEqual(env["ARENA_ISAAC_LIDAR_BACKEND"], "rtx")
        self.assertEqual(env["ARENA_ISAAC_LIDAR_DEDUPLICATE"], "true")
        self.assertEqual(env["ARENA_ISAAC_DIFF_WHEEL_DRIVE_DAMPING"], "30.0")
        self.assertEqual(env["ARENA_ISAAC_DIFF_TIRE_FORCE_ENABLED"], "true")
        self.assertEqual(env["ARENA_ISAAC_SCENE_DOOR_COLLISION_POLICY"], "restore_selected_meshes")
        self.assertEqual(env["ARENA_ISAAC_PEDESTRIAN_ROBOT_POLICY"], "stop")
        self.assertEqual(env["ARENA_ISAAC_PEDESTRIAN_PHYSICS_PROXY_ENABLED"], "false")
        self.assertEqual(env["ARENA_ISAAC_PEDESTRIAN_HARD_GUARD_ENABLED"], "true")

    def test_contact_operator_inputs_share_real_robot_limits(self):
        command = profile_module.gamepad_cmd(self.profile, "physx_diff_contact")
        self.assertIn("linear_scale:=0.4", command)
        self.assertIn("angular_scale:=0.4", command)
        self.assertIn("publish_rate_hz:=50.0", command)
        keyboard = profile_module.teleop_cmd(self.profile)
        self.assertIn("/cmd_vel_gamepad_diff", keyboard)
        self.assertIn("--linear", keyboard)
        self.assertIn("0.4", keyboard)
        self.assertNotIn("--lateral", keyboard)

if __name__ == "__main__":
    unittest.main()
