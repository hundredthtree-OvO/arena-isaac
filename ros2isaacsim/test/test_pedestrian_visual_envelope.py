import os
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from unittest import mock


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


MODULE_PATH = (
    PACKAGE_ROOT
    / "isaac_utils"
    / "services"
    / "pedestrian_visual_envelope.py"
)
SPEC = spec_from_file_location("pedestrian_visual_envelope", MODULE_PATH)
PEDESTRIAN_VISUAL_ENVELOPE = module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = PEDESTRIAN_VISUAL_ENVELOPE
SPEC.loader.exec_module(PEDESTRIAN_VISUAL_ENVELOPE)

PedestrianVisualEnvelopeConfig = (
    PEDESTRIAN_VISUAL_ENVELOPE.PedestrianVisualEnvelopeConfig
)
SkeletonVisualEnvelopeData = (
    PEDESTRIAN_VISUAL_ENVELOPE.SkeletonVisualEnvelopeData
)
collect_visual_envelope_payload = (
    PEDESTRIAN_VISUAL_ENVELOPE.collect_visual_envelope_payload
)
convex_hull_xy = PEDESTRIAN_VISUAL_ENVELOPE.convex_hull_xy
find_skeleton_prim = PEDESTRIAN_VISUAL_ENVELOPE._find_skeleton_prim
pedestrian_visual_envelope_config_from_env = (
    PEDESTRIAN_VISUAL_ENVELOPE.pedestrian_visual_envelope_config_from_env
)
sample_visual_envelope_points = (
    PEDESTRIAN_VISUAL_ENVELOPE.sample_visual_envelope_points
)


class _DummyPerson:
    def __init__(self, name: str, root_path: str, active: bool = True):
        self._requested_stage_name = name
        self.character_skel_root_stage_path = root_path
        self._active = active


class _FakePrim:
    def __init__(self, type_name: str, children=()):
        self.type_name = type_name
        self.children = tuple(children)

    def IsValid(self):
        return True

    def IsA(self, schema):
        return self.type_name == schema

    def GetAllChildren(self):
        return self.children


class TestPedestrianVisualEnvelope(unittest.TestCase):
    def test_env_defaults_are_enabled_and_tunable(self):
        env = {
            "ARENA_ISAAC_ENABLE_PEDESTRIAN_VISUAL_ENVELOPE": "false",
            "ARENA_ISAAC_PEDESTRIAN_VISUAL_ENVELOPE_HZ": "3.5",
            "ARENA_ISAAC_PEDESTRIAN_VISUAL_ENVELOPE_RADIUS_M": "0.12",
            "ARENA_ISAAC_PEDESTRIAN_VISUAL_ENVELOPE_SEGMENT_SPACING_M": "0.07",
            "ARENA_ISAAC_PEDESTRIAN_VISUAL_ENVELOPE_TOPIC": "/isaac/test_envelope",
            "ARENA_ISAAC_PEDESTRIAN_VISUAL_ENVELOPE_LOG_PATH": "/tmp/envelope.jsonl",
            "ARENA_ISAAC_PEDESTRIAN_VISUAL_ENVELOPE_EXCLUDE_PREFIXES": "/World/A,/World/B",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = pedestrian_visual_envelope_config_from_env()

        self.assertFalse(config.enabled)
        self.assertAlmostEqual(config.publish_hz, 3.5)
        self.assertAlmostEqual(config.sample_radius_m, 0.12)
        self.assertAlmostEqual(config.segment_spacing_m, 0.07)
        self.assertEqual(config.topic_name, "/isaac/test_envelope")
        self.assertEqual(config.log_path, "/tmp/envelope.jsonl")
        self.assertEqual(config.exclude_prefixes, ("/World/A", "/World/B"))

    def test_sample_points_follow_topology_parent_indices(self):
        points = sample_visual_envelope_points(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            (-1, 0, 1),
            0.5,
        )

        self.assertEqual(
            points,
            [
                (0.0, 0.0, 0.0),
                (0.5, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.5, 0.0, 0.0),
                (2.0, 0.0, 0.0),
            ],
        )

    def test_convex_hull_xy_returns_outer_boundary(self):
        hull = convex_hull_xy(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.5, 0.5, 0.0),
            )
        )

        self.assertEqual(
            hull,
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
        )

    def test_skeleton_search_does_not_treat_skel_root_as_skeleton(self):
        skeleton = _FakePrim("Skeleton")
        skel_root = _FakePrim("SkelRoot", (skeleton,))

        class _FakeUsdSkel:
            Skeleton = "Skeleton"

        self.assertIs(find_skeleton_prim(skel_root, _FakeUsdSkel), skeleton)

    def test_payload_deduplicates_aliases_and_filters_self_hits(self):
        person = _DummyPerson("toilet_agent_01", "/World/Characters/toilet_agent_01/SkelRoot")
        inactive = _DummyPerson("toilet_agent_02", "/World/Characters/toilet_agent_02/SkelRoot", active=False)
        people_by_alias = {
            "toilet_agent_01": person,
            "/World/Characters/toilet_agent_01": person,
            "toilet_agent_02": inactive,
        }

        def resolve_skeleton_data(_person, _root_path):
            return SkeletonVisualEnvelopeData(
                joint_world_positions=(
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (2.0, 0.0, 0.0),
                ),
                parent_indices=(-1, 0, 1),
            )

        def overlap_query(point, _radius):
            if point[0] < 0.5:
                return []
            return [
                "/World/Characters/toilet_agent_01/SkelRoot",
                "/World/CharactersCollision/toilet_agent_01",
                "/World/xms_mecanum/base",
                "/World/robots/xms_mecanum/gripper_link1",
                "/World/groundPlane",
                "/World/shenxinfu_841837/Meshes/Base/Floor_0000/Geom/A77",
                "/World/shenxinfu_841837/CollisionFixes/dirty_stall_clean_floor",
                "/World/static/box",
            ]

        payload = collect_visual_envelope_payload(
            people_by_alias,
            skeleton_resolver=resolve_skeleton_data,
            overlap_query=overlap_query,
            config=PedestrianVisualEnvelopeConfig(
                enabled=True,
                publish_hz=2.0,
                sample_radius_m=0.08,
                segment_spacing_m=0.5,
            ),
            timestamp_iso="2026-07-30T12:00:00.000",
        )

        self.assertEqual(payload["timestamp"], "2026-07-30T12:00:00.000")
        self.assertEqual(payload["topic"], "/isaac/pedestrian_visual_envelopes")
        self.assertEqual(len(payload["agents"]), 1)

        record = payload["agents"][0]
        self.assertEqual(record["agent_id"], "toilet_agent_01")
        self.assertTrue(record["available"])
        self.assertEqual(record["joint_count"], 3)
        self.assertEqual(record["sample_count"], 5)
        self.assertEqual(record["xy_aabb"], {"min": [0.0, 0.0], "max": [2.0, 0.0]})
        self.assertEqual(record["hull"], [[0.0, 0.0], [2.0, 0.0]])
        self.assertEqual(record["overlap_sample_count"], 4)
        self.assertEqual(record["overlap_paths"], ["/World/static/box"])
        self.assertAlmostEqual(record["overlap_ratio"], 0.8)
        self.assertEqual(record["max_consecutive_overlap_samples"], 4)


if __name__ == "__main__":
    unittest.main()
