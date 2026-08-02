import os
from types import ModuleType
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
resolve_skeleton_data = (
    PEDESTRIAN_VISUAL_ENVELOPE.resolve_skeleton_data
)
cached_skeleton_topology = (
    PEDESTRIAN_VISUAL_ENVELOPE._cached_skeleton_topology
)
prune_skeleton_topology_cache = (
    PEDESTRIAN_VISUAL_ENVELOPE._prune_skeleton_topology_cache
)
SkeletonTopologyCacheEntry = (
    PEDESTRIAN_VISUAL_ENVELOPE._SkeletonTopologyCacheEntry
)
SKELETON_TOPOLOGY_CACHE = (
    PEDESTRIAN_VISUAL_ENVELOPE._SKELETON_TOPOLOGY_CACHE
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
    def tearDown(self):
        SKELETON_TOPOLOGY_CACHE.clear()

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

    def test_topology_cache_is_scoped_to_character_instance(self):
        root_path = "/World/Characters/toilet_agent_01/SkelRoot"
        person = _DummyPerson("toilet_agent_01", root_path)
        entry = SkeletonTopologyCacheEntry(
            person_identity=id(person),
            joint_order=("Root", "Spine"),
            parent_indices=(-1, 0),
        )
        SKELETON_TOPOLOGY_CACHE[root_path] = entry

        self.assertIs(cached_skeleton_topology(person, root_path), entry)
        self.assertIsNone(
            cached_skeleton_topology(
                _DummyPerson("toilet_agent_01", root_path),
                root_path,
            )
        )

    def test_topology_cache_is_pruned_after_character_leaves(self):
        root_path = "/World/Characters/toilet_agent_01/SkelRoot"
        person = _DummyPerson("toilet_agent_01", root_path)
        SKELETON_TOPOLOGY_CACHE[root_path] = SkeletonTopologyCacheEntry(
            person_identity=id(person),
            joint_order=("Root",),
            parent_indices=(-1,),
        )

        prune_skeleton_topology_cache({"toilet_agent_01": person})
        self.assertIn(root_path, SKELETON_TOPOLOGY_CACHE)

        prune_skeleton_topology_cache({})
        self.assertNotIn(root_path, SKELETON_TOPOLOGY_CACHE)

    def test_skeleton_topology_native_cache_is_built_once_per_character(self):
        root_path = "/World/Characters/toilet_agent_01/SkelRoot"

        class _Vector(list):
            def __init__(self, *values):
                super().__init__(values)

        class _CharacterGraph:
            def get_joint_transform(self, name, position, _rotation):
                position[:] = [1.0 if name == "Root" else 2.0, 0.0, 0.0]

        class _Person(_DummyPerson):
            def __init__(self):
                super().__init__("toilet_agent_01", root_path)
                self.character_graph = _CharacterGraph()

        skeleton = _FakePrim("Skeleton")
        skeleton.IsA = lambda _schema: True
        skel_root = _FakePrim("SkelRoot", (skeleton,))
        skel_root.GetTypeName = lambda: "SkelRoot"

        class _Stage:
            def GetPrimAtPath(self, path):
                return skel_root if path == root_path else None

        class _Context:
            def get_stage(self):
                return _Stage()

        cache_builds = []

        class _Topology:
            def GetParentIndices(self):
                return (-1, 0)

        class _Query:
            def __bool__(self):
                return True

            def GetTopology(self):
                return _Topology()

            def GetJointOrder(self):
                return ("Root", "Root/Spine")

        class _Cache:
            def __init__(self):
                cache_builds.append(1)

            def Populate(self, *_args):
                return None

            def GetSkelQuery(self, _skeleton):
                return _Query()

        class _Usd:
            @staticmethod
            def TraverseInstanceProxies():
                return None

            @staticmethod
            def PrimRange(_root, *_args):
                return (skeleton,)

        class _UsdSkel:
            Skeleton = staticmethod(lambda prim: prim)
            Root = staticmethod(lambda prim: prim)
            Cache = _Cache

        carb_module = ModuleType("carb")
        carb_module.Float3 = _Vector
        carb_module.Float4 = _Vector
        omni_module = ModuleType("omni")
        omni_module.__path__ = []
        omni_usd_module = ModuleType("omni.usd")
        omni_usd_module.get_context = lambda: _Context()
        omni_module.usd = omni_usd_module
        pxr_module = ModuleType("pxr")
        pxr_module.Usd = _Usd
        pxr_module.UsdSkel = _UsdSkel
        person = _Person()

        with mock.patch.dict(
            sys.modules,
            {
                "carb": carb_module,
                "omni": omni_module,
                "omni.usd": omni_usd_module,
                "pxr": pxr_module,
            },
        ):
            first = resolve_skeleton_data(person, root_path)
            second = resolve_skeleton_data(person, root_path)

        self.assertEqual(len(cache_builds), 1)
        self.assertEqual(first, second)
        self.assertEqual(first.parent_indices, (-1, 0))

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
