import os
import unittest
from unittest import mock

from isaac_utils.scene_collision_probe import (
    SceneCollisionProbeConfig,
    build_cardinal_sweep_offsets,
    build_height_centers,
    match_target_hints,
    scene_collision_probe_config_from_env,
    target_hits,
)


class TestSceneCollisionProbe(unittest.TestCase):
    def test_defaults_include_required_target_hints_and_partition_discovery(self):
        config = SceneCollisionProbeConfig()

        self.assertEqual(
            config.target_hints,
            (
                "Edestalurinal_0000",
                "Edestalurinal_0001",
                "Edestalurinal_0002",
                "Edestalurinal_0003",
                "Edestalurinal_0004",
                "Toilet_0000",
                "Toilet_0001",
                "Door_0000",
            ),
        )
        self.assertEqual(config.discover_keywords, ("Partition", "Door"))
        self.assertEqual(config.chassis_half_extents, (0.20, 0.15, 0.20))

    def test_env_parsing_overrides_probe_settings(self):
        env = {
            "ARENA_ISAAC_SCENE_COLLISION_PROBE_OUTPUT": "/tmp/custom-scene-collisions.json",
            "ARENA_ISAAC_SCENE_COLLISION_PROBE_TARGET_HINTS": "Foo_0000,Bar_0001",
            "ARENA_ISAAC_SCENE_COLLISION_PROBE_DISCOVER_KEYWORDS": "Divider,Partition",
            "ARENA_ISAAC_SCENE_COLLISION_PROBE_CHASSIS_HALF_EXTENTS": "1.0,2.0,3.0",
            "ARENA_ISAAC_SCENE_COLLISION_PROBE_HEIGHTS": "0.1,0.2,0.3",
            "ARENA_ISAAC_SCENE_COLLISION_PROBE_MARGIN": "0.125",
            "ARENA_ISAAC_SCENE_COLLISION_PROBE_SWEEP_STEPS": "6",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            config = scene_collision_probe_config_from_env()

        self.assertEqual(config.output_path, "/tmp/custom-scene-collisions.json")
        self.assertEqual(config.target_hints, ("Foo_0000", "Bar_0001"))
        self.assertEqual(config.discover_keywords, ("Divider", "Partition"))
        self.assertEqual(config.chassis_half_extents, (1.0, 2.0, 3.0))
        self.assertEqual(config.probe_heights, (0.1, 0.2, 0.3))
        self.assertEqual(config.probe_margin, 0.125)
        self.assertEqual(config.sweep_steps, 6)

    def test_match_target_hints_is_case_insensitive_and_deduplicated(self):
        config = SceneCollisionProbeConfig(
            target_hints=("Toilet_0000", "Partition"),
            discover_keywords=("Partition",),
        )

        self.assertEqual(
            match_target_hints("/World/scene/toilet_0000/PartitionWall", "PartitionWall", config),
            ("Partition",),
        )

    def test_descendants_are_not_promoted_to_duplicate_targets(self):
        config = SceneCollisionProbeConfig(target_hints=("Edestalurinal_0000",))
        self.assertEqual(
            match_target_hints(
                "/World/scene/Edestalurinal_0000/Geom/Ceramic",
                "Ceramic",
                config,
            ),
            (),
        )

    def test_target_hits_exclude_neighboring_collision_paths(self):
        hits = [
            {"collision": "/World/scene/Toilet_0000/Geom/collider"},
            {"collision": "/World/scene/Wall_0000/Geom/collider"},
        ]
        self.assertEqual(
            target_hits(hits, "/World/scene/Toilet_0000"),
            [hits[0]],
        )

    def test_probe_geometry_helpers_stay_inside_bounds_when_possible(self):
        heights = build_height_centers(0.0, 1.0, 0.2, (0.1, 0.9, 1.2))
        self.assertEqual(heights, [0.1, 0.2, 0.5, 0.8, 0.9])

        sweep = build_cardinal_sweep_offsets(
            (0.6, 0.4),
            (0.3, 0.2),
            probe_margin=0.05,
            sweep_steps=4,
        )

        self.assertEqual(set(sweep), {"front", "rear", "left", "right"})
        self.assertEqual(sweep["front"], [0.0, 0.2375, 0.475, 0.7125, 0.95])
        self.assertEqual(sweep["right"][-1], 0.95)


if __name__ == "__main__":
    unittest.main()
