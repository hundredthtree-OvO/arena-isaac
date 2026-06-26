#!/usr/bin/env python3
"""Client for /isaac/build_voxel_map.

Run after the bridge is up and the scene has been imported into Isaac.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, List, Tuple

import rclpy
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from std_srvs.srv import Trigger


def _param(name: str, value: Any) -> Parameter:
    p = Parameter()
    p.name = name
    pv = ParameterValue()
    if isinstance(value, bool):
        pv.type = ParameterType.PARAMETER_BOOL
        pv.bool_value = bool(value)
    elif isinstance(value, int) and not isinstance(value, bool):
        pv.type = ParameterType.PARAMETER_INTEGER
        pv.integer_value = int(value)
    elif isinstance(value, float):
        pv.type = ParameterType.PARAMETER_DOUBLE
        pv.double_value = float(value)
    else:
        pv.type = ParameterType.PARAMETER_STRING
        pv.string_value = str(value)
    p.value = pv
    return p


def _call_set_params(node, params: List[Tuple[str, Any]], timeout: float) -> None:
    client = node.create_client(SetParameters, "/isaac_controller/set_parameters")
    if not client.wait_for_service(timeout_sec=timeout):
        raise RuntimeError("/isaac_controller/set_parameters service not available")
    req = SetParameters.Request()
    req.parameters = [_param(name, value) for name, value in params]
    fut = client.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=timeout)
    if not fut.done() or fut.result() is None:
        raise RuntimeError("timed out setting voxel parameters")
    bad = [r.reason for r in fut.result().results if not r.successful]
    if bad:
        raise RuntimeError("failed to set parameters: " + "; ".join(bad))


def _call_trigger(node, timeout: float):
    client = node.create_client(Trigger, "/isaac/build_voxel_map")
    if not client.wait_for_service(timeout_sec=timeout):
        raise RuntimeError("/isaac/build_voxel_map service not available")
    fut = client.call_async(Trigger.Request())
    rclpy.spin_until_future_complete(node, fut, timeout_sec=timeout)
    if not fut.done() or fut.result() is None:
        raise RuntimeError("timed out calling /isaac/build_voxel_map")
    return fut.result()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", default="/World/shenxinfu_841837")
    parser.add_argument("--output", default="/home/stardust/resources/arena_ws/arena_assets/collision_configs/shenxinfu_841837.voxel.json.gz")
    parser.add_argument("--debug-svg", default="")
    parser.add_argument("--debug-pcd", default="")
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--sample-step", type=float, default=0.05)
    parser.add_argument("--z-min", type=float, default=0.05)
    parser.add_argument("--z-max", type=float, default=1.20)
    parser.add_argument("--skip-keywords", default="")
    parser.add_argument("--include-keywords", default="")
    parser.add_argument("--max-faces-per-mesh", type=int, default=100000)
    parser.add_argument("--max-samples-per-mesh", type=int, default=250000)
    parser.add_argument("--max-stored-voxels", type=int, default=300000)
    parser.add_argument("--max-debug-stage-points", type=int, default=30000)
    parser.add_argument("--create-stage-debug-points", default="true")
    parser.add_argument("--stage-debug-path", default="")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args(argv)

    out = args.output
    debug_svg = args.debug_svg or out.replace(".json.gz", ".svg").replace(".json", ".svg")
    debug_pcd = args.debug_pcd or out.replace(".json.gz", ".pcd").replace(".json", ".pcd")

    rclpy.init()
    node = rclpy.create_node("export_voxel_map_client")
    try:
        params = [
            ("voxel_map_scene_root", args.scene_root),
            ("voxel_map_output_path", out),
            ("voxel_map_debug_svg_path", debug_svg),
            ("voxel_map_debug_pcd_path", debug_pcd),
            ("voxel_map_resolution", float(args.resolution)),
            ("voxel_map_sample_step", float(args.sample_step)),
            ("voxel_map_z_min", float(args.z_min)),
            ("voxel_map_z_max", float(args.z_max)),
            ("voxel_map_skip_keywords", args.skip_keywords),
            ("voxel_map_include_keywords", args.include_keywords),
            ("voxel_map_max_faces_per_mesh", int(args.max_faces_per_mesh)),
            ("voxel_map_max_samples_per_mesh", int(args.max_samples_per_mesh)),
            ("voxel_map_max_stored_voxels", int(args.max_stored_voxels)),
            ("voxel_map_max_debug_stage_points", int(args.max_debug_stage_points)),
            ("voxel_map_create_stage_debug_points", str(args.create_stage_debug_points)),
            ("voxel_map_stage_debug_path", str(args.stage_debug_path)),
        ]
        _call_set_params(node, params, args.timeout)
        result = _call_trigger(node, args.timeout)
        print(result.message)
        return 0 if result.success else 2
    except Exception as exc:
        print(f"export_voxel_map failed: {exc}", file=sys.stderr)
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
