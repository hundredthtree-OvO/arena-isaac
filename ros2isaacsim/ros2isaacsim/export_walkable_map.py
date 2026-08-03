#!/usr/bin/env python3
"""ROS client for the Isaac static walkable-map export service."""

from __future__ import annotations

import argparse
from typing import Any

import rclpy
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from std_srvs.srv import Trigger


def _parameter(name: str, value: Any) -> Parameter:
    parameter = Parameter()
    parameter.name = name
    wrapped = ParameterValue()
    if isinstance(value, float):
        wrapped.type = ParameterType.PARAMETER_DOUBLE
        wrapped.double_value = value
    else:
        wrapped.type = ParameterType.PARAMETER_STRING
        wrapped.string_value = str(value)
    parameter.value = wrapped
    return parameter


def _set_parameters(node, values: dict[str, Any], timeout: float) -> None:
    client = node.create_client(SetParameters, "/isaac_controller/set_parameters")
    if not client.wait_for_service(timeout_sec=timeout):
        raise RuntimeError("/isaac_controller/set_parameters is unavailable")
    request = SetParameters.Request()
    request.parameters = [_parameter(name, value) for name, value in values.items()]
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    if not future.done() or future.result() is None:
        raise RuntimeError("timed out setting walkable-map parameters")
    failures = [result.reason for result in future.result().results if not result.successful]
    if failures:
        raise RuntimeError("; ".join(failures))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", default="/World/shenxinfu_841837")
    parser.add_argument(
        "--output",
        default="/home/stardust/resources/arena_ws/arena_assets/navigation/shenxinfu_841837.walkable.json",
    )
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--origin", default="-2.24,-0.90,0.75")
    parser.add_argument("--sample-heights", default="0.15,0.45,0.75,1.05")
    parser.add_argument("--bounds", default="-4.60,3.60,-2.10,1.80")
    parser.add_argument(
        "--exclude-path-keywords",
        default="/World/Characters,/World/xms_mecanum,_debug",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)

    rclpy.init()
    node = rclpy.create_node("export_walkable_map_client")
    try:
        _set_parameters(
            node,
            {
                "walkable_map_scene_root": args.scene_root,
                "walkable_map_output_path": args.output,
                "walkable_map_resolution": args.resolution,
                "walkable_map_origin": args.origin,
                "walkable_map_sample_heights": args.sample_heights,
                "walkable_map_world_bounds": args.bounds,
                "walkable_map_exclude_path_keywords": args.exclude_path_keywords,
            },
            args.timeout,
        )
        client = node.create_client(Trigger, "/isaac/export_walkable_map")
        if not client.wait_for_service(timeout_sec=args.timeout):
            raise RuntimeError("/isaac/export_walkable_map is unavailable")
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=args.timeout)
        if not future.done() or future.result() is None:
            raise RuntimeError("walkable-map export timed out")
        response = future.result()
        print(response.message)
        return 0 if response.success else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
