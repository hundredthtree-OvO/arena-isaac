#!/usr/bin/env python3
"""Client helper for exporting currently edited Isaac Sim collision proxies to YAML.

This version avoids rclpy.parameter_client because some ROS 2 Humble builds do
not ship that Python module. It sets parameters through the standard
/isaac_controller/set_parameters service, then triggers /isaac/export_collision_proxies.
"""
from __future__ import annotations

import argparse
import sys

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from std_srvs.srv import Trigger


def _set_string_param_msg(name: str, value: str) -> Parameter:
    return Parameter(
        name=name,
        value=ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=str(value)),
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", default="/World/shenxinfu_841837")
    parser.add_argument("--output", default="/home/stardust/resources/arena_ws/arena_assets/collision_configs/shenxinfu_841837.proxies.yaml")
    parser.add_argument("--controller-node", default="/isaac_controller")
    parser.add_argument("--service", default="/isaac/export_collision_proxies")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args(argv)

    rclpy.init()
    node = Node("export_collision_proxies_client")
    try:
        controller = args.controller_node.rstrip("/") or "/isaac_controller"
        if not controller.startswith("/"):
            controller = "/" + controller
        set_params_srv = f"{controller}/set_parameters"

        param_client = node.create_client(SetParameters, set_params_srv)
        if param_client.wait_for_service(timeout_sec=args.timeout):
            req = SetParameters.Request()
            req.parameters = [
                _set_string_param_msg("collision_proxy_export_scene_root", args.scene_root),
                _set_string_param_msg("collision_proxy_export_path", args.output),
            ]
            fut = param_client.call_async(req)
            rclpy.spin_until_future_complete(node, fut, timeout_sec=args.timeout)
            if not fut.done() or fut.result() is None:
                node.get_logger().warning(f"timed out setting export parameters through {set_params_srv}; using bridge defaults")
            else:
                results = fut.result().results
                if not all(r.successful for r in results):
                    node.get_logger().warning(f"one or more export parameters were rejected by {set_params_srv}; using available/default values")
        else:
            node.get_logger().warning(f"parameter service {set_params_srv} not available; using bridge defaults")

        client = node.create_client(Trigger, args.service)
        if not client.wait_for_service(timeout_sec=args.timeout):
            node.get_logger().error(f"service {args.service} not available; is run_isaacsim running?")
            return 2
        fut = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(node, fut, timeout_sec=args.timeout)
        if not fut.done() or fut.result() is None:
            node.get_logger().error("export service timed out")
            return 3
        res = fut.result()
        if res.success:
            node.get_logger().info(res.message)
            return 0
        node.get_logger().error(res.message)
        return 4
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
