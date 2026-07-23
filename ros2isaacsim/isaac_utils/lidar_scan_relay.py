"""Relay unique RTX LaserScan frames from internal topics to public topics."""

import os
import threading


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def raw_scan_topic(topic):
    normalized = "/" + str(topic).strip("/")
    return f"{normalized}_raw"


class ScanStampDeduplicator:
    """Reject consecutive cached RTX frames with an unchanged ROS timestamp."""

    def __init__(self):
        self._last_stamp = None

    def should_forward(self, message):
        stamp = message.header.stamp
        current = (int(stamp.sec), int(stamp.nanosec))
        if current == self._last_stamp:
            return False
        self._last_stamp = current
        return True


class LidarScanRelay:
    def __init__(self, node, input_topic, output_topic):
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import LaserScan

        self._filter = ScanStampDeduplicator()
        self._publisher = node.create_publisher(
            LaserScan, output_topic, qos_profile_sensor_data
        )
        self._subscription = node.create_subscription(
            LaserScan, input_topic, self._on_scan, qos_profile_sensor_data
        )
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.forwarded = 0
        self.dropped = 0

    def _on_scan(self, message):
        if not self._filter.should_forward(message):
            self.dropped += 1
            return
        self._publisher.publish(message)
        self.forwarded += 1


class LidarScanRelayManager:
    """Run scan relays independently from the Isaac controller callback loop."""

    def __init__(self, front_topic, rear_topic):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor

        self._node = rclpy.create_node("isaac_lidar_scan_relay")
        self.relays = [
            LidarScanRelay(
                self._node, raw_scan_topic(front_topic), front_topic
            ),
            LidarScanRelay(
                self._node, raw_scan_topic(rear_topic), rear_topic
            ),
        ]
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(
            target=self._executor.spin,
            name="isaac-lidar-scan-relay",
            daemon=True,
        )
        self._thread.start()
        self._node.get_logger().info(
            "RTX lidar unique-frame relays enabled: "
            + ", ".join(
                f"{relay.input_topic} -> {relay.output_topic}"
                for relay in self.relays
            )
        )

    def shutdown(self):
        stopped = self._executor.shutdown(timeout_sec=2.0)
        self._thread.join(timeout=2.0)
        if not stopped or self._thread.is_alive():
            self._node.get_logger().warning(
                "RTX lidar scan relay executor did not stop before ROS shutdown"
            )
            return
        self._executor.remove_node(self._node)
        self._node.destroy_node()


def register_lidar_scan_relays():
    """Register front/rear relays when RTX duplicate filtering is enabled."""
    if not _env_bool("ARENA_ISAAC_LIDAR_DEDUPLICATE", False):
        return None

    front_topic = os.environ.get("ARENA_ISAAC_LIDAR_FRONT_TOPIC", "/front_scan")
    rear_topic = os.environ.get("ARENA_ISAAC_LIDAR_REAR_TOPIC", "/rear_scan")
    return LidarScanRelayManager(front_topic, rear_topic)
