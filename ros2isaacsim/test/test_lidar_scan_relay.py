import unittest
from unittest import mock

from isaac_utils.lidar_scan_relay import (
    ScanStampDeduplicator,
    raw_scan_topic,
    register_lidar_scan_relays,
)


class _Stamp:
    def __init__(self, sec, nanosec):
        self.sec = sec
        self.nanosec = nanosec


class _Header:
    def __init__(self, sec, nanosec):
        self.stamp = _Stamp(sec, nanosec)


class _Scan:
    def __init__(self, sec, nanosec):
        self.header = _Header(sec, nanosec)


class TestScanStampDeduplicator(unittest.TestCase):
    def test_forwards_only_one_message_per_scan_stamp(self):
        deduplicator = ScanStampDeduplicator()

        self.assertTrue(deduplicator.should_forward(_Scan(12, 100)))
        self.assertFalse(deduplicator.should_forward(_Scan(12, 100)))
        self.assertTrue(deduplicator.should_forward(_Scan(12, 200)))

    def test_allows_time_to_restart_after_simulation_reset(self):
        deduplicator = ScanStampDeduplicator()

        self.assertTrue(deduplicator.should_forward(_Scan(20, 0)))
        self.assertTrue(deduplicator.should_forward(_Scan(0, 0)))
        self.assertFalse(deduplicator.should_forward(_Scan(0, 0)))

    def test_raw_topic_keeps_namespace_and_avoids_output_collision(self):
        self.assertEqual(raw_scan_topic("/front_scan"), "/front_scan_raw")
        self.assertEqual(raw_scan_topic("robot/rear_scan"), "/robot/rear_scan_raw")

    def test_disabled_relay_has_no_runtime_lifecycle_object(self):
        with mock.patch.dict(
            "os.environ", {"ARENA_ISAAC_LIDAR_DEDUPLICATE": "false"}
        ):
            self.assertIsNone(register_lidar_scan_relays())


if __name__ == "__main__":
    unittest.main()
