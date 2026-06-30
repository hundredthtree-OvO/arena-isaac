import os


def get_ros_domain_id(default: int = 0) -> int:
    """Read ROS_DOMAIN_ID for Isaac OmniGraph ROS2Context nodes.

    Keeping this in one function prevents Isaac-side publishers/subscribers from
    silently using a different DDS domain than the terminal running ros2 tools.
    """
    value = os.environ.get("ROS_DOMAIN_ID", str(default)).strip()
    try:
        return int(value)
    except ValueError:
        return default
