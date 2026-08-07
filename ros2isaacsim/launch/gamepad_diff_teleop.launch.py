"""Launch the Linux joystick driver and differential gamepad mapper."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    device_id = LaunchConfiguration("device_id")
    joy_topic = LaunchConfiguration("joy_topic")
    output_topic = LaunchConfiguration("output_topic")

    return LaunchDescription([
        DeclareLaunchArgument("device_id", default_value="0"),
        DeclareLaunchArgument("joy_topic", default_value="/joy"),
        DeclareLaunchArgument(
            "output_topic", default_value="/cmd_vel_gamepad_diff"
        ),
        DeclareLaunchArgument("linear_axis", default_value="1"),
        DeclareLaunchArgument("angular_axis", default_value="0"),
        DeclareLaunchArgument("enable_button", default_value="4"),
        DeclareLaunchArgument("linear_scale", default_value="0.4"),
        DeclareLaunchArgument("angular_scale", default_value="0.4"),
        DeclareLaunchArgument("deadzone", default_value="0.1"),
        DeclareLaunchArgument("joy_timeout_sec", default_value="0.25"),
        DeclareLaunchArgument("publish_rate_hz", default_value="50.0"),
        Node(
            package="joy",
            executable="joy_node",
            name="joy_node",
            output="screen",
            parameters=[{
                "device_id": ParameterValue(device_id, value_type=int),
                "deadzone": ParameterValue(
                    LaunchConfiguration("deadzone"), value_type=float
                ),
                "autorepeat_rate": 20.0,
            }],
            remappings=[("joy", joy_topic)],
        ),
        Node(
            package="ros2isaacsim",
            executable="gamepad_diff_teleop",
            name="gamepad_diff_teleop",
            output="screen",
            parameters=[{
                "joy_topic": joy_topic,
                "output_topic": output_topic,
                "linear_axis": ParameterValue(
                    LaunchConfiguration("linear_axis"), value_type=int
                ),
                "angular_axis": ParameterValue(
                    LaunchConfiguration("angular_axis"), value_type=int
                ),
                "enable_button": ParameterValue(
                    LaunchConfiguration("enable_button"), value_type=int
                ),
                "linear_scale": ParameterValue(
                    LaunchConfiguration("linear_scale"), value_type=float
                ),
                "angular_scale": ParameterValue(
                    LaunchConfiguration("angular_scale"), value_type=float
                ),
                "deadzone": ParameterValue(
                    LaunchConfiguration("deadzone"), value_type=float
                ),
                "joy_timeout_sec": ParameterValue(
                    LaunchConfiguration("joy_timeout_sec"), value_type=float
                ),
                "publish_rate_hz": ParameterValue(
                    LaunchConfiguration("publish_rate_hz"), value_type=float
                ),
            }],
        ),
    ])
