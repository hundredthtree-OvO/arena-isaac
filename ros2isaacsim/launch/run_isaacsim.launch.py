import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    default_isaac_path = os.environ.get(
        "ISAAC_PATH",
        os.path.expanduser("~/resources/isaac-sim-4.5.0"),
    )

    isaac_path = LaunchConfiguration("isaac_path")
    logger = LaunchConfiguration("log_level")
    idle_fps = LaunchConfiguration("idle_fps")
    active_fps = LaunchConfiguration("active_fps")
    mem_log_sec = LaunchConfiguration("mem_log_sec")
    enable_people_stack = LaunchConfiguration("enable_people_stack")
    enable_character_services = LaunchConfiguration("enable_character_services")
    people_extension_mode = LaunchConfiguration("people_extension_mode")
    enable_material_stack = LaunchConfiguration("enable_material_stack")
    enable_navmesh = LaunchConfiguration("enable_navmesh")
    enable_scene_collision_repair = LaunchConfiguration("enable_scene_collision_repair")
    scene_collision_repair_mode = LaunchConfiguration("scene_collision_repair_mode")
    scene_collision_proxy_visible = LaunchConfiguration("scene_collision_proxy_visible")
    scene_collision_report_dir = LaunchConfiguration("scene_collision_report_dir")
    scene_collision_enable_wall_proxy = LaunchConfiguration("scene_collision_enable_wall_proxy")
    scene_collision_enable_door_proxy = LaunchConfiguration("scene_collision_enable_door_proxy")
    scene_collision_proxy_source = LaunchConfiguration("scene_collision_proxy_source")
    scene_collision_proxy_config = LaunchConfiguration("scene_collision_proxy_config")
    scene_collision_auto_write_config = LaunchConfiguration("scene_collision_auto_write_config")
    scene_collision_door_default_enabled = LaunchConfiguration("scene_collision_door_default_enabled")
    scene_collision_wall_default_enabled = LaunchConfiguration("scene_collision_wall_default_enabled")
    scene_collision_proxy_export_path = LaunchConfiguration("scene_collision_proxy_export_path")
    enable_kinematic_collision_guard = LaunchConfiguration("enable_kinematic_collision_guard")
    collision_guard_length = LaunchConfiguration("collision_guard_length")
    collision_guard_width = LaunchConfiguration("collision_guard_width")
    collision_guard_margin = LaunchConfiguration("collision_guard_margin")
    collision_guard_z_max = LaunchConfiguration("collision_guard_z_max")
    collision_guard_log_sec = LaunchConfiguration("collision_guard_log_sec")
    collision_guard_block_log_sec = LaunchConfiguration("collision_guard_block_log_sec")

    return LaunchDescription([
        DeclareLaunchArgument(
            "isaac_path",
            default_value=default_isaac_path,
            description="Path to Isaac Sim 4.5 installation directory.",
        ),
        DeclareLaunchArgument(
            "log_level",
            default_value="debug",
            description="ROS logging level.",
        ),
        DeclareLaunchArgument(
            "idle_fps",
            default_value="5",
            description="Idle Isaac update FPS before robots/sensors are active. Prevents busy-loop memory/CPU growth.",
        ),
        DeclareLaunchArgument(
            "active_fps",
            default_value="30",
            description="Active Isaac update FPS after a kinematic robot is registered.",
        ),
        DeclareLaunchArgument(
            "mem_log_sec",
            default_value="30",
            description="Seconds between bridge RSS memory log lines. Set 0 to disable.",
        ),
        DeclareLaunchArgument(
            "enable_people_stack",
            default_value="false",
            description="Load AnimGraph/People/Replicator stack. Keep false for robot/lidar bridge memory tests.",
        ),
        DeclareLaunchArgument(
            "enable_character_services",
            default_value="false",
            description="Register Isaac pedestrian spawn/move services and import the character backend.",
        ),
        DeclareLaunchArgument(
            "people_extension_mode",
            default_value="minimal",
            description="People extension preset: minimal, people, nav, replicator_agent_core, full, etc.",
        ),
        DeclareLaunchArgument(
            "enable_material_stack",
            default_value="false",
            description="Load material browser/library and remote MDL material binding. Keep false for lightweight bridge.",
        ),
        DeclareLaunchArgument(
            "enable_navmesh",
            default_value="false",
            description="Create and bake AnimGraph navmesh. Keep false until pedestrian tests.",
        ),
        DeclareLaunchArgument(
            "enable_scene_collision_repair",
            default_value="true",
            description="Inspect and conservatively repair imported scene collision after /isaac/import_usd.",
        ),
        DeclareLaunchArgument(
            "scene_collision_repair_mode",
            default_value="safe",
            description="Scene collision repair mode: off, report, safe, or aggressive. Use safe for v12.",
        ),
        DeclareLaunchArgument(
            "scene_collision_proxy_visible",
            default_value="false",
            description="Show generated collision proxy boxes in the Isaac viewport for debugging.",
        ),
        DeclareLaunchArgument(
            "scene_collision_report_dir",
            default_value="/tmp/arena_isaac_collision_reports",
            description="Directory where v12/v13 scene collision inspection reports are written.",
        ),
        DeclareLaunchArgument(
            "scene_collision_enable_wall_proxy",
            default_value="true",
            description="V12.2: create simplified static wall collision proxies.",
        ),
        DeclareLaunchArgument(
            "scene_collision_enable_door_proxy",
            default_value="true",
            description="V12.2: create simplified static door collision proxies while disabling original door mesh collision.",
        ),
        DeclareLaunchArgument(
            "scene_collision_proxy_source",
            default_value="auto",
            description="V13.1 proxy source: auto, config, or off. auto creates editable boxes and writes YAML; config loads edited YAML.",
        ),
        DeclareLaunchArgument(
            "scene_collision_proxy_config",
            default_value="/home/stardust/resources/arena_ws/arena_assets/collision_configs/shenxinfu_841837.proxies.yaml",
            description="YAML path for editable scene collision proxies.",
        ),
        DeclareLaunchArgument(
            "scene_collision_auto_write_config",
            default_value="true",
            description="In auto mode, write the generated editable proxy layer to scene_collision_proxy_config.",
        ),
        DeclareLaunchArgument(
            "scene_collision_door_default_enabled",
            default_value="false",
            description="In auto mode, initial enabled state for door proxies. False avoids blocking openings until manually edited.",
        ),
        DeclareLaunchArgument(
            "scene_collision_wall_default_enabled",
            default_value="true",
            description="In auto mode, initial enabled state for wall proxies.",
        ),
        DeclareLaunchArgument(
            "scene_collision_proxy_export_path",
            default_value="/home/stardust/resources/arena_ws/arena_assets/collision_configs/shenxinfu_841837.proxies.yaml",
            description="Default output path used by /isaac/export_collision_proxies Trigger service.",
        ),
        DeclareLaunchArgument(
            "enable_kinematic_collision_guard",
            default_value="true",
            description="V13: reject kinematic robot steps that would overlap scene collision proxies.",
        ),
        DeclareLaunchArgument(
            "collision_guard_length",
            default_value="0.80",
            description="V13 robot collision footprint length in meters.",
        ),
        DeclareLaunchArgument(
            "collision_guard_width",
            default_value="0.60",
            description="V13 robot collision footprint width in meters.",
        ),
        DeclareLaunchArgument(
            "collision_guard_margin",
            default_value="0.05",
            description="V13 robot collision footprint margin in meters.",
        ),
        DeclareLaunchArgument(
            "collision_guard_z_max",
            default_value="1.00",
            description="V13 obstacle z upper bound considered by the 2D collision guard.",
        ),
        DeclareLaunchArgument(
            "collision_guard_log_sec",
            default_value="5.0",
            description="V13.1 seconds between collision guard statistics logs.",
        ),
        DeclareLaunchArgument(
            "collision_guard_block_log_sec",
            default_value="2.0",
            description="V13.1 minimum seconds between blocked-motion warnings.",
        ),
        SetEnvironmentVariable("ISAAC_PATH", isaac_path),
        SetEnvironmentVariable("ARENA_ISAAC_IDLE_FPS", idle_fps),
        SetEnvironmentVariable("ARENA_ISAAC_ACTIVE_FPS", active_fps),
        SetEnvironmentVariable("ARENA_ISAAC_MEM_LOG_SEC", mem_log_sec),
        SetEnvironmentVariable("ARENA_ISAAC_ENABLE_PEOPLE_STACK", enable_people_stack),
        SetEnvironmentVariable("ARENA_ISAAC_ENABLE_CHARACTER_SERVICES", enable_character_services),
        SetEnvironmentVariable("ARENA_ISAAC_PEOPLE_EXTENSION_MODE", people_extension_mode),
        SetEnvironmentVariable("ARENA_ISAAC_ENABLE_MATERIAL_STACK", enable_material_stack),
        SetEnvironmentVariable("ARENA_ISAAC_ENABLE_NAVMESH", enable_navmesh),
        SetEnvironmentVariable("ARENA_ISAAC_ENABLE_SCENE_COLLISION_REPAIR", enable_scene_collision_repair),
        SetEnvironmentVariable("ARENA_ISAAC_SCENE_COLLISION_REPAIR_MODE", scene_collision_repair_mode),
        SetEnvironmentVariable("ARENA_ISAAC_SCENE_COLLISION_PROXY_VISIBLE", scene_collision_proxy_visible),
        SetEnvironmentVariable("ARENA_ISAAC_SCENE_COLLISION_REPORT_DIR", scene_collision_report_dir),
        SetEnvironmentVariable("ARENA_ISAAC_SCENE_COLLISION_ENABLE_WALL_PROXY", scene_collision_enable_wall_proxy),
        SetEnvironmentVariable("ARENA_ISAAC_SCENE_COLLISION_ENABLE_DOOR_PROXY", scene_collision_enable_door_proxy),
        SetEnvironmentVariable("ARENA_ISAAC_SCENE_COLLISION_PROXY_SOURCE", scene_collision_proxy_source),
        SetEnvironmentVariable("ARENA_ISAAC_SCENE_COLLISION_PROXY_CONFIG", scene_collision_proxy_config),
        SetEnvironmentVariable("ARENA_ISAAC_SCENE_COLLISION_AUTO_WRITE_CONFIG", scene_collision_auto_write_config),
        SetEnvironmentVariable("ARENA_ISAAC_SCENE_COLLISION_DOOR_DEFAULT_ENABLED", scene_collision_door_default_enabled),
        SetEnvironmentVariable("ARENA_ISAAC_SCENE_COLLISION_WALL_DEFAULT_ENABLED", scene_collision_wall_default_enabled),
        SetEnvironmentVariable("ARENA_ISAAC_COLLISION_PROXY_EXPORT_PATH", scene_collision_proxy_export_path),
        SetEnvironmentVariable("ARENA_ISAAC_ENABLE_KINEMATIC_COLLISION_GUARD", enable_kinematic_collision_guard),
        SetEnvironmentVariable("ARENA_ISAAC_COLLISION_GUARD_LENGTH", collision_guard_length),
        SetEnvironmentVariable("ARENA_ISAAC_COLLISION_GUARD_WIDTH", collision_guard_width),
        SetEnvironmentVariable("ARENA_ISAAC_COLLISION_GUARD_MARGIN", collision_guard_margin),
        SetEnvironmentVariable("ARENA_ISAAC_COLLISION_GUARD_Z_MAX", collision_guard_z_max),
        SetEnvironmentVariable("ARENA_ISAAC_COLLISION_GUARD_LOG_SEC", collision_guard_log_sec),
        SetEnvironmentVariable("ARENA_ISAAC_COLLISION_GUARD_BLOCK_LOG_SEC", collision_guard_block_log_sec),
        ExecuteProcess(
            cmd=[
                PathJoinSubstitution([isaac_path, "python.sh"]),
                "-m",
                "ros2isaacsim.run_isaacsim",
                "--ros-args",
                "--log-level",
                logger,
            ],
            output="screen",
        ),
    ])
