# Low level APIs
import math
import os
import time

import carb
import numpy as np
import omni.anim.graph.core as ag

# High level Isaac sim APIs
import omni.client
from isaacsim.core.utils import prims
try:
    from isaac_utils.dynamic_actor_guard import (
        PedestrianGuardState,
        circle_intersects_grid_cells,
        movement_allowed,
        pedestrian_robot_scores,
        predict_pedestrian_step,
    )
except Exception:  # pragma: no cover
    from ros2isaacsim.isaac_utils.dynamic_actor_guard import (
        PedestrianGuardState,
        circle_intersects_grid_cells,
        movement_allowed,
        pedestrian_robot_scores,
        predict_pedestrian_step,
    )
try:
    from isaac_utils.voxel_guard import VoxelCollisionGuard, VoxelGuardConfig
except Exception:  # pragma: no cover
    from ros2isaacsim.isaac_utils.voxel_guard import VoxelCollisionGuard, VoxelGuardConfig
from isaac_utils.utils.assets import get_assets_root_path_safe
from isaac_utils.animgraph_people import normalize_stage_name
from omni.anim.people.scripts.utils import Utils

from omni.usd import get_stage_next_free_path
from pedestrian.simulator.logic.people.person_controller import PersonController
from pedestrian.simulator.logic.people_manager import PeopleManager
from pedestrian.simulator.logic.people.external_motion import (
    ExternalMotionMode,
    ExternalMotionModeState,
    ExternalMotionState,
    animation_walk_blend,
    bounded_yaw_step,
    locomotion_path_points,
    turn_aware_animation_sample,
)
from pedestrian.simulator.logic.people.navigation_safety import (
    LateralAvoidanceCandidate,
    active_polyline_goal,
    path_target_progress_radius,
    segment_is_safe,
    select_safe_lateral_avoidance_choice,
    terminal_semantic_approach_radius,
)

# Extension APIs
from pedestrian.simulator.logic.state import State
from pxr import Gf, Sdf, Usd, Vt
from scipy.spatial.transform import Rotation

from isaacsim.replicator.agent.core.settings import PrimPaths
from isaacsim.replicator.agent.core.agent_manager import AgentManager
from isaacsim.replicator.agent.core.settings import BehaviorScriptPaths
from isaacsim.replicator.agent.core.stage_util import CharacterUtil
from omni.anim.people.scripts.global_character_position_manager import GlobalCharacterPositionManager
from pxr import UsdGeom, UsdPhysics

try:
    from pxr import PhysxSchema  # type: ignore
except Exception:  # pragma: no cover
    PhysxSchema = None  # type: ignore


def _yaw_rad_to_gf_quat(yaw_rad: float):
    return Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(float(yaw_rad))).GetQuat()


def _pedestrian_stop_radius() -> float:
    """Return the AnimGraph waypoint-switch radius configured for this bridge."""
    try:
        value = float(os.environ.get("ARENA_ISAAC_PEDESTRIAN_STOP_RADIUS_M", "0.5"))
    except (TypeError, ValueError):
        value = 0.5
    return max(0.01, value)


def _constrained_waypoint_radius() -> float:
    try:
        value = float(os.environ.get("ARENA_ISAAC_CONSTRAINED_WAYPOINT_RADIUS_M", "0.08"))
    except (TypeError, ValueError):
        value = 0.08
    return max(0.01, value)


def _env_enabled(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


class Person:
    """
    Class that implements a person in the simulation world. The person can be controlled by a controller that inherits from the PersonController class.
    """

    # Get root assets path from setting, if not set, get the Isaac-Sim asset path
    people_asset_folder = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/People/Characters"
    character_root_prim_path = PrimPaths.characters_parent_path()

    assets_root_path = None
    collision_proxy_root_path = "/World/CharactersCollision"
    collision_proxy_radius = 0.22
    collision_proxy_height = 1.00
    collision_proxy_center_z = 0.90
    anim_graph_variable_prefix = "anim:graph:variable:"
    required_anim_graph_variables = ("Action", "Walk", "PathPoints")
    _shared_static_voxel_guard = None

    if people_asset_folder:
        assets_root_path = people_asset_folder
    else:
        root_path = get_assets_root_path_safe()
        if root_path is not None:
            assets_root_path = "{}/Isaac/People/Characters".format(root_path)

    def __init__(
        self,
        world,
        stage_prefix: str,
        character_name: str | None = None,
        init_pos=[0.0, 0.0, 0.0],
        init_yaw=0.0,
        controller: PersonController | None = None,
        backend=None,
    ):
        """Initializes the person object

        Args:
            stage_prefix (str): The name the person will present in the simulator when spawned on the stage.
            character_name (str): The name of the person in the USD file. Use the Person.get_character_asset_list() method to get the list of available characters.
            init_pos (list): The initial position of the vehicle in the inertial frame (in ENU convention). Defaults to [0.0, 0.0, 0.0].
            init_yaw (float): The initial orientation of the person in rad. Defaults to 0.0.
            controller (PersonController): A controller to add some custom behaviour to the movement of the person. Defaults to None.
        """

        # Get the current world at which we want to spawn the vehicle
        self._world = world
        self._current_stage = self._world.stage

        # Variable that will hold the current state of the vehicle
        self._state = State()
        self._state.position = np.array(init_pos)
        self._state.orientation = Rotation.from_euler(
            "z", init_yaw, degrees=False
        ).as_quat()

        # Set the target position for the character
        self._target_position = np.array(init_pos)
        self._target_speed = 0.0
        self._target_yaw = None
        self._constrain_to_path = False
        self._constrained_waypoint_radius = _constrained_waypoint_radius()
        self._robot_interaction_policy = str(
            os.environ.get("ARENA_ISAAC_PEDESTRIAN_ROBOT_POLICY", "avoid")
        ).strip().lower()
        if self._robot_interaction_policy not in {"avoid", "stop", "off"}:
            carb.log_warn(
                f"Unsupported pedestrian robot policy {self._robot_interaction_policy!r}; using avoid"
            )
            self._robot_interaction_policy = "avoid"
        self._physics_proxy_enabled = _env_enabled(
            "ARENA_ISAAC_PEDESTRIAN_PHYSICS_PROXY_ENABLED",
            True,
        )
        self._hard_guard_margin_m = max(
            0.0,
            _env_float("ARENA_ISAAC_PEDESTRIAN_HARD_GUARD_MARGIN_M", 0.04),
        )
        self._hard_guard_horizon_sec = max(
            0.0,
            _env_float("ARENA_ISAAC_PEDESTRIAN_HARD_GUARD_HORIZON_SEC", 0.0),
        )
        self._hard_guard_release_margin_m = max(
            0.0,
            _env_float("ARENA_ISAAC_PEDESTRIAN_HARD_GUARD_RELEASE_MARGIN_M", 0.0),
        )
        self._hard_guard_release_hold_sec = max(
            0.0,
            _env_float("ARENA_ISAAC_PEDESTRIAN_HARD_GUARD_RELEASE_HOLD_SEC", 0.0),
        )
        self._dynamic_avoidance_default = None
        self._path_point_arrival_radius = _pedestrian_stop_radius()

        # Set the transition point for the path points
        self._transition_point = np.array(init_pos)
        self._num_path_points = 0
        self._current_point_index = 0
        self._path_direction = 1
        self._loop_path = False

        # Save the name with which the vehicle will appear in the stage
        # and the character model that will be loaded into the simulator
        self._requested_stage_name = normalize_stage_name(stage_prefix, default="Character")
        self._stage_prefix = get_stage_next_free_path(
            self._current_stage,
            os.path.join(Person.character_root_prim_path, self._requested_stage_name),
            False,
        )

        # The name of the character in the USD file
        self._character_name = character_name

        # Get the USD file corresponding to the character
        self.char_usd_file = Person.get_path_for_character_prim(character_name)

        # Spawn the agent in the world
        self.spawn_agent(self.char_usd_file, self._stage_prefix, init_pos, init_yaw)

        # Add the animation graph to the agent, such that it can move around.
        # The Biped asset may still be loading, so failed setup is retried from
        # the physics callbacks before any cached movement command is executed.
        self.character_graph = None
        self._anim_graph_setup_ok = False
        self._anim_graph_ready = False
        self._next_anim_graph_setup_attempt = 0.0
        self._last_pose_read_warn = 0.0
        self._anim_graph_setup_ok = self.add_animation_graph_to_agent()
        self._anim_motion_state = None
        self._last_walk_speed = None
        self._behavior_script_enabled = self._attach_behavior_script()
        self._motion_command_generation = 0
        self._dispatched_command_generation = 0
        self._motion_state = "idle"
        self._external_motion = ExternalMotionState(
            walk_speed_threshold=max(
                0.0,
                _env_float("ARENA_ISAAC_EXTERNAL_MOTION_WALK_THRESHOLD_MPS", 0.05),
            )
        )
        self._external_motion_mode = ExternalMotionModeState()
        self._external_hold_position = None
        self._external_hold_orientation = None
        self._external_terminal_position = None
        self._external_terminal_yaw = None
        self._external_terminal_complete_logged = False
        self._external_motion_tracking_gain = max(
            0.0,
            _env_float("ARENA_ISAAC_EXTERNAL_MOTION_TRACKING_GAIN", 1.5),
        )
        self._external_motion_max_speed_mps = max(
            0.05,
            _env_float("ARENA_ISAAC_EXTERNAL_MOTION_MAX_SPEED_MPS", 1.5),
        )
        self._external_motion_animation_full_speed_mps = max(
            0.05,
            _env_float(
                "ARENA_ISAAC_EXTERNAL_MOTION_ANIMATION_FULL_SPEED_MPS",
                1.0,
            ),
        )
        self._external_motion_animation_speed_exponent = max(
            1e-3,
            _env_float(
                "ARENA_ISAAC_EXTERNAL_MOTION_ANIMATION_SPEED_EXPONENT",
                1.0,
            ),
        )
        self._external_motion_yaw_rate_radps = max(
            0.0,
            _env_float("ARENA_ISAAC_EXTERNAL_MOTION_YAW_RATE_RADPS", 2.8),
        )
        self._external_turn_slow_angle_rad = math.radians(
            max(
                0.0,
                _env_float("ARENA_ISAAC_EXTERNAL_TURN_SLOW_ANGLE_DEG", 20.0),
            )
        )
        self._external_turn_full_slow_angle_rad = math.radians(
            max(
                1.0,
                _env_float(
                    "ARENA_ISAAC_EXTERNAL_TURN_FULL_SLOW_ANGLE_DEG",
                    55.0,
                ),
            )
        )
        self._external_turn_minimum_speed_scale = min(
            1.0,
            max(
                0.0,
                _env_float("ARENA_ISAAC_EXTERNAL_TURN_MIN_SPEED_SCALE", 0.25),
            ),
        )
        # A root rebase is visually a teleport.  Keep it opt-in for emergency
        # recovery; normal HuNav control closes the loop using the measured
        # AnimationGraph pose instead.
        self._external_motion_hard_sync_distance_m = max(
            0.0,
            _env_float("ARENA_ISAAC_EXTERNAL_MOTION_HARD_SYNC_DISTANCE_M", 0.0),
        )
        self._last_external_sync_warn = 0.0
        self._pose_valid = False
        self._last_valid_pose_time = 0.0
        self._last_static_guard_warn = 0.0
        self._static_voxel_guard = self._get_static_voxel_guard()

        # Set the controller for the person if any and initialize it
        self._controller = controller
        if self._controller:
            self._controller.initialize(self)

        # Set the backend for publishing the state of the person
        self._backend = backend
        if self._backend:
            self._backend.initialize(self)

        self._active = True
        self._parked = False
        self._state_callback_name = self._stage_prefix + "/state"
        self._update_callback_name = self._stage_prefix + "/update"
        self._timeline_callback_name = self._stage_prefix + "/start_stop_sim"

        # Add a callback to the physics engine to update the current state of the person
        self._world.add_physics_callback(self._state_callback_name, self.update_state)

        # Add the update method to the physics callback if the world was received
        # so that we can apply the new references to be tracked by the person
        self._world.add_physics_callback(self._update_callback_name, self.update)

        # Set the flag that signals if the simulation is running or not
        self._sim_running = False

        self._collision_proxy_path = None
        self._spawn_collision_proxy()
        self._guard_escape_epsilon = 1e-4
        self._last_guard_block_log = 0.0
        self._guard_blocked = False
        self._guard_block_generation = 0
        self._guard_block_count = 0
        self._guard_block_reason = ""
        self._last_navigation_voxel_wait_log = 0.0
        self._last_update_dt = 0.0
        self._robot_yield_saved_path_points = None
        self._robot_yield_blocked = False
        self._robot_guard_latched = False
        self._robot_guard_clear_since = 0.0
        carb.log_info(
            f"Pedestrian interaction policy for {self._stage_prefix}: "
            f"robot={self._robot_interaction_policy}, "
            f"physics_proxy_enabled={self._physics_proxy_enabled}"
        )

        # Add a callback to start/stop of the simulation once the play/stop button is hit
        self._world.add_timeline_callback(self._timeline_callback_name, self.sim_start_stop)

    @property
    def state(self):
        """The state of the person.

        Returns:
            State: The current state of the person, i.e., position, orientation, linear and angular velocities...
        """
        return self._state

    @property
    def anim_graph_setup_ok(self) -> bool:
        return bool(self._anim_graph_setup_ok)

    @property
    def anim_graph_ready(self) -> bool:
        return bool(self._anim_graph_ready)

    def sim_start_stop(self, event):
        """
        Callback that is called every time there is a timeline event such as starting/stoping the simulation.

        Args:
            event: A timeline event generated from Isaac Sim, such as starting or stoping the simulation.
        """

        if not self._active:
            return

        # If the start/stop button was pressed, then call the start and stop methods accordingly
        if self._world.is_playing() and not self._sim_running:
            self._sim_running = True
            self.start()

        if self._world.is_stopped() and self._sim_running:
            self._sim_running = False
            self.stop()

    def start(self):
        """
        Method that is called when the simulation starts. This method can be used to initialize any variables.
        """
        if self._controller:
            self._controller.start()

    def stop(self):
        """
        Method that is called when the simulation stops. This method can be used to reset any variables.
        """
        if self._controller:
            self._controller.stop()

    def _set_anim_variable(self, name: str, value):
        if self.character_graph is None:
            return
        try:
            self.character_graph.set_variable(name, value)
        except Exception as exc:
            now = time.monotonic()
            if float(now) - float(self._last_pose_read_warn) >= 2.0:
                self._last_pose_read_warn = float(now)
                carb.log_warn(
                    f"Failed to set animation variable {name} for {self._stage_prefix}: {exc}"
                )

    def _set_idle_animation(self):
        if self._anim_motion_state == "idle":
            return
        self._set_anim_variable("Walk", 0.0)
        # Isaac 4.5's People commands use None, not Idle, to exit locomotion.
        self._set_anim_variable("Action", "None")
        self._anim_motion_state = "idle"
        self._last_walk_speed = None

    def _set_walk_animation(self, active_goal, *, external_sample=None):
        # BaseCommand.walk() refreshes Action and PathPoints every frame.  The
        # graph consumes Python lists of carb.Float3, not USD Vt arrays.
        self._set_anim_variable("Action", "Walk")
        self._anim_motion_state = "walk"
        if external_sample is not None:
            points = locomotion_path_points(external_sample)
            self._set_anim_variable(
                "PathPoints",
                [carb.Float3(*point) for point in points],
            )
        speed = float(self._target_speed)
        if external_sample is not None:
            # Walk is an AnimGraph blend value, while the external sample is
            # m/s.  Use the tracking command so a lagging visual root can
            # smoothly catch its HuNav reference without a pose teleport.
            speed = animation_walk_blend(
                float(external_sample.speed),
                full_speed_mps=self._external_motion_animation_full_speed_mps,
                speed_exponent=self._external_motion_animation_speed_exponent,
            )
        if self._last_walk_speed is None or abs(float(self._last_walk_speed) - speed) > 1e-4:
            self._set_anim_variable("Walk", speed)
            self._last_walk_speed = speed

    def _attach_behavior_script(self) -> bool:
        """Attach Isaac 4.5's supported People behavior controller."""
        skel_root = self.character_skel_root
        if skel_root is None or not skel_root.IsValid():
            return False
        try:
            omni.kit.commands.execute(
                "ApplyScriptingAPICommand",
                paths=[Sdf.Path(skel_root.GetPrimPath())],
            )
            script_path = BehaviorScriptPaths.behavior_script_path()
            skel_root.GetAttribute("omni:scripting:scripts").Set([str(script_path)])
            carb.log_info(
                f"People behavior script attached to {self._stage_prefix}: {script_path}"
            )
            return True
        except Exception as exc:
            self._warn_pose_read_throttled(
                f"Failed to attach People behavior script to {self._stage_prefix}: {exc}"
            )
            return False

    def _behavior_agent(self):
        if not self._behavior_script_enabled:
            return None, None
        manager = AgentManager.get_instance()
        names = (
            self._requested_stage_name,
            str(self._stage_prefix).rstrip("/").split("/")[-1],
        )
        registered = getattr(manager, "_agent_name_to_script_inst", {})
        for name in names:
            agent = registered.get(name)
            if agent is not None:
                return name, agent
        return None, None

    def _dispatch_behavior_commands_if_ready(self) -> bool:
        agent_name, agent = self._behavior_agent()
        if agent is not None:
            self._apply_behavior_navigation_mode(agent)
        if self._dispatched_command_generation == self._motion_command_generation:
            if (
                agent is not None
                and self._motion_state == "executing"
                and getattr(agent, "current_command", None) is None
                and not getattr(agent, "commands", [])
            ):
                self._motion_state = "succeeded"
            return True
        if agent is None:
            return False
        commands = []
        if len(self._target_position) > 0:
            coordinates = " ".join(
                f"{float(point[0]):.9f} {float(point[1]):.9f} {float(point[2]):.9f}"
                for point in self._target_position
            )
            rotation = "_" if self._target_yaw is None else f"{math.degrees(float(self._target_yaw)):.9f}"
            # A single multi-point GoTo keeps the People controller walking
            # continuously instead of stopping and restarting at every A* bend.
            commands.append(f"{agent_name} GoTo {coordinates} {rotation}")
        try:
            current_command = getattr(agent, "current_command", None)
            if current_command is not None:
                # Keep only the currently executing entry. The official script
                # removes it after force_quit, then continues with the newly
                # injected commands at index 1.
                agent.commands = list(getattr(agent, "commands", []))[:1]
                agent.end_current_command()
            else:
                agent.commands = []
            if commands:
                agent.inject_command(commands, executeImmediately=True)
            self._dispatched_command_generation = self._motion_command_generation
            self._motion_state = "executing" if commands else "idle"
            carb.log_info(
                f"People command generation {self._motion_command_generation} dispatched to "
                f"{agent_name}: commands={len(commands)}, waypoints={len(self._target_position)}"
            )
            return True
        except Exception as exc:
            self._warn_pose_read_throttled(
                f"Failed to dispatch People commands for {self._stage_prefix}: {exc}"
            )
            return False

    def _apply_behavior_navigation_mode(self, agent) -> None:
        navigation_manager = getattr(agent, "navigation_manager", None)
        if navigation_manager is None:
            return
        if self._dynamic_avoidance_default is None:
            self._dynamic_avoidance_default = bool(
                getattr(navigation_manager, "dynamic_avoidance_enabled", True)
            )
        navigation_manager.dynamic_avoidance_enabled = (
            False if self._constrain_to_path else self._dynamic_avoidance_default
        )
        self._install_navigation_manager_voxel_safety(navigation_manager)

    def _install_navigation_manager_voxel_safety(self, navigation_manager) -> None:
        if navigation_manager is None or getattr(navigation_manager, "_arena_voxel_safety_wrapped", False):
            return
        original_update_path = getattr(navigation_manager, "update_path", None)
        original_progress = getattr(navigation_manager, "update_target_path_progress", None)
        if not callable(original_update_path) or not callable(original_progress):
            return

        def _update_target_path_progress_with_mode():
            targets = getattr(navigation_manager, "path_targets", None)
            if not targets or not self._constrain_to_path:
                return original_progress()
            radius = path_target_progress_radius(
                target_count=len(targets),
                constrain_to_path=True,
                constrained_intermediate_radius=self._constrained_waypoint_radius,
                default_intermediate_radius=float(Utils.CONFIG["MinDistanceToIntermediateTarget"]),
                final_radius=float(Utils.CONFIG["MinDistanceToFinalTarget"]),
            )
            if navigation_manager.check_proximity_to_point(targets[0], radius):
                targets.pop(0)

        def _update_path_with_voxel_safety(*args, **kwargs):
            try:
                if self._robot_interaction_policy == "stop":
                    if self._behavior_robot_yield_required(navigation_manager):
                        self._pause_behavior_path_for_robot(navigation_manager)
                        return
                    self._resume_behavior_path_after_robot(navigation_manager)
                if self._constrain_to_path or not bool(getattr(navigation_manager, "dynamic_avoidance_enabled", True)):
                    self._set_guard_block_state(False)
                    return original_update_path(*args, **kwargs)
                if bool(getattr(navigation_manager, "navmesh_enabled", False)):
                    self._set_guard_block_state(False)
                    return original_update_path(*args, **kwargs)
                return self._update_navmesh_disabled_path_with_static_voxel_safety(navigation_manager)
            except Exception as exc:  # pragma: no cover - vendor fallback
                self._set_guard_block_state(False)
                self._warn_pose_read_throttled(
                    f"Falling back to Isaac NavigationManager.update_path for {self._stage_prefix}: {exc}"
                )
                return original_update_path(*args, **kwargs)

        navigation_manager._arena_original_update_path = original_update_path
        navigation_manager._arena_original_update_target_path_progress = original_progress
        navigation_manager.update_target_path_progress = _update_target_path_progress_with_mode
        navigation_manager.update_path = _update_path_with_voxel_safety
        navigation_manager._arena_voxel_safety_wrapped = True

    def _behavior_robot_yield_required(self, navigation_manager) -> bool:
        targets = list(getattr(navigation_manager, "path_targets", []) or [])
        active_goal = self._preview_active_goal()
        if active_goal is None and targets:
            active_goal = targets[0]
        if active_goal is None:
            return False
        return not self._robot_guard_allows_motion(
            max(float(self._last_update_dt), 1.0 / 60.0),
            active_goal,
        )

    def _pause_behavior_path_for_robot(self, navigation_manager) -> None:
        if self._robot_yield_saved_path_points is None:
            self._robot_yield_saved_path_points = list(
                getattr(navigation_manager, "path_points", []) or []
            )
        navigation_manager.path_points = [
            Utils.get_character_pos(navigation_manager.character)
        ]
        _, agent = self._behavior_agent()
        command = None if agent is None else getattr(agent, "current_command", None)
        if command is not None and hasattr(command, "desired_walk_speed"):
            command.desired_walk_speed = 0.0
        self._robot_yield_blocked = True
        self._set_guard_block_state(True, reason="robot")

    def _resume_behavior_path_after_robot(self, navigation_manager) -> None:
        if self._robot_yield_saved_path_points is None:
            return
        navigation_manager.path_points = self._robot_yield_saved_path_points
        self._robot_yield_saved_path_points = None
        self._robot_yield_blocked = False
        self._set_guard_block_state(False)

    def _set_guard_block_state(
        self,
        blocked: bool,
        generation: int | None = None,
        reason: str | None = None,
    ) -> None:
        if blocked:
            if generation is None:
                generation = int(self._motion_command_generation)
            if not self._guard_blocked:
                self._guard_block_count = 0
            self._guard_blocked = True
            self._guard_block_generation = int(generation)
            self._guard_block_count += 1
            self._guard_block_reason = str(reason or self._guard_block_reason or "unknown")
            return
        self._guard_blocked = False
        self._guard_block_generation = 0
        self._guard_block_count = 0
        self._guard_block_reason = ""

    def _static_pose_check(self, position) -> tuple[bool, str | None]:
        guard = self._static_voxel_guard
        if guard is None:
            return True, None
        try:
            guard.refresh()
            if not guard.occupied_xy:
                return True, None
            radius = max(
                0.05,
                float(os.environ.get("ARENA_ISAAC_PEDESTRIAN_STATIC_GUARD_RADIUS_M", self.collision_proxy_radius)),
            )
            x, y = float(position[0]), float(position[1])
            origin_x, origin_y, _ = guard.origin
            hit = circle_intersects_grid_cells(
                center_xy=(x, y),
                radius=radius,
                resolution=float(guard.resolution),
                origin_xy=(origin_x, origin_y),
                occupied=guard.occupied_xy,
            )
            return (hit is None, None if hit is None else f"voxel:{hit[0]},{hit[1]}")
        except Exception:
            return True, None

    def _update_navmesh_disabled_path_with_static_voxel_safety(self, navigation_manager) -> None:
        navigation_manager.update_target_path_progress()
        if navigation_manager.destination_reached() or not navigation_manager.dynamic_avoidance_enabled:
            self._set_guard_block_state(False)
            return
        if not navigation_manager.detect_collision():
            self._set_guard_block_state(False)
            return

        collision_name = navigation_manager.collision_list[0]
        current_pos = Utils.get_character_pos(navigation_manager.character)
        self_current = navigation_manager.character_manager.get_character_current_pos(
            navigation_manager.character_name
        )
        self_future = navigation_manager.character_manager.get_character_future_pos(
            navigation_manager.character_name
        )
        obstacle_future = navigation_manager.character_manager.get_character_future_pos(collision_name)

        movement = np.array(
            [
                float(self_future[0]) - float(self_current[0]),
                float(self_future[1]) - float(self_current[1]),
            ],
            dtype=float,
        )
        movement_norm = float(np.linalg.norm(movement))
        if movement_norm <= 1e-6:
            self._set_guard_block_state(False)
            return

        diff = np.array(
            [
                float(obstacle_future[0]) - float(self_future[0]),
                float(obstacle_future[1]) - float(self_future[1]),
            ],
            dtype=float,
        )
        diff_norm = float(np.linalg.norm(diff))
        if diff_norm <= 1e-6:
            self._set_guard_block_state(False)
            return

        right_vector = np.array([movement[1], -movement[0]], dtype=float) / movement_norm
        direction_of_collision = float(np.dot(right_vector, diff / diff_norm))

        radius_a = float(navigation_manager.character_manager.get_character_radius(navigation_manager.character_name))
        radius_b = float(navigation_manager.character_manager.get_character_radius(collision_name))
        if radius_a <= 1e-6:
            self._set_guard_block_state(False)
            return

        avoid_angle = math.degrees(math.atan(max(0.0, radius_a + radius_b - diff_norm) / max(movement_norm, 1e-6)))
        avoid_angle = max(0.0, min(60.0, (radius_b / radius_a) * avoid_angle))
        relative = np.array(
            [
                float(self_future[0]) - float(current_pos[0]),
                float(self_future[1]) - float(current_pos[1]),
            ],
            dtype=float,
        )
        c = math.cos(math.radians(avoid_angle))
        s = math.sin(math.radians(avoid_angle))
        left_offset = np.array(
            [c * relative[0] - s * relative[1], s * relative[0] + c * relative[1]],
            dtype=float,
        )
        right_offset = np.array(
            [c * relative[0] + s * relative[1], -s * relative[0] + c * relative[1]],
            dtype=float,
        )
        left_point = carb.Float3(
            float(current_pos[0]) + float(left_offset[0]),
            float(current_pos[1]) + float(left_offset[1]),
            float(current_pos[2]),
        )
        right_point = carb.Float3(
            float(current_pos[0]) + float(right_offset[0]),
            float(current_pos[1]) + float(right_offset[1]),
            float(current_pos[2]),
        )
        start_xy = (float(current_pos[0]), float(current_pos[1]))

        def _segment_clear(candidate_point) -> tuple[bool, str | None]:
            sample_step = max(0.5 * float(getattr(self._static_voxel_guard, "resolution", 0.05)), 0.01)

            def _point_safe(point_xy: tuple[float, float]) -> bool:
                safe, _ = self._static_pose_check((float(point_xy[0]), float(point_xy[1]), float(current_pos[2])))
                return safe

            safe = segment_is_safe(
                start_xy=start_xy,
                end_xy=(float(candidate_point[0]), float(candidate_point[1])),
                is_safe_point=_point_safe,
                sample_step=sample_step,
            )
            if safe:
                return True, None
            return False, self._static_pose_check(candidate_point)[1]

        left_safe, left_obstacle = _segment_clear(left_point)
        right_safe, right_obstacle = _segment_clear(right_point)
        selected = select_safe_lateral_avoidance_choice(
            direction_of_collision=direction_of_collision,
            left=LateralAvoidanceCandidate(
                "left",
                (float(left_point[0]), float(left_point[1]), float(left_point[2])),
                left_safe,
            ),
            right=LateralAvoidanceCandidate(
                "right",
                (float(right_point[0]), float(right_point[1]), float(right_point[2])),
                right_safe,
            ),
        )
        if selected is None:
            self._set_guard_block_state(True, reason="static_voxel")
            if float(time.monotonic()) - float(self._last_navigation_voxel_wait_log) >= 0.5:
                self._last_navigation_voxel_wait_log = float(time.monotonic())
                blocked_obstacle = left_obstacle if not left_safe else right_obstacle if not right_safe else None
                carb.log_warn(
                    f"Waiting for safe lateral avoidance for {self._stage_prefix}; "
                    f"static obstacle={blocked_obstacle or 'voxel'}"
                )
            return

        self._set_guard_block_state(False)
        new_target_list = list(getattr(navigation_manager, "path_targets", []))
        new_target_list.insert(0, carb.Float3(*selected.point_xyz))
        navigation_manager.generate_path(new_target_list)
        try:
            navigation_manager.path_points.insert(0, current_pos)
        except Exception:
            pass

    def _animation_graph_prim(self):
        return self._current_stage.GetPrimAtPath(
            Person.character_root_prim_path
            + "/Biped_Setup/CharacterAnimation/AnimationGraph"
        )

    def _sync_animation_graph_instance_variables(self, animation_graph) -> bool:
        """Materialize graph variables on a runtime-spawned character.

        Isaac 4.5's VariablesService skips USD notices while the timeline is
        playing.  Runtime-spawned characters therefore need the same variable
        synchronization performed explicitly before the graph is used.
        """
        skel_root = self.character_skel_root
        if skel_root is None or not skel_root.IsValid():
            return False

        graph_variables = {}
        for graph_attr in animation_graph.GetAttributes():
            attr_name = graph_attr.GetName()
            if not attr_name.startswith(Person.anim_graph_variable_prefix):
                continue
            graph_variables[attr_name] = graph_attr

        required_names = {
            Person.anim_graph_variable_prefix + name
            for name in Person.required_anim_graph_variables
        }
        if not required_names.issubset(graph_variables):
            missing = sorted(required_names.difference(graph_variables))
            self._warn_pose_read_throttled(
                f"Biped AnimGraph is not fully loaded for {self._stage_prefix}; "
                f"missing template variables={missing}"
            )
            return False

        for attr_name, graph_attr in graph_variables.items():
            graph_type = graph_attr.GetTypeName()
            instance_attr = skel_root.GetAttribute(attr_name)
            if instance_attr and instance_attr.IsValid() and instance_attr.GetTypeName() != graph_type:
                skel_root.RemoveProperty(attr_name)
                instance_attr = None
            if not instance_attr or not instance_attr.IsValid():
                instance_attr = skel_root.CreateAttribute(attr_name, graph_type)

            graph_value = graph_attr.Get()
            if graph_value is not None:
                instance_attr.SetCustomDataByKey("default", graph_value)
            elif graph_type == Sdf.ValueTypeNames.Float3Array:
                instance_attr.Set(Vt.Vec3fArray())

        return all(skel_root.HasAttribute(name) for name in required_names)

    def _ensure_animation_graph_ready(self) -> bool:
        if self._anim_graph_ready and self.character_graph is not None:
            return True

        now = time.monotonic()
        if not self._anim_graph_setup_ok and now >= self._next_anim_graph_setup_attempt:
            self._next_anim_graph_setup_attempt = now + 0.5
            self._anim_graph_setup_ok = self.add_animation_graph_to_agent()
        if not self._anim_graph_setup_ok:
            return False

        if self.character_graph is None:
            self.character_graph = ag.get_character(self.character_skel_root_stage_path)
        if self.character_graph is None:
            self._warn_pose_read_throttled(
                f"AnimGraph runtime is not ready for {self._stage_prefix}; cached movement will wait."
            )
            return False

        skel_root = self.character_skel_root
        required_names = [
            Person.anim_graph_variable_prefix + name
            for name in Person.required_anim_graph_variables
        ]
        if skel_root is None or not all(skel_root.HasAttribute(name) for name in required_names):
            self._anim_graph_setup_ok = False
            self.character_graph = None
            return False

        self._anim_graph_ready = True
        carb.log_info(
            f"AnimGraph ready for {self._stage_prefix}: "
            f"variables={list(Person.required_anim_graph_variables)}"
        )
        return True

    def update(self, dt: float):
        """
        Method that implements the logic to make the person move around in the simulation world and also play the animation

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """

        if not self._active or self._parked or not self._stage_prim_is_valid():
            return

        if not self._ensure_animation_graph_ready():
            return

        self._last_update_dt = max(0.0, float(dt))
        external_sample = self._external_motion.sample(time.monotonic())
        if external_sample is not None:
            if self._external_hold_position is not None:
                self._set_idle_animation()
                self._apply_external_hold_pose()
                return
            if self._external_terminal_position is not None:
                self._set_idle_animation()
                self._advance_external_terminal_alignment(dt)
                return
            self._apply_external_motion_sample(external_sample)
            animation_sample = self._external_animation_sample(external_sample)
            if self._external_motion.should_walk(animation_sample):
                self._set_walk_animation(
                    animation_sample.position,
                    external_sample=animation_sample,
                )
            else:
                self._track_external_stationary_yaw(animation_sample, dt)
                self._set_idle_animation()
            return
        if self._behavior_script_enabled:
            if self._robot_interaction_policy == "avoid":
                self._publish_robot_obstacles_to_people(dt)
            self._dispatch_behavior_commands_if_ready()
            return

        # Call the controller update method that should update the reference of the target position
        if self._controller:
            self._controller.update(dt)

        # Stop when there is no active target.
        if self._num_path_points <= 0:
            self._set_idle_animation()
            return
        if self._current_point_index < 0 or self._current_point_index >= self._num_path_points:
            self._set_idle_animation()
            return

        # Compute the distance between the current position and the active goal.
        index_point = self._current_point_index
        active_goal = self._target_position[index_point]
        distance_to_target_position = np.linalg.norm(active_goal - self._state.position)

        if distance_to_target_position < self._path_point_arrival_radius:
            self._current_point_index += self._path_direction
            if self._current_point_index >= self._num_path_points:
                if self._loop_path and self._num_path_points > 1:
                    self._path_direction = -1
                    self._current_point_index = self._num_path_points - 2
                else:
                    self._set_idle_animation()
                    return
            elif self._current_point_index < 0:
                if self._loop_path and self._num_path_points > 1:
                    self._path_direction = 1
                    self._current_point_index = 1
                else:
                    self._set_idle_animation()
                    return
            index_point = self._current_point_index
            active_goal = self._target_position[index_point]

        if not self._robot_guard_allows_motion(dt, active_goal):
            self._set_idle_animation()
            return

        self._set_walk_animation(active_goal)

        # If we have a backend, update the state of the person
        if self._backend:
            self._backend.update(self._state, dt)

        # if self.character_skel_root_stage_path is not None:
        #     PeopleManager.get_people_manager().add_person(self.character_skel_root_stage_path, self)

    def update_target_position(
        self,
        position,
        walk_speed=1.0,
        loop: bool = False,
        yaw=None,
        constrain_to_path: bool = False,
    ):
        """
        Method that updates the target position of the person to which it will move towards.

        Args:
            position (list): A list with the x, y, z coordinates of the target position.
        """
        self._external_motion.clear()
        self._clear_external_hold()
        self._clear_external_terminal_alignment()
        self._reset_external_motion_mode("legacy_path")
        path = np.array(position, dtype=float)
        if path.ndim == 1:
            path = path.reshape(1, -1)
        if path.size == 0:
            self._target_position = np.empty((0, 3), dtype=float)
            self._num_path_points = 0
            self._current_point_index = 0
            self._path_direction = 1
            self._loop_path = False
            self._target_speed = 0.0
            self._target_yaw = None
            self._constrain_to_path = False
            self._set_guard_block_state(False)
            return
        if loop:
            current = np.array(self._state.position, dtype=float).reshape(1, 3)
            if np.linalg.norm(path[0] - current[0]) > 1e-4:
                path = np.vstack([current, path])
            self._current_point_index = 1 if len(path) > 1 else 0
        else:
            self._current_point_index = 0
        self._target_position = path
        self._num_path_points = len(path)
        self._path_direction = 1
        self._loop_path = bool(loop)
        self._target_speed = float(walk_speed)
        self._target_yaw = None if yaw is None else float(yaw)
        self._constrain_to_path = bool(constrain_to_path)
        self._set_guard_block_state(False)
        self._last_walk_speed = None
        self._motion_command_generation += 1
        self._motion_state = "accepted"
        return self._motion_command_generation

    def stop_motion(self):
        self._external_motion.clear()
        self._clear_external_hold()
        self._clear_external_terminal_alignment()
        self._reset_external_motion_mode("stop_motion")
        self._target_position = np.empty((0, 3), dtype=float)
        self._num_path_points = 0
        self._current_point_index = 0
        self._path_direction = 1
        self._loop_path = False
        self._target_speed = 0.0
        self._target_yaw = None
        self._constrain_to_path = False
        self._set_guard_block_state(False)
        self._robot_yield_saved_path_points = None
        self._robot_yield_blocked = False
        self._robot_guard_latched = False
        self._robot_guard_clear_since = 0.0
        self._motion_command_generation += 1
        agent_name, agent = self._behavior_agent()
        if agent is not None:
            try:
                self._apply_behavior_navigation_mode(agent)
                if getattr(agent, "current_command", None) is not None:
                    agent.commands = list(getattr(agent, "commands", []))[:1]
                    agent.end_current_command()
                else:
                    agent.commands = []
                self._dispatched_command_generation = self._motion_command_generation
            except Exception as exc:
                self._warn_pose_read_throttled(
                    f"Failed to stop People command for {agent_name}: {exc}"
                )
        else:
            self._set_idle_animation()
        self._motion_state = "idle"

    @property
    def is_parked(self) -> bool:
        return bool(self._parked)

    def _set_collision_proxy_enabled(self, enabled: bool) -> None:
        if not self._collision_proxy_path:
            return
        prim = self._current_stage.GetPrimAtPath(self._collision_proxy_path)
        if prim is None or not prim.IsValid():
            return
        try:
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True).Set(
                bool(enabled) and self._physics_proxy_enabled
            )
        except Exception:
            pass

    def park(self, position) -> None:
        """Hide an idle character without invalidating AnimGraph handles."""
        if not self._active:
            raise RuntimeError(f"Cannot park disposed pedestrian {self._stage_prefix}")
        self.set_direct_pose(position, stop=True)
        self._set_collision_proxy_enabled(False)
        try:
            if self.prim is not None and self.prim.IsValid():
                UsdGeom.Imageable(self.prim).MakeInvisible()
        except Exception:
            pass
        self._parked = True
        self._pose_valid = False

    def reactivate(self, position, yaw: float = 0.0) -> None:
        """Reset and show a parked character while preserving its graph bindings."""
        if not self._active:
            raise RuntimeError(f"Cannot reactivate disposed pedestrian {self._stage_prefix}")
        self.set_direct_pose(position, yaw=float(yaw), stop=True)
        self._guard_blocked = False
        self._guard_block_count = 0
        self._guard_block_reason = ""
        self._last_walk_speed = None
        try:
            if self.prim is not None and self.prim.IsValid():
                UsdGeom.Imageable(self.prim).MakeVisible()
        except Exception:
            pass
        self._set_collision_proxy_enabled(True)
        self._parked = False
        self._pose_valid = False

    def dispose(self):
        """Deactivate callbacks before the USD prim is destroyed."""
        if not getattr(self, "_active", True):
            return
        self.stop_motion()
        self._active = False
        self._pose_valid = False
        for method_name, callback_name in (
            ("remove_physics_callback", getattr(self, "_state_callback_name", "")),
            ("remove_physics_callback", getattr(self, "_update_callback_name", "")),
            ("remove_timeline_callback", getattr(self, "_timeline_callback_name", "")),
        ):
            remover = getattr(self._world, method_name, None)
            if callable(remover) and callback_name:
                try:
                    remover(callback_name)
                except Exception:
                    pass
        if self._collision_proxy_path:
            try:
                omni.kit.commands.execute("IsaacSimDestroyPrim", prim_path=self._collision_proxy_path)
            except Exception:
                pass
            self._collision_proxy_path = None
        self.character_graph = None

    def retire(self):
        """Make a character disappear without destroying a live AnimGraph."""
        if not getattr(self, "_active", True):
            return
        try:
            if self.prim is not None and self.prim.IsValid():
                UsdGeom.Imageable(self.prim).MakeInvisible()
        except Exception:
            pass
        self.dispose()

    def set_direct_pose(self, position, yaw: float | None = None, *, stop: bool = True):
        self._external_motion.clear()
        self._clear_external_hold()
        self._clear_external_terminal_alignment()
        self._reset_external_motion_mode("direct_pose")
        pos = np.array(position, dtype=float).reshape(3)
        if yaw is None:
            yaw = float(Rotation.from_quat(self._state.orientation).as_euler("xyz")[2])
        if stop:
            self.stop_motion()
        orientation = Rotation.from_euler("z", float(yaw), degrees=False).as_quat()
        self._set_stage_root_pose(pos, float(yaw))
        if self.character_graph is not None:
            try:
                self.character_graph.set_world_transform(
                    carb.Float3(float(pos[0]), float(pos[1]), float(pos[2])),
                    carb.Float4(
                        float(orientation[0]),
                        float(orientation[1]),
                        float(orientation[2]),
                        float(orientation[3]),
                    ),
                )
            except Exception as exc:
                self._warn_pose_read_throttled(
                    f"Failed to synchronize direct AnimGraph pose for {self._stage_prefix}: {exc}"
                )
        self._state.position = pos
        self._state.orientation = orientation
        self._update_collision_proxy()

    def set_external_motion(
        self,
        position,
        velocity,
        yaw: float,
        *,
        timeout_sec: float = 0.5,
        freeze_pose: bool = False,
        motion_mode: int = 0,
    ):
        """Give an external planner sole authority over root motion."""
        if not self._external_motion.enabled:
            self.stop_motion()
        speed = math.hypot(float(velocity[0]), float(velocity[1]))
        mode = int(motion_mode)
        if freeze_pose and mode == 0:
            mode = int(ExternalMotionMode.FREEZE)
        previous_mode, active_mode = self._external_motion_mode.transition(mode)
        if previous_mode != active_mode:
            current_yaw = float(
                Rotation.from_quat(self._state.orientation).as_euler("xyz")[2]
            )
            carb.log_info(
                f"External pedestrian mode {self._stage_prefix}: "
                f"{previous_mode.name}->{active_mode.name}, "
                f"target_yaw={float(yaw):.3f}, actual_yaw={current_yaw:.3f}, "
                f"speed={speed:.3f}"
            )
        if active_mode == ExternalMotionMode.FREEZE:
            self._clear_external_terminal_alignment()
            if self._external_hold_position is None:
                self._external_hold_position = np.array(
                    self._state.position,
                    dtype=float,
                ).reshape(3)
                self._external_hold_orientation = Rotation.from_euler(
                    "z",
                    float(yaw),
                    degrees=False,
                ).as_quat()
        elif active_mode == ExternalMotionMode.TERMINAL_ALIGN:
            self._clear_external_hold()
            if self._external_terminal_position is None:
                self._external_terminal_position = np.array(
                    self._state.position,
                    dtype=float,
                ).reshape(3)
                self._external_terminal_complete_logged = False
            self._external_terminal_yaw = float(yaw)
        else:
            self._clear_external_hold()
            self._clear_external_terminal_alignment()
        self._external_motion.set_command(
            position=position,
            velocity=velocity,
            yaw=float(yaw),
            received_at=time.monotonic(),
            timeout_sec=float(timeout_sec),
        )
        self._target_speed = speed
        self._target_yaw = float(yaw)
        self._motion_state = "executing"
        self._set_guard_block_state(False)

    def _clear_external_hold(self) -> None:
        self._external_hold_position = None
        self._external_hold_orientation = None

    def _reset_external_motion_mode(self, reason: str) -> None:
        previous, current = self._external_motion_mode.transition(
            ExternalMotionMode.LOCOMOTION
        )
        if previous != current:
            carb.log_info(
                f"External pedestrian mode {self._stage_prefix}: "
                f"{previous.name}->{current.name}, reason={reason}"
            )

    def _clear_external_terminal_alignment(self) -> None:
        self._external_terminal_position = None
        self._external_terminal_yaw = None
        self._external_terminal_complete_logged = False

    def _apply_external_hold_pose(self) -> None:
        """Keep an idle character at one world pose without moving its parent root."""
        if (
            self._external_hold_position is None
            or self._external_hold_orientation is None
        ):
            return
        position = np.array(self._external_hold_position, dtype=float).reshape(3)
        orientation = np.array(
            self._external_hold_orientation,
            dtype=float,
        ).reshape(4)
        if self.character_graph is not None:
            try:
                self.character_graph.set_world_transform(
                    carb.Float3(
                        float(position[0]),
                        float(position[1]),
                        float(position[2]),
                    ),
                    carb.Float4(
                        float(orientation[0]),
                        float(orientation[1]),
                        float(orientation[2]),
                        float(orientation[3]),
                    ),
                )
            except Exception as exc:
                self._warn_pose_read_throttled(
                    f"Failed to hold external AnimGraph pose for "
                    f"{self._stage_prefix}: {exc}"
                )
                return
        self._state.position = position
        self._state.orientation = orientation
        self._pose_valid = True
        self._last_valid_pose_time = time.monotonic()
        self._update_collision_proxy()

    def _advance_external_terminal_alignment(self, dt: float) -> None:
        """Rotate an idle character toward its semantic terminal yaw."""
        if (
            self._external_terminal_position is None
            or self._external_terminal_yaw is None
        ):
            return
        try:
            current_yaw = float(
                Rotation.from_quat(self._state.orientation).as_euler("xyz")[2]
            )
            next_yaw = bounded_yaw_step(
                current_yaw,
                float(self._external_terminal_yaw),
                max_rate_radps=self._external_motion_yaw_rate_radps,
                dt=float(dt),
            )
        except (TypeError, ValueError):
            return
        remaining_error = math.atan2(
            math.sin(float(self._external_terminal_yaw) - next_yaw),
            math.cos(float(self._external_terminal_yaw) - next_yaw),
        )
        orientation = Rotation.from_euler(
            "z",
            next_yaw,
            degrees=False,
        ).as_quat()
        self._apply_external_terminal_pose(orientation)
        if (
            abs(remaining_error) <= 1e-3
            and not self._external_terminal_complete_logged
        ):
            self._external_terminal_complete_logged = True
            carb.log_info(
                f"External terminal alignment complete {self._stage_prefix}: "
                f"target_yaw={float(self._external_terminal_yaw):.3f}, "
                f"actual_yaw={next_yaw:.3f}, reason=yaw_tolerance"
            )

    def _apply_external_terminal_pose(self, orientation=None) -> None:
        if self._external_terminal_position is None:
            return
        position = np.array(
            self._external_terminal_position,
            dtype=float,
        ).reshape(3)
        if orientation is None:
            orientation = np.array(
                self._state.orientation,
                dtype=float,
            ).reshape(4)
        else:
            orientation = np.array(orientation, dtype=float).reshape(4)
        if self.character_graph is not None:
            try:
                self.character_graph.set_world_transform(
                    carb.Float3(
                        float(position[0]),
                        float(position[1]),
                        float(position[2]),
                    ),
                    carb.Float4(
                        float(orientation[0]),
                        float(orientation[1]),
                        float(orientation[2]),
                        float(orientation[3]),
                    ),
                )
            except Exception as exc:
                self._warn_pose_read_throttled(
                    f"Failed to align external terminal pose for "
                    f"{self._stage_prefix}: {exc}"
                )
                return
        self._state.position = position
        self._state.orientation = orientation
        self._pose_valid = True
        self._last_valid_pose_time = time.monotonic()
        self._update_collision_proxy()

    def _external_animation_sample(self, reference_sample):
        """Let MotionMatching move the root while it tracks HuNav's reference."""
        try:
            current_yaw = float(
                Rotation.from_quat(self._state.orientation).as_euler("xyz")[2]
            )
        except (TypeError, ValueError):
            current_yaw = float(reference_sample.yaw)
        return turn_aware_animation_sample(
            reference_sample,
            self._state.position,
            current_yaw,
            tracking_gain=self._external_motion_tracking_gain,
            max_speed_mps=self._external_motion_max_speed_mps,
            slow_angle_rad=self._external_turn_slow_angle_rad,
            full_slow_angle_rad=self._external_turn_full_slow_angle_rad,
            minimum_speed_scale=self._external_turn_minimum_speed_scale,
        )

    def _apply_external_motion_sample(self, sample):
        # HuNav supplies a reference trajectory.  Normal locomotion is owned
        # by MotionMatching.  Root rebasing is explicitly opt-in because it
        # appears as a visible teleport when the graph falls behind.
        reference_position = np.array(sample.position, dtype=float).reshape(3)
        error_m = float(
            np.linalg.norm(reference_position[:2] - self._state.position[:2])
        )
        if (
            not sample.expired
            and self._external_motion_hard_sync_distance_m > 0.0
            and error_m > self._external_motion_hard_sync_distance_m
        ):
            self._synchronize_external_root_pose(reference_position, float(sample.yaw))
            now = time.monotonic()
            if now - self._last_external_sync_warn >= 1.0:
                self._last_external_sync_warn = now
                carb.log_warn(
                    f"External HuNav reference rebased {self._stage_prefix}: "
                    f"tracking_error_m={error_m:.3f}"
                )
        self._target_speed = float(sample.speed)
        self._motion_state = "idle" if sample.expired else "executing"

    def _track_external_stationary_yaw(self, sample, dt: float) -> None:
        """Finish a HuNav terminal heading without translating the character."""
        if self._external_motion_yaw_rate_radps <= 0.0:
            return
        try:
            current_yaw = float(
                Rotation.from_quat(self._state.orientation).as_euler("xyz")[2]
            )
            next_yaw = bounded_yaw_step(
                current_yaw,
                float(sample.yaw),
                max_rate_radps=self._external_motion_yaw_rate_radps,
                dt=float(dt),
            )
        except (TypeError, ValueError):
            return
        yaw_error = math.atan2(
            math.sin(float(sample.yaw) - current_yaw),
            math.cos(float(sample.yaw) - current_yaw),
        )
        if abs(yaw_error) <= 1e-3:
            return
        position = np.array(self._state.position, dtype=float).reshape(3)
        orientation = Rotation.from_euler("z", next_yaw, degrees=False).as_quat()
        if self.character_graph is not None:
            try:
                self.character_graph.set_world_transform(
                    carb.Float3(float(position[0]), float(position[1]), float(position[2])),
                    carb.Float4(
                        float(orientation[0]),
                        float(orientation[1]),
                        float(orientation[2]),
                        float(orientation[3]),
                    ),
                )
            except Exception as exc:
                self._warn_pose_read_throttled(
                    f"Failed to align stationary external yaw for {self._stage_prefix}: {exc}"
                )
                return
        self._state.orientation = orientation
        self._update_collision_proxy()

    def _synchronize_external_root_pose(self, position, yaw: float) -> None:
        pos = np.array(position, dtype=float).reshape(3)
        orientation = Rotation.from_euler("z", float(yaw), degrees=False).as_quat()
        self._set_stage_root_pose(pos, float(yaw))
        if self.character_graph is not None:
            try:
                self.character_graph.set_world_transform(
                    carb.Float3(float(pos[0]), float(pos[1]), float(pos[2])),
                    carb.Float4(
                        float(orientation[0]),
                        float(orientation[1]),
                        float(orientation[2]),
                        float(orientation[3]),
                    ),
                )
            except Exception as exc:
                self._warn_pose_read_throttled(
                    f"Failed to rebase external AnimGraph pose for {self._stage_prefix}: {exc}"
                )
        self._state.position = pos
        self._state.orientation = orientation
        self._pose_valid = True
        self._last_valid_pose_time = time.monotonic()
        self._update_collision_proxy()

    def _set_stage_root_pose(self, position, yaw: float):
        if self.prim is None or not self.prim.IsValid():
            return
        translate_attr = self.prim.GetAttribute("xformOp:translate")
        if translate_attr:
            translate_attr.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
        orient_attr = self.prim.GetAttribute("xformOp:orient")
        if orient_attr:
            quat = _yaw_rad_to_gf_quat(float(yaw))
            if type(orient_attr.Get()) == Gf.Quatf:
                orient_attr.Set(Gf.Quatf(quat))
            else:
                orient_attr.Set(quat)

    @staticmethod
    def _finite_vector(values, expected_len: int) -> bool:
        try:
            arr = np.array(values, dtype=float).reshape(-1)
        except Exception:
            return False
        return len(arr) >= expected_len and bool(np.all(np.isfinite(arr[:expected_len])))

    @staticmethod
    def _valid_quat_xyzw(values) -> bool:
        if not Person._finite_vector(values, 4):
            return False
        quat = np.array(values, dtype=float).reshape(-1)[:4]
        return float(np.linalg.norm(quat)) > 1e-6

    def _read_character_graph_pose(self):
        if self.character_graph is None:
            return None
        pos = carb.Float3(0, 0, 0)
        rot = carb.Float4(0, 0, 0, 0)
        try:
            self.character_graph.get_world_transform(pos, rot)
        except Exception:
            return None
        position = np.array([pos[0], pos[1], pos[2]], dtype=float)
        orientation = np.array([rot.x, rot.y, rot.z, rot.w], dtype=float)
        if not self._finite_vector(position, 3) or not self._valid_quat_xyzw(orientation):
            return None
        previous = np.array(self._state.position, dtype=float).reshape(-1)[:3]
        if (
            self._finite_vector(previous, 3)
            and np.linalg.norm(position) < 1e-3
            and np.linalg.norm(previous) > 0.75
            and np.linalg.norm(position - previous) > 0.75
        ):
            return None
        return position, orientation

    def _read_stage_root_pose(self):
        if self.prim is None or not self.prim.IsValid():
            return None
        try:
            matrix = UsdGeom.Xformable(self.prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            translation = matrix.ExtractTranslation()
            quat = matrix.ExtractRotationQuat()
            imaginary = quat.GetImaginary()
            real = quat.GetReal()
        except Exception:
            return None
        position = np.array([translation[0], translation[1], translation[2]], dtype=float)
        orientation = np.array([imaginary[0], imaginary[1], imaginary[2], real], dtype=float)
        if not self._finite_vector(position, 3) or not self._valid_quat_xyzw(orientation):
            return None
        previous = np.array(self._state.position, dtype=float).reshape(-1)[:3]
        if self._finite_vector(previous, 3):
            max_safe_jump = max(1.0, float(self._target_speed) * 1.0 + 0.5)
            if self._num_path_points > 0 and np.linalg.norm(position - previous) > max_safe_jump:
                return None
        return position, orientation

    @classmethod
    def _get_static_voxel_guard(cls):
        if cls._shared_static_voxel_guard is not None:
            return cls._shared_static_voxel_guard
        map_path = os.environ.get("ARENA_ISAAC_VOXEL_MAP_PATH", "").strip()
        if not map_path:
            return None
        try:
            diameter = 2.0 * float(cls.collision_proxy_radius)
            cls._shared_static_voxel_guard = VoxelCollisionGuard(
                robot_name="pedestrian_static_guard",
                config=VoxelGuardConfig(
                    enabled=True,
                    map_path=map_path,
                    length=diameter,
                    width=diameter,
                    margin=0.0,
                    z_min=0.05,
                    z_max=1.2,
                    refresh_sec=1.0,
                    log_sec=0.0,
                    block_log_sec=2.0,
                ),
            )
        except Exception as exc:
            carb.log_warn(f"Pedestrian static voxel guard is unavailable: {exc}")
            cls._shared_static_voxel_guard = None
        return cls._shared_static_voxel_guard

    def _static_pose_collides(self, position, orientation) -> tuple[bool, str | None]:
        safe, obstacle = self._static_pose_check(position)
        return (not safe), obstacle

    def _is_terminal_semantic_approach(self, position) -> bool:
        if self._motion_state != "executing" or len(self._target_position) == 0:
            return self._motion_state != "executing"
        final_target = np.array(self._target_position[-1], dtype=float)
        acceptance_radius = terminal_semantic_approach_radius(
            constrain_to_path=self._constrain_to_path,
            default_radius=self._path_point_arrival_radius,
            final_radius=float(Utils.CONFIG["MinDistanceToFinalTarget"]),
        )
        return float(np.linalg.norm(final_target[:2] - np.array(position, dtype=float)[:2])) <= acceptance_radius

    def _warn_pose_read_throttled(self, message: str):
        now = time.monotonic()
        if float(now) - float(self._last_pose_read_warn) >= 2.0:
            self._last_pose_read_warn = float(now)
            carb.log_warn(message)

    def update_state(self, dt: float):
        """
        Method that is called at every physics step to retrieve and update the current state of the person, i.e., get
        the current position, orientation, linear and angular velocities and acceleration of the person.

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """

        if not self._active or self._parked or not self._stage_prim_is_valid():
            return

        self._ensure_animation_graph_ready()

        # External motion supplies a reference, but MotionMatching owns normal
        # root movement. Feed the actual root back into the state stream.
        if self._external_motion.enabled:
            if self._external_hold_position is not None:
                self._apply_external_hold_pose()
                return
            if self._external_terminal_position is not None:
                self._apply_external_terminal_pose()
                return
            pose = self._read_character_graph_pose()
            if pose is None:
                pose = self._read_stage_root_pose()
            if pose is None:
                self._pose_valid = False
                self._warn_pose_read_throttled(
                    f"Keeping previous external-motion pose for {self._stage_prefix}; "
                    "AnimGraph/root transform was invalid or implausible."
                )
                return
            self._state.position = pose[0]
            self._state.orientation = pose[1]
            self._pose_valid = True
            self._last_valid_pose_time = time.monotonic()
            self._update_collision_proxy()
            return

        pose = self._read_character_graph_pose()
        if pose is None:
            pose = self._read_stage_root_pose()
        if pose is None:
            self._pose_valid = False
            self._warn_pose_read_throttled(
                f"Keeping previous pedestrian pose for {self._stage_prefix}; "
                "AnimGraph/root transform was invalid or implausible."
            )
            return

        robot_blocker = self._candidate_robot_pose_blocker(pose[0])
        if robot_blocker is not None:
            _, agent = self._behavior_agent()
            navigation_manager = None if agent is None else getattr(agent, "navigation_manager", None)
            if navigation_manager is not None:
                self._pause_behavior_path_for_robot(navigation_manager)
            else:
                self._set_guard_block_state(True, reason="robot")
            self._restore_last_safe_pose()
            now = time.monotonic()
            if now - self._last_guard_block_log >= 0.5:
                self._last_guard_block_log = now
                carb.log_warn(
                    f"Rejected pedestrian root motion into robot {robot_blocker} for "
                    f"{self._stage_prefix}; restored last safe pose."
                )
            return

        collides, obstacle = self._static_pose_collides(pose[0], pose[1])
        if collides and not self._is_terminal_semantic_approach(pose[0]):
            self._set_guard_block_state(True, reason="static_voxel")
            now = time.monotonic()
            if now - self._last_static_guard_warn >= 2.0:
                self._last_static_guard_warn = now
                carb.log_warn(
                    f"Blocked pedestrian root motion into {obstacle or 'voxel obstacle'} for {self._stage_prefix}."
                )
            return

        # Update the current state of the person only after validating the source pose.
        self._set_guard_block_state(False)
        self._state.position = pose[0]
        self._state.orientation = pose[1]
        self._pose_valid = True
        self._last_valid_pose_time = time.monotonic()
        self._update_collision_proxy()

        # Signal the controller the updated state
        if self._controller:
            self._controller.update_state(self._state)

    def _candidate_robot_pose_blocker(self, candidate_position) -> str | None:
        if self._robot_interaction_policy != "stop":
            return None
        try:
            from isaac_utils.mecanum_teleop import mecanum_teleop_manager
        except Exception:
            return None
        pedestrian = PedestrianGuardState(
            name=str(self._stage_prefix or self._requested_stage_name),
            pos_xy=(
                float(self._state.position[0]),
                float(self._state.position[1]),
            ),
            next_pos_xy=(
                float(candidate_position[0]),
                float(candidate_position[1]),
            ),
            radius=float(Person.collision_proxy_radius),
        )
        for robot in list(getattr(mecanum_teleop_manager, "robots", {}).values()):
            getter = getattr(robot, "get_dynamic_guard_state", None)
            if not callable(getter):
                continue
            robot_state = getter(0.0)
            if robot_state is None:
                continue
            current_score, next_score = pedestrian_robot_scores(pedestrian, robot_state)
            if movement_allowed(
                current_score,
                next_score,
                escape_epsilon=self._guard_escape_epsilon,
            ):
                continue
            return str(robot_state.name)
        return None

    def _restore_last_safe_pose(self) -> None:
        position = np.array(self._state.position, dtype=float).reshape(3)
        orientation = np.array(self._state.orientation, dtype=float).reshape(4)
        yaw = float(Rotation.from_quat(orientation).as_euler("xyz")[2])
        self._set_stage_root_pose(position, yaw)
        if self.character_graph is not None:
            try:
                self.character_graph.set_world_transform(
                    carb.Float3(
                        float(position[0]),
                        float(position[1]),
                        float(position[2]),
                    ),
                    carb.Float4(
                        float(orientation[0]),
                        float(orientation[1]),
                        float(orientation[2]),
                        float(orientation[3]),
                    ),
                )
            except Exception as exc:
                self._warn_pose_read_throttled(
                    f"Failed to restore safe AnimGraph pose for {self._stage_prefix}: {exc}"
                )
        self._pose_valid = True
        self._update_collision_proxy()

    def _stage_prim_is_valid(self):
        if self.prim is None or not self.prim.IsValid():
            return False
        if not self.character_skel_root_stage_path:
            return False
        prim = self._current_stage.GetPrimAtPath(self.character_skel_root_stage_path)
        return prim is not None and prim.IsValid()

    def _preview_active_goal(self):
        if self._num_path_points <= 0:
            return None
        if self._behavior_script_enabled:
            goal = active_polyline_goal(
                self._target_position,
                self._state.position,
            )
            return None if goal is None else np.array(goal, dtype=float)
        if self._current_point_index < 0 or self._current_point_index >= self._num_path_points:
            return None
        index_point = int(self._current_point_index)
        active_goal = np.array(self._target_position[index_point], dtype=float)
        if np.linalg.norm(active_goal - self._state.position) >= self._path_point_arrival_radius:
            return active_goal
        index_point += int(self._path_direction)
        if index_point >= self._num_path_points:
            if self._loop_path and self._num_path_points > 1:
                index_point = self._num_path_points - 2
            else:
                return None
        elif index_point < 0:
            if self._loop_path and self._num_path_points > 1:
                index_point = 1
            else:
                return None
        if index_point < 0 or index_point >= self._num_path_points:
            return None
        return np.array(self._target_position[index_point], dtype=float)

    def get_dynamic_guard_state(self, dt: float, active_goal=None, *, assume_moving: bool = False):
        pos_xy = (float(self._state.position[0]), float(self._state.position[1]))
        goal = active_goal
        if goal is None:
            goal = self._preview_active_goal()
        goal_xy = None if goal is None else (float(goal[0]), float(goal[1]))
        prediction_speed = (
            float(self._target_speed)
            if assume_moving or not self._robot_yield_blocked
            else 0.0
        )
        next_pos_xy = predict_pedestrian_step(
            pos_xy=pos_xy,
            goal_xy=goal_xy,
            speed=prediction_speed,
            dt=float(max(0.0, dt)),
        )
        return PedestrianGuardState(
            name=str(self._stage_prefix or self._requested_stage_name),
            pos_xy=pos_xy,
            next_pos_xy=next_pos_xy,
            radius=float(Person.collision_proxy_radius),
        )

    def _robot_guard_allows_motion(self, dt: float, active_goal) -> bool:
        if dt <= 0.0 or self._target_speed <= 0.0:
            return True
        try:
            from isaac_utils.mecanum_teleop import mecanum_teleop_manager
        except Exception:
            return True
        prediction_dt = max(float(dt), self._hard_guard_horizon_sec)
        now = time.monotonic()
        blocked_by = self._robot_guard_blocker(
            mecanum_teleop_manager,
            prediction_dt,
            active_goal,
            extra_margin_m=0.0,
        )
        if blocked_by is not None:
            self._robot_guard_latched = True
            self._robot_guard_clear_since = 0.0
        elif self._robot_guard_latched:
            blocked_by = self._robot_guard_blocker(
                mecanum_teleop_manager,
                prediction_dt,
                active_goal,
                extra_margin_m=self._hard_guard_release_margin_m,
            )
            if blocked_by is not None:
                self._robot_guard_clear_since = 0.0
            elif self._robot_guard_clear_since <= 0.0:
                self._robot_guard_clear_since = float(now)
                blocked_by = "release_hold"
            elif now - self._robot_guard_clear_since < self._hard_guard_release_hold_sec:
                blocked_by = "release_hold"
            else:
                self._robot_guard_latched = False
                self._robot_guard_clear_since = 0.0
        if blocked_by is None:
            return True
        if float(now) - float(self._last_guard_block_log) >= 0.5:
            self._last_guard_block_log = float(now)
            carb.log_info(
                f"[dynamic_actor_guard] pedestrian {self._stage_prefix} paused for robot {blocked_by}"
            )
        return False

    def _robot_guard_blocker(
        self,
        manager,
        prediction_dt: float,
        active_goal,
        *,
        extra_margin_m: float,
    ) -> str | None:
        # Resume checks must predict the first walking step, not the currently
        # paused pose, otherwise the pedestrian chatters back into the robot.
        ped_state = self.get_dynamic_guard_state(
            prediction_dt,
            active_goal=active_goal,
            assume_moving=True,
        )
        guarded_ped_state = PedestrianGuardState(
            name=ped_state.name,
            pos_xy=ped_state.pos_xy,
            next_pos_xy=ped_state.next_pos_xy,
            radius=(
                float(ped_state.radius)
                + self._hard_guard_margin_m
                + max(0.0, float(extra_margin_m))
            ),
        )
        for robot in list(getattr(manager, "robots", {}).values()):
            getter = getattr(robot, "get_dynamic_guard_state", None)
            if not callable(getter):
                continue
            robot_state = getter(prediction_dt)
            if robot_state is None:
                continue
            current_score, next_score = pedestrian_robot_scores(guarded_ped_state, robot_state)
            if movement_allowed(current_score, next_score, escape_epsilon=self._guard_escape_epsilon):
                continue
            return robot_state.name
        return None

    def _publish_robot_obstacles_to_people(self, dt: float) -> None:
        """Expose bridge robots to the official People dynamic avoidance manager."""
        try:
            from isaac_utils.mecanum_teleop import mecanum_teleop_manager
        except Exception:
            return
        manager = GlobalCharacterPositionManager.get_instance()
        for robot in list(getattr(mecanum_teleop_manager, "robots", {}).values()):
            getter = getattr(robot, "get_dynamic_guard_state", None)
            if not callable(getter):
                continue
            state = getter(max(float(dt), 0.0))
            if state is None:
                continue
            key = f"arena_robot:{state.name}"
            current = carb.Float3(float(state.pos_xy[0]), float(state.pos_xy[1]), 0.0)
            future = carb.Float3(float(state.next_pos_xy[0]), float(state.next_pos_xy[1]), 0.0)
            radius = math.hypot(
                max(float(state.forward), float(state.rear)),
                max(float(state.left), float(state.right)),
            )
            manager.set_character_current_pos(key, current)
            manager.set_character_future_pos(key, future)
            manager.set_character_radius(key, max(radius, 0.1))

    def spawn_agent(self, usd_file, stage_name, init_pos, init_yaw):

        # If there is no XForm primitive in the stage to hold all the people, create one
        if not self._current_stage.GetPrimAtPath(Person.character_root_prim_path):
            prims.create_prim(Person.character_root_prim_path, "Xform")

        # If the base biped character is not present in the stage, spawn it
        if not self._current_stage.GetPrimAtPath(
            Person.character_root_prim_path + "/Biped_Setup"
        ):
            prim = prims.create_prim(
                Person.character_root_prim_path + "/Biped_Setup",
                "Xform",
                usd_path=Person.assets_root_path + "/Biped_Setup.usd",
            )
            prim.GetAttribute("visibility").Set("invisible")

        # Spawn the person in the world.  Isaac Sim 4.5 CharacterUtil expects a
        # relative character name; it creates the prim under /World/Characters.
        stage_name_leaf = normalize_stage_name(stage_name, default=self._requested_stage_name)
        self.prim = CharacterUtil.load_character_usd_to_stage(
            usd_file, init_pos, float(init_yaw), stage_name_leaf
        )
        if self.prim is None or not self.prim.IsValid():
            self.prim = self._current_stage.GetPrimAtPath(
                os.path.join(Person.character_root_prim_path, stage_name_leaf)
            )
        if self.prim is not None and self.prim.IsValid():
            self._stage_prefix = str(self.prim.GetPath())

        # Set the initial position and orientation of the person defensively.
        if self.prim is not None and self.prim.IsValid():
            translate_attr = self.prim.GetAttribute("xformOp:translate")
            if translate_attr:
                translate_attr.Set(Gf.Vec3d(float(init_pos[0]), float(init_pos[1]), float(init_pos[2])))
            orient_attr = self.prim.GetAttribute("xformOp:orient")
            if orient_attr:
                quat = _yaw_rad_to_gf_quat(float(init_yaw))
                if type(orient_attr.Get()) == Gf.Quatf:
                    orient_attr.Set(Gf.Quatf(quat))
                else:
                    orient_attr.Set(quat)

        # Get the Skeleton root of the character
        self.character_skel_root, root_path = Person._transverse_prim(
            self._current_stage, self._stage_prefix
        )
        self.character_skel_root_stage_path = root_path

        # Add the current person to the person manager
        PeopleManager.get_people_manager().add_person(
            self.character_skel_root_stage_path, self
        )

    def _spawn_collision_proxy(self):
        stage = self._current_stage
        if stage is None:
            return
        if not stage.GetPrimAtPath(Person.collision_proxy_root_path):
            prims.create_prim(Person.collision_proxy_root_path, "Xform")
        proxy_path = get_stage_next_free_path(
            stage,
            os.path.join(Person.collision_proxy_root_path, self._requested_stage_name),
            False,
        )
        capsule = UsdGeom.Capsule.Define(stage, proxy_path)
        capsule.CreateRadiusAttr(float(Person.collision_proxy_radius))
        capsule.CreateHeightAttr(float(Person.collision_proxy_height))
        capsule.CreateAxisAttr("Z")
        prim = capsule.GetPrim()
        imageable = UsdGeom.Imageable(prim)
        imageable.MakeInvisible()
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True).Set(
            bool(self._physics_proxy_enabled)
        )
        try:
            UsdPhysics.RigidBodyAPI.Apply(prim)
            prim.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool, custom=False).Set(True)
            prim.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool, custom=False).Set(True)
            prim.CreateAttribute("physics:startsAsleep", Sdf.ValueTypeNames.Bool, custom=False).Set(False)
            if PhysxSchema is not None:
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        except Exception:
            pass
        self._collision_proxy_path = proxy_path
        self._update_collision_proxy()

    def _update_collision_proxy(self):
        if not self._collision_proxy_path:
            return
        prim = self._current_stage.GetPrimAtPath(self._collision_proxy_path)
        if prim is None or not prim.IsValid():
            return
        xform = UsdGeom.Xformable(prim)
        try:
            xform.ClearXformOpOrder()
        except Exception:
            pass
        pos = self._state.position
        xform.AddTranslateOp().Set(
            Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2] + Person.collision_proxy_center_z))
        )

    def add_animation_graph_to_agent(self) -> bool:
        animation_graph = self._animation_graph_prim()
        if animation_graph is None or not animation_graph.IsValid():
            self._warn_pose_read_throttled(
                f"Biped AnimGraph asset is not loaded for {self._stage_prefix}; setup will retry."
            )
            return False
        if self.character_skel_root is None or not self.character_skel_root.IsValid():
            self._warn_pose_read_throttled(
                f"Character SkelRoot is invalid for {self._stage_prefix}; AnimGraph setup will retry."
            )
            return False

        try:
            omni.kit.commands.execute(
                "RemoveAnimationGraphAPICommand",
                paths=[Sdf.Path(self.character_skel_root.GetPrimPath())],
            )
            if not self._sync_animation_graph_instance_variables(animation_graph):
                return False
            omni.kit.commands.execute(
                "ApplyAnimationGraphAPICommand",
                paths=[Sdf.Path(self.character_skel_root.GetPrimPath())],
                animation_graph_path=Sdf.Path(animation_graph.GetPrimPath()),
            )
            if not self._sync_animation_graph_instance_variables(animation_graph):
                return False
        except Exception as exc:
            self._warn_pose_read_throttled(
                f"AnimGraph setup failed for {self._stage_prefix}: {exc}"
            )
            return False

        self.character_graph = None
        self._anim_graph_ready = False
        carb.log_info(
            f"AnimGraph variables prepared for {self._stage_prefix}; waiting for runtime character."
        )
        return True

    @staticmethod
    def _transverse_prim(stage, stage_prefix):

        # Check if the prim is the one we are looking for
        prim = stage.GetPrimAtPath(stage_prefix)

        # If the prim is the one we are looking for, return it
        if prim.GetTypeName() == "SkelRoot":
            return prim, stage_prefix

        # Otherwise, get all the children of the prim and keep transversing until we find the SkelRoot
        children = prim.GetAllChildren()

        # If there are no children, return
        if not children or len(children) == 0:
            return None, None

        # Recursively look through the children to get the SkelRoot
        for child in children:
            prim_child, child_stage_prefix = Person._transverse_prim(
                stage, stage_prefix + "/" + child.GetName()
            )

            if prim_child is not None:
                return prim_child, child_stage_prefix

        return None, None

    @staticmethod
    def get_character_asset_list():
        # List all files in characters directory
        result, folder_list = omni.client.list("{}/".format(Person.assets_root_path))

        if result != omni.client.Result.OK:
            carb.log_error(
                "Unable to get character assets from provided asset root path."
            )
            return

        # Prune items from folder list that are not directories.
        pruned_folder_list = [
            folder.relative_path
            for folder in folder_list
            if (folder.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN)
            and not folder.relative_path.startswith(".")
        ]

        return pruned_folder_list

    @staticmethod
    def get_path_for_character_prim(agent_name):
        # Check if a folder with agent_name exists. If exists we load the character, else we load a random character
        agent_folder = os.path.join(Person.assets_root_path, agent_name)
        result, properties = omni.client.stat(agent_folder)

        # Attempt to load the character if it exists, otherwise load a random character
        if result != omni.client.Result.OK:
            carb.log_error(
                f"Character folder does not exist: {agent_name}. Available: {Person.get_character_asset_list()}"
            )
            return None

        # Get the usd present in the character folder
        character_folder = "{}/{}".format(Person.assets_root_path, agent_name)
        character_usd = Person.get_usd_in_folder(character_folder)

        # Return the character name (folder name) and the usd path to the character
        return "{}/{}".format(character_folder, character_usd)

    @staticmethod
    def get_usd_in_folder(character_folder_path):
        result, folder_list = omni.client.list(character_folder_path)

        if result != omni.client.Result.OK:
            carb.log_error(
                "Unable to read character folder path at {}".format(
                    character_folder_path
                )
            )
            return

        for item in folder_list:
            if item.relative_path.endswith(".usd"):
                return item.relative_path

        carb.log_error(
            "Unable to file a .usd file in {} character folder".format(
                character_folder_path
            )
        )
