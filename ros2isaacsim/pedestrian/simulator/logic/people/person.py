# Low level APIs
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
        movement_allowed,
        pedestrian_robot_scores,
        predict_pedestrian_step,
    )
except Exception:  # pragma: no cover
    from ros2isaacsim.isaac_utils.dynamic_actor_guard import (
        PedestrianGuardState,
        movement_allowed,
        pedestrian_robot_scores,
        predict_pedestrian_step,
    )
from isaac_utils.utils.assets import get_assets_root_path_safe
from isaac_utils.animgraph_people import normalize_stage_name

from omni.usd import get_stage_next_free_path
from pedestrian.simulator.logic.people.person_controller import PersonController
from pedestrian.simulator.logic.people_manager import PeopleManager

# Extension APIs
from pedestrian.simulator.logic.state import State
from pxr import Gf, Sdf
from scipy.spatial.transform import Rotation

import isaacsim.replicator.agent.core
from isaacsim.replicator.agent.core.settings import PrimPaths
from isaacsim.replicator.agent.core.stage_util import CharacterUtil
from isaacsim.replicator.agent.core.simulation import SimulationManager
from pxr import UsdGeom, UsdPhysics

try:
    from pxr import PhysxSchema  # type: ignore
except Exception:  # pragma: no cover
    PhysxSchema = None  # type: ignore


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

    if people_asset_folder:
        assets_root_path = people_asset_folder
    else:
        root_path = get_assets_root_path()
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

        # Add the animation graph to the agent, such that it can move around
        self.character_graph = None
        self.add_animation_graph_to_agent()

        # Set the controller for the person if any and initialize it
        self._controller = controller
        if self._controller:
            self._controller.initialize(self)

        # Set the backend for publishing the state of the person
        self._backend = backend
        if self._backend:
            self._backend.initialize(self)

        # Add a callback to the physics engine to update the current state of the person
        self._world.add_physics_callback(
            self._stage_prefix + "/state", self.update_state
        )

        # Add the update method to the physics callback if the world was received
        # so that we can apply the new references to be tracked by the person
        self._world.add_physics_callback(self._stage_prefix + "/update", self.update)

        # Set the flag that signals if the simulation is running or not
        self._sim_running = False

        self._collision_proxy_path = None
        self._spawn_collision_proxy()
        self._guard_escape_epsilon = 1e-4
        self._last_guard_block_log = 0.0

        # Add a callback to start/stop of the simulation once the play/stop button is hit
        self._world.add_timeline_callback(
            self._stage_prefix + "/start_stop_sim", self.sim_start_stop
        )

    @property
    def state(self):
        """The state of the person.

        Returns:
            State: The current state of the person, i.e., position, orientation, linear and angular velocities...
        """
        return self._state

    def sim_start_stop(self, event):
        """
        Callback that is called every time there is a timeline event such as starting/stoping the simulation.

        Args:
            event: A timeline event generated from Isaac Sim, such as starting or stoping the simulation.
        """

        # If the start/stop button was pressed, then call the start and stop methods accordingly
        if self._world.is_playing() and self._sim_running == False:
            self._sim_running = True
            self.start()

        if self._world.is_stopped() and self._sim_running == True:
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

    def update(self, dt: float):
        """
        Method that implements the logic to make the person move around in the simulation world and also play the animation

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """

        # Note: this is done to avoid the error of the character_graph being None. The animation graph is only created after the simulation starts
        if not self.character_graph or self.character_graph is None:
            self.character_graph = ag.get_character(self.character_skel_root_stage_path)
        if self.character_graph is None:
            return

        # Call the controller update method that should update the reference of the target position
        if self._controller:
            self._controller.update(dt)

        # Stop when there is no active target.
        if self._num_path_points <= 0:
            self.character_graph.set_variable("Walk", 0.0)
            self.character_graph.set_variable("Action", "Idle")
            return
        if self._current_point_index < 0 or self._current_point_index >= self._num_path_points:
            self.character_graph.set_variable("Walk", 0.0)
            self.character_graph.set_variable("Action", "Idle")
            return

        # Compute the distance between the current position and the active goal.
        index_point = self._current_point_index
        active_goal = self._target_position[index_point]
        distance_to_target_position = np.linalg.norm(active_goal - self._state.position)

        if distance_to_target_position < 0.5:
            self._current_point_index += self._path_direction
            if self._current_point_index >= self._num_path_points:
                if self._loop_path and self._num_path_points > 1:
                    self._path_direction = -1
                    self._current_point_index = self._num_path_points - 2
                else:
                    self.character_graph.set_variable("Walk", 0.0)
                    self.character_graph.set_variable("Action", "Idle")
                    return
            elif self._current_point_index < 0:
                if self._loop_path and self._num_path_points > 1:
                    self._path_direction = 1
                    self._current_point_index = 1
                else:
                    self.character_graph.set_variable("Walk", 0.0)
                    self.character_graph.set_variable("Action", "Idle")
                    return
            index_point = self._current_point_index
            active_goal = self._target_position[index_point]

        if not self._robot_guard_allows_motion(dt, active_goal):
            self.character_graph.set_variable("Walk", 0.0)
            self.character_graph.set_variable("Action", "Idle")
            return

        self.character_graph.set_variable("Action", "Walk")
        self.character_graph.set_variable(
            "PathPoints",
            [
                carb.Float3(float(self._state.position[0]), float(self._state.position[1]), float(self._state.position[2])),
                carb.Float3(float(active_goal[0]), float(active_goal[1]), float(active_goal[2])),
            ],
        )
        self.character_graph.set_variable("Walk", self._target_speed)

        # If we have a backend, update the state of the person
        if self._backend:
            self._backend.update(self._state, dt)

        # if self.character_skel_root_stage_path is not None:
        #     PeopleManager.get_people_manager().add_person(self.character_skel_root_stage_path, self)

    def update_target_position(self, position, walk_speed=1.0, loop: bool = False):
        """
        Method that updates the target position of the person to which it will move towards.

        Args:
            position (list): A list with the x, y, z coordinates of the target position.
        """
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

    def update_state(self, dt: float):
        """
        Method that is called at every physics step to retrieve and update the current state of the person, i.e., get
        the current position, orientation, linear and angular velocities and acceleration of the person.

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """

        # Note: this is done to avoid the error of the character_graph being None. The animation graph is only created after the simulation starts
        if not self.character_graph or self.character_graph is None:
            self.character_graph = ag.get_character(self.character_skel_root_stage_path)
        if self.character_graph is None:
            return

        # Get the current position of the person
        pos = carb.Float3(0, 0, 0)
        rot = carb.Float4(0, 0, 0, 0)
        self.character_graph.get_world_transform(pos, rot)

        # Update the current state of the person
        self._state.position = np.array([pos[0], pos[1], pos[2]])
        self._state.orientation = np.array([rot.x, rot.y, rot.z, rot.w])
        self._update_collision_proxy()

        # Signal the controller the updated state
        if self._controller:
            self._controller.update_state(self._state)

    def _preview_active_goal(self):
        if self._num_path_points <= 0:
            return None
        if self._current_point_index < 0 or self._current_point_index >= self._num_path_points:
            return None
        index_point = int(self._current_point_index)
        active_goal = np.array(self._target_position[index_point], dtype=float)
        if np.linalg.norm(active_goal - self._state.position) >= 0.5:
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

    def get_dynamic_guard_state(self, dt: float, active_goal=None):
        pos_xy = (float(self._state.position[0]), float(self._state.position[1]))
        goal = active_goal
        if goal is None:
            goal = self._preview_active_goal()
        goal_xy = None if goal is None else (float(goal[0]), float(goal[1]))
        next_pos_xy = predict_pedestrian_step(
            pos_xy=pos_xy,
            goal_xy=goal_xy,
            speed=float(self._target_speed),
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
        ped_state = self.get_dynamic_guard_state(dt, active_goal=active_goal)
        blocked_by = None
        for robot in list(getattr(mecanum_teleop_manager, "robots", {}).values()):
            getter = getattr(robot, "get_dynamic_guard_state", None)
            if not callable(getter):
                continue
            robot_state = getter(dt)
            if robot_state is None:
                continue
            current_score, next_score = pedestrian_robot_scores(ped_state, robot_state)
            if movement_allowed(current_score, next_score, escape_epsilon=self._guard_escape_epsilon):
                continue
            blocked_by = robot_state.name
            break
        if blocked_by is None:
            return True
        now = time.monotonic()
        if float(now) - float(self._last_guard_block_log) >= 0.5:
            self._last_guard_block_log = float(now)
            carb.log_info(
                f"[dynamic_actor_guard] pedestrian {self._stage_prefix} paused for robot {blocked_by}"
            )
        return False

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
                quat = Gf.Rotation(Gf.Vec3d(0, 0, 1), float(init_yaw)).GetQuat()
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
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True).Set(True)
        try:
            rb = UsdPhysics.RigidBodyAPI.Apply(prim)
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

    def add_animation_graph_to_agent(self):

        # Get the animation graph that we are going to add to the person
        animation_graph = self._current_stage.GetPrimAtPath(
            Person.character_root_prim_path
            + "/Biped_Setup/CharacterAnimation/AnimationGraph"
        )

        # Remove the animation graph attribute if it exists
        if self.character_skel_root is not None:
            omni.kit.commands.execute(
                "RemoveAnimationGraphAPICommand",
                paths=[Sdf.Path(self.character_skel_root.GetPrimPath())],
            )

        # Add the animation graph to the character
        if self.character_skel_root is not None:
            omni.kit.commands.execute(
                "ApplyAnimationGraphAPICommand",
                paths=[Sdf.Path(self.character_skel_root.GetPrimPath())],
                animation_graph_path=Sdf.Path(animation_graph.GetPrimPath()),
            )

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
