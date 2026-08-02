"""Command-driven SMPL-H motion-graph runtime independent of Isaac Sim."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

try:
    from ..analysis.gait import upright_stance_frame
    from ..analysis.rotations import (
        axis_angle_to_matrix,
        matrix_to_quaternion,
        planar_heading,
        quaternion_slerp,
        quaternion_to_matrix,
        z_rotation,
    )
    from ..builders.smplh_numpy import (
        load_model,
        shaped_vertices_and_joints,
        skin_frame_rotations,
    )
    from .motion_graph import MotionClip
    from .motion_library import load_manifest
except ImportError:
    from analysis.gait import upright_stance_frame
    from analysis.rotations import (
        axis_angle_to_matrix,
        matrix_to_quaternion,
        planar_heading,
        quaternion_slerp,
        quaternion_to_matrix,
        z_rotation,
    )
    from builders.smplh_numpy import (
        load_model,
        shaped_vertices_and_joints,
        skin_frame_rotations,
    )
    from runtime.motion_graph import MotionClip
    from runtime.motion_library import load_manifest


class MotionState(str, Enum):
    IDLE = "idle"
    RESUME = "resume"
    WALK = "walk"
    TURN_RIGHT = "turn_right"
    TURN_LEFT = "turn_left"
    STOP = "stop"


@dataclass(frozen=True)
class MotionCommand:
    speed_mps: float = 0.0
    yaw_rate_rps: float = 0.0


@dataclass(frozen=True)
class AvatarFrame:
    rotations: np.ndarray
    root_position: np.ndarray
    yaw_rad: float
    state: MotionState
    motion_id: str
    clip_index: int
    commanded_speed_mps: float
    animated_speed_mps: float
    gait_phase: float
    pending_action: str | None


@dataclass
class _Blend:
    source_rotations: np.ndarray
    target_rotations: np.ndarray
    source_position: np.ndarray
    support_joint: int | None
    support_anchor: np.ndarray | None
    source_velocity: np.ndarray
    root_velocity: np.ndarray
    frames: int
    frame: int
    target_state: MotionState
    target_clip: MotionClip
    target_index: int
    target_base_yaw: float


class SmplUsdAvatarBackend:
    """Map continuous speed/yaw commands onto a small contact-aware motion graph."""

    def __init__(
        self,
        manifest_path: Path,
        asset_root: Path,
        *,
        transition_sec: float = 0.12,
        resume_transition_sec: float = 0.30,
        stop_speed_mps: float = 0.08,
        turn_rate_threshold_rps: float = 0.25,
    ) -> None:
        self.asset_root = asset_root.expanduser().resolve()
        manifest = load_manifest(manifest_path.expanduser().resolve())
        entries = {entry["id"]: entry for entry in manifest["motions"]}

        def load(motion_id: str) -> MotionClip:
            entry = entries[motion_id]
            return MotionClip.load(
                motion_id, self.asset_root / entry["clip"]["relative_path"]
            )

        walk_ids = (
            "walk_slow_37_01",
            "walk_forward_83_37",
            "walk_fast_38_01",
        )
        self.walk_profiles = [load(motion_id) for motion_id in walk_ids]
        self.walk_profile_speeds = np.asarray(
            [entries[motion_id]["gait"]["nominal_speed_mps"] for motion_id in walk_ids],
            dtype=np.float64,
        )
        self.walk_profile_durations = np.asarray(
            [entries[motion_id]["gait"]["cycle_duration_sec"] for motion_id in walk_ids],
            dtype=np.float64,
        )

        self.clips = {
            MotionState.WALK: self.walk_profiles[1],
            MotionState.TURN_RIGHT: load("turn_right_90_83_37"),
            MotionState.TURN_LEFT: load("turn_left_90_83_36"),
            MotionState.STOP: load("stop_walk_83_36"),
            MotionState.RESUME: load("start_walk_83_37"),
        }
        self.fps = self.clips[MotionState.WALK].fps
        if not all(np.isclose(clip.fps, self.fps) for clip in self.clips.values()):
            raise ValueError("all avatar runtime clips must have one FPS")

        gender = self.clips[MotionState.WALK].gender
        self.model = load_model(self.asset_root / "smplh" / gender / "model.npz")
        self.shaped, self.rest_joints = shaped_vertices_and_joints(
            self.model, self.clips[MotionState.WALK].betas
        )
        self.transition_frames = max(2, int(round(transition_sec * self.fps)))
        self.resume_transition_frames = max(
            self.transition_frames,
            int(round(resume_transition_sec * self.fps)),
        )
        self.walk_overlap_frames = self.transition_frames
        self.stop_speed_mps = float(stop_speed_mps)
        self.turn_rate_threshold_rps = float(turn_rate_threshold_rps)
        self._rotations = {
            state: axis_angle_to_matrix(clip.poses.reshape(-1, 52, 3))
            for state, clip in self.clips.items()
        }
        self._heading = {
            state: planar_heading(clip.poses[:, :3])
            for state, clip in self.clips.items()
        }
        self._walk_rotations = {
            clip.motion_id: axis_angle_to_matrix(clip.poses.reshape(-1, 52, 3))
            for clip in self.walk_profiles
        }
        self._walk_heading = {
            clip.motion_id: planar_heading(clip.poses[:, :3])
            for clip in self.walk_profiles
        }

        stop_clip = self.clips[MotionState.STOP]
        self.idle_index, self.idle_diagnostics = upright_stance_frame(
            stop_clip.joint_positions,
            stop_clip.root_trajectory,
            self.fps,
            start_frame=max(0, len(stop_clip.poses) - int(round(0.8 * self.fps))),
        )
        self.command = MotionCommand()
        self.state = MotionState.IDLE
        self.clip = stop_clip
        self.clip_index = self.idle_index
        self.clip_start_index = self.idle_index
        self.base_yaw = 0.0
        self.root_position = np.zeros(3, dtype=np.float64)
        rotations, yaw = self._global_rotations(
            MotionState.STOP, self.idle_index, self.idle_index, 0.0
        )
        self.current_rotations = rotations
        self.current_yaw = yaw
        vertices, _ = skin_frame_rotations(
            self.model,
            self.shaped,
            self.rest_joints,
            self.current_rotations,
            self.root_position,
        )
        self.root_position[2] -= float(vertices[:, 2].min())
        self.current_contacts = stop_clip.foot_contacts[self.idle_index].copy()
        self._blend: _Blend | None = None
        self._walk_loop_base_yaw: float | None = None
        self._turn_latched = False
        self._pending_action: MotionState | None = None
        self._walk_phase = 0.0
        self._animated_speed_mps = float(self.walk_profile_speeds[1])
        self._last_root_velocity = np.zeros(3, dtype=np.float64)
        self.max_speed_accel_mps2 = 0.8

    @property
    def frame(self) -> AvatarFrame:
        return AvatarFrame(
            rotations=self.current_rotations.copy(),
            root_position=self.root_position.copy(),
            yaw_rad=float(self.current_yaw),
            state=self.state,
            motion_id=self.clip.motion_id,
            clip_index=int(self.clip_index),
            commanded_speed_mps=float(self.command.speed_mps),
            animated_speed_mps=(
                float(self._animated_speed_mps)
                if self.state is MotionState.WALK
                else self._current_speed()
            ),
            gait_phase=(
                float(self._walk_phase % 1.0)
                if self.state is MotionState.WALK
                else float("nan")
            ),
            pending_action=(
                self._pending_action.value
                if self._pending_action is not None
                else None
            ),
        )

    def set_command(self, speed_mps: float, yaw_rate_rps: float) -> None:
        self.command = MotionCommand(float(speed_mps), float(yaw_rate_rps))
        if abs(yaw_rate_rps) < 0.5 * self.turn_rate_threshold_rps:
            self._turn_latched = False

    def update(self) -> AvatarFrame:
        """Advance one authored frame at the manifest FPS."""
        if self._blend is not None:
            self._advance_blend()
            return self.frame

        if self.state is MotionState.IDLE:
            if self.command.speed_mps > self.stop_speed_mps:
                self._begin_action(MotionState.RESUME, head_frames=72)
            return self.frame

        if self.state is MotionState.WALK:
            if self.command.speed_mps <= self.stop_speed_mps:
                self._pending_action = MotionState.STOP
            elif (
                self.command.yaw_rate_rps < -self.turn_rate_threshold_rps
                and not self._turn_latched
            ):
                self._turn_latched = True
                self._pending_action = MotionState.TURN_RIGHT
            elif (
                self.command.yaw_rate_rps > self.turn_rate_threshold_rps
                and not self._turn_latched
            ):
                self._turn_latched = True
                self._pending_action = MotionState.TURN_LEFT

            if (
                self._pending_action is not None
                and self._is_safe_transition_phase()
            ):
                action = self._pending_action
                self._pending_action = None
                head_frames = 90 if action is MotionState.STOP else 60
                self._begin_action(action, head_frames=head_frames)
                self._advance_blend()
                return self.frame

        self._advance_clip()
        return self.frame

    def _is_safe_transition_phase(self) -> bool:
        """Allow action changes at foot transfer or stable support phases."""
        phase = self._walk_phase % 1.0
        safe_phases = (0.0, 0.25, 0.5, 0.75)
        distance = min(
            min(abs(phase - candidate), 1.0 - abs(phase - candidate))
            for candidate in safe_phases
        )
        return distance <= 0.04

    def _global_rotations(
        self,
        state: MotionState,
        index: int,
        start_index: int,
        base_yaw: float,
    ) -> tuple[np.ndarray, float]:
        rotations = self._rotations[state][index].copy()
        heading = self._heading[state]
        local_root = z_rotation(-heading[index]) @ rotations[0]
        yaw = base_yaw + float(heading[index] - heading[start_index])
        rotations[0] = z_rotation(yaw) @ local_root
        return rotations, yaw

    def _best_target(self, state: MotionState, head_frames: int) -> int:
        clip = self.clips[state]
        if state is MotionState.RESUME:
            return 0
        current_speed = self._current_speed()
        speed = clip.planar_speed()
        first_index = 0
        if state is MotionState.STOP:
            first_index = min(
                int(round(clip.event_start_sec * clip.fps)),
                len(clip.poses) - 2,
            )
        smooth_speed = np.convolve(
            np.pad(speed, (4, 4), mode="edge"),
            np.ones(9, dtype=np.float64) / 9.0,
            mode="valid",
        )
        best_cost = np.inf
        best_index = first_index
        for index in range(first_index, min(head_frames, len(clip.poses) - 1)):
            target_rotations, _ = self._global_rotations(
                state, index, index, self.current_yaw
            )
            pose_cost = float(
                np.sqrt(
                    np.mean(
                        (
                            self.current_rotations[1:22]
                            - target_rotations[1:22]
                        )
                        ** 2
                    )
                )
            )
            speed_cost = abs(current_speed - speed[index])
            contact_cost = float(
                np.mean(self.current_contacts != clip.foot_contacts[index])
            )
            if state is MotionState.STOP:
                speed_weight = 1.4
            elif state is MotionState.WALK and self.state is MotionState.RESUME:
                # The phase gait immediately regulates speed. At this boundary
                # limb phase and support continuity matter more than a source
                # clip's within-step pelvis-speed oscillation.
                speed_weight = 0.2
            else:
                speed_weight = 0.6
            contact_weight = 0.15 if state is MotionState.STOP else 0.35
            total = (
                pose_cost
                + speed_weight * speed_cost
                + contact_weight * contact_cost
            )
            if state is MotionState.STOP:
                future = smooth_speed[index : min(index + 16, len(speed))]
                speed_rise = max(0.0, float(future.max() - future[0]))
                end_rise = max(0.0, float(future[-1] - future[0]))
                total += 2.0 * speed_rise + 2.0 * end_rise
            if total < best_cost:
                best_cost = total
                best_index = index
        return best_index

    def _current_speed(self) -> float:
        if self.state is MotionState.IDLE:
            return 0.0
        if self.state is MotionState.WALK:
            return float(self._animated_speed_mps)
        speed = self.clip.planar_speed()
        return float(speed[min(self.clip_index, len(speed) - 1)])

    def _current_velocity_world(self) -> np.ndarray:
        if self.state is MotionState.IDLE:
            return np.zeros(3, dtype=np.float64)
        if self.state is MotionState.WALK:
            return self._last_root_velocity.copy()
        previous = max(0, self.clip_index - 1)
        following = min(len(self.clip.root_trajectory) - 1, self.clip_index + 1)
        if following == previous:
            return np.zeros(3, dtype=np.float64)
        local_velocity = (
            self.clip.root_trajectory[following]
            - self.clip.root_trajectory[previous]
        ) * (self.fps / (following - previous))
        return local_velocity @ z_rotation(self.base_yaw).T

    def _begin_action(self, state: MotionState, *, head_frames: int) -> None:
        target_clip = self.clips[state]
        target_index = self._best_target(state, head_frames)
        target_rotations, _ = self._global_rotations(
            state, target_index, target_index, self.current_yaw
        )
        target_delta = (
            target_clip.root_trajectory[target_index + 1]
            - target_clip.root_trajectory[target_index]
        )
        # Keep locomotion advancing during pose blends. Freezing the root while
        # the legs continue changing phase produces a visible cadence hitch.
        root_velocity = (
            np.zeros(3, dtype=np.float64)
            if state is MotionState.RESUME
            else target_delta @ z_rotation(self.current_yaw).T * self.fps
        )
        common_contacts = (
            self.current_contacts & target_clip.foot_contacts[target_index]
        )
        support_joint = (
            10 + int(np.flatnonzero(common_contacts)[0])
            if common_contacts.any()
            else None
        )
        if state is MotionState.WALK and self.state is MotionState.RESUME:
            # Both clips are already translating. Treating their noisy
            # double-contact labels as a planted foot pulls the pelvis forward
            # during the blend and creates a visible resume-to-walk jump.
            support_joint = None
        support_anchor = None
        if support_joint is not None:
            _, joints = skin_frame_rotations(
                self.model,
                self.shaped,
                self.rest_joints,
                self.current_rotations,
                self.root_position,
            )
            support_anchor = joints[support_joint].copy()
            _, target_joints = skin_frame_rotations(
                self.model,
                self.shaped,
                self.rest_joints,
                target_rotations,
                np.zeros(3),
            )
            required_position = support_anchor - target_joints[support_joint]
            if (
                np.linalg.norm(
                    required_position[:2] - self.root_position[:2]
                )
                > 0.15
            ):
                support_joint = None
                support_anchor = None
        self._blend = _Blend(
            source_rotations=self.current_rotations.copy(),
            target_rotations=target_rotations,
            source_position=self.root_position.copy(),
            support_joint=support_joint,
            support_anchor=support_anchor,
            source_velocity=self._current_velocity_world(),
            root_velocity=root_velocity,
            frames=(
                self.resume_transition_frames
                if state is MotionState.RESUME
                else self.transition_frames
            ),
            frame=0,
            target_state=state,
            target_clip=target_clip,
            target_index=target_index,
            target_base_yaw=self.current_yaw,
        )

    def _advance_blend(self) -> None:
        blend = self._blend
        if blend is None:
            return
        previous_position = self.root_position.copy()
        blend.frame += 1
        linear_fraction = blend.frame / blend.frames
        fraction = linear_fraction * linear_fraction * (3.0 - 2.0 * linear_fraction)
        elapsed = blend.frame / self.fps
        duration = blend.frames / self.fps
        blend_displacement = (
            blend.source_velocity * elapsed
            + 0.5
            * (blend.root_velocity - blend.source_velocity)
            * (elapsed * elapsed / duration)
        )
        quaternions = quaternion_slerp(
            matrix_to_quaternion(blend.source_rotations),
            matrix_to_quaternion(blend.target_rotations),
            fraction,
        )
        self.current_rotations = quaternion_to_matrix(quaternions)
        if blend.support_joint is not None and blend.support_anchor is not None:
            _, local_joints = skin_frame_rotations(
                self.model,
                self.shaped,
                self.rest_joints,
                self.current_rotations,
                np.zeros(3),
            )
            moving_anchor = (
                blend.support_anchor
                + blend_displacement
            )
            self.root_position = moving_anchor - local_joints[blend.support_joint]
        else:
            self.root_position = (
                blend.source_position
                + blend_displacement
            )
        planar_delta = self.root_position[:2] - previous_position[:2]
        planar_step = float(np.linalg.norm(planar_delta))
        if planar_step > 0.025:
            self.root_position[:2] = (
                previous_position[:2] + planar_delta * (0.025 / planar_step)
            )
        if blend.frame < blend.frames:
            return
        self.state = blend.target_state
        self.clip = blend.target_clip
        self.clip_index = blend.target_index
        self.clip_start_index = blend.target_index
        self.base_yaw = blend.target_base_yaw
        self.current_yaw = blend.target_base_yaw
        self.current_contacts = self.clip.foot_contacts[self.clip_index].copy()
        if self.state is MotionState.WALK:
            self._walk_loop_base_yaw = None
            self._walk_phase = self.clip_index / max(1, len(self.clip.poses) - 1)
            # Keep the cycle-average gait speed latent across action clips.
            # A single pelvis frame can be much faster than the cycle mean and
            # would incorrectly switch the first phase-walk frame to fast gait.
        self._blend = None

    def _advance_clip(self) -> None:
        previous_index = self.clip_index
        next_index = previous_index + 1
        if self.state is MotionState.STOP and next_index > self.idle_index:
            self.state = MotionState.IDLE
            self.clip_index = self.idle_index
            self.current_contacts = self.clip.foot_contacts[
                self.idle_index
            ].copy()
            return
        if (
            self.state in (MotionState.TURN_RIGHT, MotionState.TURN_LEFT)
            and next_index >= len(self.clip.poses) - 1
        ):
            self._begin_action(MotionState.WALK, head_frames=30)
            self._advance_blend()
            return
        if self.state is MotionState.WALK:
            self._advance_phase_walk()
            return
        if (
            self.state is MotionState.WALK
            and next_index
            >= len(self.clip.poses) - self.walk_overlap_frames
        ):
            self._advance_walk_overlap(previous_index, next_index)
            return
        if next_index >= len(self.clip.poses):
            # A natural start may finish at any point in the gait cycle. Match
            # the complete cycle instead of forcing it back near phase zero.
            head_frames = (
                len(self.clips[MotionState.WALK].poses)
                if self.state is MotionState.RESUME
                else 30
            )
            self._begin_action(MotionState.WALK, head_frames=head_frames)
            self._advance_blend()
            return

        local_delta = (
            self.clip.root_trajectory[next_index]
            - self.clip.root_trajectory[previous_index]
        )
        self.root_position += local_delta @ z_rotation(self.base_yaw).T
        self.clip_index = next_index
        self.current_rotations, self.current_yaw = self._global_rotations(
            self.state, next_index, self.clip_start_index, self.base_yaw
        )
        self.current_contacts = self.clip.foot_contacts[next_index].copy()

    def _walk_profile_pair(self, speed_mps: float) -> tuple[int, int, float, float]:
        speeds = self.walk_profile_speeds
        if speed_mps <= speeds[0]:
            scale = np.sqrt(max(speed_mps, 1.0e-3) / speeds[0])
            return 0, 0, 0.0, float(np.clip(scale, 0.85, 1.0))
        if speed_mps >= speeds[-1]:
            scale = np.sqrt(speed_mps / speeds[-1])
            return 2, 2, 0.0, float(np.clip(scale, 1.0, 1.15))
        upper = int(np.searchsorted(speeds, speed_mps))
        lower = upper - 1
        weight = float(
            (speed_mps - speeds[lower]) / (speeds[upper] - speeds[lower])
        )
        return lower, upper, weight, 1.0

    def _sample_walk_profile(
        self, profile_index: int, phase: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        clip = self.walk_profiles[profile_index]
        rotations = self._walk_rotations[clip.motion_id]
        heading = self._walk_heading[clip.motion_id]
        cycle = int(np.floor(phase))
        local_phase = phase - cycle
        sample = local_phase * (len(clip.poses) - 1)
        lower = int(np.floor(sample))
        upper = min(lower + 1, len(clip.poses) - 1)
        fraction = sample - lower

        def facing_rotation(index: int) -> np.ndarray:
            value = rotations[index].copy()
            local_root = z_rotation(-heading[index]) @ value[0]
            value[0] = z_rotation(self.base_yaw) @ local_root
            return value

        pose = quaternion_to_matrix(
            quaternion_slerp(
                matrix_to_quaternion(facing_rotation(lower)),
                matrix_to_quaternion(facing_rotation(upper)),
                fraction,
            )
        )
        trajectory = clip.root_trajectory
        cycle_delta = trajectory[-1] - trajectory[0]
        relative_position = (
            (1.0 - fraction) * trajectory[lower]
            + fraction * trajectory[upper]
            - trajectory[0]
        )
        # Close root velocity without changing the cycle displacement.
        phase_scale = len(trajectory) - 1
        start_derivative = (trajectory[1] - trajectory[0]) * phase_scale
        end_derivative = (trajectory[-1] - trajectory[-2]) * phase_scale
        derivative_delta = start_derivative - end_derivative
        relative_position += derivative_delta * (
            local_phase**3 - local_phase**2
        )
        position = relative_position + cycle * cycle_delta

        seam_start = 0.88
        if local_phase > seam_start:
            seam = (local_phase - seam_start) / (1.0 - seam_start)
            seam = seam * seam * (3.0 - 2.0 * seam)
            start_pose = facing_rotation(0)
            pose = quaternion_to_matrix(
                quaternion_slerp(
                    matrix_to_quaternion(pose),
                    matrix_to_quaternion(start_pose),
                    seam,
                )
            )
        contacts = clip.foot_contacts[lower if fraction < 0.5 else upper]
        return pose, position, contacts.copy(), int(round(sample))

    def _advance_phase_walk(self) -> None:
        target = max(0.0, float(self.command.speed_mps))
        maximum_delta = self.max_speed_accel_mps2 / self.fps
        self._animated_speed_mps += float(
            np.clip(target - self._animated_speed_mps, -maximum_delta, maximum_delta)
        )
        low, high, weight, scale = self._walk_profile_pair(
            self._animated_speed_mps
        )
        frequency = (
            (1.0 - weight) / self.walk_profile_durations[low]
            + weight / self.walk_profile_durations[high]
        ) * scale
        previous_phase = self._walk_phase
        self._walk_phase += frequency / self.fps
        low_pose, low_position, low_contacts, low_index = self._sample_walk_profile(
            low, self._walk_phase
        )
        high_pose, high_position, high_contacts, high_index = self._sample_walk_profile(
            high, self._walk_phase
        )
        _, previous_low, _, _ = self._sample_walk_profile(low, previous_phase)
        _, previous_high, _, _ = self._sample_walk_profile(high, previous_phase)
        pose = quaternion_to_matrix(
            quaternion_slerp(
                matrix_to_quaternion(low_pose),
                matrix_to_quaternion(high_pose),
                weight,
            )
        )
        local_delta = (
            (1.0 - weight) * (low_position - previous_low)
            + weight * (high_position - previous_high)
        ) * scale
        world_delta = local_delta @ z_rotation(self.base_yaw).T
        self.root_position += world_delta
        self._last_root_velocity = world_delta * self.fps
        self.current_rotations = pose
        self.current_yaw = self.base_yaw
        self.current_contacts = (
            low_contacts if weight < 0.5 else high_contacts
        )
        selected = low if weight < 0.5 else high
        self.clip = self.walk_profiles[selected]
        self.clip_index = low_index if selected == low else high_index

    def _advance_walk_overlap(
        self, previous_index: int, next_index: int
    ) -> None:
        overlap_start = len(self.clip.poses) - self.walk_overlap_frames
        target_index = next_index - overlap_start
        if self._walk_loop_base_yaw is None:
            self._walk_loop_base_yaw = self.current_yaw

        source_rotations, source_yaw = self._global_rotations(
            MotionState.WALK,
            next_index,
            self.clip_start_index,
            self.base_yaw,
        )
        target_rotations, target_yaw = self._global_rotations(
            MotionState.WALK,
            target_index,
            0,
            self._walk_loop_base_yaw,
        )
        fraction = (target_index + 1) / self.walk_overlap_frames
        self.current_rotations = quaternion_to_matrix(
            quaternion_slerp(
                matrix_to_quaternion(source_rotations),
                matrix_to_quaternion(target_rotations),
                fraction,
            )
        )
        local_delta = (
            self.clip.root_trajectory[next_index]
            - self.clip.root_trajectory[previous_index]
        )
        self.root_position += local_delta @ z_rotation(self.base_yaw).T
        self.current_yaw = (
            (1.0 - fraction) * source_yaw + fraction * target_yaw
        )

        if target_index == self.walk_overlap_frames - 1:
            self.clip_index = target_index
            self.clip_start_index = 0
            self.base_yaw = self._walk_loop_base_yaw
            self.current_contacts = self.clip.foot_contacts[
                target_index
            ].copy()
            self._walk_loop_base_yaw = None
            return

        self.clip_index = next_index
        self.current_contacts = (
            self.clip.foot_contacts[target_index].copy()
            if fraction >= 0.5
            else self.clip.foot_contacts[next_index].copy()
        )
