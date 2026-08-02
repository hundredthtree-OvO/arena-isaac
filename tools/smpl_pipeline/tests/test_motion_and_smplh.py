from pathlib import Path
import sys
import unittest

import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

from analysis.motion_features import (  # noqa: E402
    rank_natural_turn_windows,
    rank_stable_walk_windows,
)
from analysis.gait import upright_stance_frame  # noqa: E402
from analysis.rotations import (  # noqa: E402
    axis_angle_to_matrix,
    matrix_to_quaternion,
    planar_heading,
    quaternion_slerp,
    quaternion_to_matrix,
    z_rotation,
)
from builders.smplh_numpy import (  # noqa: E402
    SmplhModel,
    shaped_vertices_and_joints,
    skin_frame,
)
from runtime.motion_library import validate_manifest  # noqa: E402
from runtime.motion_graph import (  # noqa: E402
    MotionClip,
    find_shared_source_frame,
    select_transition,
)
from runtime.avatar_backend import (  # noqa: E402
    MotionState,
    SmplUsdAvatarBackend,
)


class RotationTest(unittest.TestCase):
    def test_z_rotation_and_smpl_forward_heading(self):
        pose = np.asarray([[0.0, 0.0, np.pi / 2.0]])
        matrix = axis_angle_to_matrix(pose)[0]
        np.testing.assert_allclose(matrix @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-8)
        self.assertAlmostEqual(planar_heading(np.zeros((1, 3)))[0], 0.0)

    def test_stable_window_ranking(self):
        fps = 20.0
        count = 81
        trans = np.zeros((count, 3))
        trans[:, 0] = np.arange(count) / fps
        poses = np.zeros((count, 156))
        ranked = rank_stable_walk_windows(trans, poses, fps)
        self.assertTrue(ranked)
        self.assertGreaterEqual(ranked[0].displacement_m, 1.0)

    def test_heading_decomposition_reconstructs_root_rotation(self):
        source = axis_angle_to_matrix(
            np.asarray([[0.1, -0.05, 1.2], [-0.05, 0.1, 1.35]])
        )
        heading = np.unwrap(
            np.arctan2(
                (source @ [0.0, 0.0, 1.0])[:, 1],
                (source @ [0.0, 0.0, 1.0])[:, 0],
            )
        )
        local = z_rotation(-heading) @ source
        reconstructed = z_rotation(heading) @ local
        np.testing.assert_allclose(reconstructed, source, atol=1e-10)
        quaternions = matrix_to_quaternion(source)
        np.testing.assert_allclose(np.linalg.norm(quaternions, axis=1), 1.0)
        np.testing.assert_allclose(
            quaternion_to_matrix(quaternions), source, atol=1e-10
        )

    def test_quaternion_slerp_endpoints(self):
        start = np.asarray([1.0, 0.0, 0.0, 0.0])
        end = matrix_to_quaternion(z_rotation(np.pi / 2.0))
        np.testing.assert_allclose(quaternion_slerp(start, end, 0.0), start)
        np.testing.assert_allclose(
            quaternion_to_matrix(quaternion_slerp(start, end, 1.0)),
            z_rotation(np.pi / 2.0),
            atol=1e-10,
        )

    def test_natural_turn_window_ranking(self):
        fps = 20.0
        count = 61
        angle = np.linspace(0.0, np.pi / 2.0, count)
        trans = np.column_stack((np.sin(angle), 1.0 - np.cos(angle), angle * 0.0))
        poses = np.zeros((count, 156))
        # Root local +Z is rotated into the tangent direction.
        poses[:, 0] = -np.sin(angle) * np.pi / 2.0
        poses[:, 1] = np.cos(angle) * np.pi / 2.0
        ranked = rank_natural_turn_windows(trans, poses, fps)
        self.assertTrue(ranked)
        self.assertAlmostEqual(abs(ranked[0].yaw_change_deg), 90.0, delta=5.0)


class SmplhNumpyTest(unittest.TestCase):
    def test_zero_pose_identity_skinning(self):
        vertices = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        model = SmplhModel(
            vertices_template=vertices,
            shapedirs=np.zeros((2, 3, 1)),
            posedirs=np.zeros((2, 3, 9)),
            joint_regressor=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            weights=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            parents=np.asarray([-1, 0]),
            faces=np.zeros((0, 3), dtype=np.int32),
        )
        shaped, joints = shaped_vertices_and_joints(model, np.zeros(1))
        result, result_joints = skin_frame(
            model, shaped, joints, np.zeros(6), np.asarray([2.0, 3.0, 4.0])
        )
        np.testing.assert_allclose(result, vertices + [2.0, 3.0, 4.0])
        np.testing.assert_allclose(result_joints, joints + [2.0, 3.0, 4.0])


class MotionLibraryTest(unittest.TestCase):
    def test_upright_stance_prefers_straight_knees(self):
        joints = np.zeros((2, 22, 3))
        for frame in range(2):
            joints[frame, 0] = [0.0, 0.0, 1.0]
            joints[frame, 12] = [0.0, 0.0, 1.6]
            for hip, knee, ankle, foot, x in (
                (1, 4, 7, 10, -0.1),
                (2, 5, 8, 11, 0.1),
            ):
                joints[frame, hip] = [x, 0.0, 1.0]
                joints[frame, knee] = [x, 0.0, 0.55]
                joints[frame, ankle] = [x, 0.0, 0.1]
                joints[frame, foot] = [x, 0.0, 0.0]
        joints[0, [4, 5], 1] = 0.25
        trajectory = np.zeros((2, 3))
        index, diagnostics = upright_stance_frame(joints, trajectory, 60.0)
        self.assertEqual(index, 1)
        self.assertLess(diagnostics["mean_knee_flex_deg"], 1.0)

    def test_manifest_contract_rejects_duplicate_ids(self):
        motion = {
            "id": "turn_right",
            "category": "turn",
            "direction": "right",
            "source": {},
            "clip": {},
            "timing": {"frames": 10, "fps": 60.0},
            "kinematics": {},
            "contacts": {"left_fraction": 0.5, "right_fraction": 0.5},
        }
        payload = {
            "schema_version": 1,
            "library_id": "test",
            "motions": [motion, dict(motion)],
        }
        with self.assertRaisesRegex(ValueError, "duplicate motion id"):
            validate_manifest(payload)

    def test_transition_and_shared_source_selection(self):
        def clip(motion_id, frames, source_frames, speed):
            poses = np.zeros((frames, 156))
            trajectory = np.zeros((frames, 3))
            trajectory[:, 0] = np.arange(frames) * speed / 10.0
            return MotionClip(
                motion_id,
                poses,
                trajectory,
                np.asarray(source_frames),
                np.tile([[True, False]], (frames, 1)),
                np.zeros((frames, 22, 3)),
                10.0,
                np.zeros(16),
                "male",
            )

        source = clip("source", 5, [0, 1, 2, 3, 4], 1.0)
        target = clip("target", 5, [4, 5, 6, 7, 8], 1.0)
        match = select_transition(
            source, target, source_start_index=3, target_end_index=2
        )
        self.assertLess(match.speed_cost, 1.0e-8)
        self.assertEqual(find_shared_source_frame(source, target), (4, 0))

    def test_walk_manifest_requires_speed_and_phase_contract(self):
        motion = {
            "id": "walk",
            "category": "walk",
            "direction": "forward",
            "source": {},
            "clip": {},
            "timing": {"frames": 10, "fps": 60.0},
            "kinematics": {},
            "contacts": {"left_fraction": 0.5, "right_fraction": 0.5},
        }
        payload = {
            "schema_version": 1,
            "library_id": "test",
            "motions": [motion],
        }
        with self.assertRaisesRegex(ValueError, "requires gait metadata"):
            validate_manifest(payload)


class AvatarBackendIntegrationTest(unittest.TestCase):
    ASSET_ROOT = Path(
        "/home/stardust/resources/arena_ws/arena_assets/smpl"
    )
    MANIFEST = (
        ASSET_ROOT
        / "derived/manifests/cmu_smplh_locomotion_seed.json"
    )

    @unittest.skipUnless(MANIFEST.exists(), "generated SMPL motion library absent")
    def test_sustained_walk_loop_does_not_insert_periodic_zero_speed(self):
        backend = SmplUsdAvatarBackend(self.MANIFEST, self.ASSET_ROOT)
        positions = []
        states = []
        for _ in range(650):
            backend.set_command(0.95, 0.0)
            frame = backend.update()
            positions.append(frame.root_position)
            states.append(frame.state)
        positions = np.asarray(positions)
        speed = np.linalg.norm(
            np.diff(positions[:, :2], axis=0), axis=1
        ) * backend.fps
        sustained = [
            speed[index]
            for index in range(200, len(speed))
            if states[index] is MotionState.WALK
            and states[index + 1] is MotionState.WALK
        ]
        self.assertTrue(sustained)
        self.assertGreater(min(sustained), 0.5)

    @unittest.skipUnless(MANIFEST.exists(), "generated SMPL motion library absent")
    def test_command_sequence_turns_stops_and_holds_without_root_jump(self):
        backend = SmplUsdAvatarBackend(self.MANIFEST, self.ASSET_ROOT)
        positions = []
        states = []
        yaw_before_turn = None
        yaw_after_turn = None
        for frame_index in range(1200):
            speed = 0.95 if frame_index < 800 else 0.0
            yaw_rate = -0.8 if 300 <= frame_index < 330 else 0.0
            backend.set_command(speed, yaw_rate)
            frame = backend.update()
            positions.append(frame.root_position)
            states.append(frame.state)
            if frame_index == 299:
                yaw_before_turn = frame.yaw_rad
            if (
                yaw_after_turn is None
                and frame_index > 330
                and frame.state is MotionState.WALK
            ):
                yaw_after_turn = frame.yaw_rad

        positions = np.asarray(positions)
        self.assertIn(MotionState.RESUME, states)
        self.assertIn(MotionState.TURN_RIGHT, states)
        self.assertIn(MotionState.STOP, states)
        self.assertEqual(states[-1], MotionState.IDLE)
        self.assertLess(
            np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1).max(),
            0.03,
        )
        self.assertLess(np.linalg.norm(positions[-1] - positions[-100]), 1.0e-8)
        self.assertIsNotNone(yaw_before_turn)
        self.assertIsNotNone(yaw_after_turn)
        self.assertLess(yaw_after_turn - yaw_before_turn, -1.2)

    @unittest.skipUnless(MANIFEST.exists(), "generated SMPL motion library absent")
    def test_stop_transition_uses_annotated_deceleration_event(self):
        backend = SmplUsdAvatarBackend(self.MANIFEST, self.ASSET_ROOT)
        for _ in range(700):
            backend.set_command(0.95, 0.0)
            backend.update()
        backend.set_command(0.0, 0.0)
        for _ in range(30):
            backend.update()
            if backend._blend is not None:
                break

        self.assertIsNotNone(backend._blend)
        event_start = int(
            round(
                backend.clips[MotionState.STOP].event_start_sec
                * backend.fps
            )
        )
        self.assertGreaterEqual(backend._blend.target_index, event_start)
        target_speed = backend.clips[MotionState.STOP].planar_speed()[
            backend._blend.target_index
        ]
        self.assertLess(abs(target_speed - backend._current_speed()), 0.12)

    @unittest.skipUnless(MANIFEST.exists(), "generated SMPL motion library absent")
    def test_walk_action_waits_for_safe_gait_phase(self):
        backend = SmplUsdAvatarBackend(self.MANIFEST, self.ASSET_ROOT)
        for _ in range(500):
            backend.set_command(0.95, 0.0)
            backend.update()

        backend._walk_phase = np.floor(backend._walk_phase) + 0.12
        backend.set_command(0.0, 0.0)
        frame = backend.update()
        self.assertIsNone(backend._blend)
        self.assertEqual(frame.pending_action, MotionState.STOP.value)

        for _ in range(30):
            backend.update()
            if backend._blend is not None:
                break
        self.assertIsNotNone(backend._blend)
        self.assertIs(backend._blend.target_state, MotionState.STOP)

    @unittest.skipUnless(MANIFEST.exists(), "generated SMPL motion library absent")
    def test_turn_request_remains_latched_until_safe_phase(self):
        backend = SmplUsdAvatarBackend(self.MANIFEST, self.ASSET_ROOT)
        for _ in range(500):
            backend.set_command(0.95, 0.0)
            backend.update()

        backend._walk_phase = np.floor(backend._walk_phase) + 0.12
        backend.set_command(0.95, -0.8)
        backend.update()
        backend.set_command(0.95, 0.0)
        self.assertIs(backend._pending_action, MotionState.TURN_RIGHT)

        for _ in range(30):
            backend.update()
            if backend._blend is not None:
                break
        self.assertIsNotNone(backend._blend)
        self.assertIs(backend._blend.target_state, MotionState.TURN_RIGHT)

    @unittest.skipUnless(MANIFEST.exists(), "generated SMPL motion library absent")
    def test_resume_consumes_full_83_37_start(self):
        backend = SmplUsdAvatarBackend(self.MANIFEST, self.ASSET_ROOT)
        backend.set_command(0.95, 0.0)
        backend.update()
        self.assertIsNotNone(backend._blend)
        self.assertEqual(backend._blend.target_index, 0)
        self.assertEqual(backend._blend.frames, backend.resume_transition_frames)
        self.assertGreater(backend.resume_transition_frames, backend.transition_frames)

        rotations = [backend.frame.rotations]
        for _ in range(backend.resume_transition_frames):
            rotations.append(backend.update().rotations)
        frame_steps = [
            float(np.sqrt(np.mean((following - previous) ** 2)))
            for previous, following in zip(rotations[:-1], rotations[1:])
        ]
        self.assertLess(max(frame_steps), 0.03)

    @unittest.skipUnless(MANIFEST.exists(), "generated SMPL motion library absent")
    def test_resume_matches_walk_phase_in_full_cycle(self):
        backend = SmplUsdAvatarBackend(self.MANIFEST, self.ASSET_ROOT)
        backend.set_command(0.75, 0.0)
        walk_blend = None
        for _ in range(400):
            backend.update()
            if (
                backend._blend is not None
                and backend._blend.target_state is MotionState.WALK
            ):
                walk_blend = backend._blend
                break

        self.assertIsNotNone(walk_blend)
        self.assertGreater(walk_blend.target_index, 30)
        target = backend._rotations[MotionState.WALK][walk_blend.target_index]
        phase_zero = backend._rotations[MotionState.WALK][0]
        matched_error = np.sqrt(
            np.mean((backend.current_rotations[1:22] - target[1:22]) ** 2)
        )
        phase_zero_error = np.sqrt(
            np.mean((backend.current_rotations[1:22] - phase_zero[1:22]) ** 2)
        )
        self.assertLess(matched_error, phase_zero_error)
        self.assertIsNone(walk_blend.support_joint)

        positions = [backend.root_position.copy()]
        for _ in range(walk_blend.frames + 2):
            positions.append(backend.update().root_position.copy())
        speeds = np.linalg.norm(
            np.diff(np.asarray(positions)[:, :2], axis=0), axis=1
        ) * backend.fps
        self.assertLess(float(speeds.max()), 1.35)

    @unittest.skipUnless(MANIFEST.exists(), "generated SMPL motion library absent")
    def test_native_left_turn_changes_yaw_counterclockwise(self):
        backend = SmplUsdAvatarBackend(self.MANIFEST, self.ASSET_ROOT)
        yaw_before = None
        yaw_after = None
        for frame_index in range(900):
            yaw_rate = 0.8 if 300 <= frame_index < 330 else 0.0
            backend.set_command(0.95, yaw_rate)
            frame = backend.update()
            if frame_index == 299:
                yaw_before = frame.yaw_rad
            if (
                yaw_after is None
                and frame_index > 330
                and frame.state is MotionState.WALK
            ):
                yaw_after = frame.yaw_rad
        self.assertIsNotNone(yaw_before)
        self.assertIsNotNone(yaw_after)
        self.assertGreater(yaw_after - yaw_before, 1.2)

    @unittest.skipUnless(MANIFEST.exists(), "generated SMPL motion library absent")
    def test_locomotion_transitions_do_not_freeze_root(self):
        backend = SmplUsdAvatarBackend(self.MANIFEST, self.ASSET_ROOT)
        positions = []
        states = []
        for frame_index in range(720):
            yaw_rate = -0.8 if 300 <= frame_index < 330 else 0.0
            backend.set_command(0.95, yaw_rate)
            frame = backend.update()
            positions.append(frame.root_position.copy())
            states.append(frame.state)

        positions = np.asarray(positions)
        speed = np.linalg.norm(
            np.diff(positions[:, :2], axis=0), axis=1
        ) * backend.fps
        transition_steps = [
            speed[index]
            for index in range(250, len(speed))
            if states[index] is not states[index + 1]
            and states[index]
            in (MotionState.WALK, MotionState.TURN_RIGHT)
            and states[index + 1]
            in (MotionState.WALK, MotionState.TURN_RIGHT)
        ]
        self.assertTrue(transition_steps)
        self.assertGreater(min(transition_steps), 0.15)
        moving_steps = [
            speed[index]
            for index in range(250, len(speed))
            if states[index]
            in (MotionState.WALK, MotionState.TURN_RIGHT)
            and states[index + 1]
            in (MotionState.WALK, MotionState.TURN_RIGHT)
        ]
        self.assertGreater(min(moving_steps), 0.005)

    @unittest.skipUnless(MANIFEST.exists(), "generated SMPL motion library absent")
    def test_phase_gait_tracks_slow_normal_and_fast_commands(self):
        backend = SmplUsdAvatarBackend(self.MANIFEST, self.ASSET_ROOT)
        measured = []
        for target in (0.75, 0.95, 1.20):
            positions = []
            for _ in range(600):
                backend.set_command(target, 0.0)
                positions.append(backend.update().root_position.copy())
            speed = np.linalg.norm(
                np.diff(np.asarray(positions)[-121:, :2], axis=0), axis=1
            ) * backend.fps
            measured.append(float(np.median(speed)))
            self.assertAlmostEqual(measured[-1], target, delta=0.05)
        self.assertLess(measured[0], measured[1])
        self.assertLess(measured[1], measured[2])

    @unittest.skipUnless(MANIFEST.exists(), "generated SMPL motion library absent")
    def test_phase_gait_speed_command_is_acceleration_limited(self):
        backend = SmplUsdAvatarBackend(self.MANIFEST, self.ASSET_ROOT)
        for _ in range(500):
            backend.set_command(0.75, 0.0)
            backend.update()
        before = backend._animated_speed_mps
        backend.set_command(1.20, 0.0)
        backend.update()
        self.assertLessEqual(
            backend._animated_speed_mps - before,
            backend.max_speed_accel_mps2 / backend.fps + 1.0e-9,
        )

    @unittest.skipUnless(MANIFEST.exists(), "generated SMPL motion library absent")
    def test_phase_gait_cycle_pose_and_velocity_are_closed(self):
        backend = SmplUsdAvatarBackend(self.MANIFEST, self.ASSET_ROOT)
        for profile in range(3):
            start_pose, start_position, _, _ = backend._sample_walk_profile(
                profile, 0.0
            )
            end_pose, end_position, _, _ = backend._sample_walk_profile(
                profile, 1.0 - 1.0e-6
            )
            next_pose, next_position, _, _ = backend._sample_walk_profile(
                profile, 1.0
            )
            self.assertLess(
                float(np.sqrt(np.mean((end_pose - next_pose) ** 2))),
                1.0e-4,
            )
            before = next_position - end_position
            _, after_position, _, _ = backend._sample_walk_profile(
                profile, 1.0 + 1.0e-6
            )
            after = after_position - next_position
            self.assertLess(np.linalg.norm(before - after), 1.0e-4)
            self.assertTrue(np.isfinite(start_pose).all())
            self.assertTrue(np.isfinite(start_position).all())


if __name__ == "__main__":
    unittest.main()
