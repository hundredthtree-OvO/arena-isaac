#!/usr/bin/env python3
"""Run a live single-SMPL-avatar command mirror in an empty Isaac stage."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_ASSET_ROOT = Path("/home/stardust/resources/arena_ws/arena_assets/smpl")


def _command_at(time_sec: float) -> tuple[float, float, str]:
    if time_sec < 1.0:
        return 0.0, 0.0, "initial idle"
    if time_sec < 5.0:
        return 0.95, 0.0, "resume and walk"
    if time_sec < 5.35:
        return 0.95, -0.8, "right-turn trigger"
    if time_sec < 13.0:
        return 0.95, 0.0, "finish turn and walk"
    if time_sec < 17.0:
        return 0.0, 0.0, "natural stop and hold"
    if time_sec < 22.0:
        return 0.95, 0.0, "resume walking"
    return 0.0, 0.0, "final stop"


def _speed_command_at(time_sec: float) -> tuple[float, float, str]:
    if time_sec < 1.0:
        return 0.0, 0.0, "initial idle"
    if time_sec < 6.0:
        return 0.75, 0.0, "slow walk"
    if time_sec < 11.0:
        return 0.95, 0.0, "normal walk"
    if time_sec < 16.0:
        return 1.20, 0.0, "fast walk"
    if time_sec < 21.0:
        return 0.75, 0.0, "decelerate to slow walk"
    return 0.0, 0.0, "natural stop"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_ASSET_ROOT
        / "derived/manifests/cmu_smplh_locomotion_seed.json",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--exit-after-sec", type=float, default=0.0)
    parser.add_argument(
        "--follow-camera",
        action="store_true",
        help="continuously follow the avatar instead of allowing free navigation",
    )
    parser.add_argument(
        "--scenario",
        choices=("motion-graph", "speed-ramp", "interactive"),
        default="motion-graph",
    )
    args = parser.parse_args()
    if args.headless and args.scenario == "interactive":
        parser.error("interactive scenario requires the Isaac GUI")

    from isaacsim import SimulationApp

    isaac_root = Path(
        os.environ.get(
            "ISAAC_ROOT", "/home/stardust/resources/isaac-sim-4.5.0"
        )
    )
    experience = isaac_root / "apps" / (
        "isaacsim.exp.base.python.kit"
        if args.headless
        else "isaacsim.exp.base.kit"
    )
    simulation_app = SimulationApp(
        {"headless": args.headless, "width": 1280, "height": 720},
        experience=str(experience),
    )
    import omni.timeline
    import omni.usd
    from pxr import Gf, UsdGeom, UsdLux

    from tools.smpl_pipeline.runtime.avatar_backend import SmplUsdAvatarBackend
    from tools.smpl_pipeline.runtime.isaac_avatar_adapter import (
        IsaacSmplAvatarAdapter,
    )
    from tools.smpl_pipeline.runtime.interactive_commands import (
        InteractiveCommandController,
    )

    context = omni.usd.get_context()
    context.new_stage()
    simulation_app.update()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.CreateSizeAttr(1.0)
    ground.AddScaleOp().Set(Gf.Vec3d(20.0, 20.0, 0.02))
    ground.AddTranslateOp().Set(Gf.Vec3d(4.0, -2.0, -0.011))
    ground.CreateDisplayColorAttr([Gf.Vec3f(0.18, 0.20, 0.22)])
    light = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    light.CreateIntensityAttr(1800.0)
    light.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 25.0, -35.0))

    camera = UsdGeom.Camera.Define(stage, "/World/FollowCamera")
    camera.CreateFocalLengthAttr(32.0)
    camera_transform = camera.AddTransformOp()
    try:
        from omni.kit.viewport.utility import get_active_viewport

        get_active_viewport().set_active_camera(camera.GetPath().pathString)
    except Exception as error:
        print(f"[SMPL mirror] camera activation skipped: {error}")

    backend = SmplUsdAvatarBackend(
        args.manifest.expanduser().resolve(),
        args.asset_root.expanduser().resolve(),
    )
    adapter = IsaacSmplAvatarAdapter(stage, backend)
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    input_interface = None
    keyboard = None
    keyboard_subscription = None
    interactive = None
    if args.scenario == "interactive":
        import carb.input
        import omni.appwindow

        interactive = InteractiveCommandController()
        input_interface = carb.input.acquire_input_interface()
        keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        key_actions = {
            carb.input.KeyboardInput.I: "move",
            carb.input.KeyboardInput.K: "stop",
            carb.input.KeyboardInput.SPACE: "stop",
            carb.input.KeyboardInput.J: "turn_left",
            carb.input.KeyboardInput.L: "turn_right",
            carb.input.KeyboardInput.KEY_1: "slow",
            carb.input.KeyboardInput.KEY_2: "normal",
            carb.input.KeyboardInput.KEY_3: "fast",
        }

        def on_keyboard(event, *_) -> bool:
            action = key_actions.get(event.input)
            if action is None:
                return True
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                interactive.handle(action, True)
            elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                interactive.handle(action, False)
            return True

        keyboard_subscription = input_interface.subscribe_to_keyboard_events(
            keyboard, on_keyboard
        )
        print(
            "[SMPL mirror] interactive controls: I=move/resume, "
            "K/Space=stop, J/L=turn left/right, 1/2/3=slow/normal/fast"
        )

    frame_index = 0
    last_label = ""
    last_state = ""
    camera_initialized = False
    try:
        while simulation_app.is_running():
            time_sec = frame_index / backend.fps
            if interactive is not None:
                speed, yaw_rate, label = interactive.command()
            else:
                command_source = (
                    _speed_command_at
                    if args.scenario == "speed-ramp"
                    else _command_at
                )
                speed, yaw_rate, label = command_source(time_sec)
            backend.set_command(speed, yaw_rate)
            frame = backend.update()
            adapter.update(frame)
            if label != last_label:
                print(
                    "[SMPL mirror] "
                    f"t={time_sec:.2f}s command={label}, "
                    f"state={frame.state.value}"
                )
                last_label = label
            if frame.state.value != last_state:
                print(
                    "[SMPL mirror] "
                    f"t={time_sec:.2f}s state_transition={last_state or 'none'}"
                    f"->{frame.state.value}, clip={frame.motion_id}, "
                    f"clip_index={frame.clip_index}"
                )
                last_state = frame.state.value

            if args.follow_camera or not camera_initialized:
                target = frame.root_position + [0.0, 0.0, 1.0]
                eye = frame.root_position + [-5.5, -5.5, 3.5]
                camera_matrix = Gf.Matrix4d().SetLookAt(
                    Gf.Vec3d(*eye.tolist()),
                    Gf.Vec3d(*target.tolist()),
                    Gf.Vec3d(0.0, 0.0, 1.0),
                ).GetInverse()
                camera_transform.Set(camera_matrix)
                camera_initialized = True

            simulation_app.update()
            frame_index += 1
            if args.exit_after_sec > 0.0 and time_sec >= args.exit_after_sec:
                break
    finally:
        if (
            input_interface is not None
            and keyboard is not None
            and keyboard_subscription is not None
        ):
            input_interface.unsubscribe_to_keyboard_events(
                keyboard, keyboard_subscription
            )
        timeline.stop()
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
