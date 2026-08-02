"""Calibrate Isaac People Walk blend against measured root-motion speed."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import time
from typing import Iterable, Sequence

from isaacsim_msgs.msg import NavPed, Person
from isaacsim_msgs.srv import MovePed, Pedestrian
from people_msgs.msg import People
import rclpy
from rclpy.node import Node

from pedestrian.simulator.logic.people.external_motion import animation_walk_blend


@dataclass(frozen=True)
class SpeedTrial:
    commanded_speed_mps: float
    walk_blend: float
    measured_speed_mps: float
    sample_count: int
    position_fit_rmse_m: float
    path_speed_mps: float = 0.0
    lateral_speed_mps: float = 0.0


@dataclass(frozen=True)
class PowerLawFit:
    full_speed_mps: float
    speed_exponent: float
    rmse_mps: float
    max_abs_error_mps: float


def parse_speed_levels(value: str) -> tuple[float, ...]:
    levels = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(levels) < 2 or any(not math.isfinite(item) or item <= 0.0 for item in levels):
        raise argparse.ArgumentTypeError("speeds must contain at least two positive values")
    if tuple(sorted(set(levels))) != levels:
        raise argparse.ArgumentTypeError("speeds must be strictly increasing and unique")
    return levels


def summarize_speed_samples(samples: Iterable[float]) -> tuple[float, float, int]:
    clean = sorted(float(value) for value in samples if math.isfinite(value) and value >= 0.0)
    if not clean:
        raise ValueError("no valid root-speed samples")
    # Trim callback jitter without hiding a systematic speed error.
    trim = int(len(clean) * 0.1) if len(clean) >= 10 else 0
    retained = clean[trim : len(clean) - trim] if trim else clean
    return (
        statistics.median(retained),
        statistics.pstdev(retained) if len(retained) > 1 else 0.0,
        len(retained),
    )


def estimate_projected_root_speed(
    samples: Sequence[tuple[float, float, float]],
    direction_xy: Sequence[float],
) -> tuple[float, float, int]:
    """Estimate cycle-average speed from a position-vs-time slope."""
    if len(samples) < 3:
        raise ValueError("at least three root-position samples are required")
    direction_norm = math.hypot(float(direction_xy[0]), float(direction_xy[1]))
    if direction_norm <= 1e-9:
        raise ValueError("calibration direction must be non-zero")
    direction = (
        float(direction_xy[0]) / direction_norm,
        float(direction_xy[1]) / direction_norm,
    )
    ordered = sorted(
        (float(t), float(x), float(y)) for t, x, y in samples
        if all(math.isfinite(value) for value in (t, x, y))
    )
    accepted = [ordered[0]]
    for sample in ordered[1:]:
        previous = accepted[-1]
        dt = sample[0] - previous[0]
        if dt <= 0.0:
            continue
        if math.hypot(sample[1] - previous[1], sample[2] - previous[2]) / dt <= 2.5:
            accepted.append(sample)
    if len(accepted) < 3:
        raise ValueError("too few root-position samples remain after outlier rejection")
    times = [sample[0] - accepted[0][0] for sample in accepted]
    projected = [sample[1] * direction[0] + sample[2] * direction[1] for sample in accepted]
    time_mean = statistics.fmean(times)
    position_mean = statistics.fmean(projected)
    denominator = sum((value - time_mean) ** 2 for value in times)
    if denominator <= 1e-12:
        raise ValueError("root-position samples do not span time")
    slope = sum(
        (time_value - time_mean) * (position_value - position_mean)
        for time_value, position_value in zip(times, projected)
    ) / denominator
    intercept = position_mean - slope * time_mean
    residuals = [
        position_value - (intercept + slope * time_value)
        for time_value, position_value in zip(times, projected)
    ]
    return (
        max(0.0, slope),
        math.sqrt(statistics.fmean(value * value for value in residuals)),
        len(accepted),
    )


def root_motion_diagnostics(
    samples: Sequence[tuple[float, float, float]],
    direction_xy: Sequence[float],
) -> tuple[float, float]:
    if len(samples) < 2:
        raise ValueError("at least two root-position samples are required")
    norm = math.hypot(float(direction_xy[0]), float(direction_xy[1]))
    if norm <= 1e-9:
        raise ValueError("calibration direction must be non-zero")
    direction = (float(direction_xy[0]) / norm, float(direction_xy[1]) / norm)
    lateral = (-direction[1], direction[0])
    ordered = sorted((float(t), float(x), float(y)) for t, x, y in samples)
    duration = ordered[-1][0] - ordered[0][0]
    if duration <= 1e-9:
        raise ValueError("root-position samples do not span time")
    path_length = sum(
        math.hypot(current[1] - previous[1], current[2] - previous[2])
        for previous, current in zip(ordered, ordered[1:])
    )
    lateral_displacement = (
        (ordered[-1][1] - ordered[0][1]) * lateral[0]
        + (ordered[-1][2] - ordered[0][2]) * lateral[1]
    )
    return path_length / duration, abs(lateral_displacement) / duration


def fit_walk_blend_power_law(trials: Sequence[SpeedTrial]) -> PowerLawFit:
    usable = [
        trial
        for trial in trials
        if 0.0 < trial.walk_blend < 1.0 and trial.measured_speed_mps > 0.0
    ]
    if len(usable) < 2:
        raise ValueError("at least two non-saturated positive trials are required")
    xs = [math.log(trial.walk_blend) for trial in usable]
    ys = [math.log(trial.measured_speed_mps) for trial in usable]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    if denominator <= 1e-12:
        raise ValueError("walk blend samples do not span a usable range")
    exponent = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(xs, ys)
    ) / denominator
    exponent = max(1e-3, exponent)
    full_speed = math.exp(y_mean - exponent * x_mean)
    errors = [
        full_speed * trial.walk_blend**exponent - trial.measured_speed_mps
        for trial in usable
    ]
    return PowerLawFit(
        full_speed_mps=full_speed,
        speed_exponent=exponent,
        rmse_mps=math.sqrt(statistics.fmean(error * error for error in errors)),
        max_abs_error_mps=max(abs(error) for error in errors),
    )


class PedestrianSpeedCalibration(Node):
    def __init__(self, agent_id: str):
        super().__init__("pedestrian_speed_calibration")
        self._agent_id = str(agent_id)
        self._spawn = self.create_client(Pedestrian, "/isaac/spawn_pedestrian")
        self._move = self.create_client(MovePed, "/isaac/move_pedestrians")
        self._samples: list[tuple[float, float, float]] = []
        self._latest_pose: tuple[float, float, float] | None = None
        self.create_subscription(People, "/isaac/pedestrian_states", self._on_people, 20)

    def wait_for_bridge(self, timeout_sec: float) -> None:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if self._spawn.wait_for_service(timeout_sec=0.25) and self._move.wait_for_service(
                timeout_sec=0.25
            ):
                return
        raise RuntimeError("Isaac pedestrian services did not become available")

    def spawn(self, character: str, pose: Sequence[float], yaw: float) -> None:
        person = Person()
        person.stage_prefix = self._agent_id
        person.character_name = str(character)
        person.initial_pose = [float(value) for value in pose]
        person.goal_pose = person.initial_pose
        person.loop_path = False
        person.orientation = float(yaw)
        person.controller_stats = False
        person.velocity = 0.0
        request = Pedestrian.Request()
        request.people = [person]
        response = self._call(self._spawn, request, timeout_sec=30.0)
        if response is None or not response.ret:
            raise RuntimeError("Isaac rejected calibration pedestrian spawn")

    def direct_pose(self, pose: Sequence[float], yaw: float) -> None:
        command = NavPed()
        command.path = self._agent_id
        command.goal_pose = [float(value) for value in pose]
        command.use_direct_pose = True
        command.direct_pose = command.goal_pose
        command.orientation = float(yaw)
        command.stop = True
        self._send_move(command)

    def start_external_motion(
        self,
        pose: Sequence[float],
        velocity: Sequence[float],
        yaw: float,
        timeout_sec: float,
    ) -> None:
        command = NavPed()
        command.path = self._agent_id
        command.goal_pose = [float(value) for value in pose]
        command.direct_pose = command.goal_pose
        command.use_external_motion = True
        command.external_velocity = [float(value) for value in velocity]
        command.external_timeout_sec = float(timeout_sec)
        command.external_motion_mode = NavPed.EXTERNAL_MOTION_LOCOMOTION
        command.orientation = float(yaw)
        self._send_move(command)

    def drive_and_collect_root_poses(
        self,
        velocity: Sequence[float],
        yaw: float,
        warmup_sec: float,
        sample_sec: float,
        command_hz: float,
    ) -> list[tuple[float, float, float]]:
        self._samples.clear()
        started = time.monotonic()
        deadline = started + warmup_sec + sample_sec
        period = 1.0 / max(1.0, float(command_hz))
        next_command_at = started
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_command_at and self._latest_pose is not None:
                # Keep tracking error near zero so this measures the Walk blend,
                # not the reference catch-up controller.
                self.start_external_motion(
                    self._latest_pose,
                    velocity,
                    yaw,
                    max(0.5, 4.0 * period),
                )
                next_command_at = now + period
            rclpy.spin_once(self, timeout_sec=min(0.02, max(0.0, next_command_at - now)))
        return [sample for sample in self._samples if sample[0] >= started + warmup_sec]

    def settle(self, duration_sec: float) -> None:
        deadline = time.monotonic() + max(0.0, duration_sec)
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _on_people(self, message: People) -> None:
        for person in message.people:
            if str(person.name) == self._agent_id:
                self._latest_pose = (
                    float(person.position.x),
                    float(person.position.y),
                    float(person.position.z),
                )
                self._samples.append(
                    (time.monotonic(), self._latest_pose[0], self._latest_pose[1])
                )
                return

    def _send_move(self, command: NavPed) -> None:
        request = MovePed.Request()
        request.nav_list = [command]
        response = self._call(self._move, request, timeout_sec=10.0)
        if response is None or not response.ret:
            raise RuntimeError("Isaac rejected calibration pedestrian motion")

    def _call(self, client, request, timeout_sec: float):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        if not future.done() or future.exception() is not None:
            raise RuntimeError(f"pedestrian service call failed: {future.exception()}")
        return future.result()


def _vector3(values: Sequence[str]) -> tuple[float, float, float]:
    parsed = tuple(float(value) for value in values)
    if len(parsed) != 3 or not all(math.isfinite(value) for value in parsed):
        raise argparse.ArgumentTypeError("pose must contain three finite values")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-id", default="walk_speed_calibration_agent")
    parser.add_argument("--character", default="original_female_adult_business_02")
    parser.add_argument("--start-pose", nargs=3, default=("-4.4", "-2.0", "0.0"))
    parser.add_argument("--heading-rad", type=float, default=math.pi)
    parser.add_argument("--speeds", type=parse_speed_levels, default=parse_speed_levels("0.15,0.25,0.35,0.45,0.60,0.75"))
    parser.add_argument("--warmup-sec", type=float, default=2.0)
    parser.add_argument("--sample-sec", type=float, default=4.0)
    parser.add_argument("--settle-sec", type=float, default=0.75)
    parser.add_argument("--command-hz", type=float, default=10.0)
    parser.add_argument("--current-full-speed-mps", type=float, default=0.8)
    parser.add_argument("--current-speed-exponent", type=float, default=3.0)
    parser.add_argument("--output", default="/tmp/pedestrian_walk_speed_calibration.json")
    parser.add_argument("--service-timeout-sec", type=float, default=30.0)
    return parser


def main(args=None) -> None:
    parser = build_parser()
    parsed, ros_args = parser.parse_known_args(args)
    start = _vector3(parsed.start_pose)
    if parsed.warmup_sec < 0.0 or parsed.sample_sec <= 0.0:
        parser.error("warmup-sec must be non-negative and sample-sec must be positive")
    rclpy.init(args=ros_args)
    node = PedestrianSpeedCalibration(parsed.agent_id)
    trials: list[SpeedTrial] = []
    direction = (math.cos(parsed.heading_rad), math.sin(parsed.heading_rad))
    try:
        node.wait_for_bridge(parsed.service_timeout_sec)
        node.spawn(parsed.character, start, parsed.heading_rad)
        node.settle(1.0)
        for speed in parsed.speeds:
            node.direct_pose(start, parsed.heading_rad)
            node.settle(parsed.settle_sec)
            velocity = (speed * direction[0], speed * direction[1], 0.0)
            root_poses = node.drive_and_collect_root_poses(
                velocity,
                parsed.heading_rad,
                parsed.warmup_sec,
                parsed.sample_sec,
                parsed.command_hz,
            )
            measured, fit_rmse, count = estimate_projected_root_speed(root_poses, direction)
            path_speed, lateral_speed = root_motion_diagnostics(root_poses, direction)
            blend = animation_walk_blend(
                speed,
                full_speed_mps=parsed.current_full_speed_mps,
                speed_exponent=parsed.current_speed_exponent,
            )
            trial = SpeedTrial(
                speed,
                blend,
                measured,
                count,
                fit_rmse,
                path_speed,
                lateral_speed,
            )
            trials.append(trial)
            node.get_logger().info(
                f"Walk calibration: command={speed:.3f} m/s, blend={blend:.4f}, "
                f"root={measured:.3f} m/s, samples={count}, "
                f"path_speed={path_speed:.3f} m/s, lateral_speed={lateral_speed:.3f} m/s, "
                f"position_fit_rmse={fit_rmse:.3f} m"
            )
        fit = fit_walk_blend_power_law(trials)
        payload = {
            "schema_version": 1,
            "current_mapping": {
                "full_speed_mps": float(parsed.current_full_speed_mps),
                "speed_exponent": float(parsed.current_speed_exponent),
            },
            "recommended_mapping": asdict(fit),
            "trials": [asdict(trial) for trial in trials],
        }
        output = Path(parsed.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    finally:
        try:
            node.direct_pose((1000.0, 1000.0, 0.0), 0.0)
        except Exception as exc:
            node.get_logger().warning(f"Failed to park calibration pedestrian: {exc}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
