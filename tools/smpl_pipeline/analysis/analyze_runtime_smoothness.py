#!/usr/bin/env python3
"""Render dependency-free CSV/SVG diagnostics for the avatar command mirror."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from ..runtime.avatar_backend import SmplUsdAvatarBackend
from ..runtime.mirror_single_avatar import (
    DEFAULT_ASSET_ROOT,
    _command_at,
    _speed_command_at,
)


def _series_path(values: np.ndarray, left: float, top: float, width: float, height: float) -> str:
    finite = values[np.isfinite(values)]
    maximum = max(float(np.percentile(finite, 99)) if len(finite) else 1.0, 1.0e-6)
    points = []
    for index, value in enumerate(values):
        x = left + width * index / max(1, len(values) - 1)
        y = top + height * (1.0 - min(abs(float(value)) / maximum, 1.0))
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_ASSET_ROOT / "derived/manifests/cmu_smplh_locomotion_seed.json",
    )
    parser.add_argument("--duration-sec", type=float, default=24.0)
    parser.add_argument(
        "--scenario",
        choices=("motion-graph", "speed-ramp"),
        default="motion-graph",
    )
    parser.add_argument("--output-prefix", type=Path, default=Path("/tmp/smpl_runtime_smoothness"))
    args = parser.parse_args()

    backend = SmplUsdAvatarBackend(args.manifest, args.asset_root)
    count = int(round(args.duration_sec * backend.fps)) + 1
    positions = []
    states = []
    clips = []
    indices = []
    labels = []
    commanded_speeds = []
    animated_speeds = []
    gait_phases = []
    pending_actions = []
    for frame_index in range(count):
        command_source = (
            _speed_command_at if args.scenario == "speed-ramp" else _command_at
        )
        speed_command, yaw_rate, label = command_source(
            frame_index / backend.fps
        )
        backend.set_command(speed_command, yaw_rate)
        frame = backend.update()
        positions.append(frame.root_position.copy())
        states.append(frame.state.value)
        clips.append(frame.motion_id)
        indices.append(frame.clip_index)
        labels.append(label)
        commanded_speeds.append(frame.commanded_speed_mps)
        animated_speeds.append(frame.animated_speed_mps)
        gait_phases.append(frame.gait_phase)
        pending_actions.append(frame.pending_action or "")

    positions = np.asarray(positions)
    dt = 1.0 / backend.fps
    speed = np.r_[0.0, np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1) / dt]
    acceleration = np.gradient(speed, dt)
    jerk = np.gradient(acceleration, dt)
    commanded_speeds = np.asarray(commanded_speeds)
    animated_speeds = np.asarray(animated_speeds)
    gait_phases = np.asarray(gait_phases)
    # A full-cycle moving average separates intended travel speed from the
    # normal within-step pelvis acceleration visible in instantaneous speed.
    cycle_window = max(3, int(round(float(np.median(backend.walk_profile_durations)) * backend.fps)))
    kernel = np.ones(cycle_window, dtype=np.float64) / cycle_window
    cycle_speed = np.convolve(speed, kernel, mode="same")
    times = np.arange(count) * dt
    prefix = args.output_prefix.expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)

    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_sec", "state", "motion_id", "clip_index", "commanded_speed_mps", "animated_speed_mps", "root_speed_mps", "cycle_average_speed_mps", "gait_phase", "pending_action", "acceleration_mps2", "jerk_mps3", "command"))
        writer.writerows(zip(times, states, clips, indices, commanded_speeds, animated_speeds, speed, cycle_speed, gait_phases, pending_actions, acceleration, jerk, labels))

    width, height = 1280, 760
    left, plot_width, panel_height = 85, 1160, 175
    colors = ("#167d9a", "#d97706", "#b42318")
    series = (("root speed (m/s)", speed), ("cycle-average speed (m/s)", cycle_speed), ("|jerk| (m/s3)", np.abs(jerk)))
    state_colors = {"idle": "#d8dee9", "resume": "#a7d8bd", "walk": "#8ecae6", "turn_right": "#f4c95d", "turn_left": "#d4a95d", "stop": "#ef9a9a"}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="#fbfaf6"/>', '<text x="85" y="34" font-family="sans-serif" font-size="22" font-weight="bold">SMPL runtime smoothness</text>']
    transitions = [0] + [i for i in range(1, count) if states[i] != states[i - 1]] + [count - 1]
    for panel, ((title, values), color) in enumerate(zip(series, colors)):
        top = 65 + panel * 220
        for start, end in zip(transitions[:-1], transitions[1:]):
            x = left + plot_width * start / (count - 1)
            w = plot_width * (end - start) / (count - 1)
            parts.append(f'<rect x="{x:.1f}" y="{top}" width="{w:.1f}" height="{panel_height}" fill="{state_colors[states[start]]}" opacity="0.22"/>')
        parts.extend((f'<rect x="{left}" y="{top}" width="{plot_width}" height="{panel_height}" fill="none" stroke="#7b7b73"/>', f'<text x="{left}" y="{top - 9}" font-family="sans-serif" font-size="15">{title}</text>', f'<polyline points="{_series_path(values, left, top, plot_width, panel_height)}" fill="none" stroke="{color}" stroke-width="2"/>'))
    for second in range(0, int(args.duration_sec) + 1):
        x = left + plot_width * second / args.duration_sec
        parts.append(f'<text x="{x:.1f}" y="735" text-anchor="middle" font-family="sans-serif" font-size="11">{second}</text>')
    parts.append('<text x="665" y="754" text-anchor="middle" font-family="sans-serif" font-size="13">time (s); background color denotes motion state</text></svg>')
    prefix.with_suffix(".svg").write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {prefix.with_suffix('.csv')}")
    print(f"Wrote {prefix.with_suffix('.svg')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
