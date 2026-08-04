"""Mode policies for the mecanum controller.

The Isaac-facing implementation remains on ``MecanumRobot`` for now. These
small strategies make mode ownership explicit without duplicating ROS or
articulation lifecycle code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ros2isaacsim.physx_diff_contact import wheel_slip_diagnostics


@dataclass(frozen=True)
class MotionBackend:
    mode: str
    uses_collision_guard: bool
    applies_navigation_hold: bool
    uses_measured_odom_twist: bool

    def apply(
        self,
        robot: Any,
        vx: float,
        vy: float,
        wz: float,
        dt: float,
        drive_mode: str,
    ) -> None:
        raise NotImplementedError


class PhysxWheelsBackend(MotionBackend):
    def __init__(self) -> None:
        super().__init__(
            mode="physx_wheels",
            uses_collision_guard=True,
            applies_navigation_hold=True,
            uses_measured_odom_twist=False,
        )

    def apply(self, robot: Any, vx: float, vy: float, wz: float, dt: float, drive_mode: str) -> None:
        pos, quat, heading = robot._current_base_pose_for_motion()
        vx, vy, wz, guard_mode = robot._filter_with_collision_guard(
            pos, heading, vx, vy, wz, dt, self.mode
        )
        robot._applied_vx, robot._applied_vy, robot._applied_wz = float(vx), float(vy), float(wz)
        robot._last_guard_mode = guard_mode
        robot._publish_applied_cmd_vel()

        wheel_speeds = robot._wheel_speeds(vx, vy, wz, drive_mode)
        robot._apply_navigation_hold_targets()
        robot._apply_joint_velocity_targets(wheel_speeds)
        wrote = robot._apply_motion_based_root(pos, quat, vx, vy, wz, dt)
        if not wrote and robot.config.physx_fallback_kinematic:
            robot._apply_kinematic_base(vx, vy, wz, dt)
            return
        robot._publish_actual_odom_tf()


class PhysxDiffContactBackend(MotionBackend):
    def __init__(self) -> None:
        super().__init__(
            mode="physx_diff_contact",
            uses_collision_guard=False,
            applies_navigation_hold=False,
            uses_measured_odom_twist=True,
        )

    def apply(self, robot: Any, vx: float, vy: float, wz: float, dt: float, drive_mode: str) -> None:
        del vy, dt, drive_mode
        if robot._articulation is None or robot._joint_indices is None:
            return
        # Pausing the timeline to rebuild a pooled pedestrian's AnimGraph can
        # invalidate every PhysX tensor view, including the robot articulation.
        # Recover it before wheel targets are sent; otherwise target writes and
        # tire-force feedback silently remain unavailable for the whole episode.
        robot._ensure_diff_articulation_force_view()
        robot._publish_actual_odom_tf()
        robot._applied_vx = float(vx)
        robot._applied_vy = 0.0
        robot._applied_wz = float(wz)
        robot._publish_applied_cmd_vel()
        wheel_targets = robot._physx_diff_drive.apply(vx, wz)
        actual = robot._read_joint_velocities(robot._joint_indices)
        contacts = robot._current_physx_diff_contacts()
        tire_wrench = robot._apply_separated_tire_forces(actual, contacts)
        self._log_diagnostics(robot, vx, wz, wheel_targets, actual, contacts, tire_wrench)

    @staticmethod
    def _log_diagnostics(robot: Any, vx: float, wz: float, wheel_targets, actual, contacts, tire_wrench) -> None:
        import time

        now = time.monotonic()
        if (abs(vx) <= 1.0e-4 and abs(wz) <= 1.0e-4) or now - robot._last_wheel_target_log_t < 1.0:
            return
        robot._last_wheel_target_log_t = now
        actual_text = "unavailable" if actual is None else np.asarray(actual).round(3).tolist()
        slip = None
        if actual is not None and np.asarray(actual).size == 4:
            slip = wheel_slip_diagnostics(
                joint_velocities=np.asarray(actual).reshape(-1),
                wheel_radius=robot.config.wheel_radius,
                track_width=robot.config.differential_track_width,
                wheel_signs=robot.config.wheel_signs,
                measured_vx=robot._measured_vx,
                measured_vy=robot._measured_vy,
                measured_wz=robot._measured_wz,
                effective_wz=robot._physx_diff_drive.last_effective_wz,
            )
        contact_flags = [sample.get("contact") for sample in contacts]
        slip_text = "unavailable" if slip is None else np.asarray(slip["slip_ratios"]).round(3).tolist()
        yaw_efficiency = 0.0 if slip is None else float(slip["yaw_efficiency"])
        wrench_text = (
            "disabled"
            if not robot.config.separated_tire_force_enabled
            else "unavailable"
            if tire_wrench is None
            else [
                *np.asarray(tire_wrench["net_force_body_n"][:2]).round(2).tolist(),
                round(float(tire_wrench["net_torque_body_nm"][2]), 2),
            ]
        )
        robot._log_info(
            f"[{robot.name}] PhysX differential drive: smoothed_vx={vx:+.3f}, smoothed_wz={wz:+.3f}, "
            f"effective_vx={robot._physx_diff_drive.last_effective_vx:+.3f}, "
            f"effective_wz={robot._physx_diff_drive.last_effective_wz:+.3f}, "
            f"limit_scale={robot._physx_diff_drive.last_limit_scale:.3f}, "
            f"targets={np.asarray(wheel_targets).round(3).tolist()} rad/s, actual={actual_text} rad/s, "
            f"measured_base=(vx={robot._measured_vx:+.3f}, vy={robot._measured_vy:+.3f}, "
            f"wz={robot._measured_wz:+.3f}), slip={slip_text}, "
            f"yaw_efficiency={yaw_efficiency:.3f}, contacts={contact_flags}, "
            f"tire_wrench=[Fx,Fy,Tz]={wrench_text}"
        )
        robot._append_physx_diff_diagnostic(
            {
                "timestamp_monotonic": now,
                "run_label": robot._diff_diagnostics_run_label,
                "command": {"vx": float(vx), "wz": float(wz)},
                "effective_command": {
                    "vx": robot._physx_diff_drive.last_effective_vx,
                    "wz": robot._physx_diff_drive.last_effective_wz,
                },
                "limit_scale": robot._physx_diff_drive.last_limit_scale,
                "wheel_targets_rad_s": np.asarray(wheel_targets).reshape(-1).tolist(),
                "wheel_actual_rad_s": None if actual is None else np.asarray(actual).reshape(-1).tolist(),
                "measured_base": {
                    "vx": robot._measured_vx,
                    "vy": robot._measured_vy,
                    "wz": robot._measured_wz,
                },
                "slip": slip,
                "contacts": contacts,
                "separated_tire_wrench": tire_wrench,
                "parameters": {
                    "wheel_radius": robot.config.wheel_radius,
                    "track_width": robot.config.differential_track_width,
                    "linear_gain": robot.config.differential_linear_gain,
                    "angular_gain": robot.config.differential_angular_gain,
                    "drive_damping": robot.config.wheel_drive_damping,
                    "drive_max_force": robot.config.wheel_drive_max_force,
                    "static_friction": robot.config.wheel_static_friction,
                    "dynamic_friction": robot.config.wheel_dynamic_friction,
                    "separated_tire_force_enabled": robot.config.separated_tire_force_enabled,
                    "tire_longitudinal_stiffness": robot.config.tire_longitudinal_stiffness,
                    "tire_lateral_stiffness": robot.config.tire_lateral_stiffness,
                    "tire_max_longitudinal_force": robot.config.tire_max_longitudinal_force,
                    "tire_max_lateral_force": robot.config.tire_max_lateral_force,
                },
            }
        )


def motion_backend_for_mode(mode: str) -> Optional[MotionBackend]:
    if mode == "physx_wheels":
        return PhysxWheelsBackend()
    if mode == "physx_diff_contact":
        return PhysxDiffContactBackend()
    return None
