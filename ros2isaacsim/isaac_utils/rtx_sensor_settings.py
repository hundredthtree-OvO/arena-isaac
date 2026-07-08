"""Helpers for stabilizing Isaac RTX sensor global settings in bridge flows."""

from __future__ import annotations

import os

RTX_SENSOR_COORD_FRAME_SETTING = "/rtx/rtxsensor/coordinateFrameQuaternion"
RTX_SENSOR_COORD_FRAME_DEFAULT = "0.0,0.0,0.0,1.0"
RTX_SENSOR_COORD_FRAME_REPLICATOR_AGENT = "0.5,-0.5,-0.5,-0.5"

_SYNTHETIC_LIDAR_BACKENDS = {"synthetic", "synthetic_2d", "2d", "nav_2d"}
_AUTO_MODES = {"", "auto"}
_DEFAULT_MODES = {"default", "identity", "isaac", "isaac_default"}
_REPLICATOR_MODES = {"replicator", "replicator_agent", "replicator_agent_core"}
_KEEP_MODES = {"keep", "preserve", "off", "disabled"}


def _normalize_csv_quaternion(value) -> str:
    if value is None:
        return ""
    text = str(value).strip().strip("\"'")
    if not text:
        return ""
    parts = [part.strip() for part in text.split(",")]
    return ",".join(parts)


def desired_rtx_sensor_coord_frame(*, lidar_backend: str | None = None, mode: str | None = None) -> str | None:
    """Return the desired global RTX sensor coordinate frame override.

    Modes:
      auto        -> keep synthetic backends untouched, otherwise restore Isaac default
      default     -> force Isaac RTX default frame
      replicator  -> force Replicator Agent frame
      keep        -> leave the current global setting as-is
    """

    resolved_mode = (mode or os.environ.get("ARENA_ISAAC_RTX_SENSOR_COORD_FRAME_MODE", "auto")).strip().lower()
    if resolved_mode in _KEEP_MODES:
        return None
    if resolved_mode in _DEFAULT_MODES:
        return RTX_SENSOR_COORD_FRAME_DEFAULT
    if resolved_mode in _REPLICATOR_MODES:
        return RTX_SENSOR_COORD_FRAME_REPLICATOR_AGENT

    backend = (lidar_backend or os.environ.get("ARENA_ISAAC_LIDAR_BACKEND", "rtx")).strip().lower()
    if resolved_mode not in _AUTO_MODES:
        return RTX_SENSOR_COORD_FRAME_DEFAULT
    if backend in _SYNTHETIC_LIDAR_BACKENDS:
        return None
    return RTX_SENSOR_COORD_FRAME_DEFAULT


def ensure_rtx_sensor_coord_frame(logger=None, *, lidar_backend: str | None = None, mode: str | None = None, reason: str = "") -> bool:
    """Apply the requested global RTX sensor coordinate frame if needed.

    Returns ``True`` when a setting override was applied.
    """

    target = desired_rtx_sensor_coord_frame(lidar_backend=lidar_backend, mode=mode)
    if target is None:
        return False

    try:
        import carb.settings
    except Exception:
        return False

    settings = carb.settings.get_settings()
    current = _normalize_csv_quaternion(settings.get(RTX_SENSOR_COORD_FRAME_SETTING))
    if current == target:
        return False

    settings.set(RTX_SENSOR_COORD_FRAME_SETTING, target)
    message = (
        f"[rtx_sensor_settings] {RTX_SENSOR_COORD_FRAME_SETTING}: "
        f"{current or '<unset>'} -> {target}"
    )
    if reason:
        message += f" ({reason})"

    if logger is None:
        print(message, flush=True)
        return True

    for attr in ("warn", "warning", "info"):
        fn = getattr(logger, attr, None)
        if fn is not None:
            fn(message)
            return True
    print(message, flush=True)
    return True
