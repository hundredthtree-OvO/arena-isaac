import math


def stable_person_name(person) -> str:
    requested = str(getattr(person, "_requested_stage_name", "") or "").strip()
    if requested:
        return requested
    stage_prefix = str(getattr(person, "_stage_prefix", "") or "").strip().strip("/")
    if stage_prefix:
        return stage_prefix.split("/")[-1]
    return "pedestrian"


def iter_unique_people(people_by_alias: dict):
    seen: set[int] = set()
    for person in list(people_by_alias.values()):
        key = id(person)
        if key in seen:
            continue
        seen.add(key)
        yield stable_person_name(person), person


def pedestrian_state_publishable(person) -> bool:
    return bool(getattr(person, "_active", True)) and not bool(
        getattr(person, "is_parked", False)
    )


def pedestrian_state_reliable(person) -> bool:
    """Keep an idle active pedestrian present through AnimGraph read gaps."""
    if bool(getattr(person, "_pose_valid", False)):
        return True
    return str(getattr(person, "_motion_state", "")).strip().lower() == "idle"


def pedestrian_state_tagnames() -> tuple[str, ...]:
    return (
        "isaac",
        "pedestrian_state",
        "pose_valid",
        "motion_state",
        "command_generation",
        "guard_blocked",
        "guard_block_generation",
        "guard_block_count",
        "guard_block_reason",
        "yaw_rad",
        "yaw_valid",
        "embodiment_generation",
        "reactivation_ready",
    )


def pedestrian_state_yaw(person) -> float | None:
    state = getattr(person, "_state", None)
    orientation = getattr(state, "orientation", None)
    try:
        x, y, z, w = (float(orientation[index]) for index in range(4))
    except (TypeError, ValueError, IndexError):
        return None
    if not all(math.isfinite(value) for value in (x, y, z, w)):
        return None
    norm_sq = x * x + y * y + z * z + w * w
    if norm_sq <= 1e-12:
        return None
    scale = 1.0 / math.sqrt(norm_sq)
    x, y, z, w = x * scale, y * scale, z * scale, w * scale
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def estimate_pedestrian_velocity(
    previous,
    position,
    observed_at_sec: float,
    *,
    max_gap_sec: float = 1.0,
    max_speed_mps: float = 5.0,
) -> tuple[float, float, float]:
    if previous is None:
        return (0.0, 0.0, 0.0)
    previous_at, previous_position = previous
    dt = float(observed_at_sec) - float(previous_at)
    if dt <= 1e-6 or dt > float(max_gap_sec):
        return (0.0, 0.0, 0.0)
    try:
        velocity = tuple(
            (float(position[index]) - float(previous_position[index])) / dt
            for index in range(3)
        )
    except (TypeError, ValueError, IndexError):
        return (0.0, 0.0, 0.0)
    if not all(math.isfinite(value) for value in velocity):
        return (0.0, 0.0, 0.0)
    if math.sqrt(sum(value * value for value in velocity)) > float(
        max_speed_mps
    ):
        return (0.0, 0.0, 0.0)
    return velocity


def pedestrian_state_tags(person) -> list[str]:
    guard_blocked = bool(getattr(person, "_guard_blocked", False))
    guard_block_generation = getattr(person, "_guard_block_generation", 0) if guard_blocked else 0
    guard_block_count = getattr(person, "_guard_block_count", 0) if guard_blocked else 0
    guard_block_reason = getattr(person, "_guard_block_reason", "") if guard_blocked else ""
    pose_valid = bool(getattr(person, "_pose_valid", False))
    yaw = pedestrian_state_yaw(person)
    return [
        str(getattr(person, "_stage_prefix", "") or ""),
        str(getattr(person, "character_skel_root_stage_path", "") or ""),
        "true" if pose_valid else "false",
        str(getattr(person, "_motion_state", "unknown")),
        str(getattr(person, "_motion_command_generation", 0)),
        "true" if guard_blocked else "false",
        str(guard_block_generation),
        str(guard_block_count),
        str(guard_block_reason),
        "nan" if yaw is None else f"{yaw:.9f}",
        "false" if yaw is None else "true",
        str(getattr(person, "embodiment_generation", 0)),
        "true" if bool(getattr(person, "reactivation_ready", False)) else "false",
    ]
