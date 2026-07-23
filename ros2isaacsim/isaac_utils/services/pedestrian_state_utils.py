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
    )


def pedestrian_state_tags(person) -> list[str]:
    guard_blocked = bool(getattr(person, "_guard_blocked", False))
    guard_block_generation = getattr(person, "_guard_block_generation", 0) if guard_blocked else 0
    guard_block_count = getattr(person, "_guard_block_count", 0) if guard_blocked else 0
    guard_block_reason = getattr(person, "_guard_block_reason", "") if guard_blocked else ""
    pose_valid = bool(getattr(person, "_pose_valid", False))
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
    ]
