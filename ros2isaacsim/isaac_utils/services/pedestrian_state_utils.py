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
