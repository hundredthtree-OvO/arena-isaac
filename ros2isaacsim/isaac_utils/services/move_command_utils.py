def resolve_nav_velocity(value, *, default: float = 1.0) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)
