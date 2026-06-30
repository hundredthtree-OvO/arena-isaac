from __future__ import annotations

from pathlib import Path

_assets_root_path: str | None = None


def get_assets_root_path_safe(fallback: str = '/', *, reload: bool = False) -> str:
    """Try to get Isaac asset root path; fall back to `fallback`."""
    global _assets_root_path
    if not reload and _assets_root_path is not None:
        return _assets_root_path

    try:
        from isaacsim.storage.native import get_assets_root_path

        root = get_assets_root_path()
        _assets_root_path = root or fallback
        return _assets_root_path
    except Exception:
        pass

    try:
        import omni.isaac.nucleus

        root = omni.isaac.nucleus.get_assets_root_path()
        _assets_root_path = root or fallback
    except Exception:
        _assets_root_path = fallback

    return _assets_root_path


def get_package_asset_path(*relative_parts: str) -> str | None:
    """Resolve a ros2isaacsim asset path from either install-share or source tree."""
    candidates: list[Path] = []

    try:
        from ament_index_python.packages import get_package_share_directory

        candidates.append(Path(get_package_share_directory("ros2isaacsim")) / "assets" / Path(*relative_parts))
    except Exception:
        pass

    candidates.append(Path(__file__).resolve().parents[2] / "ros2isaacsim" / "assets" / Path(*relative_parts))

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return None
