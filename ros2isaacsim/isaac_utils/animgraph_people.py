"""Isaac Sim 4.5 AnimGraph/People compatibility helpers for arena-isaac.

The original arena-isaac people backend used older omni.anim.people/AnimGraph
startup assumptions.  In Isaac Sim 4.5 these extensions must be available when
Kit starts; enabling them after SimulationApp creation can leave the USD schema
registry in a bad state.  This module keeps the workaround in one place.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

ANIMGRAPH_PEOPLE_EXTENSIONS: tuple[str, ...] = (
    "omni.anim.graph.schema",
    "omni.anim.graph.core",
    "omni.anim.graph.ui",
    "omni.anim.navigation.schema",
    "omni.anim.navigation.core",
    "omni.anim.navigation.recast",
    "omni.anim.people",
    "isaacsim.replicator.agent.core",
    "isaacsim.replicator.agent.ui",
)

DEFAULT_PEOPLE_ASSET_ROOT = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/4.5/Isaac/People/Characters"
)


def _guess_isaac_path() -> Optional[Path]:
    for env_name in ("ISAAC_PATH", "ISAACSIM_PATH", "ISAAC_SIM_PATH"):
        value = os.environ.get(env_name)
        if value:
            p = Path(value).expanduser()
            if p.exists():
                return p
    common = Path.home() / "resources" / "isaac-sim-4.5.0"
    if common.exists():
        return common
    return None


def ensure_animgraph_experience(isaac_path: str | os.PathLike | None = None) -> str | None:
    """Create a small .kit experience that preloads AnimGraph/People extensions.

    Returns the generated kit path, or ``None`` if the base Isaac kit cannot be
    found.  It is safe to call before importing/constructing SimulationApp.
    """

    explicit = os.environ.get("ISAACSIM_ROSNAV_EXPERIENCE")
    if explicit:
        return str(Path(explicit).expanduser())

    root = Path(isaac_path).expanduser() if isaac_path else _guess_isaac_path()
    if root is None:
        return None

    base = root / "apps" / "isaacsim.exp.base.python.kit"
    if not base.exists():
        return None

    out_dir = Path.home() / ".cache" / "arena_rosnav" / "isaacsim45"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "isaacsim_exp_base_python_animgraph.kit"

    text = base.read_text()
    deps = [f'"{ext}" = {{}}' for ext in ANIMGRAPH_PEOPLE_EXTENSIONS]

    marker = "[dependencies]"
    idx = text.find(marker)
    if idx == -1:
        text += "\n\n[dependencies]\n" + "\n".join(deps) + "\n"
    else:
        next_section = text.find("\n[", idx + len(marker))
        if next_section == -1:
            next_section = len(text)
        block = text[idx:next_section]
        add = []
        for dep in deps:
            key = dep.split("=", 1)[0].strip()
            if key not in block:
                add.append(dep)
        if add:
            text = text[:next_section] + "\n" + "\n".join(add) + "\n" + text[next_section:]

    out.write_text(text)
    return str(out)


def enable_animgraph_people_extensions(extensions_module) -> None:
    """Enable extensions again after startup; harmless when already loaded."""
    for ext in ANIMGRAPH_PEOPLE_EXTENSIONS:
        try:
            extensions_module.enable_extension(ext)
        except Exception as exc:  # noqa: BLE001 - Isaac logs are better than hard failure here.
            try:
                import carb

                carb.log_warn(f"Failed to enable extension {ext}: {exc}")
            except Exception:
                print(f"Failed to enable extension {ext}: {exc}")


def normalize_stage_name(value: str | None, default: str = "Character") -> str:
    """Return a relative, USD-path-safe stage name for CharacterUtil."""
    if not value:
        value = default
    value = str(value).strip().strip("/")
    if not value:
        value = default
    # If the caller accidentally provided an absolute prim path, keep only its leaf.
    if "/" in value:
        value = value.rstrip("/").split("/")[-1]
    # CharacterUtil will create the prim under /World/Characters, so this must be relative.
    value = value.replace(" ", "_").replace(":", "_")
    if value and value[0].isdigit():
        value = f"_{value}"
    return value or default


def character_asset_path(character_name: str | None, asset_root: str | None = None) -> str:
    """Resolve a character folder/name into a USD file path.

    ``character_name`` may be an absolute path/URL to a USD file or a folder name
    in the Isaac People/Characters asset root.
    """
    if character_name and str(character_name).lower().endswith((".usd", ".usda", ".usdc")):
        return str(character_name)

    name = character_name or os.environ.get("ISAACSIM_ROSNAV_DEFAULT_CHARACTER") or "biped_demo"
    root = (asset_root or os.environ.get("ISAACSIM_PEOPLE_ASSET_ROOT") or DEFAULT_PEOPLE_ASSET_ROOT).rstrip("/")
    return f"{root}/{name}/{name}.usd"


def find_skelroot_with_anim_graph(stage, root_path: str) -> tuple[object | None, str | None]:
    """Find the first SkelRoot under ``root_path`` with AnimationGraphAPI applied."""
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return None, None

    def _visit(prim):
        try:
            applied = {str(x) for x in prim.GetAppliedSchemas()}
            if prim.GetTypeName() == "SkelRoot" and "AnimationGraphAPI" in applied:
                return prim
        except Exception:
            pass
        for child in prim.GetAllChildren():
            found = _visit(child)
            if found is not None:
                return found
        return None

    found = _visit(root)
    if found is None:
        return None, None
    return found, str(found.GetPath())


def as_float3_list(points: Iterable[Iterable[float]]) -> list:
    """Convert numeric point iterables to carb.Float3 values lazily at runtime."""
    import carb

    out = []
    for p in points:
        out.append(carb.Float3(float(p[0]), float(p[1]), float(p[2])))
    return out
