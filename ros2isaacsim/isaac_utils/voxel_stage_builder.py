"""Build an OctoMap-like occupied voxel map from the current Isaac USD stage.

The output is a compact JSON or JSON.GZ file consumed by ``VoxelCollisionGuard``.
It deliberately samples visual mesh geometry rather than PhysX colliders, because
some scanned/imported bathroom USDs contain invalid PhysX transforms.  This keeps
runtime collision independent of broken mesh colliders and avoids room-level bbox
proxy artifacts.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_SKIP_KEYWORDS = (
    "_collision_proxy",
    "/world/robots",
    "groundplane",
    "ground_plane",
    "/light",
    "dlight",
    "downlight",
    "camera",
    "lidar",
    "sensor",
    "ceiling",
    "floor_floor",
    "/floor/",
    "tuyere",
)


@dataclass
class VoxelBuildConfig:
    scene_root: str = "/World/shenxinfu_841837"
    output_path: str = "/home/stardust/resources/arena_ws/arena_assets/collision_configs/shenxinfu_841837.voxel.json.gz"
    debug_svg_path: str = "/home/stardust/resources/arena_ws/arena_assets/collision_configs/shenxinfu_841837.voxel.svg"
    debug_pcd_path: str = "/home/stardust/resources/arena_ws/arena_assets/collision_configs/shenxinfu_841837.voxel.pcd"
    resolution: float = 0.05
    sample_step: float = 0.05
    z_min: float = 0.05
    z_max: float = 1.20
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    skip_keywords: Sequence[str] = field(default_factory=lambda: list(DEFAULT_SKIP_KEYWORDS))
    include_keywords: Sequence[str] = field(default_factory=list)
    max_faces_per_mesh: int = 100000
    max_samples_per_mesh: int = 250000
    max_stored_voxels: int = 300000
    max_debug_stage_points: int = 30000
    create_stage_debug_points: bool = True
    stage_debug_path: str = ""  # default: <scene_root>/_voxel_debug_points


def _as_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _as_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v if str(x).strip()]
    return [x.strip() for x in str(v).split(",") if x.strip()]


def config_from_env() -> VoxelBuildConfig:
    def env_float(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, str(default)))
        except Exception:
            return float(default)

    def env_int(name: str, default: int) -> int:
        try:
            return int(float(os.environ.get(name, str(default))))
        except Exception:
            return int(default)

    scene_root = os.environ.get("ARENA_ISAAC_VOXEL_SCENE_ROOT", "/World/shenxinfu_841837")
    output = os.environ.get(
        "ARENA_ISAAC_VOXEL_MAP_PATH",
        "/home/stardust/resources/arena_ws/arena_assets/collision_configs/shenxinfu_841837.voxel.json.gz",
    )
    return VoxelBuildConfig(
        scene_root=scene_root,
        output_path=output,
        debug_svg_path=os.environ.get("ARENA_ISAAC_VOXEL_DEBUG_SVG_PATH", output.replace(".json.gz", ".svg").replace(".json", ".svg")),
        debug_pcd_path=os.environ.get("ARENA_ISAAC_VOXEL_DEBUG_PCD_PATH", output.replace(".json.gz", ".pcd").replace(".json", ".pcd")),
        resolution=env_float("ARENA_ISAAC_VOXEL_RESOLUTION", 0.05),
        sample_step=env_float("ARENA_ISAAC_VOXEL_SAMPLE_STEP", env_float("ARENA_ISAAC_VOXEL_RESOLUTION", 0.05)),
        z_min=env_float("ARENA_ISAAC_VOXEL_Z_MIN", 0.05),
        z_max=env_float("ARENA_ISAAC_VOXEL_Z_MAX", 1.20),
        skip_keywords=_as_list(os.environ.get("ARENA_ISAAC_VOXEL_SKIP_KEYWORDS")) or list(DEFAULT_SKIP_KEYWORDS),
        include_keywords=_as_list(os.environ.get("ARENA_ISAAC_VOXEL_INCLUDE_KEYWORDS")),
        max_faces_per_mesh=env_int("ARENA_ISAAC_VOXEL_MAX_FACES_PER_MESH", 100000),
        max_samples_per_mesh=env_int("ARENA_ISAAC_VOXEL_MAX_SAMPLES_PER_MESH", 250000),
        max_stored_voxels=env_int("ARENA_ISAAC_VOXEL_MAX_STORED_VOXELS", 300000),
        max_debug_stage_points=env_int("ARENA_ISAAC_VOXEL_MAX_DEBUG_STAGE_POINTS", 30000),
        create_stage_debug_points=_as_bool(os.environ.get("ARENA_ISAAC_VOXEL_CREATE_STAGE_DEBUG_POINTS"), True),
        stage_debug_path=os.environ.get("ARENA_ISAAC_VOXEL_STAGE_DEBUG_PATH", ""),
    )


def _should_skip_path(path: str, cfg: VoxelBuildConfig) -> bool:
    low = path.lower()
    for kw in cfg.skip_keywords:
        if kw and str(kw).lower() in low:
            return True
    if cfg.include_keywords:
        return not any(str(kw).lower() in low for kw in cfg.include_keywords)
    return False


def _vsub(a, b):
    return (float(a[0]) - float(b[0]), float(a[1]) - float(b[1]), float(a[2]) - float(b[2]))


def _vlen(v) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _sample_triangle(p0, p1, p2, step: float, max_samples: int) -> Iterable[Tuple[float, float, float]]:
    e0 = _vlen(_vsub(p1, p0))
    e1 = _vlen(_vsub(p2, p1))
    e2 = _vlen(_vsub(p0, p2))
    n = max(1, int(math.ceil(max(e0, e1, e2) / max(1e-4, step))))
    # Cap pathological long faces; still place enough samples along the surface.
    n = min(n, int(math.sqrt(max_samples)) if max_samples > 0 else n)
    count = 0
    for i in range(n + 1):
        for j in range(n + 1 - i):
            if count >= max_samples:
                return
            a = i / n
            b = j / n
            c = 1.0 - a - b
            yield (
                c * float(p0[0]) + a * float(p1[0]) + b * float(p2[0]),
                c * float(p0[1]) + a * float(p1[1]) + b * float(p2[1]),
                c * float(p0[2]) + a * float(p1[2]) + b * float(p2[2]),
            )
            count += 1


def _write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def _load_json(path: str) -> Dict[str, Any]:
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_debug_svg_from_data(data: Dict[str, Any], out_path: str, max_points: int = 120000) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    res = float(data.get("resolution", 0.05))
    origin = data.get("origin", [0.0, 0.0, 0.0])
    ox, oy = float(origin[0]), float(origin[1])
    cells = []
    if data.get("columns"):
        for c in data.get("columns", []):
            cells.append((int(c[0]), int(c[1])))
    else:
        seen = set()
        for v in data.get("voxels", []):
            key = (int(v[0]), int(v[1]))
            if key not in seen:
                seen.add(key)
                cells.append(key)
    if not cells:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write('<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600"><text x="20" y="40">No occupied voxel columns</text></svg>')
        return
    if len(cells) > max_points:
        stride = max(1, len(cells) // max_points)
        cells = cells[::stride]
    xs = [ix for ix, _ in cells]
    ys = [iy for _, iy in cells]
    min_x = ox + min(xs) * res
    max_x = ox + (max(xs) + 1) * res
    min_y = oy + min(ys) * res
    max_y = oy + (max(ys) + 1) * res
    width = 1100
    height = 800
    margin = 40
    sx = (width - 2 * margin) / max(1e-6, max_x - min_x)
    sy = (height - 2 * margin) / max(1e-6, max_y - min_y)
    scale = min(sx, sy)

    def map_pt(x, y):
        px = margin + (x - min_x) * scale
        py = height - margin - (y - min_y) * scale
        return px, py

    rect_size = max(1.0, res * scale)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="24" font-size="16" font-family="monospace">{data.get("scene_name", "scene")} voxel map: columns={len(data.get("columns", [])) or len(cells)}, res={res:.3f}m</text>',
    ]
    for ix, iy in cells:
        x = ox + ix * res
        y = oy + iy * res
        px, py = map_pt(x, y + res)
        parts.append(f'<rect x="{px:.2f}" y="{py:.2f}" width="{rect_size:.2f}" height="{rect_size:.2f}" fill="#2563eb" fill-opacity="0.65" stroke="none"/>')
    # axes / bounding box
    x0, y0 = map_pt(min_x, min_y)
    x1, y1 = map_pt(max_x, max_y)
    parts.append(f'<rect x="{x0:.2f}" y="{y1:.2f}" width="{(max_x-min_x)*scale:.2f}" height="{(max_y-min_y)*scale:.2f}" fill="none" stroke="#111" stroke-width="1"/>')
    parts.append(f'<text x="20" y="{height-20}" font-size="12" font-family="monospace">world bbox x=[{min_x:.2f},{max_x:.2f}] y=[{min_y:.2f},{max_y:.2f}]</text>')
    parts.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def write_debug_pcd_from_data(data: Dict[str, Any], out_path: str, max_points: int = 200000) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    res = float(data.get("resolution", 0.05))
    origin = data.get("origin", [0.0, 0.0, 0.0])
    ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
    pts: List[Tuple[float, float, float]] = []
    voxels = data.get("voxels") or []
    if voxels:
        stride = max(1, len(voxels) // max_points)
        for ix, iy, iz in voxels[::stride]:
            pts.append((ox + (int(ix) + 0.5) * res, oy + (int(iy) + 0.5) * res, oz + (int(iz) + 0.5) * res))
    else:
        columns = data.get("columns") or []
        stride = max(1, len(columns) // max_points)
        for item in columns[::stride]:
            ix, iy, iz0, iz1 = int(item[0]), int(item[1]), int(item[2]), int(item[3])
            iz = int(0.5 * (iz0 + iz1))
            pts.append((ox + (ix + 0.5) * res, oy + (iy + 0.5) * res, oz + (iz + 0.5) * res))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n")
        f.write(f"WIDTH {len(pts)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(pts)}\nDATA ascii\n")
        for x, y, z in pts:
            f.write(f"{x:.5f} {y:.5f} {z:.5f}\n")


def _make_stage_points(stage, path: str, points: List[Tuple[float, float, float]], width: float) -> None:
    if not points:
        return
    from pxr import Gf, Sdf, UsdGeom

    prim_path = Sdf.Path(path)
    try:
        old = stage.GetPrimAtPath(prim_path)
        if old and old.IsValid():
            stage.RemovePrim(prim_path)
    except Exception:
        pass
    pts_prim = UsdGeom.Points.Define(stage, prim_path)
    pts_prim.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in points])
    pts_prim.CreateWidthsAttr([float(width)] * len(points))
    pts_prim.CreateDisplayColorAttr([Gf.Vec3f(0.0, 0.55, 1.0)])


def build_voxel_map_from_stage(cfg: Optional[VoxelBuildConfig] = None, *, logger=None) -> Dict[str, Any]:
    cfg = cfg or config_from_env()
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import Gf, Usd, UsdGeom

    stage = get_current_stage()
    if stage is None:
        raise RuntimeError("no current USD stage")
    root = stage.GetPrimAtPath(cfg.scene_root)
    if not root or not root.IsValid():
        raise RuntimeError(f"scene root not found: {cfg.scene_root}")

    res = float(cfg.resolution)
    step = max(float(cfg.sample_step), res * 0.5)
    ox, oy, oz = cfg.origin
    voxel_set = set()
    columns: Dict[Tuple[int, int], List[int]] = {}
    source_stats: List[Dict[str, Any]] = []
    mesh_count = 0
    used_mesh_count = 0
    sample_count_total = 0
    start = time.monotonic()

    def log_info(msg: str):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    def add_point(x: float, y: float, z: float):
        nonlocal sample_count_total
        if z < cfg.z_min or z > cfg.z_max:
            return
        ix = math.floor((x - ox) / res)
        iy = math.floor((y - oy) / res)
        iz = math.floor((z - oz) / res)
        key3 = (int(ix), int(iy), int(iz))
        if len(voxel_set) < cfg.max_stored_voxels:
            voxel_set.add(key3)
        key2 = (int(ix), int(iy))
        old = columns.get(key2)
        if old is None:
            columns[key2] = [int(iz), int(iz), 1]
        else:
            old[0] = min(old[0], int(iz))
            old[1] = max(old[1], int(iz))
            old[2] += 1
        sample_count_total += 1

    for prim in stage.Traverse():
        if not prim.IsValid() or not prim.IsActive():
            continue
        path = prim.GetPath().pathString
        if not path.startswith(cfg.scene_root + "/"):
            continue
        if _should_skip_path(path, cfg):
            continue
        if prim.GetTypeName() != "Mesh":
            continue
        mesh_count += 1
        try:
            mesh = UsdGeom.Mesh(prim)
            pts = mesh.GetPointsAttr().Get(Usd.TimeCode.Default()) or []
            counts = mesh.GetFaceVertexCountsAttr().Get(Usd.TimeCode.Default()) or []
            indices = mesh.GetFaceVertexIndicesAttr().Get(Usd.TimeCode.Default()) or []
            if not pts or not counts or not indices:
                continue
            xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            world_pts = [xf.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2]))) for p in pts]
        except Exception as exc:
            source_stats.append({"path": path, "error": str(exc)[:200]})
            continue

        local_samples = 0
        local_vox_before = len(voxel_set)
        used_mesh_count += 1
        idx = 0
        face_idx = 0
        stop_mesh = False
        for count in counts:
            if face_idx >= cfg.max_faces_per_mesh:
                break
            face_indices = indices[idx: idx + int(count)]
            idx += int(count)
            face_idx += 1
            if len(face_indices) < 3:
                continue
            # fan triangulation
            p0 = world_pts[int(face_indices[0])]
            for k in range(1, len(face_indices) - 1):
                p1 = world_pts[int(face_indices[k])]
                p2 = world_pts[int(face_indices[k + 1])]
                remaining = max(0, cfg.max_samples_per_mesh - local_samples)
                if remaining <= 0:
                    stop_mesh = True
                    break
                for x, y, z in _sample_triangle(p0, p1, p2, step, remaining):
                    add_point(x, y, z)
                    local_samples += 1
                    if local_samples >= cfg.max_samples_per_mesh:
                        stop_mesh = True
                        break
                if stop_mesh:
                    break
            if stop_mesh:
                break
        if local_samples > 0:
            source_stats.append({
                "path": path,
                "samples": int(local_samples),
                "stored_voxels_delta": int(max(0, len(voxel_set) - local_vox_before)),
                "faces_used": int(face_idx),
            })

    columns_list = [[ix, iy, vals[0], vals[1], vals[2]] for (ix, iy), vals in columns.items()]
    columns_list.sort(key=lambda v: (v[0], v[1]))
    voxels_list = [list(v) for v in sorted(voxel_set)]
    data: Dict[str, Any] = {
        "format": "arena_voxel_guard_v1",
        "scene_name": cfg.scene_root.rstrip("/").split("/")[-1],
        "scene_root": cfg.scene_root,
        "resolution": res,
        "sample_step": step,
        "origin": [float(ox), float(oy), float(oz)],
        "z_min": float(cfg.z_min),
        "z_max": float(cfg.z_max),
        "created_unix": time.time(),
        "builder": "isaac_stage_visual_mesh_surface_sampler",
        "skip_keywords": list(cfg.skip_keywords),
        "include_keywords": list(cfg.include_keywords),
        "mesh_count_seen": int(mesh_count),
        "mesh_count_used": int(used_mesh_count),
        "sample_count": int(sample_count_total),
        "stored_voxel_count": int(len(voxels_list)),
        "column_count": int(len(columns_list)),
        "voxels_truncated": bool(len(voxel_set) >= cfg.max_stored_voxels),
        "voxels": voxels_list,
        "columns": columns_list,
        "sources": source_stats[:5000],
        "elapsed_sec": round(time.monotonic() - start, 3),
    }
    if columns_list:
        xs = [c[0] for c in columns_list]
        ys = [c[1] for c in columns_list]
        data["grid_bounds"] = [min(xs), max(xs), min(ys), max(ys)]
        data["world_bounds_xy"] = [ox + min(xs) * res, ox + (max(xs) + 1) * res, oy + min(ys) * res, oy + (max(ys) + 1) * res]

    _write_json(cfg.output_path, data)
    if cfg.debug_svg_path:
        write_debug_svg_from_data(data, cfg.debug_svg_path)
    if cfg.debug_pcd_path:
        write_debug_pcd_from_data(data, cfg.debug_pcd_path)
    if cfg.create_stage_debug_points:
        points: List[Tuple[float, float, float]] = []
        # Prefer actual 3D voxels. If truncated or absent, fallback to column mid-height.
        seq = voxels_list or [[c[0], c[1], int(0.5 * (c[2] + c[3]))] for c in columns_list]
        if seq:
            stride = max(1, len(seq) // max(1, cfg.max_debug_stage_points))
            for ix, iy, iz in seq[::stride]:
                points.append((ox + (int(ix) + 0.5) * res, oy + (int(iy) + 0.5) * res, oz + (int(iz) + 0.5) * res))
        debug_path = cfg.stage_debug_path or (cfg.scene_root.rstrip("/") + "/_voxel_debug_points")
        try:
            _make_stage_points(stage, debug_path, points[: cfg.max_debug_stage_points], max(res * 0.6, 0.015))
        except Exception as exc:
            data.setdefault("warnings", []).append(f"failed to create stage debug points: {exc}")
    log_info(
        f"[voxel_builder] wrote {cfg.output_path}: columns={len(columns_list)} stored_voxels={len(voxels_list)} "
        f"meshes={used_mesh_count}/{mesh_count} samples={sample_count_total} elapsed={data['elapsed_sec']}s"
    )
    return data


def load_voxel_map(path: str) -> Dict[str, Any]:
    return _load_json(path)
