#!/usr/bin/env python3
"""Offline helpers for Arena Isaac voxel maps and derived occupancy maps."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import yaml


def load_map(path: str) -> Dict[str, Any]:
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_columns(data: Dict[str, Any]) -> Iterable[Tuple[int, int]]:
    if data.get("columns"):
        for c in data.get("columns", []) or []:
            if len(c) >= 2:
                yield int(c[0]), int(c[1])
        return
    seen = set()
    for v in data.get("voxels", []) or []:
        if len(v) < 2:
            continue
        key = (int(v[0]), int(v[1]))
        if key in seen:
            continue
        seen.add(key)
        yield key


def _read_pgm_token(stream) -> Optional[bytes]:
    token = bytearray()
    while True:
        ch = stream.read(1)
        if not ch:
            break
        if ch == b"#":
            stream.readline()
            continue
        if ch.isspace():
            if token:
                break
            continue
        token.extend(ch)
    return bytes(token) if token else None


def load_pgm(path: str) -> Tuple[int, int, int, List[int]]:
    with open(path, "rb") as f:
        magic = _read_pgm_token(f)
        if magic not in {b"P2", b"P5"}:
            raise ValueError(f"unsupported image format for {path}; expected P2/P5 PGM")
        width_token = _read_pgm_token(f)
        height_token = _read_pgm_token(f)
        maxval_token = _read_pgm_token(f)
        if width_token is None or height_token is None or maxval_token is None:
            raise ValueError(f"incomplete PGM header: {path}")
        width = int(width_token)
        height = int(height_token)
        maxval = int(maxval_token)
        total = width * height
        if magic == b"P2":
            values: List[int] = []
            while len(values) < total:
                token = _read_pgm_token(f)
                if token is None:
                    break
                values.append(int(token))
        else:
            bytes_per = 1 if maxval < 256 else 2
            raw = f.read(total * bytes_per)
            if len(raw) != total * bytes_per:
                raise ValueError(f"unexpected PGM payload length in {path}")
            if bytes_per == 1:
                values = list(raw)
            else:
                values = [
                    int.from_bytes(raw[i:i + 2], byteorder="big", signed=False)
                    for i in range(0, len(raw), 2)
                ]
        if len(values) != total:
            raise ValueError(f"unexpected PGM pixel count in {path}")
        return width, height, maxval, values


def load_occ_map_yaml(yaml_path: str) -> Dict[str, Any]:
    yaml_path_obj = Path(yaml_path)
    with yaml_path_obj.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    image_path = Path(str(data.get("image", "")).strip())
    if not image_path.is_absolute():
        image_path = (yaml_path_obj.parent / image_path).resolve()
    width, height, maxval, pixels = load_pgm(str(image_path))
    resolution = float(data.get("resolution", 0.05))
    origin_raw = data.get("origin", [0.0, 0.0, 0.0])
    origin = (float(origin_raw[0]), float(origin_raw[1]), float(origin_raw[2]))
    negate = bool(int(data.get("negate", 0)))
    occupied_thresh = float(data.get("occupied_thresh", 0.65))
    occupied: Set[Tuple[int, int]] = set()

    for img_y in range(height):
        iy = height - 1 - img_y
        for ix in range(width):
            value = float(pixels[img_y * width + ix]) / float(maxval)
            occ = value if negate else 1.0 - value
            if occ >= occupied_thresh:
                occupied.add((ix, iy))

    return {
        "yaml_path": str(yaml_path_obj),
        "image_path": str(image_path),
        "resolution": resolution,
        "origin": origin,
        "width": width,
        "height": height,
        "occupied_xy": occupied,
    }


def _grid_world_bounds(columns: Sequence[Tuple[int, int]], origin: Tuple[float, float, float], res: float) -> Tuple[float, float, float, float]:
    xs = [ix for ix, _ in columns]
    ys = [iy for _, iy in columns]
    min_x = origin[0] + min(xs) * res
    max_x = origin[1 - 1] + 0  # placeholder to keep line lengths small
    max_x = origin[0] + (max(xs) + 1) * res
    min_y = origin[1] + min(ys) * res
    max_y = origin[1] + (max(ys) + 1) * res
    return min_x, max_x, min_y, max_y


def _world_cells(columns: Iterable[Tuple[int, int]], origin: Tuple[float, float, float], res: float) -> Set[Tuple[int, int]]:
    scale = max(1, int(round(1.0 / max(res, 1e-9))))
    return {
        (int(round((origin[0] + ix * res) * scale)), int(round((origin[1] + iy * res) * scale)))
        for ix, iy in columns
    }


def _ray_grid_distance(
    ox: float,
    oy: float,
    dx: float,
    dy: float,
    occupied: Set[Tuple[int, int]],
    origin: Tuple[float, float, float],
    res: float,
    range_max: float,
) -> Optional[float]:
    if not occupied:
        return None
    gx0 = (ox - origin[0]) / res
    gy0 = (oy - origin[1]) / res
    ix = math.floor(gx0)
    iy = math.floor(gy0)
    start_cell = (int(ix), int(iy))
    step_x = 1 if dx > 0.0 else -1 if dx < 0.0 else 0
    step_y = 1 if dy > 0.0 else -1 if dy < 0.0 else 0
    inf = float("inf")

    if step_x == 0:
        t_max_x = inf
        t_delta_x = inf
    else:
        next_x = ix + 1 if step_x > 0 else ix
        world_next_x = origin[0] + next_x * res
        t_max_x = (world_next_x - ox) / dx
        t_delta_x = res / abs(dx)

    if step_y == 0:
        t_max_y = inf
        t_delta_y = inf
    else:
        next_y = iy + 1 if step_y > 0 else iy
        world_next_y = origin[1] + next_y * res
        t_max_y = (world_next_y - oy) / dy
        t_delta_y = res / abs(dy)

    max_steps = max(1, int(math.ceil(range_max / max(res, 1e-6))) + 4)
    xs = [p[0] for p in occupied]
    ys = [p[1] for p in occupied]
    bx0, bx1, by0, by1 = min(xs), max(xs), min(ys), max(ys)

    for _ in range(max_steps):
        cell = (int(ix), int(iy))
        if cell != start_cell and cell in occupied:
            t_hit = min(t_max_x, t_max_y)
            return t_hit if math.isfinite(t_hit) and t_hit <= range_max else None

        if t_max_x < t_max_y:
            t = t_max_x
            ix += step_x
            t_max_x += t_delta_x
        else:
            t = t_max_y
            iy += step_y
            t_max_y += t_delta_y

        if t > range_max:
            return None
        if (ix < bx0 - 2 and step_x <= 0) or (ix > bx1 + 2 and step_x >= 0):
            return None
        if (iy < by0 - 2 and step_y <= 0) or (iy > by1 + 2 and step_y >= 0):
            return None
    return None


def _sample_free_poses(
    occupied: Set[Tuple[int, int]],
    origin: Tuple[float, float, float],
    res: float,
    max_samples: int,
) -> List[Tuple[float, float]]:
    if not occupied:
        return []
    xs = [p[0] for p in occupied]
    ys = [p[1] for p in occupied]
    min_ix, max_ix = min(xs), max(xs)
    min_iy, max_iy = min(ys), max(ys)
    candidate_fracs = [
        (0.5, 0.5),
        (0.2, 0.2),
        (0.8, 0.2),
        (0.2, 0.8),
        (0.8, 0.8),
        (0.5, 0.2),
        (0.5, 0.8),
        (0.3, 0.5),
        (0.7, 0.5),
    ]
    poses: List[Tuple[float, float]] = []
    seen = set()
    for fx, fy in candidate_fracs:
        ix = int(round(min_ix + fx * (max_ix - min_ix)))
        iy = int(round(min_iy + fy * (max_iy - min_iy)))
        if (ix, iy) in occupied:
            continue
        if (ix, iy) in seen:
            continue
        seen.add((ix, iy))
        poses.append((origin[0] + (ix + 0.5) * res, origin[1] + (iy + 0.5) * res))
        if len(poses) >= max_samples:
            return poses

    for iy in range(min_iy, max_iy + 1):
        for ix in range(min_ix, max_ix + 1):
            if (ix, iy) in occupied or (ix, iy) in seen:
                continue
            seen.add((ix, iy))
            poses.append((origin[0] + (ix + 0.5) * res, origin[1] + (iy + 0.5) * res))
            if len(poses) >= max_samples:
                return poses
    return poses


def write_svg(data: Dict[str, Any], out_path: str, max_points: int = 160000) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    res = float(data.get("resolution", 0.05))
    origin = data.get("origin", [0.0, 0.0, 0.0])
    ox, oy = float(origin[0]), float(origin[1])
    cells = list(iter_columns(data))
    if not cells:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write('<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600"><text x="20" y="40">No occupied voxel columns</text></svg>')
        return
    total_cells = len(cells)
    if len(cells) > max_points:
        stride = max(1, len(cells) // max_points)
        cells = cells[::stride]
    xs = [ix for ix, _ in cells]
    ys = [iy for _, iy in cells]
    min_x = ox + min(xs) * res
    max_x = ox + (max(xs) + 1) * res
    min_y = oy + min(ys) * res
    max_y = oy + (max(ys) + 1) * res
    width, height, margin = 1200, 850, 45
    scale = min((width - 2 * margin) / max(1e-6, max_x - min_x), (height - 2 * margin) / max(1e-6, max_y - min_y))

    def mp(x: float, y: float):
        return margin + (x - min_x) * scale, height - margin - (y - min_y) * scale

    size = max(1.0, res * scale)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="25" font-size="16" font-family="monospace">{data.get("scene_name", "scene")} voxel map: columns={total_cells}, res={res:.3f}m, z=[{data.get("z_min", "?")},{data.get("z_max", "?")}]</text>',
    ]
    for ix, iy in cells:
        x = ox + ix * res
        y = oy + iy * res
        px, py = mp(x, y + res)
        parts.append(f'<rect x="{px:.2f}" y="{py:.2f}" width="{size:.2f}" height="{size:.2f}" fill="#2563eb" fill-opacity="0.70"/>')
    x0, y0 = mp(min_x, min_y)
    x1, y1 = mp(max_x, max_y)
    parts.append(f'<rect x="{x0:.2f}" y="{y1:.2f}" width="{(max_x-min_x)*scale:.2f}" height="{(max_y-min_y)*scale:.2f}" fill="none" stroke="#111" stroke-width="1"/>')
    parts.append(f'<text x="20" y="{height-20}" font-size="12" font-family="monospace">world bbox x=[{min_x:.2f},{max_x:.2f}] y=[{min_y:.2f},{max_y:.2f}], displayed={len(cells)}</text>')
    parts.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def write_pcd(data: Dict[str, Any], out_path: str, max_points: int = 250000) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    res = float(data.get("resolution", 0.05))
    origin = data.get("origin", [0.0, 0.0, 0.0])
    ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
    pts = []
    voxels = data.get("voxels") or []
    if voxels:
        stride = max(1, len(voxels) // max_points)
        for ix, iy, iz in voxels[::stride]:
            pts.append((ox + (int(ix) + 0.5) * res, oy + (int(iy) + 0.5) * res, oz + (int(iz) + 0.5) * res))
    else:
        columns = data.get("columns") or []
        stride = max(1, len(columns) // max_points)
        for c in columns[::stride]:
            ix, iy, iz0, iz1 = int(c[0]), int(c[1]), int(c[2]), int(c[3])
            iz = int(0.5 * (iz0 + iz1))
            pts.append((ox + (ix + 0.5) * res, oy + (iy + 0.5) * res, oz + (iz + 0.5) * res))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n")
        f.write(f"WIDTH {len(pts)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(pts)}\nDATA ascii\n")
        for x, y, z in pts:
            f.write(f"{x:.5f} {y:.5f} {z:.5f}\n")


def write_occ_map(data: Dict[str, Any], yaml_path: str, image_path: str = "", margin_cells: int = 1) -> Tuple[str, str]:
    columns = list(iter_columns(data))
    if not columns:
        raise ValueError("voxel map does not contain any occupied columns")

    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    if not image_path:
        image_path = yaml_path.replace(".yaml", ".pgm")

    res = float(data.get("resolution", data.get("voxel_size", 0.05)))
    origin = data.get("origin", [0.0, 0.0, 0.0])
    ox, oy = float(origin[0]), float(origin[1])
    pad = max(0, int(margin_cells))
    xs = [ix for ix, _ in columns]
    ys = [iy for _, iy in columns]
    min_ix = min(xs) - pad
    max_ix = max(xs) + pad
    min_iy = min(ys) - pad
    max_iy = max(ys) + pad
    width = max_ix - min_ix + 1
    height = max_iy - min_iy + 1

    pixels = bytearray([254]) * (width * height)
    for ix, iy in columns:
        px = ix - min_ix
        py = iy - min_iy
        img_y = height - 1 - py
        pixels[img_y * width + px] = 0

    with open(image_path, "wb") as f:
        f.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        f.write(pixels)

    yaml_data = {
        "image": os.path.basename(image_path),
        "resolution": res,
        "origin": [round(ox + min_ix * res, 6), round(oy + min_iy * res, 6), 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
        "type": "occupancy",
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        for key in ("image", "resolution", "origin", "negate", "occupied_thresh", "free_thresh", "type"):
            f.write(f"{key}: {yaml_data[key]}\n")
    return yaml_path, image_path


def check_occ_map(
    voxel_data: Dict[str, Any],
    occ_map: Dict[str, Any],
    *,
    sample_poses: int = 6,
    beams: int = 72,
    range_max: float = 12.0,
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    voxel_columns = list(iter_columns(voxel_data))
    voxel_res = float(voxel_data.get("resolution", voxel_data.get("voxel_size", 0.05)))
    voxel_origin = tuple(float(x) for x in voxel_data.get("origin", [0.0, 0.0, 0.0]))
    occ_res = float(occ_map["resolution"])
    occ_origin = tuple(float(x) for x in occ_map["origin"])
    occ_columns = occ_map["occupied_xy"]

    voxel_world = _world_cells(voxel_columns, voxel_origin, voxel_res)
    occ_world = _world_cells(occ_columns, occ_origin, occ_res)
    missing = voxel_world - occ_world
    extra = occ_world - voxel_world

    voxel_bounds = _grid_world_bounds(voxel_columns, voxel_origin, voxel_res)
    occ_bounds = _grid_world_bounds(list(occ_columns), occ_origin, occ_res)

    poses = _sample_free_poses(set(voxel_columns), voxel_origin, voxel_res, sample_poses)
    pose_reports = []
    max_ray_error = 0.0
    total_rays = 0
    mismatched_rays = 0
    hard_mismatched_rays = 0
    corner_tolerance = max(voxel_res, occ_res) * math.sqrt(2.0) + tolerance
    for ox, oy in poses:
        pose_max = 0.0
        for beam_idx in range(beams):
            angle = (2.0 * math.pi * (beam_idx + 0.5)) / float(beams)
            dx = math.cos(angle)
            dy = math.sin(angle)
            hit_voxel = _ray_grid_distance(ox, oy, dx, dy, set(voxel_columns), voxel_origin, voxel_res, range_max)
            hit_occ = _ray_grid_distance(ox, oy, dx, dy, occ_columns, occ_origin, occ_res, range_max)
            a = range_max if hit_voxel is None else float(hit_voxel)
            b = range_max if hit_occ is None else float(hit_occ)
            err = abs(a - b)
            pose_max = max(pose_max, err)
            max_ray_error = max(max_ray_error, err)
            total_rays += 1
            if err > tolerance:
                mismatched_rays += 1
            if err > corner_tolerance:
                hard_mismatched_rays += 1
        pose_reports.append({"x": round(ox, 4), "y": round(oy, 4), "max_error": pose_max})

    scan_alignment_ok = hard_mismatched_rays == 0
    return {
        "voxel_resolution": voxel_res,
        "occ_resolution": occ_res,
        "resolution_match": abs(voxel_res - occ_res) <= tolerance,
        "voxel_origin": [round(x, 6) for x in voxel_origin],
        "occ_origin": [round(x, 6) for x in occ_origin],
        "voxel_world_bounds": [round(x, 6) for x in voxel_bounds],
        "occ_world_bounds": [round(x, 6) for x in occ_bounds],
        "voxel_columns": len(voxel_columns),
        "occ_cells": len(occ_columns),
        "missing_world_cells": len(missing),
        "extra_world_cells": len(extra),
        "sample_pose_count": len(poses),
        "beam_count": beams,
        "total_rays": total_rays,
        "mismatched_rays": mismatched_rays,
        "hard_mismatched_rays": hard_mismatched_rays,
        "corner_tolerance": corner_tolerance,
        "max_ray_error": max_ray_error,
        "scan_alignment_ok": scan_alignment_ok,
        "pose_reports": pose_reports,
        "aligned": not missing and not extra and scan_alignment_ok and abs(voxel_res - occ_res) <= tolerance,
    }


def print_summary(data: Dict[str, Any]) -> None:
    keys = [
        "format", "scene_name", "scene_root", "resolution", "sample_step", "z_min", "z_max",
        "mesh_count_seen", "mesh_count_used", "sample_count", "stored_voxel_count", "column_count",
        "voxels_truncated", "world_bounds_xy", "elapsed_sec",
    ]
    for key in keys:
        if key in data:
            print(f"{key}: {data[key]}")
    sources = data.get("sources") or []
    if sources:
        print("top_sources:")
        for item in sorted(sources, key=lambda x: int(x.get("samples", 0)), reverse=True)[:20]:
            print(f"  samples={item.get('samples', 0):>8} faces={item.get('faces_used', 0):>6} path={item.get('path', '')}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_sum = sub.add_parser("summary")
    p_sum.add_argument("--map", required=True)
    p_svg = sub.add_parser("render")
    p_svg.add_argument("--map", required=True)
    p_svg.add_argument("--out", required=True)
    p_pcd = sub.add_parser("pcd")
    p_pcd.add_argument("--map", required=True)
    p_pcd.add_argument("--out", required=True)
    p_occ = sub.add_parser("occ-map")
    p_occ.add_argument("--map", required=True)
    p_occ.add_argument("--yaml-out", required=True)
    p_occ.add_argument("--image-out", default="")
    p_occ.add_argument("--margin-cells", type=int, default=1)
    p_chk = sub.add_parser("check-occ-map")
    p_chk.add_argument("--map", required=True)
    p_chk.add_argument("--yaml", required=True)
    p_chk.add_argument("--sample-poses", type=int, default=6)
    p_chk.add_argument("--beams", type=int, default=72)
    p_chk.add_argument("--range-max", type=float, default=12.0)
    p_chk.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args(argv)

    if args.cmd == "check-occ-map":
        report = check_occ_map(
            load_map(args.map),
            load_occ_map_yaml(args.yaml),
            sample_poses=args.sample_poses,
            beams=args.beams,
            range_max=args.range_max,
            tolerance=args.tolerance,
        )
        print(yaml.safe_dump(report, sort_keys=False))
        return 0 if report.get("aligned") else 2

    data = load_map(args.map)
    if args.cmd == "summary":
        print_summary(data)
    elif args.cmd == "render":
        write_svg(data, args.out)
        print(args.out)
    elif args.cmd == "pcd":
        write_pcd(data, args.out)
        print(args.out)
    elif args.cmd == "occ-map":
        yaml_path, image_path = write_occ_map(
            data,
            yaml_path=args.yaml_out,
            image_path=args.image_out,
            margin_cells=args.margin_cells,
        )
        print(yaml_path)
        print(image_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
