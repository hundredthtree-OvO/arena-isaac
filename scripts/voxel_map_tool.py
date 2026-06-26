#!/usr/bin/env python3
"""Offline helpers for Arena Isaac v15 voxel maps.

This script does not require Isaac/pxr.  It can summarize and render a voxel
map generated inside Isaac by /isaac/build_voxel_map.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
from typing import Any, Dict


def load_map(path: str) -> Dict[str, Any]:
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_svg(data: Dict[str, Any], out_path: str, max_points: int = 160000) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    res = float(data.get("resolution", 0.05))
    origin = data.get("origin", [0.0, 0.0, 0.0])
    ox, oy = float(origin[0]), float(origin[1])
    cells = []
    if data.get("columns"):
        cells = [(int(c[0]), int(c[1])) for c in data.get("columns", [])]
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


def print_summary(data: Dict[str, Any]) -> None:
    keys = [
        "format", "scene_name", "scene_root", "resolution", "sample_step", "z_min", "z_max",
        "mesh_count_seen", "mesh_count_used", "sample_count", "stored_voxel_count", "column_count",
        "voxels_truncated", "world_bounds_xy", "elapsed_sec",
    ]
    for k in keys:
        if k in data:
            print(f"{k}: {data[k]}")
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
    args = parser.parse_args(argv)
    data = load_map(args.map)
    if args.cmd == "summary":
        print_summary(data)
    elif args.cmd == "render":
        write_svg(data, args.out)
        print(args.out)
    elif args.cmd == "pcd":
        write_pcd(data, args.out)
        print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
