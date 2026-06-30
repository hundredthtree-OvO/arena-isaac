#!/usr/bin/env python3
"""2D semantic collision-map helper for Arena Isaac.

This tool runs outside Isaac. It can bootstrap a semantic 2D collision YAML from an
exported editable proxy YAML, render an SVG top-down debug map, and print a summary.
The runtime guard uses the same YAML via ``ARENA_ISAAC_COLLISION_GUARD_BACKEND=2d``.
"""
from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml


def _as_bool(v, default=True):
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _xy(v, default=(0.0, 0.0)):
    try:
        return float(v[0]), float(v[1])
    except Exception:
        return float(default[0]), float(default[1])


def _safe_name(text: str) -> str:
    t = re.sub(r"[^A-Za-z0-9_]+", "_", str(text)).strip("_")
    return t or "item"


def _proxy_to_line(item: Dict[str, Any], *, kind: str) -> Dict[str, Any]:
    cx, cy = _xy(item.get("center", [0, 0]))
    sx, sy = _xy(item.get("size", [1, 0.1]), default=(1.0, 0.1))
    yaw = float(item.get("yaw", 0.0))
    # Long local axis is the line direction. If sy is longer, rotate by 90 deg.
    if sx >= sy:
        length = sx
        thickness = sy
        theta = yaw
    else:
        length = sy
        thickness = sx
        theta = yaw + math.pi / 2.0
    dx = 0.5 * length * math.cos(theta)
    dy = 0.5 * length * math.sin(theta)
    return {
        "name": _safe_name(item.get("name", item.get("prim_path", kind))),
        "kind": kind,
        "from": [round(cx - dx, 4), round(cy - dy, 4)],
        "to": [round(cx + dx, 4), round(cy + dy, 4)],
        "thickness": round(max(0.03, min(float(thickness), 0.40)), 4),
        "enabled": _as_bool(item.get("enabled", True), True),
        "source": item.get("source", item.get("prim_path", "proxy_yaml")),
    }


def _proxy_to_box(item: Dict[str, Any], *, kind: str) -> Dict[str, Any]:
    cx, cy = _xy(item.get("center", [0, 0]))
    sx, sy = _xy(item.get("size", [0.5, 0.5]), default=(0.5, 0.5))
    return {
        "name": _safe_name(item.get("name", item.get("prim_path", kind))),
        "kind": kind,
        "type": "box",
        "center": [round(cx, 4), round(cy, 4)],
        "size": [round(max(0.03, sx), 4), round(max(0.03, sy), 4)],
        "yaw": round(float(item.get("yaw", 0.0)), 6),
        "enabled": _as_bool(item.get("enabled", True), True),
        "source": item.get("source", item.get("prim_path", "proxy_yaml")),
    }


def _is_room_bbox(item: Dict[str, Any]) -> bool:
    name = str(item.get("name", "")).lower()
    kind = str(item.get("kind", "")).lower()
    sx, sy = _xy(item.get("size", [0, 0]))
    # Room envelope/bbox: both dimensions are large. A true wall segment is long but thin.
    if kind == "wall" and sx > 2.0 and sy > 2.0:
        return True
    if name.startswith("wall_wall_0000") and sx > 2.0 and sy > 2.0:
        return True
    return False


def init_from_proxies(proxy_yaml: str, out: str, *, scene: str, root_path: str, resolution: float) -> Dict[str, Any]:
    with open(proxy_yaml, "r", encoding="utf-8") as f:
        src = yaml.safe_load(f) or {}
    walls: List[Dict[str, Any]] = []
    partitions: List[Dict[str, Any]] = []
    objects: List[Dict[str, Any]] = []
    doors: List[Dict[str, Any]] = []
    ignored: List[Dict[str, Any]] = []

    for item in src.get("proxies", []) or []:
        kind = str(item.get("kind", "proxy")).lower()
        name = _safe_name(item.get("name", item.get("prim_path", kind)))
        if _is_room_bbox(item):
            ignored.append({"name": name, "reason": "room_level_wall_bbox_not_a_real_wall_segment", "source": item.get("source", item.get("prim_path", ""))})
            continue
        if kind == "wall":
            walls.append(_proxy_to_line(item, kind="wall"))
        elif kind == "partition":
            partitions.append(_proxy_to_line(item, kind="partition"))
        elif kind == "door":
            # Doors are semantic. Default to ignore/open because many scanned bathroom doors are open or partial meshes.
            cx, cy = _xy(item.get("center", [0, 0]))
            sx, sy = _xy(item.get("size", [0.75, 0.05]), default=(0.75, 0.05))
            width = max(sx, sy)
            thickness = max(0.03, min(sx, sy))
            doors.append({
                "name": name,
                "state": "ignore",
                "center": [round(cx, 4), round(cy, 4)],
                "width": round(width, 4),
                "thickness": round(thickness, 4),
                "yaw": round(float(item.get("yaw", 0.0)), 6),
                "enabled": False,
                "source": item.get("source", item.get("prim_path", "proxy_yaml")),
                "note": "Set state=closed and enabled=true only if this door should block navigation.",
            })
        elif kind in {"toilet", "urinal", "fixture", "cabinet", "sink", "basin"}:
            objects.append(_proxy_to_box(item, kind=kind))
        else:
            objects.append(_proxy_to_box(item, kind=kind))

    data = {
        "schema_version": 1,
        "scene": scene,
        "scene_root_path": root_path,
        "frame": "world",
        "resolution": float(resolution),
        "notes": [
            "V14 semantic 2D collision source. Runtime guard uses this file when ARENA_ISAAC_COLLISION_GUARD_BACKEND=2d.",
            "Large room-level wall bbox proxies are moved to ignored; replace them with explicit wall line segments.",
            "Door items default to state=ignore/enabled=false; enable only doors that should be physically closed.",
        ],
        "robot": {"footprint": {"length": 0.80, "width": 0.60, "margin": 0.05}},
        "walls": walls,
        "partitions": partitions,
        "objects": objects,
        "doors": doors,
        "ignored_from_proxy_yaml": ignored,
        "manual_todo": [
            "Add four outer wall line segments if the room-level wall bbox was ignored.",
            "Add missing stall partition wall line segments manually from top-down debug SVG.",
            "Set any truly closed door to enabled=true and state=closed.",
        ],
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return data


def _iter_rects(data: Dict[str, Any]) -> Iterable[Tuple[str, str, float, float, float, float, float, bool]]:
    # name, kind, cx, cy, sx, sy, yaw, enabled
    for section, kind_default in (("walls", "wall"), ("partitions", "partition")):
        for item in data.get(section, []) or []:
            enabled = _as_bool(item.get("enabled", True), True)
            if "from" in item and "to" in item:
                x0, y0 = _xy(item.get("from")); x1, y1 = _xy(item.get("to"))
                dx, dy = x1 - x0, y1 - y0
                length = math.hypot(dx, dy)
                yaw = math.atan2(dy, dx)
                yield str(item.get("name", kind_default)), str(item.get("kind", kind_default)), 0.5*(x0+x1), 0.5*(y0+y1), length, float(item.get("thickness", 0.1)), yaw, enabled
            elif "center" in item and "size" in item:
                cx, cy = _xy(item.get("center")); sx, sy = _xy(item.get("size"), default=(1,0.1))
                yield str(item.get("name", kind_default)), str(item.get("kind", kind_default)), cx, cy, sx, sy, float(item.get("yaw",0)), enabled
    for item in data.get("objects", []) or []:
        if str(item.get("type", "box")).lower() == "box":
            cx, cy = _xy(item.get("center", [0,0])); sx, sy = _xy(item.get("size", [0.5,0.5]), default=(0.5,0.5))
            yield str(item.get("name", "object")), str(item.get("kind", "object")), cx, cy, sx, sy, float(item.get("yaw",0)), _as_bool(item.get("enabled", True), True)
    for item in data.get("doors", []) or []:
        state = str(item.get("state", "ignore")).lower()
        if state in {"open", "ignore", "disabled"} or not _as_bool(item.get("enabled", True), True):
            continue
        cx, cy = _xy(item.get("center", [0,0])); width=float(item.get("width",0.75)); thickness=float(item.get("thickness",0.05))
        yield str(item.get("name", "door")), "door", cx, cy, width, thickness, float(item.get("yaw",0)), True


def _rect_corners(cx, cy, sx, sy, yaw):
    c=math.cos(yaw); s=math.sin(yaw); hx=sx*0.5; hy=sy*0.5
    pts=[]
    for dx,dy in [(-hx,-hy),(hx,-hy),(hx,hy),(-hx,hy)]:
        pts.append((cx+dx*c-dy*s, cy+dx*s+dy*c))
    return pts


def _bounds(data: Dict[str, Any], margin: float = 0.5):
    pts=[]
    for _name,_kind,cx,cy,sx,sy,yaw,_enabled in _iter_rects(data):
        pts.extend(_rect_corners(cx,cy,sx,sy,yaw))
    for item in data.get("objects", []) or []:
        if str(item.get("type", "box")).lower() in {"circle", "cylinder"}:
            cx,cy=_xy(item.get("center",[0,0])); r=float(item.get("radius",0.25)); pts.extend([(cx-r,cy-r),(cx+r,cy+r)])
    if not pts:
        return -1, -1, 1, 1
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    return min(xs)-margin, min(ys)-margin, max(xs)+margin, max(ys)+margin


def render_svg(config: str, out: str) -> None:
    with open(config, "r", encoding="utf-8") as f:
        data=yaml.safe_load(f) or {}
    minx,miny,maxx,maxy=_bounds(data)
    width=max(1e-6,maxx-minx); height=max(1e-6,maxy-miny)
    scale=120.0
    W=int(width*scale); H=int(height*scale)
    def tx(x): return (x-minx)*scale
    def ty(y): return H-(y-miny)*scale
    elems=[]
    elems.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#f7f7f7" stroke="#999"/>')
    color_map={"wall":"#222222","partition":"#555555","door":"#f2b200","toilet":"#00a6d6","urinal":"#00a6d6","fixture":"#00a6d6","cabinet":"#41b883","object":"#8888cc"}
    for i,(name,kind,cx,cy,sx,sy,yaw,enabled) in enumerate(_iter_rects(data),1):
        pts=_rect_corners(cx,cy,sx,sy,yaw)
        d=" ".join(f"{tx(x):.1f},{ty(y):.1f}" for x,y in pts)
        col=color_map.get(kind,"#8888cc")
        opacity="0.80" if enabled else "0.18"
        elems.append(f'<polygon points="{d}" fill="{col}" opacity="{opacity}" stroke="#000" stroke-width="1"/>')
        elems.append(f'<text x="{tx(cx):.1f}" y="{ty(cy):.1f}" font-size="10" fill="#111">{i}:{name}</text>')
    for i,item in enumerate(data.get("objects",[]) or [],1):
        if str(item.get("type","box")).lower() in {"circle","cylinder"} and _as_bool(item.get("enabled",True),True):
            cx,cy=_xy(item.get("center",[0,0])); r=float(item.get("radius",0.25));
            elems.append(f'<circle cx="{tx(cx):.1f}" cy="{ty(cy):.1f}" r="{r*scale:.1f}" fill="#00a6d6" opacity="0.8" stroke="#000"/>')
    elems.append(f'<text x="8" y="18" font-size="14" fill="#000">{data.get("scene","collision2d")} semantic 2D debug map</text>')
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out,"w",encoding="utf-8") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n' + "\n".join(elems) + "\n</svg>\n")


def summarize(config: str) -> Dict[str, int]:
    with open(config, "r", encoding="utf-8") as f:
        data=yaml.safe_load(f) or {}
    counts={"walls":len(data.get("walls",[]) or []),"partitions":len(data.get("partitions",[]) or []),"objects":len(data.get("objects",[]) or []),"doors":len(data.get("doors",[]) or []),"ignored":len(data.get("ignored_from_proxy_yaml",[]) or [])}
    return counts


def main(argv=None):
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest="cmd",required=True)
    p=sub.add_parser("init-from-proxies")
    p.add_argument("--proxy-yaml",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--scene",default="shenxinfu_841837")
    p.add_argument("--root-path",default="/World/shenxinfu_841837")
    p.add_argument("--resolution",type=float,default=0.03)
    p=sub.add_parser("render")
    p.add_argument("--config",required=True)
    p.add_argument("--out",required=True)
    p=sub.add_parser("summary")
    p.add_argument("--config",required=True)
    args=ap.parse_args(argv)
    if args.cmd=="init-from-proxies":
        data=init_from_proxies(args.proxy_yaml,args.out,scene=args.scene,root_path=args.root_path,resolution=args.resolution)
        print(f"wrote {args.out}")
        print(yaml.safe_dump({"counts":{"walls":len(data.get('walls',[])),"partitions":len(data.get('partitions',[])),"objects":len(data.get('objects',[])),"doors":len(data.get('doors',[])),"ignored":len(data.get('ignored_from_proxy_yaml',[]))}}, sort_keys=False))
    elif args.cmd=="render":
        render_svg(args.config,args.out)
        print(f"wrote {args.out}")
    elif args.cmd=="summary":
        print(yaml.safe_dump(summarize(args.config), sort_keys=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
