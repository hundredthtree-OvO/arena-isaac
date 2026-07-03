#!/usr/bin/env python3
"""Generate a unified robot URDF by merging calibration sensor fragments.

This mirrors the calibration repo's merge_multi_urdf_file() idea, but uses the
mobile-base/arm URDF as the primary source of truth and appends calibration
fragments such as base_scan_01/base_scan_02/imu as fixed joints under
base_footprint. Mesh/file references are rewritten to absolute paths so the
merged URDF is relocatable inside arena-isaac assets.
"""
from __future__ import annotations

import argparse
import copy
import xml.etree.ElementTree as ET
from pathlib import Path


def _default_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "ros2isaacsim"
        / "ros2isaacsim"
        / "assets"
        / "robots"
        / "unified_v25730"
        / "mecanum730_xms5_gripper_v25730.urdf"
    )


def _default_base_urdf() -> Path:
    return Path(
        "/home/stardust/resources/mcp_service/curobo_v3/urdf/"
        "motion_wheel_arm_simple_sphere_urdf/"
        "mecanum730_xms5_gripper_joint_robotbuilder_link2plus_ee_motion_safe_collision.urdf"
    )


def _default_calibration_dir() -> Path:
    return Path("/home/stardust/resources/calibration/src/stardust_calibration/urdf/v25730")


def _abs_reference(base_path: Path, ref: str) -> str:
    ref_path = Path(ref)
    if ref_path.is_absolute():
        return str(ref_path)
    return str((base_path.parent / ref_path).resolve())


def _rewrite_references(root: ET.Element, source_path: Path):
    file_path = root.attrib.get("file_path")
    if file_path:
        root.attrib["file_path"] = _abs_reference(source_path, file_path)
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if filename:
            mesh.attrib["filename"] = _abs_reference(source_path, filename)


def _child_key(elem: ET.Element) -> tuple[str, str]:
    return elem.tag, str(elem.attrib.get("name", ""))


def merge_robot_description(base_urdf: Path, fragments: list[Path], output_path: Path):
    base_tree = ET.parse(base_urdf)
    base_root = base_tree.getroot()
    _rewrite_references(base_root, base_urdf)

    merged_root = ET.Element("robot", attrib=dict(base_root.attrib))
    seen: set[tuple[str, str]] = set()

    for child in list(base_root):
        merged_root.append(copy.deepcopy(child))
        seen.add(_child_key(child))

    for fragment_path in fragments:
        fragment_tree = ET.parse(fragment_path)
        fragment_root = fragment_tree.getroot()
        _rewrite_references(fragment_root, fragment_path)
        for child in list(fragment_root):
            key = _child_key(child)
            if key in seen:
                continue
            merged_root.append(copy.deepcopy(child))
            seen.add(key)

    try:
        ET.indent(merged_root)  # type: ignore[attr-defined]
    except Exception:
        pass

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(merged_root).write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate a unified robot URDF with calibration sensor fragments.")
    parser.add_argument("--base-urdf", default=str(_default_base_urdf()))
    parser.add_argument("--calibration-dir", default=str(_default_calibration_dir()))
    parser.add_argument(
        "--fragments",
        nargs="*",
        default=["base_scan_01.urdf.xml", "base_scan_02.urdf.xml", "imu.urdf.xml"],
        help="Fragment file names relative to --calibration-dir",
    )
    parser.add_argument("--output", default=str(_default_output_path()))
    args = parser.parse_args()

    base_urdf = Path(args.base_urdf).expanduser().resolve()
    calibration_dir = Path(args.calibration_dir).expanduser().resolve()
    fragments = [(calibration_dir / name).resolve() for name in args.fragments]
    output_path = Path(args.output).expanduser().resolve()

    missing = [str(path) for path in [base_urdf, *fragments] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))

    merged_path = merge_robot_description(base_urdf, fragments, output_path)
    print(merged_path)


if __name__ == "__main__":
    main()
