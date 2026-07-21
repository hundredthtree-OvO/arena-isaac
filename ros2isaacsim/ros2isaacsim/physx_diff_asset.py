"""Build the repository-local URDF used by the PhysX differential contact mode."""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
ROBOT_ASSETS = PACKAGE_ROOT / "assets" / "robots"
SOURCE_URDF = ROBOT_ASSETS / "unified_v25730" / "mecanum730_xms5_gripper_v25730.urdf"
OUTPUT_URDF = ROBOT_ASSETS / "physx_diff_contact" / "mecanum730_xms5_physx_diff_contact.urdf"
LOCAL_MESH_DIR = ROBOT_ASSETS / "unified_v25730" / "xms_mecanum_meshes"

WHEEL_LINKS = ("wheel_fl_link", "wheel_fr_link", "wheel_rl_link", "wheel_rr_link")
WHEEL_JOINTS = ("wheel_fl_joint", "wheel_fr_joint", "wheel_rl_joint", "wheel_rr_joint")
ROLLER_JOINT_TOKEN = "_roller_"
WHEEL_MASS_KG = 2.0
WHEEL_IXX = 0.0064
WHEEL_IYY_IZZ = 0.0036166667
ARM_FOLDED_POSITIONS = {
    "joint1": -0.04719,
    "joint2": -0.2495,
    "joint3": -0.477,
    "joint4": -0.0703,
    "joint5": -2.51,
    "joint6": 1.74,
    "gripper_joint1": 0.0,
    "gripper_joint2": 0.0,
}

PHYSX_DIFF_CONTACT_BASE_COLLIDER_SPECS = (
    {
        "link": "base_link",
        "name": "physx_chassis_lower_collision",
        "kind": "cube",
        "size": (0.50, 0.42, 0.16),
        "translate": (0.0, 0.0, 0.17),
    },
    {
        "link": "base_link",
        "name": "physx_chassis_upper_collision",
        "kind": "cube",
        "size": (0.54, 0.47, 0.47),
        "translate": (0.0, 0.0, 0.485),
    },
)

# Compact non-instance LOD fitted to the original 212-sphere collision cloud.
# Shapes remain link-local so they can follow the arm if its joints are later unfixed.
PHYSX_DIFF_CONTACT_UPPER_BODY_COLLIDER_SPECS = (
    {
        "link": "base_link",
        "name": "physx_upper_body_collision",
        "kind": "cube",
        "size": (0.54, 0.44, 0.38),
        "translate": (0.0, 0.0, 0.90),
    },
    {
        "link": "XMS5_R800_W4G3B4C_link2",
        "name": "physx_arm_link2_collision",
        "kind": "capsule",
        "axis": "x",
        "radius": 0.075,
        "height": 0.39,
        "translate": (0.195, 0.0, -0.095),
    },
    {
        "link": "XMS5_R800_W4G3B4C_link3",
        "name": "physx_arm_link3_collision",
        "kind": "cube",
        "size": (0.17, 0.27, 0.12),
        "translate": (-0.02, 0.073, 0.0),
    },
    {
        "link": "XMS5_R800_W4G3B4C_link4",
        "name": "physx_arm_link4_collision",
        "kind": "capsule",
        "axis": "z",
        "radius": 0.055,
        "height": 0.16,
        "translate": (0.0, 0.0, -0.08),
    },
    {
        "link": "XMS5_R800_W4G3B4C_link5",
        "name": "physx_arm_link5_collision",
        "kind": "capsule",
        "axis": "y",
        "radius": 0.06,
        "height": 0.065,
        "translate": (0.0, 0.0, 0.01),
    },
    {
        "link": "link6",
        "name": "physx_wrist_collision",
        "kind": "cube",
        "size": (0.096, 0.10, 0.055),
        "translate": (0.0, 0.0, -0.018),
    },
    {
        "link": "tool_link",
        "name": "physx_tool_collision",
        "kind": "capsule",
        "axis": "z",
        "radius": 0.05,
        "height": 0.36,
        "translate": (0.0, 0.0, 0.21),
    },
    {
        "link": "gripper_link1",
        "name": "physx_gripper_left_collision",
        "kind": "capsule",
        "axis": "y",
        "radius": 0.012,
        "height": 0.04,
        "translate": (0.0, -0.021, 0.0),
    },
    {
        "link": "gripper_link2",
        "name": "physx_gripper_right_collision",
        "kind": "capsule",
        "axis": "y",
        "radius": 0.012,
        "height": 0.04,
        "translate": (0.0, -0.023, 0.0),
    },
)


def _local_meshes(mesh_dir: Path) -> dict[str, Path]:
    meshes: dict[str, Path] = {}
    for path in mesh_dir.iterdir():
        stem = path.stem.rsplit("_", 1)[0].replace("-", "_").lower()
        meshes[stem] = path
    return meshes


def _portable_mesh_path(filename: str, mesh_dir: Path, output: Path) -> str:
    source_stem = Path(filename).stem.replace("-", "_").lower()
    local_path = _local_meshes(mesh_dir).get(source_stem)
    if local_path is None:
        raise ValueError(f"no repository-local mesh matches {filename}")
    return Path("..").joinpath("unified_v25730", "xms_mecanum_meshes", local_path.name).as_posix()


def _fixed_origin_rpy(rpy: str, angle: float) -> str:
    """Compose a URDF joint-origin rotation with a local-Z joint angle."""
    roll, pitch, yaw = map(float, rpy.split())
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    ca, sa = math.cos(angle), math.sin(angle)
    origin = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    local_z = ((ca, -sa, 0.0), (sa, ca, 0.0), (0.0, 0.0, 1.0))
    rotation = tuple(
        tuple(sum(origin[row][k] * local_z[k][col] for k in range(3)) for col in range(3))
        for row in range(3)
    )
    fixed_pitch = math.asin(max(-1.0, min(1.0, -rotation[2][0])))
    fixed_roll = math.atan2(rotation[2][1], rotation[2][2])
    fixed_yaw = math.atan2(rotation[1][0], rotation[0][0])
    return f"{fixed_roll:.12g} {fixed_pitch:.12g} {fixed_yaw:.12g}"


def build_physx_diff_urdf(
    source: Path = SOURCE_URDF,
    output: Path = OUTPUT_URDF,
    mesh_dir: Path = LOCAL_MESH_DIR,
) -> Path:
    tree = ET.parse(source)
    robot = tree.getroot()
    robot.set("name", "mecanum730_xms5_physx_diff_contact")
    robot.attrib.pop("file_path", None)

    for link in robot.findall("link"):
        link.attrib.pop("prim", None)
    for mesh in robot.findall(".//mesh"):
        mesh.set("filename", _portable_mesh_path(mesh.attrib["filename"], mesh_dir, output))

    if robot.find("./link[@name='base_link']") is None:
        raise ValueError("source URDF has no base_link")

    # Isaac Sim 4.5 may register instance-proxy collision shapes at the world
    # origin. Keep this importer input collision-free and author the six
    # contact-mode colliders directly into the resulting USD instead.
    for link in robot.findall("link"):
        for collision in list(link.findall("collision")):
            link.remove(collision)

    for link_name in WHEEL_LINKS:
        link = robot.find(f"./link[@name='{link_name}']")
        if link is None:
            raise ValueError(f"source URDF has no {link_name}")
        inertial = link.find("inertial")
        if inertial is None:
            inertial = ET.SubElement(link, "inertial")
        origin = inertial.find("origin")
        if origin is None:
            origin = ET.SubElement(inertial, "origin")
        origin.attrib.update({"xyz": "0 0 0", "rpy": "0 0 0"})
        mass = inertial.find("mass")
        if mass is None:
            mass = ET.SubElement(inertial, "mass")
        mass.set("value", f"{WHEEL_MASS_KG:g}")
        inertia = inertial.find("inertia")
        if inertia is None:
            inertia = ET.SubElement(inertial, "inertia")
        inertia.attrib.update(
            {
                "ixx": f"{WHEEL_IXX:g}",
                "ixy": "0",
                "ixz": "0",
                "iyy": f"{WHEEL_IYY_IZZ:g}",
                "iyz": "0",
                "izz": f"{WHEEL_IYY_IZZ:g}",
            }
        )
    for joint_name in WHEEL_JOINTS:
        joint = robot.find(f"./joint[@name='{joint_name}']")
        if joint is None:
            raise ValueError(f"source URDF has no {joint_name}")
        dynamics = joint.find("dynamics")
        if dynamics is None:
            dynamics = ET.SubElement(joint, "dynamics")
        dynamics.set("damping", "0.05")
        dynamics.set("friction", "0")

    # Differential contact uses one solid cylinder per wheel. The mecanum
    # rollers remain visible, but must not add 28 unconstrained articulation
    # DOFs that compete with the four driven wheel joints.
    fixed_roller_joints = set()
    for joint in robot.findall("joint"):
        joint_name = joint.attrib.get("name", "")
        if ROLLER_JOINT_TOKEN not in joint_name.lower():
            continue
        joint.set("type", "fixed")
        fixed_roller_joints.add(joint_name)
        for child in list(joint):
            if child.tag in {"axis", "limit", "dynamics", "mimic"}:
                joint.remove(child)

    for joint_name, angle in ARM_FOLDED_POSITIONS.items():
        joint = robot.find(f"./joint[@name='{joint_name}']")
        if joint is None:
            raise ValueError(f"source URDF has no {joint_name}")
        axis = joint.find("axis")
        if axis is not None and axis.attrib.get("xyz") != "0 0 1":
            raise ValueError(f"folded-arm baking supports local-Z joints only, got {joint_name}")
        origin = joint.find("origin")
        if origin is None:
            origin = ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        origin.set("rpy", _fixed_origin_rpy(origin.attrib.get("rpy", "0 0 0"), angle))
        joint.set("type", "fixed")
        for child in list(joint):
            if child.tag in {"axis", "limit", "dynamics", "mimic"}:
                joint.remove(child)

    fixed_names = set(ARM_FOLDED_POSITIONS) | fixed_roller_joints
    for transmission in list(robot.findall("transmission")):
        referenced = {
            elem.attrib.get("name")
            for elem in transmission.findall("joint")
            if elem.attrib.get("name")
        }
        if referenced & fixed_names:
            robot.remove(transmission)

    ET.indent(tree, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_URDF)
    parser.add_argument("--output", type=Path, default=OUTPUT_URDF)
    parser.add_argument("--mesh-dir", type=Path, default=LOCAL_MESH_DIR)
    args = parser.parse_args()
    print(build_physx_diff_urdf(args.source, args.output, args.mesh_dir))


if __name__ == "__main__":
    main()
