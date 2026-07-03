#!/usr/bin/env python3

import sys
import xml.etree.ElementTree as etree
import argparse
import os

def pose2origin(root):
    """
    Converts a <pose> element to an <origin> element with xyz and rpy attributes.
    """
    pose = root.text.strip()
    pose_list = pose.split()

    if len(pose_list) != 6:
        print("Error: <pose> element must contain exactly 6 values (x y z roll pitch yaw).")
        sys.exit(1)

    new_root = etree.Element("origin")
    new_root.set("xyz", " ".join(pose_list[:3]))
    new_root.set("rpy", " ".join(pose_list[3:]))

    return new_root

def children2attributes(root):
    """
    Converts child elements into attributes of the parent element.
    """
    children = list(root)

    new_root = etree.Element(root.tag)
    for child in children:
        if list(child):
            print(f"Error: Element <{child.tag}> is too nested and cannot be converted.")
            sys.exit(1)
        text = child.text.strip() if child.text else ''
        new_root.set(child.tag, text)

    return new_root

def convert_sdf2urdf(root):
    """
    Recursively converts SDF XML elements to URDF XML elements.
    """
    if root.tag == "pose":
        return pose2origin(root)
    elif root.tag in ["box", "sphere", "cylinder", "inertia", "limit"]:
        return children2attributes(root)
    elif root.tag in ["joint", "link", "robot", "geometry", "inertial", "visual", "collision"]:
        new_root = etree.Element(root.tag, attrib=root.attrib)
        for child in list(root):
            converted_child = convert_sdf2urdf(child)
            new_root.append(converted_child)
        return new_root
    else:
        # Add a warning comment for unrecognized elements
        warn_comment = etree.Comment(f"Warning: <{root.tag}> element is not recognized and left unchanged.")
        new_root = etree.Element(root.tag, attrib=root.attrib)
        new_root.insert(0, warn_comment)
        return new_root

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Convert an SDF file to a URDF file.")
    parser.add_argument("input_sdf", help="Path to the input SDF file.")
    parser.add_argument("output_urdf", help="Path to the output URDF file.")

    args = parser.parse_args()

    # Validate input file
    if not os.path.isfile(args.input_sdf):
        print(f"Error: Input SDF file '{args.input_sdf}' does not exist.")
        sys.exit(1)

    # Read and parse the input SDF file
    try:
        tree = etree.parse(args.input_sdf)
        root = tree.getroot()
    except etree.ParseError as e:
        print(f"Error parsing SDF file: {e}")
        sys.exit(1)

    # Convert SDF to URDF
    urdf_root = convert_sdf2urdf(root)

    # Write the resulting URDF to the output file
    try:
        urdf_tree = etree.ElementTree(urdf_root)
        urdf_tree.write(args.output_urdf, encoding="utf-8", xml_declaration=True)
        print(f"Successfully converted '{args.input_sdf}' to '{args.output_urdf}'.")
    except IOError as e:
        print(f"Error writing URDF file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
