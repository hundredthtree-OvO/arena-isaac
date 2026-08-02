#!/usr/bin/env python3
"""Validate a motion manifest and all referenced derived clip hashes."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from ..analysis.amass_io import file_sha256
    from .motion_library import load_manifest
except ImportError:
    from analysis.amass_io import file_sha256
    from runtime.motion_library import load_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--asset-root", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest.expanduser().resolve())
    asset_root = args.asset_root.expanduser().resolve()
    for motion in manifest["motions"]:
        clip = asset_root / motion["clip"]["relative_path"]
        if not clip.is_file():
            raise FileNotFoundError(f"{motion['id']}: missing clip {clip}")
        digest = file_sha256(clip)
        if digest != motion["clip"]["sha256"]:
            raise ValueError(f"{motion['id']}: clip SHA-256 mismatch")
        print(
            f"OK {motion['id']}: {motion['category']}/{motion['direction']} "
            f"{motion['timing']['frames']} frames"
        )
    print(
        f"Validated motion library {manifest['library_id']}: "
        f"{len(manifest['motions'])} clips"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
