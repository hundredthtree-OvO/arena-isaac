#!/usr/bin/env python3
"""Extract an AMASS time interval and separate planar root translation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from ..analysis.amass_io import extract_clip_arrays, file_sha256
except ImportError:
    from analysis.amass_io import extract_clip_arrays, file_sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", type=Path)
    parser.add_argument("--start-sec", type=float, required=True)
    parser.add_argument("--end-sec", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.motion.expanduser().resolve()
    arrays = extract_clip_arrays(
        source,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)

    fps = float(arrays["mocap_framerate"].item())
    metadata = {
        "schema_version": 1,
        "source_path": str(source),
        "source_sha256": file_sha256(source),
        "output_path": str(output),
        "output_sha256": file_sha256(output),
        "start_sec": args.start_sec,
        "end_sec": args.end_sec,
        "start_frame": int(arrays["source_start_frame"].item()),
        "end_frame_exclusive": int(arrays["source_end_frame_exclusive"].item()),
        "frames": int(arrays["poses"].shape[0]),
        "fps": fps,
        "duration_sec": (arrays["poses"].shape[0] - 1) / fps,
        "root_displacement_m": arrays["root_trajectory"][-1].tolist(),
        "notes": [
            "local_trans removes planar translation only",
            "root orientation remains in poses and is not yet converted to in-place heading",
            "no pose resampling is performed",
        ],
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote clip to {output}")
    print(f"Wrote metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
