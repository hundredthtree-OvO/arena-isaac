#!/usr/bin/env python3
"""Print a JSON summary for one AMASS motion file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .amass_io import summarize_motion
except ImportError:
    from amass_io import summarize_motion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", type=Path)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Root used for relative_path; defaults to the motion parent.",
    )
    parser.add_argument("--stationary-speed-mps", type=float, default=0.08)
    args = parser.parse_args()

    path = args.motion.expanduser().resolve()
    root = (
        args.dataset_root.expanduser().resolve()
        if args.dataset_root
        else path.parent
    )
    summary = summarize_motion(
        path,
        dataset_root=root,
        stationary_speed_mps=args.stationary_speed_mps,
    )
    print(json.dumps(summary.to_dict(), ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
