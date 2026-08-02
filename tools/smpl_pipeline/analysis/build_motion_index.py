#!/usr/bin/env python3
"""Build a deterministic JSON index for an AMASS dataset directory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

try:
    from .amass_io import summarize_motion
except ImportError:
    from amass_io import summarize_motion


DEFAULT_DATASET = Path(
    "/home/stardust/resources/arena_ws/arena_assets/smpl/CMU/CMU"
)
DEFAULT_OUTPUT = Path(
    "/home/stardust/resources/arena_ws/arena_assets/smpl/derived/indexes/cmu.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stationary-speed-mps", type=float, default=0.08)
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Skip per-file hashes for a faster exploratory index.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    paths = sorted(dataset_root.rglob("*_poses.npz"))
    if not paths:
        print(f"No AMASS motion files found below {dataset_root}", file=sys.stderr)
        return 2

    motions = []
    failures = []
    for index, path in enumerate(paths, start=1):
        try:
            motions.append(
                summarize_motion(
                    path,
                    dataset_root=dataset_root,
                    stationary_speed_mps=args.stationary_speed_mps,
                    include_sha256=not args.skip_sha256,
                ).to_dict()
            )
        except (OSError, ValueError, KeyError) as exc:
            failures.append({"path": str(path), "error": str(exc)})
        if index % 100 == 0:
            print(f"Indexed {index}/{len(paths)} motions...", file=sys.stderr)

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "stationary_speed_mps": args.stationary_speed_mps,
        "motion_count": len(motions),
        "failure_count": len(failures),
        "motions": motions,
        "failures": failures,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(motions)} motions to {output}")
    if failures:
        print(f"Warning: {len(failures)} motions failed validation", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
