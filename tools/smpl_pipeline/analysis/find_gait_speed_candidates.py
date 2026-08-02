#!/usr/bin/env python3
"""Find deterministic slow/normal/fast straight-walk windows in AMASS CMU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .motion_features import rank_stable_walk_windows


DEFAULT_DATASET = Path(
    "/home/stardust/resources/arena_ws/arena_assets/smpl/CMU/CMU"
)
DEFAULT_OUTPUT = Path(
    "/home/stardust/resources/arena_ws/arena_assets/smpl/derived/indexes/"
    "cmu_gait_speed_candidates.json"
)
SPEED_BANDS = {
    "slow": (0.35, 0.72),
    "normal": (0.72, 1.10),
    "fast": (1.10, 1.55),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-band", type=int, default=12)
    args = parser.parse_args()

    root = args.dataset_root.expanduser().resolve()
    ranked: dict[str, list[dict]] = {name: [] for name in SPEED_BANDS}
    failures = []
    for path in sorted(root.rglob("*_poses.npz")):
        try:
            with np.load(path, allow_pickle=False) as data:
                fps = float(np.asarray(data["mocap_framerate"]).item())
                trans = np.asarray(data["trans"], dtype=np.float64)
                poses = np.asarray(data["poses"], dtype=np.float64)
            windows = rank_stable_walk_windows(
                trans,
                poses,
                fps,
                window_sec=2.0,
                hop_sec=0.5,
            )
            for window in windows[:8]:
                for band, (minimum, maximum) in SPEED_BANDS.items():
                    if minimum <= window.median_speed_mps < maximum:
                        payload = window.to_dict()
                        payload.update(
                            {
                                "motion_id": path.stem.removesuffix("_poses"),
                                "relative_path": str(path.relative_to(root)),
                                "speed_band": band,
                            }
                        )
                        ranked[band].append(payload)
                        break
        except (OSError, KeyError, ValueError) as error:
            failures.append({"path": str(path), "error": str(error)})

    for band, values in ranked.items():
        center = 0.5 * sum(SPEED_BANDS[band])
        values.sort(
            key=lambda item: (
                abs(item["median_speed_mps"] - center),
                -item["median_forward_alignment"],
                abs(item["yaw_change_rad"]),
                item["relative_path"],
                item["start_frame"],
            )
        )
        ranked[band] = values[: args.per_band]

    payload = {
        "schema_version": 1,
        "dataset_root": str(root),
        "speed_bands_mps": SPEED_BANDS,
        "candidates": ranked,
        "failures": failures,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for band, values in ranked.items():
        print(f"{band}: {len(values)} candidates")
        for item in values[:3]:
            print(
                f"  {item['relative_path']} "
                f"{item['start_sec']:.2f}-{item['end_sec']:.2f}s "
                f"speed={item['median_speed_mps']:.3f}m/s"
            )
    print(f"Wrote {output}")
    return 0 if all(ranked.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
