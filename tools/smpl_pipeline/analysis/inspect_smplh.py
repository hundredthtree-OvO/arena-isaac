#!/usr/bin/env python3
"""Validate the AMASS-compatible SMPL-H model files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_MODEL_ROOT = Path(
    "/home/stardust/resources/arena_ws/arena_assets/smpl/smplh"
)
EXPECTED_SHAPES = {
    "v_template": (6890, 3),
    "f": (13776, 3),
    "J_regressor": (52, 6890),
    "kintree_table": (2, 52),
    "weights": (6890, 52),
    "shapedirs": (6890, 3, 16),
}


def inspect_model(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        missing = set(EXPECTED_SHAPES).difference(data.files)
        mismatches = {}
        for key, expected in EXPECTED_SHAPES.items():
            if key in data.files and tuple(data[key].shape) != expected:
                mismatches[key] = {
                    "actual": list(data[key].shape),
                    "expected": list(expected),
                }
        return {
            "path": str(path.resolve()),
            "keys": sorted(data.files),
            "missing_keys": sorted(missing),
            "shape_mismatches": mismatches,
            "valid": not missing and not mismatches,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    args = parser.parse_args()

    root = args.model_root.expanduser().resolve()
    results = {
        gender: inspect_model(root / gender / "model.npz")
        for gender in ("female", "male", "neutral")
    }
    print(json.dumps(results, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if all(result["valid"] for result in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
