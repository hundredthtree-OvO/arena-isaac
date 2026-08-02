#!/usr/bin/env bash
set -euo pipefail

ISAAC_ROOT="${ISAAC_ROOT:-/home/stardust/resources/isaac-sim-4.5.0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$ISAAC_ROOT/python.sh" \
  -m tools.smpl_pipeline.runtime.mirror_single_avatar "$@"
