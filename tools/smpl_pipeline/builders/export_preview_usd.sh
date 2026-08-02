#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 PREVIEW_NPZ OUTPUT_USD" >&2
  exit 2
fi

ISAAC_ROOT="${ISAAC_ROOT:-/home/stardust/resources/isaac-sim-4.5.0}"
USD_LIB="$ISAAC_ROOT/extscache/omni.usd.libs-1.0.1+d02c707b.lx64.r.cp310"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ ! -d "$USD_LIB/pxr" ]]; then
  echo "Isaac USD Python package not found: $USD_LIB" >&2
  echo "Set ISAAC_ROOT to the Isaac Sim 4.5 installation directory." >&2
  exit 1
fi

export PYTHONPATH="$PIPELINE_DIR:$USD_LIB${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$USD_LIB/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$ISAAC_ROOT/python.sh" \
  "$SCRIPT_DIR/export_preview_usd.py" \
  "$1" \
  --output "$2"
