#!/bin/bash
# process_data.sh — download or pin the official benchmark source and write the processed task stream,
# calibration pool and evaluation sidecar under data/processed/<benchmark>/ (frozen bundles in data/frozen/
# are used when present).
# Usage: bash process_data.sh [config.yaml]
set -e
CONFIG=${1:-config.yaml}
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python3 -m data.prepare --config "$CONFIG"
