#!/bin/bash
# run_ucoe.sh — run the UCoE lifecycle: Direct Generation drafts -> relation discovery (if enabled) ->
# response-contrast banks -> initial verifier selection and the per-task deployment loop with checkpoint updates.
# Usage: bash run_ucoe.sh [config.yaml] [stage]
#   stage = all | raw | discover | banks | run | status   (default: all)
set -e
CONFIG=${1:-config.yaml}
STAGE=${2:-all}
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python3 -m src.run --config "$CONFIG" --stage "$STAGE"
