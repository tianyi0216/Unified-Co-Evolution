#!/bin/bash
# eval.sh — evaluate the arms of a run: tie-adjusted preference against Direct Generation (blinded both-order
# judge) and the benchmark's native metric. Results land in <output.work_dir>/eval/.
# Usage: bash eval.sh [config.yaml] [mode] [arms]
#   mode = preference | native | all          (default: all)
#   arms = comma-separated method arm names    (default: ucoe; the baseline arm is raw)
# Examples:
#   bash eval.sh                                      # config.yaml, both evaluations, ucoe vs raw
#   bash eval.sh configs/wildbench_qwen8b.yaml native  # WB-Score only
set -e
CONFIG=${1:-config.yaml}
MODE=${2:-all}
ARMS=${3:-ucoe}
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python3 -m src.eval.run --config "$CONFIG" --mode "$MODE" --arms "$ARMS"
