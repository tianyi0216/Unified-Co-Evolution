#!/bin/bash
# run_agent_harness.sh — UCoE inside the Codex agent harness on SpreadsheetBench-2.
# Usage: bash run_agent_harness.sh [configs/ssb2_codex.yaml] <cmd> [--arm native|judge|ucoe] [--initial-candidates K]
#   cmd: manifest | pin-probe | smoke | calib-drafts | banks | h0 | run | status | official | report | audit | all
#   all = manifest -> pin-probe -> smoke -> run (native, judge, ucoe) -> official -> report; the remaining
#   arguments (e.g. --initial-candidates K, frozen into the manifest) are passed to every step
# Prerequisites: `python3 -m data.prepare_ssb2 --config <cfg>` (official checker + data + task tree),
# the `codex` CLI logged in, LibreOffice installed (tools.soffice or SOFFICE).
set -e
CONFIG=configs/ssb2_codex.yaml
if [[ "${1:-}" == *.yaml || "${1:-}" == *.yml ]]; then
  CONFIG=$1
  shift
fi
CMD=${1:-all}
if [[ $# -gt 0 ]]; then shift; fi
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
run() { python3 -m src.agent_harness.run --config "$CONFIG" "$@"; }
case "$CMD" in
  all)
    echo "[1/6] Freezing the manifest..."
    run manifest "$@"
    echo "[2/6] Pin probe (served model/effort per role)..."
    run pin-probe "$@"
    echo "[3/6] Live smoke (two reserved calibration tasks)..."
    run smoke "$@"
    echo "[4/6] Running the three arms..."
    for arm in native judge ucoe; do run run --arm "$arm" "$@"; done
    echo "[5/6] Official evaluation..."
    run official "$@"
    echo "[6/6] Report..."
    run report "$@"
    ;;
  *)
    run "$CMD" "$@"
    ;;
esac
