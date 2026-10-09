"""Command-line entry point of the agent-harness study (paper App. "Agent-harness study on SpreadsheetBench-2").

    python -m src.agent_harness.run --config configs/ssb2_codex.yaml <cmd> [--arm native|judge|ucoe]
    [--initial-candidates K]

Commands, in the order of a full study:
  manifest       freeze the deployment stream, the deployment-disjoint calibration sources and the protocol hashes
  pin-probe      one probe turn per role; the served model and effort must equal the configured ones
  smoke          two live calibration tasks on the judge arm in `<work_dir>/smoke`, its own work directory; all live
  checks must pass
  calib-drafts   actor drafts on the stage-0 calibration chunks (otherwise generated lazily by `banks`)
  banks          stage-0 selection and held-out contrast banks
  h0             initial verifier synthesis and selection for the ucoe arm (otherwise run by `run --arm ucoe`)
  run --arm A    the task stream for one arm (resumable; fail-closed on any hard stop)
  status         rows, statuses, verifier lineage, experience-block size and usage per arm
  official       official LibreOffice recalculation + checker on the sealed archives (after every arm is complete)
  report         accuracy per arm and category, paired differences, harness statistics
  audit          post-hoc integrity audit of one arm
"""

from __future__ import annotations

import argparse
import os
import shutil

from src.agent_harness import evaluate as EV
from src.agent_harness import runner as R
from src.utils.config import load_config

COMMANDS = (
    "manifest",
    "pin-probe",
    "smoke",
    "calib-drafts",
    "banks",
    "h0",
    "run",
    "status",
    "official",
    "report",
    "audit",
)


def run_smoke(cfg, main, initial_candidates=None):
    """Live smoke in `<work_dir>/smoke` (its own manifest copy, state, logs and LibreOffice profile) with the main
    manifest; the report is copied to the main work directory where `run` checks it."""
    sm = R.Harness(cfg, root=os.path.join(main.root, "smoke"), initial_candidates=initial_candidates)
    os.makedirs(sm.root, exist_ok=True)
    if not os.path.exists(sm.p("manifest.json")):
        if os.path.exists(main.p("manifest.json")):
            shutil.copyfile(main.p("manifest.json"), sm.p("manifest.json"))
        else:
            sm.manifest()
    pin = R._load(main.p("private", "pin_probe.json"))
    if not pin or not pin.get("ok"):
        raise SystemExit("[ucoe-agent smoke] run `pin-probe` on the main work_dir first")
    try:
        rep = sm.smoke()
    finally:
        sm.backend.close()
    R._save(main.p("private", "smoke_report.json"), {**rep, "smoke_root": os.path.relpath(sm.root, main.root)})
    return rep


def run_h0(h):
    sp = h.p("state", "ucoe.json")
    st = R._load(sp, {"arm": "ucoe", "rows": [], "V": None, "lineage_V": ["null"], "roles": h.roles_dict(), "ckpt": {}})
    h.ucoe_h0(st)
    R._save(sp, st)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m src.agent_harness.run", description="UCoE inside the Codex agent harness on SpreadsheetBench-2"
    )
    ap.add_argument("cmd", choices=COMMANDS)
    ap.add_argument("--config", default="configs/ssb2_codex.yaml")
    ap.add_argument("--arm", default="ucoe", choices=R.ARMS)
    ap.add_argument(
        "--initial-candidates",
        type=int,
        default=None,
        help="K initial verifier candidates (frozen into the manifest; config default)",
    )
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    h = R.Harness(cfg, initial_candidates=a.initial_candidates)
    os.makedirs(h.root, exist_ok=True)
    try:
        if a.cmd == "manifest":
            h.manifest()
        elif a.cmd == "pin-probe":
            h.pin_probe()
        elif a.cmd == "smoke":
            run_smoke(cfg, h, a.initial_candidates)
        elif a.cmd == "calib-drafts":
            for k in ("0_select", "0_cert"):
                for t in h.manifest()["calibration"]["chunks"][k]:
                    h.calib_draft(t)
        elif a.cmd == "banks":
            h.build_bank(0, "select")
            h.build_bank(0, "cert")
        elif a.cmd == "h0":
            run_h0(h)
        elif a.cmd == "run":
            h.run(a.arm)
        elif a.cmd == "status":
            h.status()
        elif a.cmd == "official":
            EV.official(h)
        elif a.cmd == "report":
            EV.report(h)
        elif a.cmd == "audit":
            h.audit_arm(a.arm)
    finally:
        h.backend.close()


if __name__ == "__main__":
    main()
