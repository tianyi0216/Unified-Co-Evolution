"""Recalculation helper copied into every agent-visible directory as `recalc.py`.

Part of the fixed tool set of the agent-harness study (paper App. "Agent-harness study on
SpreadsheetBench-2": "a helper that recalculates a copy of a workbook"). Usage inside a
session: `python recalc.py <file.xlsx>` recalculates the workbook IN PLACE with LibreOffice
(headless convert-to under a profile with "always recalculate on load") so cached values
become available to openpyxl. No network, no other side effects.

The LibreOffice binary and profile are resolved from, in order: the environment variables
SOFFICE / LO_PROFILE, a `recalc.json` file next to this script ({"soffice": ..., "profile": ...},
written by the runner when it stages a directory; a relative profile path is taken relative to
this script's directory), the PATH, and the macOS application path.
"""

import json
import os
import shutil
import subprocess
import sys


def _settings():
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = {}
    p = os.path.join(here, "recalc.json")
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, ValueError):
            cfg = {}
    soffice = (
        os.environ.get("SOFFICE")
        or cfg.get("soffice")
        or shutil.which("soffice")
        or shutil.which("libreoffice")
        or "/Applications/LibreOffice.app/Contents/MacOS/soffice"
    )
    profile = os.environ.get("LO_PROFILE") or cfg.get("profile")
    if profile and not os.path.isabs(profile):
        profile = os.path.normpath(os.path.join(here, profile))
    return soffice, profile


def recalc(path):
    soffice, profile = _settings()
    if not os.path.isfile(soffice) and not shutil.which(soffice):
        print("recalc failed: LibreOffice not found (%s)" % soffice)
        return 1
    out = os.path.join(os.path.dirname(os.path.abspath(path)) or ".", "_recalc_out")
    os.makedirs(out, exist_ok=True)
    cmd = [soffice, "--headless"]
    if profile:
        cmd.append("-env:UserInstallation=file://%s" % profile)
    cmd += ["--convert-to", "xlsx", "--outdir", out, path]
    subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    got = os.path.join(out, os.path.basename(path))
    if not os.path.exists(got):
        print("recalc failed")
        return 1
    shutil.move(got, path)
    shutil.rmtree(out, ignore_errors=True)
    print("recalculated %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(recalc(sys.argv[1]))
