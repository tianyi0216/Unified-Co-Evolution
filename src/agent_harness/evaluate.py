"""LibreOffice recalculation and the official SpreadsheetBench-2 checker (evaluation only).

Implements the evaluation of the agent-harness study (paper App. "Agent-harness study on
SpreadsheetBench-2", "Benchmark and evaluation"): the benchmark's checker recalculates the
returned workbook with LibreOffice and compares it cell by cell with the reference workbook,
so a task counts as correct only on an exact match. This is the ONLY module that opens the
reference workbooks and `answer_position`; the method path never reads them. It also owns
the LibreOffice profile with "always recalculate on load", which it creates when missing,
and the recalculation function the evidence module binds to every scored candidate.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import random
import shutil
import subprocess

from src.agent_harness import core as C
from src.utils.io import atomic_write, read_json

CATS = ("Template", "Financial_Model", "Debugging")
MACOS_SOFFICE = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
REGISTRY_XCU = """<?xml version="1.0" encoding="UTF-8"?>
<oor:items xmlns:oor="http://openoffice.org/2001/registry"
           xmlns:xs="http://www.w3.org/2001/XMLSchema"
           xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<item oor:path="/org.openoffice.Office.Calc/Formula/Load">
  <prop oor:name="OOXMLRecalcMode" oor:op="fuse"><value>0</value></prop>
</item>
<item oor:path="/org.openoffice.Office.Calc/Formula/Load">
  <prop oor:name="ODFRecalcMode" oor:op="fuse"><value>0</value></prop>
</item>
</oor:items>
"""


def _cfg(cfg, path, default=None):
    cur = cfg
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return default if cur is None else cur


# ======================================================================= LibreOffice
def soffice_path(cfg):
    """LibreOffice binary: config `tools.soffice`, then env SOFFICE, then PATH, then the macOS application path; None
    when absent."""
    for c in (_cfg(cfg, "tools.soffice"), os.environ.get("SOFFICE"), "soffice", "libreoffice", MACOS_SOFFICE):
        if not c:
            continue
        p = shutil.which(c) or (c if os.path.isfile(c) else None)
        if p:
            return p
    return None


def lo_profile(cfg, root):
    """Isolated LibreOffice profile directory (config `tools.lo_profile` or `<work_dir>/lo_profile`), created with
    OOXMLRecalcMode/ODFRecalcMode = 0 ("always recalculate on load") when missing."""
    prof = os.path.abspath(_cfg(cfg, "tools.lo_profile") or os.path.join(root, "lo_profile"))
    xcu = os.path.join(prof, "user", "registrymodifications.xcu")
    if not os.path.exists(xcu):
        os.makedirs(os.path.dirname(xcu), exist_ok=True)
        atomic_write(xcu, REGISTRY_XCU)
    return prof


def recalc_dir(d, soffice, profile, timeout=900):
    """FULL LibreOffice recalculation of every .xlsx in `d` (evaluation copies only, replaced in place): headless
    `--convert-to xlsx`
    under a profile with "always recalculate on load". ONE file per soffice process: workbooks with iterative
    calculation converge to
    a different value when batched; single-file runs are deterministic."""
    files = sorted(glob.glob(os.path.join(d, "*.xlsx")))
    if not files:
        return True, "no xlsx"
    if not soffice:
        return False, "LibreOffice (soffice) not found: set tools.soffice or SOFFICE"
    out = os.path.join(d, "_recalc_out")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    ok, note = True, ""
    for f in files:
        r = subprocess.run(
            [
                soffice,
                "--headless",
                "-env:UserInstallation=file://%s" % profile,
                "--convert-to",
                "xlsx",
                "--outdir",
                out,
                f,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        note = (r.stdout + r.stderr)[-200:]
        g = os.path.join(out, os.path.basename(f))
        if os.path.exists(g):
            shutil.move(g, f)
        else:
            ok = False
    shutil.rmtree(out, ignore_errors=True)
    return ok, note


def make_recalc_fn(cfg, root):
    """(dir) -> (ok, note) bound to the configured LibreOffice binary and profile."""
    soffice, profile = soffice_path(cfg), lo_profile(cfg, root)

    def fn(d):
        return recalc_dir(d, soffice, profile)

    return fn


def soffice_setting(cfg):
    """The LibreOffice binary as configured (`tools.soffice`, env SOFFICE, a PATH name or the macOS application path):
    the first candidate that resolves, recorded as given rather than as a resolved machine path."""
    for c in (_cfg(cfg, "tools.soffice"), os.environ.get("SOFFICE"), "soffice", "libreoffice", MACOS_SOFFICE):
        if c and (shutil.which(c) or os.path.isfile(c)):
            return c
    return None


def recalc_settings(cfg, root):
    """Settings written next to every staged `recalc.py` helper so agent sessions use the same binary and profile
    (the runner stores the profile path relative to the helper)."""
    return {"soffice": soffice_setting(cfg), "profile": lo_profile(cfg, root)}


# ======================================================================= official checker
def dataset_root(data_dir):
    """Directory holding `<category>/dataset.json` for every category (the extracted official data)."""
    pin = read_json(os.path.join(data_dir, "pin.json"), None) or {}
    cands = [os.path.join(data_dir, pin["data_root"])] if pin.get("data_root") else []
    cands += [os.path.join(data_dir, "spreadsheetbench-v2"), data_dir]
    for dp, dns, _ in os.walk(data_dir):
        if dp.count(os.sep) - data_dir.count(os.sep) > 3:
            dns[:] = []
            continue
        cands.append(dp)
    for c in cands:
        if all(os.path.exists(os.path.join(c, cat, "dataset.json")) for cat in CATS):
            return c
    raise SystemExit(
        "[ucoe-agent official] official data not found under %s; run `python -m data.prepare_ssb2 --config <cfg>` first"
        % data_dir
    )


def load_evaluator(data_dir):
    """The official `evaluation.py` of the pinned SpreadsheetBench-2 checkout (imported, never modified)."""
    import importlib.util

    path = os.path.join(data_dir, "SpreadsheetBench-2", "evaluation", "evaluation.py")
    if not os.path.exists(path):
        raise SystemExit("[ucoe-agent official] official checker missing: %s (run data preparation)" % path)
    spec = importlib.util.spec_from_file_location("ssb2_official_evaluation", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _stage_reference(h, src, dst):
    """Copy one reference workbook into the evaluation tree and recalculate it exactly once: the copy is
    recalculated alone in a staging directory and moved into place afterwards, so workbooks sharing a family
    directory are never recalculated again (a second recalculation drifts iterative-calculation workbooks) and an
    interrupted run resumes without repeating the step."""
    stage = h.p("evalcopy", "_staging")
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    tmp = os.path.join(stage, os.path.basename(dst))
    shutil.copyfile(src, tmp)
    ok, note = h.recalc_fn(stage)
    if not ok:
        raise SystemExit("[ucoe-agent official] recalculation of the reference copy failed: %s" % note)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.replace(tmp, dst)
    shutil.rmtree(stage, ignore_errors=True)


def _record(data_root, cat, tid):
    full = json.load(open(os.path.join(data_root, cat, "dataset.json"), encoding="utf-8"))
    return next(x for x in full if str(x["id"]) == tid)


def official(h, arms=None):
    """Official recalculation + checker on the sealed archives of every arm (COPIES only; the reference copy is
    recalculated once).
    Opens only after every requested arm has sealed the whole stream; an archive hash mismatch refuses to evaluate."""
    man = h.manifest()
    arms = tuple(arms or h.arms)
    for arm in arms:
        st = read_json(h.p("state", "%s.json" % arm), None) or {"rows": []}
        if len(st["rows"]) != len(man["deployment_tasks"]):
            raise SystemExit(
                "%s incomplete (%d/%d); official evaluation opens only after every arm is sealed"
                % (arm, len(st["rows"]), len(man["deployment_tasks"]))
            )
    ev = load_evaluator(h.data_dir)
    data_root = dataset_root(h.data_dir)
    res = read_json(h.p("official.json"), None) or {}
    for t in man["deployment_tasks"]:
        cat, tid = t["category"], t["id"]
        rec = _record(data_root, cat, tid)
        ec = h.p("evalcopy", cat)
        for k in ("spreadsheet_path", "golden_response_path"):
            dst = os.path.join(ec, rec[k])
            if not os.path.exists(dst):
                _stage_reference(h, os.path.join(data_root, cat, rec[k]), dst)
        for arm in arms:
            key = "%s/%s|%s" % (arm, cat, tid)
            if key in res:
                continue
            ad = h.p("private", "archive", arm, "t%03d_%s_%s" % (t["i"], cat, tid))
            arec, src = read_json(os.path.join(ad, "record.json"), None), os.path.join(ad, "served.xlsx")
            if not arec or not os.path.exists(src):
                res[key] = {"missing": True, "accuracy": 0, "modification_accuracy": 0, "regression_accuracy": 0}
                continue
            if C.sha_file(src) != arec.get("served_sha"):
                raise SystemExit("%s: archive hash mismatch -- refusing to evaluate" % key)
            od = h.p(
                "evalcopy", "outputs", arm, cat, tid
            )  # one directory per task: the checker recalculates only this file
            shutil.rmtree(od, ignore_errors=True)
            os.makedirs(od)
            shutil.copyfile(src, os.path.join(od, "%s_output.xlsx" % tid))
            ok, note = h.recalc_fn(od)
            if not ok:
                raise SystemExit("[ucoe-agent official] recalculation of %s failed: %s" % (key, note))
            item, _ = ev.process_single_item(rec, ec, od, {}, cat)
            res[key] = {k: item.get(k) for k in ("accuracy", "modification_accuracy", "regression_accuracy")}
        atomic_write(h.p("official.json"), json.dumps(res, indent=1, default=str))
    print("[ucoe-agent official] %d entries" % len(res), flush=True)
    return res


def report(h):
    """Task accuracy per arm and category, paired differences with a bootstrap interval, harness statistics. Refuses
    incomplete arms."""
    man = h.manifest()
    res = read_json(h.p("official.json"), None) or {}
    arms = tuple(h.arms)
    ids = ["%s|%s" % (t["category"], t["id"]) for t in man["deployment_tasks"]]
    missing = [
        "%s/%s" % (a, k)
        for a in arms
        for k in ids
        if not isinstance(res.get("%s/%s" % (a, k)), dict)
        or res["%s/%s" % (a, k)].get("missing")
        or res["%s/%s" % (a, k)].get("accuracy") is None
    ]
    if missing:
        raise SystemExit(
            "[ucoe-agent report] %d of %d official results missing (e.g. %s); report refuses to compare incomplete "
            "arms -- run `official` after every arm is sealed" % (len(missing), len(ids) * len(arms), missing[:3])
        )
    acc = {a: [float(res["%s/%s" % (a, k)]["accuracy"]) for k in ids] for a in arms}
    rng = random.Random(42)

    def paired(a, b):
        d = [x - y for x, y in zip(acc[a], acc[b])]
        W, Ls = sum(1 for x in d if x > 0), sum(1 for x in d if x < 0)
        bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(5000)) if d else [0, 0]
        return {
            "delta": round(sum(d) / max(1, len(d)), 4),
            "ci95": [round(bs[125], 3), round(bs[4874], 3)],
            "W_L_T": [W, Ls, len(d) - W - Ls],
        }

    pairs = [(a, b) for a, b in (("ucoe", "native"), ("ucoe", "judge"), ("judge", "native")) if a in arms and b in arms]
    out = {
        "label": man["label"],
        "backend": h.backend_name,
        "n": len(ids),
        "task_accuracy": {a: round(sum(acc[a]) / max(1, len(ids)), 4) for a in arms},
        "paired": {"%s_vs_%s" % (a, b): paired(a, b) for a, b in pairs},
        "by_category": {
            c: {
                a: round(
                    sum(x for x, k in zip(acc[a], ids) if k.startswith(c))
                    / max(1, sum(1 for k in ids if k.startswith(c))),
                    4,
                )
                for a in arms
            }
            for c in CATS
        },
        "arms": {},
        "usage": {a: read_json(h.p("state", "usage_%s.json" % a), None) or {} for a in arms + ("calib",)},
    }
    for a in arms:
        st = read_json(h.p("state", "%s.json" % a), None) or {"rows": []}
        out["arms"][a] = {
            "statuses": dict(collections.Counter(r["status"] for r in st["rows"])),
            "H_entries": len(h.h_load(a)),
            "V_lineage": st.get("lineage_V"),
            "checkpoints": {k: v.get("decision") for k, v in (st.get("ckpt") or {}).items()},
            "incidents": st.get("incidents", []),
            "h0": (st.get("h0") or {}).get("decision"),
        }
    atomic_write(h.p("report.json"), json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str), flush=True)
    return out
