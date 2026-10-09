"""BigCodeBench-Hard native metric (paper Table "Native metric and scorer": execution harness, Pass@1;
Table "Verifiable tasks with Qwen3-8B").

The last fenced code block of each served response (or the whole response when unfenced) is executed
together with the official test of the task in an isolated subprocess: a fresh temporary working
directory, a minimal environment and a hard timeout. A task passes when the unittest run succeeds.
Tests, entry points and canonical solutions come from the evaluation sidecar and never reach the method
path. Per-task outcomes are cached by solution content under ``<work_dir>/eval`` so re-runs and identical
outputs across arms share one execution. ``evaluation.native.bcb_canonical_check`` additionally runs
every canonical solution first to validate the interpreter environment (failures are reported, never
used to drop tasks).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

from src.eval import eval_dir, load_arms
from src.eval.native import baseline_of, boot_cfg, load_sidecar, paired_vs_baseline, write_results
from src.llm.client import pmap
from src.utils.config import cfg_get
from src.utils.io import read_json, sha, write_json

TIMEOUT = 120
HARNESS = (
    "\n\nif __name__ == '__main__':\n"
    "    import unittest, sys\n"
    "    _r = unittest.main(exit=False, argv=['x']).result\n"
    "    sys.exit(0 if _r.wasSuccessful() else 1)\n")
CACHE_FILE = "bigcodebench_exec.json"
BATCH = 32


def extract_code(text):
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return (m[-1] if m else text).strip()


def run_one(solution, test, python=None, timeout=TIMEOUT):
    """Execute ``solution + test`` under ``python`` in a throwaway directory; returns (passed, stderr tail)."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "prog.py")
        with open(p, "w", encoding="utf-8") as f:
            f.write(solution + "\n\n" + test + HARNESS)
        try:
            r = subprocess.run([python or sys.executable, p], cwd=d, capture_output=True, text=True,
                               timeout=timeout,
                               env={"PATH": os.environ.get("PATH", ""), "HOME": d, "MPLBACKEND": "Agg",
                                    "PYTHONDONTWRITEBYTECODE": "1"})
            ok = r.returncode == 0
            err = (r.stderr or "")[-600:]
        except subprocess.TimeoutExpired:
            ok, err = False, "TIMEOUT"
    return ok, err


def settings(cfg):
    """``(python, timeout, workers)`` for the executor from ``evaluation.native``."""
    return (cfg_get(cfg, "evaluation.native.bcb_python") or sys.executable,
            int(cfg_get(cfg, "evaluation.native.bcb_timeout", TIMEOUT)),
            int(cfg_get(cfg, "evaluation.native.bcb_workers", 6)))


def execute(jobs, cache, cache_path, python, timeout, workers, label):
    """Run the jobs ``{key: (solution, test)}`` missing from ``cache``; persists after every batch when a
    cache path is given. Returns the cache."""
    todo = sorted(k for k in jobs if k not in cache)

    def one(key):
        ok, err = run_one(jobs[key][0], jobs[key][1], python, timeout)
        return key, {"pass": ok, "err": None if ok else err}

    for j in range(0, len(todo), BATCH):
        cache.update(dict(pmap(one, todo[j:j + BATCH], workers, label, every=BATCH)))
        if cache_path:
            write_json(cache_path, cache)
    return cache


def evaluate(cfg, bench, work_dir, arms):
    tasks, texts, missing = load_arms(cfg, bench, work_dir, arms)
    side = load_sidecar(cfg)
    python, timeout, workers = settings(cfg)
    ids = [t["task_id"] for t in tasks]
    baseline = baseline_of(arms)
    extract = getattr(bench, "extract_code", extract_code)
    results = {"label": "BigCodeBench-Hard Pass@1: served solution + official test executed in an isolated "
                        "subprocess with a hard timeout",
               "n": len(ids), "executor": python, "timeout_s": timeout, "baseline": baseline,
               "missing_rows": missing}
    if cfg_get(cfg, "evaluation.native.bcb_canonical_check", False):
        canon = execute({tid: ((side[tid].get("code_prompt") or "") + "\n" + side[tid]["canonical_solution"],
                               side[tid]["test"]) for tid in ids}, {}, None, python, timeout, workers,
                        "bcb_canonical")
        fails = {k: v["err"] for k, v in canon.items() if not v["pass"]}
        results["canonical_check"] = {"pass": len(ids) - len(fails), "fail": fails}
        print(f"[eval] bigcodebench canonical solutions pass {len(ids) - len(fails)}/{len(ids)} under {python}",
              flush=True)
    cache_path = os.path.join(eval_dir(work_dir), CACHE_FILE)
    cache = read_json(cache_path, default={}) or {}
    jobs, slots = {}, {}
    for arm in arms:
        for tid in ids:
            code = extract(texts[arm][tid])
            key = f"{tid}|{sha(code)[:16]}"
            jobs[key] = (code, side[tid]["test"])
            slots[(arm, tid)] = key
    print(f"[eval] bigcodebench: {len(jobs)} unique solutions across {len(arms)} arms "
          f"({sum(1 for k in jobs if k in cache)} cached)", flush=True)
    execute(jobs, cache, cache_path, python, timeout, workers, "bcb_exec")
    scores = {arm: {tid: float(cache[slots[(arm, tid)]]["pass"]) for tid in ids} for arm in arms}
    for arm in arms:
        results[f"pass1_{arm}"] = round(100 * sum(scores[arm].values()) / len(ids), 2)
        results[f"timeouts_{arm}"] = sum(1 for tid in ids if cache[slots[(arm, tid)]]["err"] == "TIMEOUT")
        print(f"[eval] bigcodebench {arm}: pass@1 {results[f'pass1_{arm}']} (timeouts {results[f'timeouts_{arm}']})",
              flush=True)
    results.update(paired_vs_baseline(scores, texts, ids, arms, baseline, boot_cfg(cfg), scale=100.0, digits=2))
    write_results(work_dir, bench.name, results)
    return results
