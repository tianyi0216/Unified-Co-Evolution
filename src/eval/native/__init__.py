"""Native benchmark metrics (paper App. "Evaluation", paragraph "Benchmark-specific metrics", Table
"Native metric and scorer").

Each wrapper in this package reproduces one benchmark's own scoring procedure on the served responses
of every arm and exposes ``evaluate(cfg, bench, work_dir, arms) -> dict``, writing
``<work_dir>/eval/native_<benchmark>.json``. Evaluation-only fields (reference answers, checklists,
rubrics, tests) are read from ``eval_sidecar.json.gz`` in the benchmark data directory and never from
the method path. This module holds the registry (``get_native_evaluator``) and the pieces the wrappers
share: sidecar loading, the deterministic retry schedule for judge calls, per-task paired statistics
against the Direct Generation baseline (task-resampling bootstrap interval, exact two-sided sign test
over decisive tasks, Holm correction over the arms) and slice means.
"""
from __future__ import annotations

import importlib
import os
import random

from src.eval import eval_dir
from src.utils.config import cfg_get
from src.utils.io import read_gz_json, seed_from, write_json
from src.utils.stats import holm, sign_test_p
from src.utils.text import extract_json

SIDECAR_FILE = "eval_sidecar.json.gz"
EVALUATORS = {
    "arena_hard": "src.eval.native.arena_hard",
    "ifbench": "src.eval.native.ifbench",
    "hard_math": "src.eval.native.hard_math",
    "bigcodebench": "src.eval.native.bigcodebench",
    "wildbench_score": "src.eval.native.wildbench_score",
    "multichallenge_rubric": "src.eval.native.multichallenge_rubric",
    "refinebench_checklist": "src.eval.native.refinebench_checklist",
    "writingbench_critic": "src.eval.native.writingbench_critic",
}
BENCHMARK_EVALUATOR = {
    "arena_hard": "arena_hard",
    "ifbench": "ifbench",
    "hard_math": "hard_math",
    "bigcodebench": "bigcodebench",
    "bigcodebench_hard": "bigcodebench",
    "wildbench": "wildbench_score",
    "multichallenge": "multichallenge_rubric",
    "refinebench": "refinebench_checklist",
    "writingbench": "writingbench_critic",
}


class NativeEvaluator:
    """One native metric. ``evaluate`` accepts ``(cfg, bench, work_dir, arms)`` and also the adapter-side
    form ``(cfg, rows_by_arm)``, where ``rows_by_arm`` maps arm names to their rows and the benchmark and
    work directory come from the config."""

    def __init__(self, name, module):
        self.name = name
        self.module = module

    def evaluate(self, cfg, bench, work_dir=None, arms=None):
        if isinstance(bench, dict):
            from src.benchmarks.registry import get_benchmark
            arms, bench = list(bench), get_benchmark(cfg)
        work_dir = work_dir or cfg_get(cfg, "output.work_dir")
        return self.module.evaluate(cfg, bench, work_dir, list(arms or ("raw", "ucoe")))


def get_native_evaluator(name):
    """Evaluator for a native metric name (``'ifbench'``, ``'wildbench_score'``, ...); None when unknown."""
    key = str(name or "").strip().lower()
    if key not in EVALUATORS:
        return None
    return NativeEvaluator(key, importlib.import_module(EVALUATORS[key]))


def native_for_benchmark(bench_name):
    """The paper's native metric for a benchmark name, or None (WritingPrompts has none; Arena-Hard's entry
    exports the served answers for the official scorer, which is not part of this release)."""
    return get_native_evaluator(BENCHMARK_EVALUATOR.get(str(bench_name or "").strip().lower()))


def baseline_of(arms):
    """The Direct Generation arm when present, else the first arm."""
    return "raw" if "raw" in arms else arms[0]


def boot_cfg(cfg):
    return {"reps": int(cfg_get(cfg, "evaluation.bootstrap.reps", 50000)),
            "seed": int(cfg_get(cfg, "evaluation.bootstrap.seed", 42))}


def load_sidecar(cfg):
    """Evaluation-only fields keyed by task id (a mapping, or a list of records carrying ``task_id``)."""
    path = os.path.join(cfg_get(cfg, "benchmark.data_dir"), SIDECAR_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"[eval] evaluation sidecar missing: {path} (run process_data.sh)")
    side = read_gz_json(path)
    if isinstance(side, list):
        side = {str(r["task_id"]): r for r in side}
    return {str(k): v for k, v in side.items()}


def write_results(work_dir, bench_name, results):
    path = os.path.join(eval_dir(work_dir), f"native_{bench_name}.json")
    write_json(path, results, indent=1)
    print(f"[eval] wrote {path}", flush=True)
    return path


def judge_rows(client, meter, role, model, system, user, kw, key, attempts):
    """Deterministic retry schedule for a judge call: attempts 0 and 1 reuse the base seed under a new
    request nonce; later attempts also vary the seed so deterministic decoding can produce a different
    output. Yields ``(attempt, row)`` and counts every retry on the meter."""
    base = seed_from("eval", "call", key)
    for attempt in range(attempts):
        if attempt:
            meter.syntax_retry()
        k = dict(kw)
        k["seed"] = base if attempt < 2 else base + attempt
        yield attempt, client.call(role, model, system, user, k, attempt)


def judge_verdict(client, meter, role, model, system, user, kw, key, attempts, validate):
    """Validated JSON verdict from the judge, or None when no attempt produced one."""
    for _, row in judge_rows(client, meter, role, model, system, user, kw, key, attempts):
        obj = extract_json(row.get("text") or "")
        if isinstance(obj, dict):
            try:
                v = validate(obj)
            except (TypeError, KeyError, ValueError):
                v = None
            if v is not None:
                return v
    return None


def boot_ci(d, reps, seed, digits=4, scale=1.0):
    """Percentile 95% interval of the mean of ``d`` under task resampling with replacement."""
    rng = random.Random(seed)
    n = len(d)
    boots = sorted(sum(rng.choices(d, k=n)) / n for _ in range(reps))
    return [round(scale * boots[int(0.025 * reps)], digits), round(scale * boots[int(0.975 * reps) - 1], digits)]


def paired_vs_baseline(scores, texts, ids, arms, baseline, boot, scale=1.0, digits=4):
    """Per-arm paired comparison against the baseline on the same tasks: mean delta (scaled), bootstrap
    CI95, W/L/T over decisive tasks, exact sign test (the exact McNemar test for binary scores), changed
    output counts and the delta over changed outputs only; Holm correction over the arms' sign tests.
    Returns ``{"<arm>_vs_<baseline>": {...}}``."""
    out, pvals = {}, {}
    for arm in arms:
        if arm == baseline:
            continue
        d = [scores[arm][t] - scores[baseline][t] for t in ids]
        wins = sum(1 for x in d if x > 0)
        losses = sum(1 for x in d if x < 0)
        changed = [t for t in ids if texts[arm][t] != texts[baseline][t]]
        dc = [scores[arm][t] - scores[baseline][t] for t in changed]
        name = f"{arm}_vs_{baseline}"
        pvals[name] = round(sign_test_p(wins, losses), 6)
        out[name] = {"delta": round(scale * sum(d) / len(d), digits),
                     "ci95": boot_ci(d, boot["reps"], boot["seed"], digits, scale),
                     "W_L_T": [wins, losses, len(d) - wins - losses], "sign_p": pvals[name],
                     "changed_n": len(changed), "output_change_rate": round(len(changed) / len(ids), 4),
                     "changed_only_delta": round(scale * sum(dc) / len(dc), digits) if dc else 0.0}
    if pvals:
        for name, p in holm(pvals).items():
            out[name]["holm_p"] = round(p, 6)
    return out


def slice_means(values, ids, key_of, scale=1.0, digits=4):
    """Mean of ``values[task_id]`` per slice label ``key_of(task_id)``, in sorted label order."""
    groups = {}
    for t in ids:
        groups.setdefault(key_of(t), []).append(values[t])
    return {k: round(scale * sum(v) / len(v), digits) for k, v in sorted(groups.items())}
