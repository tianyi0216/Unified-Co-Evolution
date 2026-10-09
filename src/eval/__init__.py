"""Final evaluation of UCoE runs (paper App. "Evaluation"; Table "Native metric and scorer").

This package is the only part of the release that opens evaluation-only data: the benchmark's
``eval_sidecar.json.gz`` (reference answers, checklists, rubrics, tests) and the final judge
model. The method path never reads either. The helpers here load the served responses of every
arm from ``<work_dir>/<arm>/rows.jsonl`` (fields ``i``, ``task_id``, ``final``, ``final_sha256``;
the UCoE arm lives in ``<work_dir>/ucoe`` and Direct Generation in ``<work_dir>/raw``) and restrict
the frozen task stream to the evaluated window, so the preference score and every native metric
are computed on exactly the same tasks.
"""
from __future__ import annotations

import os

from src.utils.config import cfg_get
from src.utils.io import ensure_dir, read_jsonl, sha

ROWS_FILE = "rows.jsonl"
EVAL_SUBDIR = "eval"


def eval_tasks(cfg, bench):
    """Deployment tasks in the evaluated window: the first ``benchmark.n_tasks`` tasks of the frozen
    stream, narrowed to the adapter's ``eval_range`` (a ``(start, end)`` pair of stream indices)."""
    tasks = [dict(t, task_id=str(t["task_id"])) for t in bench.load_tasks()]
    n = cfg_get(cfg, "benchmark.n_tasks")
    if n:
        tasks = tasks[: int(n)]
    window = getattr(bench, "eval_range", None)
    if window:
        lo, hi = int(window[0]), int(window[1])
        tasks = [t for t in tasks if lo <= int(t["i"]) < hi]
    return tasks


def eval_dir(work_dir):
    """``<work_dir>/eval``, created on demand; every evaluation artifact lives here."""
    path = os.path.join(work_dir, EVAL_SUBDIR)
    ensure_dir(path)
    return path


def load_arm(work_dir, arm):
    """Served texts of one arm keyed by task id. Each row's text must match its recorded sha256, so a
    corrupted or hand-edited rows file is refused rather than scored."""
    path = os.path.join(work_dir, arm, ROWS_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"[eval] arm '{arm}' has no {path}; run the pipeline for that arm first")
    texts = {}
    for row in read_jsonl(path):
        final = row.get("final") or ""
        if row.get("final_sha256") and sha(final) != row["final_sha256"]:
            raise ValueError(f"[eval] {arm}/{row.get('task_id')}: final text does not match final_sha256")
        texts[str(row["task_id"])] = final
    return texts


def load_arms(cfg, bench, work_dir, arms):
    """Tasks and texts shared by every arm: ``(tasks, texts, missing)`` where ``texts[arm][task_id]`` is
    the served response and ``missing[arm]`` counts evaluated tasks the arm has not produced. Tasks
    absent from any arm are dropped for all arms so that every comparison stays paired."""
    tasks = eval_tasks(cfg, bench)
    texts = {arm: load_arm(work_dir, arm) for arm in arms}
    missing = {arm: sum(1 for t in tasks if t["task_id"] not in texts[arm]) for arm in arms}
    common = [t for t in tasks if all(t["task_id"] in texts[a] for a in arms)]
    if not common:
        raise RuntimeError(f"[eval] no evaluated task has rows in every arm {list(arms)} (missing {missing})")
    if len(common) < len(tasks):
        print(f"[eval] {len(tasks) - len(common)}/{len(tasks)} evaluated tasks lack rows in some arm "
              f"(missing per arm {missing}); scoring the {len(common)} tasks present in every arm", flush=True)
    keep = {t["task_id"] for t in common}
    texts = {a: {k: v for k, v in texts[a].items() if k in keep} for a in arms}
    return common, texts, missing


def judge_client(cfg, work_dir, tag):
    """Cache-first client for the final judge model (``models.judge`` through the configured backend).
    Its request ledger and cost meter live under ``<work_dir>/eval`` and never mix with the method's
    own ledger."""
    from src.llm.base import make_transport
    from src.llm.client import Client, Meter
    d = eval_dir(work_dir)
    meter = Meter(os.path.join(d, f"cost_meter_{tag}.json"), cfg_get(cfg, "output.ceilings"))
    client = Client(os.path.join(d, f"llm_cache_{tag}.jsonl"), make_transport(cfg["models"]), meter, cfg["models"])
    return client, meter
