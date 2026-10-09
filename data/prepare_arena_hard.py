"""Prepare Arena-Hard (paper App. "Benchmarks and dataset splits": 500 hard_prompt tasks in official
order; calibration from a frozen prompt-only pool of coding and math development prompts).

Usage: python -m data.prepare_arena_hard --config config.yaml [--from-frozen | --from-source]

Official source: Hugging Face ``lmarena-ai/arena-hard-auto``, file ``data/arena-hard-v2.0/question.jsonl``
at the pinned revision, filtered to ``category == "hard_prompt"`` (253 coding / 247 math; the 250
creative_writing rows are excluded). Deployment exposes only {uid, prompt}; category/subcategory/
language live in the evaluation sidecar.

Frozen bundle (``data/frozen/arena_hard/``):
    public.json.gz   {"revision", "file_sha256", "items": [{"i", "uid", "prompt"}]}        500 items
    sidecar.json.gz  {uid: {"category", "subcategory", "language"}}
    pool.json.gz     [{"id", "prompt"}]  prompt-only calibration sources, ids ``code_*`` (HumanEval+/MBPP+
                     instructions) and ``math_*`` (competition-math problems), content-deduplicated and
                     disjoint from deployment by prompt sha256

Processed output: tasks.jsonl (task = prompt), calibration.jsonl (domain = code | math),
eval_sidecar.json.gz, manifest.json. ``--from-source`` downloads the pinned question file and
rebuilds the pool from the sibling frozen bundles (bigcodebench pool + hard_math battery pool) when
they are present, else from the frozen Arena pool.
"""
from __future__ import annotations

import os
from collections import Counter

from data.prepare import (choose_mode, file_sha, frozen_dir, hf_download, processed_dir, read_gz, run_cli,
                          sha, source_row, task_row, write_outputs)

NAME = "arena_hard"
ARENA_REPO = "lmarena-ai/arena-hard-auto"
ARENA_FILE = "data/arena-hard-v2.0/question.jsonl"
ARENA_REV = "15f3746e21432264ce9b453999bde4f3c946d2e6"
ARENA_SHA = "6cfe75abb0e09cd39e7f9b0b18ea96d41be2d4d9ddf76dc1d07d78b6accb2039"
N_SRC = 500
SUBCATS = {"coding": 253, "math": 247}
N_CREATIVE = 250
MAX_PROMPT_CHARS = 90000


def _domain(source_id):
    return "code" if source_id.startswith("code_") else "math" if source_id.startswith("math_") else None


def _pool_from_siblings(cfg, dep_shas):
    """Rebuild the calibration pool from the sibling frozen bundles (prompt-only fields)."""
    bcb = os.path.join(frozen_dir(cfg, "bigcodebench"), "pool.json.gz")
    math = os.path.join(frozen_dir(cfg, "hard_math"), "public.json.gz")
    if not (os.path.exists(bcb) and os.path.exists(math)):
        return None
    pool, seen = [], set()
    for r in read_gz(bcb):
        h = sha(r["instruction"])
        if h not in seen:
            seen.add(h)
            pool.append({"id": f"code_{r['id']}", "prompt": r["instruction"]})
    for r in read_gz(math)["battery_pool"]:
        h = sha(r["problem"])
        if h not in seen:
            seen.add(h)
            pool.append({"id": f"math_{r['id']}", "prompt": r["problem"]})
    return [r for r in pool if sha(r["prompt"]) not in dep_shas]


def load_frozen(cfg):
    fz = frozen_dir(cfg, NAME)
    pub = read_gz(os.path.join(fz, "public.json.gz"))
    if pub.get("revision") != ARENA_REV or pub.get("file_sha256") != ARENA_SHA or len(pub["items"]) != N_SRC:
        raise RuntimeError("frozen Arena-Hard public bundle does not match the pinned release")
    side = read_gz(os.path.join(fz, "sidecar.json.gz"))
    pool = read_gz(os.path.join(fz, "pool.json.gz"))
    inputs = {k: file_sha(os.path.join(fz, f"{k}.json.gz")) for k in ("public", "sidecar", "pool")}
    return pub["items"], side, pool, inputs


def load_source(cfg):
    p = hf_download(ARENA_REPO, ARENA_FILE, ARENA_REV)
    raw = open(p, "rb").read()
    got = sha_bytes(raw)
    if got != ARENA_SHA:
        raise RuntimeError(f"question.jsonl sha256 {got[:12]} != pinned {ARENA_SHA[:12]}")
    import json
    rows = [json.loads(l) for l in raw.decode("utf-8").split("\n") if l.strip()]
    hard = [r for r in rows if r.get("category") == "hard_prompt"]
    sub = Counter(r.get("subcategory") for r in hard)
    if len(hard) != N_SRC or len({r["uid"] for r in hard}) != N_SRC or any(sub.get(k) != v for k, v in SUBCATS.items()):
        raise RuntimeError(f"hard_prompt contract violated: {len(hard)} rows, {dict(sub)}")
    if sum(1 for r in rows if r.get("category") == "creative_writing") != N_CREATIVE:
        raise RuntimeError("creative_writing row count drifted from the pinned release")
    for r in hard:
        if len(r["prompt"]) > MAX_PROMPT_CHARS:
            raise RuntimeError(f"prompt too long for the context guard: {r['uid']}")
    items = [{"i": i, "uid": r["uid"], "prompt": r["prompt"]} for i, r in enumerate(hard)]
    side = {r["uid"]: {"category": r["category"], "subcategory": r.get("subcategory"), "language": r.get("language")}
            for r in hard}
    dep_shas = {sha(t["prompt"]) for t in items}
    pool = _pool_from_siblings(cfg, dep_shas)
    pool_origin = "sibling frozen bundles (bigcodebench pool + hard_math battery pool)"
    if pool is None:
        fp = os.path.join(frozen_dir(cfg, NAME), "pool.json.gz")
        if not os.path.exists(fp):
            raise RuntimeError("no calibration pool available: add data/frozen/arena_hard/pool.json.gz or the "
                               "bigcodebench / hard_math frozen bundles")
        pool = read_gz(fp)
        pool_origin = "frozen Arena-Hard pool"
    return items, side, pool, {"question_jsonl": got, "pool_origin": pool_origin}


def sha_bytes(raw):
    import hashlib
    return hashlib.sha256(raw).hexdigest()


def main(cfg, mode=None, out_dir=None):
    out = processed_dir(cfg, NAME, out_dir)
    mode = choose_mode(mode, os.path.join(frozen_dir(cfg, NAME), "public.json.gz"))
    items, side, pool, inputs = load_frozen(cfg) if mode == "frozen" else load_source(cfg)
    tasks = [task_row(i, t["uid"], t["prompt"]) for i, t in enumerate(items)]
    dep = {sha(t["task"]) for t in tasks}
    seen, sources = set(), []
    for r in pool:
        h = sha(r["prompt"])
        if h in seen or h in dep:
            continue
        seen.add(h)
        sources.append(source_row(r["id"], r["prompt"], domain=_domain(r["id"])))
    sidecar = {t["task_id"]: side[t["task_id"]] for t in tasks}
    manifest = {"built_from": mode,
                "source": {"repo": ARENA_REPO, "file": ARENA_FILE, "revision": ARENA_REV, "file_sha256": ARENA_SHA,
                           "filter": "category == hard_prompt (official order)"},
                "order": "official",
                "calibration": {"origin": "prompt-only coding (HumanEval+/MBPP+) and competition-math development "
                                          "prompts, content-deduplicated, disjoint from deployment",
                                "domains": dict(Counter(s["domain"] for s in sources))},
                "inputs": inputs}
    return write_outputs(out, NAME, tasks, sources, sidecar, manifest)


if __name__ == "__main__":
    run_cli(main, NAME)
