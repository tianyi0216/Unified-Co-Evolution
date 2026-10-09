"""Prepare WildBench (paper App. "Benchmarks and dataset splits": 1,024 real-user conversations in
official order; calibration from a frozen mixed pool of external conversations).

Usage: python -m data.prepare_wildbench --config config.yaml [--from-frozen | --from-source]

Official source: Hugging Face ``allenai/WildBench``, config v2, test split (``v2/test-00000-of-00001.parquet``
at the pinned revision), exactly 1,024 unique conversations. The public view keeps only the
conversation turns ({role, content}); checklist, references, intent, tags and moderation metadata
live in the evaluation sidecar.

Frozen bundle (``data/frozen/wildbench/``):
    public.json.gz   {"revision", "file_sha256", "items": [{"i", "task_id", "conv": [{"role", "content"}]}]}
    sidecar.json.gz  {task_id: {"checklist", "references", "intent", "primary_tag", "secondary_tags",
                                "toxic", "redacted", "turn_meta"}}
    pool.json.gz     [{"id", "conv"}]  6,940 prompt-only external conversations: ids ``hb_*`` (health
                     assistant conversations) and ``ar_*`` (general single-turn requests), deduplicated by
                     the sha256 of the rendered conversation and disjoint from deployment

Processed output: tasks.jsonl (task = role-tagged rendering, messages = native turns),
calibration.jsonl (domain = hb | ar), eval_sidecar.json.gz, manifest.json. ``--from-source`` downloads the
pinned parquet and keeps the calibration pool from the frozen bundle (its conversations have no other
public source in this repository).
"""
from __future__ import annotations

import os
from collections import Counter

from data.prepare import (choose_mode, file_sha, frozen_dir, hf_download, processed_dir, read_gz,
                          read_parquet_rows, run_cli, sha, source_row, task_row, write_outputs)
from src.benchmarks.base import render_convo

NAME = "wildbench"
WB_REPO = "allenai/WildBench"
WB_FILE = "v2/test-00000-of-00001.parquet"
WB_REV = "26c49eb39d7d5ce2099b0bbafed5a88dcce954ec"
WB_SHA = "e1fbdc5d46deca34e77f92991c23b0476860e26adf2bf28379a3bb3bf00d3ed6"
N_SRC = 1024
MAX_CONV_CHARS = 90000
SIDE_FIELDS = ("checklist", "references", "intent", "primary_tag", "secondary_tags", "toxic", "redacted")


def _domain(source_id):
    return source_id.split("_", 1)[0] if "_" in source_id else None


def _pool(cfg, dep_shas):
    fp = os.path.join(frozen_dir(cfg, NAME), "pool.json.gz")
    if not os.path.exists(fp):
        raise RuntimeError("calibration pool missing: data/frozen/wildbench/pool.json.gz")
    seen, sources = set(), []
    for r in read_gz(fp):
        conv = [{"role": m["role"], "content": m["content"]} for m in r["conv"]]
        text = render_convo(conv)
        h = sha(text)
        if h in seen or h in dep_shas:
            continue
        seen.add(h)
        sources.append(source_row(r["id"], text, messages=conv, domain=_domain(r["id"])))
    return sources, file_sha(fp)


def load_frozen(cfg):
    fz = frozen_dir(cfg, NAME)
    pub = read_gz(os.path.join(fz, "public.json.gz"))
    if pub.get("revision") != WB_REV or pub.get("file_sha256") != WB_SHA or len(pub["items"]) != N_SRC:
        raise RuntimeError("frozen WildBench public bundle does not match the pinned release")
    side = read_gz(os.path.join(fz, "sidecar.json.gz"))
    inputs = {k: file_sha(os.path.join(fz, f"{k}.json.gz")) for k in ("public", "sidecar")}
    return pub["items"], side, inputs


def load_source(cfg):
    p = hf_download(WB_REPO, WB_FILE, WB_REV)
    got = file_sha(p)
    if got != WB_SHA:
        raise RuntimeError(f"parquet sha256 {got[:12]} != pinned {WB_SHA[:12]}")
    rows = read_parquet_rows(p)
    if len(rows) != N_SRC:
        raise RuntimeError(f"rows {len(rows)} != {N_SRC}")
    ids = [r.get("session_id") or r["id"] for r in rows]
    if len(set(ids)) != N_SRC:
        raise RuntimeError("conversation ids are not unique")
    items, side = [], {}
    for i, r in enumerate(rows):
        conv = [{"role": m["role"], "content": m["content"]} for m in r["conversation_input"]]
        if sum(len(m["content"]) for m in conv) > MAX_CONV_CHARS:
            raise RuntimeError(f"{ids[i]}: conversation exceeds the context guard; refusing silent truncation")
        items.append({"i": i, "task_id": ids[i], "conv": conv})
        side[ids[i]] = {k: r.get(k) for k in SIDE_FIELDS}
        side[ids[i]]["turn_meta"] = [{k: m.get(k) for k in ("language", "redacted", "toxic", "timestamp")}
                                     for m in r["conversation_input"]]
    return items, side, {"parquet_sha256": got}


def main(cfg, mode=None, out_dir=None):
    out = processed_dir(cfg, NAME, out_dir)
    mode = choose_mode(mode, os.path.join(frozen_dir(cfg, NAME), "public.json.gz"))
    items, side, inputs = load_frozen(cfg) if mode == "frozen" else load_source(cfg)
    tasks = [task_row(i, t["task_id"], render_convo(t["conv"]), messages=t["conv"]) for i, t in enumerate(items)]
    sources, pool_sha = _pool(cfg, {sha(t["task"]) for t in tasks})
    inputs["pool"] = pool_sha
    sidecar = {t["task_id"]: side[t["task_id"]] for t in tasks}
    manifest = {"built_from": mode,
                "source": {"repo": WB_REPO, "file": WB_FILE, "revision": WB_REV, "file_sha256": WB_SHA},
                "order": "official",
                "calibration": {"origin": "mixed external conversations (health-assistant and general-request "
                                          "pools), deduplicated and disjoint from deployment",
                                "domains": dict(Counter(s["domain"] for s in sources))},
                "inputs": inputs}
    return write_outputs(out, NAME, tasks, sources, sidecar, manifest)


if __name__ == "__main__":
    run_cli(main, NAME)
