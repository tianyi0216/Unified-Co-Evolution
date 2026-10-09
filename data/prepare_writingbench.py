"""Prepare WritingBench (paper App. "Benchmarks and dataset splits": the 1,000 official queries are
split by hash into a 500-query calibration pool and a 500-task deployment stream).

Usage: python -m data.prepare_writingbench --config config.yaml [--from-frozen | --from-source]

Official source: GitHub ``X-PLUG/WritingBench`` at the pinned commit, ``benchmark_query/benchmark_all.jsonl``
(1,000 rows, unique ``index``) plus the style/format/length requirement subsets. The model-visible view
is the query text only (user materials are embedded in it); the five per-task scoring criteria,
domains, language and subset membership live in the evaluation sidecar.

Frozen bundle (``data/frozen/writingbench/``):
    public.json.gz   {"commit", "jsonl_sha16", "items": [{"i", "task_id", "query"}]}   1,000 items
    sidecar.json.gz  {task_id: {"checklist": [{"name", "criteria_description", "1-2", "3-4", "5-6", "7-8",
                                "9-10"}], "domain1", "domain2", "lang", "subsets": {"style", "style_C",
                                "format", "format_C", "length", "length_C"}}}

Split (sealed before any generation): ids ordered by sha256("<jsonl_sha16>|<salt>|<index>");
the first 500 are calibration sources and the remaining 500 the deployment stream in that order.

Processed output: tasks.jsonl (task = query), calibration.jsonl, eval_sidecar.json.gz, manifest.json.
"""
from __future__ import annotations

import hashlib
import json
import os

from data.prepare import (choose_mode, file_sha, frozen_dir, http_bytes, processed_dir, read_gz, run_cli, sha,
                          source_row, task_row, write_outputs)

NAME = "writingbench"
WB_REPO = "X-PLUG/WritingBench"
WB_COMMIT = "ae2d5176449b7b769815482641d35926f26793eb"
WB_JSONL_SHA16 = "18fee37c645166eb"
WB_JSONL = "benchmark_query/benchmark_all.jsonl"
RAW_BASE = f"https://raw.githubusercontent.com/{WB_REPO}/{WB_COMMIT}/"
N_SRC = 1000
N_CAL, N_DEP = 500, 500
SPLIT_SALT = "matchedbank_v2"


def split_key(task_id):
    return sha(f"{WB_JSONL_SHA16}|{SPLIT_SALT}|{task_id}")


def load_frozen(cfg):
    fz = frozen_dir(cfg, NAME)
    pub = read_gz(os.path.join(fz, "public.json.gz"))
    if pub.get("commit") != WB_COMMIT or pub.get("jsonl_sha16") != WB_JSONL_SHA16 or len(pub["items"]) != N_SRC:
        raise RuntimeError("frozen WritingBench public bundle does not match the pinned release")
    side = read_gz(os.path.join(fz, "sidecar.json.gz"))
    inputs = {k: file_sha(os.path.join(fz, f"{k}.json.gz")) for k in ("public", "sidecar")}
    return pub["items"], side, inputs


def load_source(cfg):
    raw = http_bytes(RAW_BASE + WB_JSONL)
    h16 = hashlib.sha256(raw).hexdigest()[:16]
    if h16 != WB_JSONL_SHA16:
        raise RuntimeError(f"benchmark_all.jsonl sha256[:16] {h16} != pinned {WB_JSONL_SHA16}")
    rows = [json.loads(ln) for ln in raw.decode("utf-8").split("\n") if ln.strip()]
    if len(rows) != N_SRC or len({r["index"] for r in rows}) != N_SRC:
        raise RuntimeError("row count / unique index mismatch")
    member = {}
    for cat in ("style", "format", "length"):
        for variant in ("", "_C"):
            body = http_bytes(RAW_BASE + f"benchmark_query/requirement/{cat}/{cat}_subset{variant}.jsonl")
            member[f"{cat}{variant}"] = {json.loads(ln)["index"] for ln in body.decode("utf-8").split("\n")
                                         if ln.strip()}
    items, side = [], {}
    for i, r in enumerate(rows):
        items.append({"i": i, "task_id": str(r["index"]), "query": r["query"]})
        side[str(r["index"])] = {"checklist": r["checklist"], "domain1": r["domain1"], "domain2": r["domain2"],
                                 "lang": r["lang"], "subsets": {k: (r["index"] in v) for k, v in member.items()}}
    return items, side, {"benchmark_all_jsonl_sha16": h16}


def main(cfg, mode=None, out_dir=None):
    out = processed_dir(cfg, NAME, out_dir)
    mode = choose_mode(mode, os.path.join(frozen_dir(cfg, NAME), "public.json.gz"))
    items, side, inputs = load_frozen(cfg) if mode == "frozen" else load_source(cfg)
    ordered = sorted(items, key=lambda t: split_key(t["task_id"]))
    cal, dep = ordered[:N_CAL], ordered[N_CAL:]
    if len(cal) != N_CAL or len(dep) != N_DEP or {t["task_id"] for t in cal} & {t["task_id"] for t in dep}:
        raise RuntimeError("split sizes or overlap violate the sealed 500/500 rule")
    tasks = [task_row(i, t["task_id"], t["query"]) for i, t in enumerate(dep)]
    sources = [source_row(t["task_id"], t["query"]) for t in cal]
    sidecar = {t["task_id"]: side[t["task_id"]] for t in items}
    manifest = {"built_from": mode,
                "source": {"repo": WB_REPO, "commit": WB_COMMIT, "file": WB_JSONL, "jsonl_sha16": WB_JSONL_SHA16},
                "split": {"rule": f"sort by sha256('<jsonl_sha16>|{SPLIT_SALT}|<index>'); calibration = first "
                                  f"{N_CAL}, deployment = remaining {N_DEP} in that order",
                          "n_calibration": N_CAL, "n_deployment": N_DEP},
                "inputs": inputs}
    return write_outputs(out, NAME, tasks, sources, sidecar, manifest)


if __name__ == "__main__":
    run_cli(main, NAME)
