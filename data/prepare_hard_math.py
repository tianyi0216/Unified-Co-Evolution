"""Prepare Hard-MATH for the UCoE pipeline (paper App. "Benchmarks and dataset splits": Hard-MATH, 382 evaluated tasks).

The stream is 262 MATH-500 problems of levels 4-5 (official order) followed by 120 AIME problems
(2022-2025, 30 per year), a fixed difficulty order. Calibration sources for response contrasts are
2,400 MATH training problems of levels 4-5, disjoint from the stream by id and by problem text.
Gold answers are evaluation-only and are written to a separate sidecar that the method path never
opens (``src.eval.native.hard_math`` scores the last boxed answer against it).

Frozen bundles (data/frozen/hard_math/, pinned by sha256):
  public.json.gz   {"domain_order": ["math500_l45", "aime2022", "aime2023", "aime2024", "aime2025"],
                    "test_stream":  [{"id": "mh000001".., "problem", "domain", "subject", "level"}, ...]   382 rows
                    "battery_pool": [{"id": "mp000601".."mp003000", "problem", "domain": "math_l45_pool",
                                      "subject", "level"}, ...]                                          2,400 rows
                    "u_cal":        [... same fields, ids mp000001-mp000600 ...]                           600 rows
                   (``u_cal`` belonged to an earlier calibration protocol and is not used here)
  sidecar.json.gz  {id: gold answer string} for the 382 stream problems and the 3,000 pool problems
                   (evaluation only)
  pool.json.gz     [{"subject", "level", "problem", "solution", "gold"}, ...]  5,582 MATH training problems of
                   levels 3-5: the parent pool the calibration problems were drawn from (reference only)

Processed outputs (<data_dir>, default data/processed/hard_math/):
  tasks.jsonl           382 rows in stream order: {"i", "task_id", "task", "messages": null,
                        "domain": "math500" | "aime", "source_domain", "subject", "level"}
  calibration.jsonl     2,400 rows: {"source_id", "task", "messages": null, "domain": null, "subject", "level"}
  eval_sidecar.json.gz  {task_id: gold} for the 382 stream problems (opened only by src/eval)
  manifest.json          pins, counts and file hashes

Usage:
  python -m data.prepare_hard_math --config configs/hard_math_qwen8b.yaml [--from-frozen | --from-source]
``--from-frozen`` is the default whenever the frozen bundle exists. ``--from-source`` rebuilds the
stream from the official releases on the Hugging Face hub (MATH-500, AIME 2022-2024, AIME 2025 and
the MATH training set; needs ``datasets``); when the frozen bundle is also present the rebuilt
problem sets must match it and the frozen ids and order are kept, otherwise the rebuilt order is
used and a warning is printed.
"""
from __future__ import annotations

import argparse
import json
import os
import re

from src.utils.config import cfg_get, load_config
from src.utils.io import atomic_write, ensure_dir, file_sha, read_gz_json, sha, write_gz_json, write_json

BENCH = "hard_math"
N_STREAM = 382
N_MATH500 = 262
N_AIME = 120
N_POOL = 2400
DOMAIN_ORDER = ("math500_l45", "aime2022", "aime2023", "aime2024", "aime2025")
IDS_SHA256 = "4d0f6262c4bdf97ed17e6511085233ebe1855b42caff31a232590ebb9587bf33"
PROMPTS_SHA256 = "1b311c7a3b31431e7ce6eaacf141791fc4d6e49b6b9dc834ece54b0ef6bdf74c"
FROZEN_SHA256 = {
    "public.json.gz": "b48cd1157b93470fa2513c89bed17fa6f407ff4ac48770c1f8576a83adc165e3",
    "sidecar.json.gz": "7a3c97895e44f710bb06838d4f4386566c120a3abefd81506c4c65207a7d36c0",
    "pool.json.gz": "6696c3289b30c188aa3d9e4a262970b665160d112ccbcc176c9b9188d8c88d19",
}
HF_MATH500 = "HuggingFaceH4/MATH-500"
HF_AIME_22_24 = "AI-MO/aimo-validation-aime"
HF_AIME_25 = "MathArena/aime_2025"
HF_MATH_TRAIN = "EleutherAI/hendrycks_math"
MATH_SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                 "number_theory", "prealgebra", "precalculus")
TAG = f"[prepare:{BENCH}]"


def frozen_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "frozen", BENCH)


def out_dir(cfg, override=None):
    if override:
        return override
    if str(cfg_get(cfg, "benchmark.name", "")) == BENCH and cfg_get(cfg, "benchmark.data_dir"):
        return str(cfg_get(cfg, "benchmark.data_dir"))
    return os.path.join("data", "processed", BENCH)


def _write_jsonl(path, rows):
    atomic_write(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def _frozen_file(name):
    path = os.path.join(frozen_dir(), name)
    if not os.path.exists(path):
        return None
    got = file_sha(path)
    if got != FROZEN_SHA256[name]:
        raise RuntimeError(f"{path}: sha256 {got[:12]} != pinned {FROZEN_SHA256[name][:12]}")
    return path


def domain_tag(source_domain):
    return "aime" if str(source_domain).startswith("aime") else "math500"


def _check_stream(stream, strict):
    if len(stream) != N_STREAM:
        raise RuntimeError(f"stream has {len(stream)} problems, expected {N_STREAM}")
    ids_sha = sha(json.dumps([r["id"] for r in stream]))
    prompts_sha = sha(json.dumps([(r["id"], sha(r["problem"])) for r in stream], sort_keys=True))
    if strict and (ids_sha != IDS_SHA256 or prompts_sha != PROMPTS_SHA256):
        raise RuntimeError("stream ids/order or problem texts differ from the pinned stream")
    return ids_sha, prompts_sha


def load_frozen():
    pub_p = _frozen_file("public.json.gz")
    side_p = _frozen_file("sidecar.json.gz")
    if pub_p is None or side_p is None:
        raise FileNotFoundError(f"frozen Hard-MATH bundle missing under {frozen_dir()}; use --from-source")
    public = read_gz_json(pub_p)
    gold = read_gz_json(side_p)
    return public["test_stream"], public["battery_pool"], gold


def _aime_key(row):
    text = f"{row.get('url', '')} {row.get('id', '')}"
    year = re.search(r"(20[0-9]{2})", text)
    contest = re.search(r"AIME[_ ]*(I{1,2})\b", text)
    number = re.search(r"Problem[_ ]*(\d+)", text)
    return (int(year.group(1)) if year else 0, len(contest.group(1)) if contest else 0,
            int(number.group(1)) if number else 0)


def load_source():
    import datasets  # optional dependency, imported lazily
    stream, gold = [], {}
    for r in datasets.load_dataset(HF_MATH500, split="test"):
        if int(r["level"]) in (4, 5):
            stream.append({"problem": r["problem"], "domain": "math500_l45",
                           "subject": str(r.get("subject", "")), "level": str(r["level"]),
                           "gold": str(r["answer"])})
    if len(stream) != N_MATH500:
        raise RuntimeError(f"MATH-500 levels 4-5 gave {len(stream)} problems, expected {N_MATH500}")
    aime = []
    for r in datasets.load_dataset(HF_AIME_22_24, split="train"):
        year = _aime_key(r)[0]
        aime.append((_aime_key(r), {"problem": r["problem"], "domain": f"aime{year}", "subject": "aime",
                                    "level": "AIME", "gold": str(r["answer"])}))
    for r in datasets.load_dataset(HF_AIME_25, split="train"):
        idx = int(r.get("problem_idx", len(aime)))
        aime.append(((2025, 1 if idx <= 15 else 2, idx), {"problem": r["problem"], "domain": "aime2025",
                                                           "subject": "aime", "level": "AIME",
                                                           "gold": str(r["answer"])}))
    aime.sort(key=lambda kv: kv[0])
    stream.extend(row for _, row in aime)
    if len(stream) != N_STREAM:
        raise RuntimeError(f"rebuilt stream has {len(stream)} problems, expected {N_STREAM}")
    for k, row in enumerate(stream):
        row["id"] = f"mh{k + 1:06d}"
        gold[row["id"]] = row.pop("gold")
    stream_shas = {sha(r["problem"]) for r in stream}
    pool, seen = [], set()
    for subject in MATH_SUBJECTS:
        for r in datasets.load_dataset(HF_MATH_TRAIN, subject, split="train"):
            if str(r.get("level", "")) not in ("Level 4", "Level 5"):
                continue
            h = sha(r["problem"])
            if h in stream_shas or h in seen:
                continue
            seen.add(h)
            pool.append({"id": f"mp{len(pool) + 1:06d}", "problem": r["problem"], "domain": "math_l45_pool",
                         "subject": subject, "level": str(r["level"])})
    pub_p = _frozen_file("public.json.gz")
    if pub_p is not None:
        frozen = read_gz_json(pub_p)
        frozen_stream = frozen["test_stream"]
        if {sha(r["problem"]) for r in frozen_stream} != stream_shas:
            raise RuntimeError("rebuilt stream does not contain the same problems as the frozen stream")
        by_sha = {sha(r["problem"]): gold[r["id"]] for r in stream}
        stream = frozen_stream
        gold = {r["id"]: by_sha[sha(r["problem"])] for r in stream}
        pool = frozen["battery_pool"]
        print(f"{TAG} rebuilt problem sets match the frozen bundle; keeping its ids, order and calibration pool",
              flush=True)
    else:
        print(f"{TAG} WARNING: no frozen bundle; the rebuilt ids/order and the {len(pool)}-problem pool are "
              f"not the paper's pinned stream", flush=True)
    return stream, pool, gold


def build(stream, pool, gold, out, strict):
    ids_sha, prompts_sha = _check_stream(stream, strict)
    missing = [r["id"] for r in stream if r["id"] not in gold]
    if missing:
        raise RuntimeError(f"gold sidecar lacks {len(missing)} stream ids (e.g. {missing[:3]})")
    stream_ids = {r["id"] for r in stream}
    stream_shas = {sha(r["problem"]) for r in stream}
    if any(r["id"] in stream_ids or sha(r["problem"]) in stream_shas for r in pool):
        raise RuntimeError("calibration pool overlaps the deployment stream")
    ensure_dir(out)
    tasks = [{"i": i, "task_id": r["id"], "task": r["problem"], "messages": None,
              "domain": domain_tag(r["domain"]), "source_domain": r["domain"],
              "subject": r.get("subject", "?"), "level": str(r.get("level", "?"))}
             for i, r in enumerate(stream)]
    calib = [{"source_id": r["id"], "task": r["problem"], "messages": None, "domain": None,
              "subject": r.get("subject", "?"), "level": str(r.get("level", "?"))} for r in pool]
    _write_jsonl(os.path.join(out, "tasks.jsonl"), tasks)
    _write_jsonl(os.path.join(out, "calibration.jsonl"), calib)
    write_gz_json(os.path.join(out, "eval_sidecar.json.gz"), {r["id"]: gold[r["id"]] for r in stream})
    domains = {}
    for t in tasks:
        domains[t["source_domain"]] = domains.get(t["source_domain"], 0) + 1
    manifest = {
        "benchmark": BENCH,
        "stream": {"n": len(tasks), "order": list(DOMAIN_ORDER), "per_domain": domains,
                   "ids_sha256": ids_sha, "prompts_sha256": prompts_sha,
                   "pinned": ids_sha == IDS_SHA256 and prompts_sha == PROMPTS_SHA256},
        "sources": {"math500": HF_MATH500, "aime_2022_2024": HF_AIME_22_24, "aime_2025": HF_AIME_25,
                    "calibration_pool": f"{HF_MATH_TRAIN} levels 4-5"},
        "counts": {"tasks": len(tasks), "calibration": len(calib), "sidecar": len(tasks)},
        "files": {n: file_sha(os.path.join(out, n)) for n in sorted(os.listdir(out)) if n != "manifest.json"},
    }
    write_json(os.path.join(out, "manifest.json"), manifest, indent=2)
    print(f"{TAG} wrote {len(tasks)} stream tasks ({domains}), {len(calib)} calibration problems and the gold "
          f"sidecar to {out}", flush=True)
    return manifest


def main(cfg, mode="auto", out=None):
    """Build the processed Hard-MATH files; ``mode`` is 'auto', 'frozen' or 'source'."""
    out = out_dir(cfg, out)
    have_frozen = all(os.path.exists(os.path.join(frozen_dir(), n)) for n in ("public.json.gz", "sidecar.json.gz"))
    if mode == "auto":
        mode = "frozen" if have_frozen else "source"
    print(f"{TAG} mode={mode} out={out}", flush=True)
    if mode == "frozen":
        stream, pool, gold = load_frozen()
        if len(pool) != N_POOL:
            raise RuntimeError(f"frozen calibration pool has {len(pool)} problems, expected {N_POOL}")
        return build(stream, pool, gold, out, strict=True)
    stream, pool, gold = load_source()
    return build(stream, pool, gold, out, strict=have_frozen)


def _cli():
    ap = argparse.ArgumentParser(description="Prepare Hard-MATH for UCoE")
    ap.add_argument("--config", default="config.yaml", help="pipeline config (benchmark.data_dir is the output)")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--from-frozen", action="store_true", help="build from data/frozen/hard_math (default)")
    group.add_argument("--from-source", action="store_true", help="rebuild from the Hugging Face releases")
    ap.add_argument("--out_dir", default=None, help="override the output directory")
    args = ap.parse_args()
    cfg = load_config(args.config) if os.path.exists(args.config) else {}
    mode = "frozen" if args.from_frozen else "source" if args.from_source else "auto"
    main(cfg, mode, args.out_dir)


if __name__ == "__main__":
    _cli()
