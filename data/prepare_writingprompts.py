"""Prepare WritingPrompts (paper App. "Benchmarks and dataset splits": a 2,048-task curriculum stream
whose final 1,024 tasks are evaluated; calibration from 1,400 frozen battery prompts).

Usage: python -m data.prepare_writingprompts --config config.yaml [--from-frozen | --from-source]

Official source: Hugging Face ``euclaise/writingprompts`` (train split). Only the ``prompt`` column is
read; the paired human stories never enter the pipeline. Prompts are cleaned (``[ WP ]``-style tags
removed, whitespace collapsed), restricted to 40-300 characters, deduplicated case-insensitively and
shuffled with a fixed seed; the first 500 form the state-off calibration pool and the next 6,400 the
frozen prompt stream.

Frozen bundle (``data/frozen/writingprompts/``):
    public.json.gz     [{"id": "wps000001", "prompt"}]  6,400 prompts in the seeded shuffle order
    pool_ucal.json.gz  [{"id": "wpu000001", "prompt"}]    500 state-off calibration prompts (kept for
                       completeness; the default pipeline does not use them)

Stream construction (both modes): the first 2,048 prompts of the frozen stream form the deployment
pool and the next 1,400 the calibration pool; the deployment pool is ordered as a curriculum by
``complexity`` (word count plus 25 per matched instruction-lexicon cue), ties by prompt sha256, and
only the last 1,024 tasks of the stream are evaluated (``eval_range``).

Processed output: tasks.jsonl (task = instruction + prompt, extra ``prompt`` and ``cx`` fields),
calibration.jsonl, an empty eval_sidecar.json.gz (no native metric) and manifest.json.
"""
from __future__ import annotations

import os
import random
import re

from data.prepare import (choose_mode, file_sha, frozen_dir, processed_dir, read_gz, run_cli, sha, source_row,
                          task_row, write_outputs)
from src.benchmarks.writingprompts import EVAL_RANGE, TASK_PREFIX

NAME = "writingprompts"
HF_DATASET = "euclaise/writingprompts"
SEED = 42
N_UCAL, N_STREAM = 500, 6400
N_DEPLOY, N_CALIB = 2048, 1400
CURR_LEX = {
    "explicit": ("must", "should", "include", "requirement", "exactly",
                 "without using", "cannot", "can't", "forbidden", "rule"),
    "sequence": ("then", "before", "after", "finally", "first", "second",
                 "begins", "ends with", "twist"),
    "pov": ("first person", "second person", "third person", "you are",
            "your", "pov", "perspective", "narrator", "tense"),
    "quote": ('"', "'", "says", "said", "quote", "sentence"),
    "multi": (" and ", " two ", " three ", " both ", "list", "each")}
_TAG = re.compile(r"\[\s*[A-Z]{2,3}\s*\]")
_WS = re.compile(r"\s+")


def clean(p):
    return _WS.sub(" ", _TAG.sub("", p)).strip()


def complexity(prompt):
    """Curriculum key: word count plus 25 per instruction-lexicon cue found in the prompt."""
    low = prompt.lower()
    s = len(prompt.split())
    for toks in CURR_LEX.values():
        s += 25 * sum(1 for t in toks if t in low)
    return s


def load_frozen(cfg):
    fz = frozen_dir(cfg, NAME)
    stream = read_gz(os.path.join(fz, "public.json.gz"))
    if len(stream) != N_STREAM:
        raise RuntimeError(f"frozen stream has {len(stream)} prompts, expected {N_STREAM}")
    inputs = {"public": file_sha(os.path.join(fz, "public.json.gz"))}
    ucal_p = os.path.join(fz, "pool_ucal.json.gz")
    if os.path.exists(ucal_p):
        inputs["pool_ucal"] = file_sha(ucal_p)
    return stream, inputs


def load_source(cfg):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("the `datasets` package is required for --from-source") from e
    ds = load_dataset(HF_DATASET, split="train")
    seen, prompts = set(), []
    for row in ds:
        c = clean(row["prompt"])
        if not (40 <= len(c) <= 300):
            continue
        k = c.lower()
        if k in seen:
            continue
        seen.add(k)
        prompts.append(c)
    rng = random.Random(SEED)
    rng.shuffle(prompts)
    need = N_UCAL + N_STREAM
    if len(prompts) < need:
        raise RuntimeError(f"only {len(prompts)} usable prompts, need {need}")
    stream = [{"id": f"wps{i:06d}", "prompt": p} for i, p in enumerate(prompts[N_UCAL:need], 1)]
    return stream, {"dataset": HF_DATASET, "cleaned_unique_prompts": len(prompts)}


def build(stream):
    deploy = stream[:N_DEPLOY]
    calib = stream[N_DEPLOY:N_DEPLOY + N_CALIB]
    ordered = sorted(deploy, key=lambda r: (complexity(r["prompt"]), sha(r["prompt"])))
    tasks = [task_row(j, r["id"], TASK_PREFIX + r["prompt"], prompt=r["prompt"], cx=complexity(r["prompt"]))
             for j, r in enumerate(ordered)]
    sources = [source_row(r["id"], TASK_PREFIX + r["prompt"], prompt=r["prompt"]) for r in calib]
    return tasks, sources


def main(cfg, mode=None, out_dir=None):
    out = processed_dir(cfg, NAME, out_dir)
    mode = choose_mode(mode, os.path.join(frozen_dir(cfg, NAME), "public.json.gz"))
    stream, inputs = load_frozen(cfg) if mode == "frozen" else load_source(cfg)
    tasks, sources = build(stream)
    manifest = {"built_from": mode,
                "source": {"dataset": HF_DATASET, "split": "train", "field": "prompt", "seed": SEED,
                           "clean": "tags removed, whitespace collapsed, 40-300 chars, case-insensitive dedup"},
                "order": "curriculum by complexity (word count + 25 per instruction cue), ties by prompt sha256",
                "stream": {"deployment": N_DEPLOY, "calibration": N_CALIB, "eval_range": list(EVAL_RANGE)},
                "eval_range": list(EVAL_RANGE),
                "inputs": inputs}
    return write_outputs(out, NAME, tasks, sources, {}, manifest)


if __name__ == "__main__":
    run_cli(main, NAME)
