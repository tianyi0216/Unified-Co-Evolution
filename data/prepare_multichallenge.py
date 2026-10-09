"""Prepare MultiChallenge (paper App. "Benchmarks and dataset splits": the 266 official conversations
are split by hash into a 138-conversation calibration pool and a 128-task deployment stream).

Usage: python -m data.prepare_multichallenge --config config.yaml [--from-frozen | --from-source]

Official source: Hugging Face ``ScaleAI/MultiChallenge``, test split (``data/test-00000-of-00001.parquet``
at the pinned revision), 266 conversations with unique ``question_id``. Every conversation alternates
user/assistant turns, has 3-19 turns and ends with a user turn; only the next assistant response is
generated. The public view holds question_id + the exact conversation; axis, target_question,
pass_criteria and num_turns live in the evaluation sidecar.

Frozen bundle (``data/frozen/multichallenge/``):
    public.json.gz   {"revision", "parquet_sha", "items": [{"i", "question_id", "messages": [{"role", "content"}]}]}
    sidecar.json.gz  {question_id: {"axis", "target_question", "pass_criteria", "num_turns"}}

Split (sealed before any generation): ids ordered by sha256("<parquet_sha>|<salt>|<question_id>");
the first 138 are calibration conversations and the remaining 128 the deployment stream in that order.

Processed output: tasks.jsonl (task = role-tagged rendering, messages = native turns),
calibration.jsonl (native turns), eval_sidecar.json.gz, manifest.json.
"""
from __future__ import annotations

import os

from data.prepare import (choose_mode, file_sha, frozen_dir, hf_download, processed_dir, read_gz,
                          read_parquet_rows, run_cli, sha, source_row, task_row, write_outputs)
from src.benchmarks.base import render_convo

NAME = "multichallenge"
MC_REPO = "ScaleAI/MultiChallenge"
MC_FILE = "data/test-00000-of-00001.parquet"
MC_REV = "670fd81f270314d0f8f82b34fcfe2187fd28f699"
MC_SHA = "2711646aba61f814823297714aaf757e3a1b71c66c501402231de48490650515"
N_SRC = 266
N_CAL, N_DEP = 138, 128
SPLIT_SALT = "matchedbank_v2"


def split_key(question_id):
    return sha(f"{MC_SHA}|{SPLIT_SALT}|{question_id}")


def normalize_conv(c):
    """Conversation column as a list of {role, content} (parquet may yield a dict of columns)."""
    if isinstance(c, list):
        return [{"role": str(m["role"]), "content": str(m["content"])} for m in c]
    roles, conts = list(c["role"]), list(c["content"])
    if len(roles) != len(conts):
        raise RuntimeError("role/content length mismatch")
    return [{"role": str(r), "content": str(t)} for r, t in zip(roles, conts)]


def check_conv(msgs):
    if not 3 <= len(msgs) <= 19:
        raise RuntimeError(f"conversation length {len(msgs)} outside 3-19")
    for i, m in enumerate(msgs):
        want = "user" if i % 2 == 0 else "assistant"
        if m["role"] != want or not m["content"]:
            raise RuntimeError(f"turn {i}: expected a non-empty {want} turn")
    if msgs[-1]["role"] != "user":
        raise RuntimeError("conversation must end with a user turn")


def load_frozen(cfg):
    fz = frozen_dir(cfg, NAME)
    pub = read_gz(os.path.join(fz, "public.json.gz"))
    if pub.get("revision") != MC_REV or pub.get("parquet_sha") != MC_SHA or len(pub["items"]) != N_SRC:
        raise RuntimeError("frozen MultiChallenge public bundle does not match the pinned release")
    side = read_gz(os.path.join(fz, "sidecar.json.gz"))
    inputs = {k: file_sha(os.path.join(fz, f"{k}.json.gz")) for k in ("public", "sidecar")}
    return pub["items"], side, inputs


def load_source(cfg):
    p = hf_download(MC_REPO, MC_FILE, MC_REV)
    got = file_sha(p)
    if got != MC_SHA:
        raise RuntimeError(f"parquet sha256 {got[:12]} != pinned {MC_SHA[:12]}")
    rows = read_parquet_rows(p)
    if len(rows) != N_SRC or len({r["question_id"] for r in rows}) != N_SRC:
        raise RuntimeError(f"rows {len(rows)} / unique ids != {N_SRC}")
    items, side = [], {}
    for i, r in enumerate(rows):
        msgs = normalize_conv(r["conversation"])
        check_conv(msgs)
        if int(r["num_turns"]) != len(msgs):
            raise RuntimeError(f"{r['question_id']}: num_turns {r['num_turns']} != {len(msgs)}")
        items.append({"i": i, "question_id": r["question_id"], "messages": msgs})
        side[r["question_id"]] = {"axis": r["axis"], "target_question": r["target_question"],
                                  "pass_criteria": r["pass_criteria"], "num_turns": int(r["num_turns"])}
    return items, side, {"parquet_sha256": got}


def main(cfg, mode=None, out_dir=None):
    out = processed_dir(cfg, NAME, out_dir)
    mode = choose_mode(mode, os.path.join(frozen_dir(cfg, NAME), "public.json.gz"))
    items, side, inputs = load_frozen(cfg) if mode == "frozen" else load_source(cfg)
    for t in items:
        check_conv(t["messages"])
    ordered = sorted(items, key=lambda t: split_key(t["question_id"]))
    cal, dep = ordered[:N_CAL], ordered[N_CAL:]
    if len(cal) != N_CAL or len(dep) != N_DEP or {t["question_id"] for t in cal} & {t["question_id"] for t in dep}:
        raise RuntimeError("split sizes or overlap violate the sealed 138/128 rule")
    tasks = [task_row(i, t["question_id"], render_convo(t["messages"]), messages=t["messages"])
             for i, t in enumerate(dep)]
    sources = [source_row(t["question_id"], render_convo(t["messages"]), messages=t["messages"]) for t in cal]
    sidecar = {t["question_id"]: side[t["question_id"]] for t in items}
    manifest = {"built_from": mode,
                "source": {"repo": MC_REPO, "file": MC_FILE, "revision": MC_REV, "parquet_sha256": MC_SHA},
                "split": {"rule": f"sort by sha256('<parquet_sha>|{SPLIT_SALT}|<question_id>'); calibration = "
                                  f"first {N_CAL}, deployment = remaining {N_DEP} in that order",
                          "n_calibration": N_CAL, "n_deployment": N_DEP},
                "inputs": inputs}
    return write_outputs(out, NAME, tasks, sources, sidecar, manifest)


if __name__ == "__main__":
    run_cli(main, NAME)
