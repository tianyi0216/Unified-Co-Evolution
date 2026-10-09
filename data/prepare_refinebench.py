"""Prepare RefineBench Free-Form (paper App. "Benchmarks and dataset splits": the 478 Free-Form exam
tasks are split, stratified by field, into a 200-task calibration pool and a 278-task deployment stream
that keeps the official order).

Usage: python -m data.prepare_refinebench --config config.yaml [--from-frozen | --from-source]

Official source: Hugging Face ``RefineBench/RefineBench`` (``data/train-00000-of-00001.parquet`` at the
pinned revision, 1,000 rows); rows with ``exam_type == "Free Form"`` give exactly 478 tasks with unique
``index`` in the official order. The model-visible task is the question plus any provided passages and
materials; reference answers, checklist, comments and field/subject/institution live in the evaluation
sidecar.

Frozen bundle (``data/frozen/refinebench/``):
    public.json.gz   {"revision", "items": [{"i", "task_id", "question", "passages": [...], "materials": [...]}]}
    sidecar.json.gz  {task_id: {"reference_answer": [...], "checklist": [...], "comment": [...], "field",
                                "subject", "institution"}}
    pool.json.gz     [{"id", "task"}]  the 522 Exact-Match tasks (public task info only) kept as an extra
                     prompt-only pool; the default pipeline calibrates on the Free-Form split instead

Split (sealed before any generation): field-stratified with largest-remainder quotas (200 calibration
tasks over the fields read from the sidecar by this split builder only); within a field, tasks are
ordered by sha256("<revision>|<salt>|<task_id>") and the first ``quota`` are calibration. Deployment keeps
the official order.

Processed output: tasks.jsonl (task = bundled question/passages/materials), calibration.jsonl,
eval_sidecar.json.gz, manifest.json.
"""
from __future__ import annotations

import os

from data.prepare import (choose_mode, file_sha, frozen_dir, hf_download, processed_dir, read_gz,
                          read_parquet_rows, run_cli, sha, source_row, task_row, write_outputs)
from src.benchmarks.refinebench import bundle

NAME = "refinebench"
RB_REPO = "RefineBench/RefineBench"
RB_FILE = "data/train-00000-of-00001.parquet"
RB_REV = "2777137e7c489f5049608f41d2432326429ea619"
N_TOTAL, N_SRC = 1000, 478
N_CAL, N_DEP = 200, 278
SPLIT_SALT = "rewardbank_v3"


def split_key(task_id):
    return sha(f"{RB_REV}|{SPLIT_SALT}|{task_id}")


def _strs(x):
    return [str(v) for v in list(x)] if x is not None else []


def load_frozen(cfg):
    fz = frozen_dir(cfg, NAME)
    pub = read_gz(os.path.join(fz, "public.json.gz"))
    if pub.get("revision") != RB_REV or len(pub["items"]) != N_SRC:
        raise RuntimeError("frozen RefineBench public bundle does not match the pinned release")
    side = read_gz(os.path.join(fz, "sidecar.json.gz"))
    inputs = {k: file_sha(os.path.join(fz, f"{k}.json.gz")) for k in ("public", "sidecar")}
    return pub["items"], side, inputs


def load_source(cfg):
    p = hf_download(RB_REPO, RB_FILE, RB_REV)
    rows = read_parquet_rows(p)
    if len(rows) != N_TOTAL:
        raise RuntimeError(f"total rows {len(rows)} != {N_TOTAL}")
    ff = [r for r in rows if r["exam_type"] == "Free Form"]
    if len(ff) != N_SRC or len({r["index"] for r in ff}) != N_SRC:
        raise RuntimeError(f"Free Form rows {len(ff)} != {N_SRC}")
    items, side = [], {}
    for i, r in enumerate(ff):
        items.append({"i": i, "task_id": r["index"], "question": str(r["question"]),
                      "passages": _strs(r["passages"]), "materials": _strs(r["materials"])})
        side[r["index"]] = {"reference_answer": _strs(r["reference_answer"]), "checklist": _strs(r["checklist"]),
                            "comment": _strs(r["comment"]), "field": str(r["field"]), "subject": str(r["subject"]),
                            "institution": str(r["institution"])}
    return items, side, {"parquet_sha256": file_sha(p)}


def compute_split(items, field_of):
    """Field-stratified largest-remainder calibration quota; deployment keeps the official order."""
    by_field = {}
    for t in items:
        by_field.setdefault(field_of[t["task_id"]], []).append(t)
    exact = {f: N_CAL * len(v) / len(items) for f, v in by_field.items()}
    quota = {f: int(x) for f, x in exact.items()}
    rem = N_CAL - sum(quota.values())
    for f in sorted(exact, key=lambda f: -(exact[f] - quota[f]))[:rem]:
        quota[f] += 1
    cal = set()
    for f, v in by_field.items():
        order = sorted(v, key=lambda t: split_key(t["task_id"]))
        cal.update(t["task_id"] for t in order[:quota[f]])
    cal_items = [t for t in items if t["task_id"] in cal]
    dep_items = [t for t in items if t["task_id"] not in cal]
    if len(cal_items) != N_CAL or len(dep_items) != N_DEP:
        raise RuntimeError(f"split sizes {len(cal_items)}/{len(dep_items)} violate the sealed {N_CAL}/{N_DEP} rule")
    counts = {f: [sum(1 for t in by_field[f] if t["task_id"] in cal), len(by_field[f])] for f in by_field}
    return cal_items, dep_items, counts


def main(cfg, mode=None, out_dir=None):
    out = processed_dir(cfg, NAME, out_dir)
    mode = choose_mode(mode, os.path.join(frozen_dir(cfg, NAME), "public.json.gz"))
    items, side, inputs = load_frozen(cfg) if mode == "frozen" else load_source(cfg)
    field_of = {k: v["field"] for k, v in side.items()}
    cal, dep, counts = compute_split(items, field_of)
    tasks = [task_row(i, t["task_id"], bundle(t["question"], t["passages"], t["materials"])) for i, t in enumerate(dep)]
    sources = [source_row(t["task_id"], bundle(t["question"], t["passages"], t["materials"])) for t in cal]
    sidecar = {t["task_id"]: side[t["task_id"]] for t in items}
    manifest = {"built_from": mode,
                "source": {"repo": RB_REPO, "file": RB_FILE, "revision": RB_REV,
                           "filter": "exam_type == Free Form (official order)"},
                "split": {"rule": f"field-stratified largest-remainder; within field sort by sha256('<revision>|"
                                  f"{SPLIT_SALT}|<task_id>'); deployment keeps official order",
                          "n_calibration": N_CAL, "n_deployment": N_DEP, "field_counts_cal_of_total": counts},
                "order": "official",
                "inputs": inputs}
    return write_outputs(out, NAME, tasks, sources, sidecar, manifest)


if __name__ == "__main__":
    run_cli(main, NAME)
