"""Prepare BigCodeBench-Hard (Instruct split) for the UCoE pipeline (paper App. "Benchmarks and dataset splits").

Official source: the ``bigcode/bigcodebench-hard`` dataset on the Hugging Face hub, parquet file
``data/v0.1.4-00000-of-00001.parquet`` (148 tasks, official order). The method sees only the task
id and the natural-language ``instruct_prompt``; tests, entry points, canonical solutions and the
code prompts are evaluation-only and are written to a separate sidecar that the method path never
opens (``src.eval.native.bigcodebench`` executes the served solution against the official test).
Calibration sources for response contrasts are the prompt texts of HumanEval+ and MBPP+
(``evalplus/humanevalplus``, ``evalplus/mbppplus``), without their tests or solutions.

Frozen bundles (data/frozen/bigcodebench/, pinned by sha256):
  public.json.gz   {"revision": "data/v0.1.4-00000-of-00001.parquet",
                    "items": [{"i": int, "task_id": "BigCodeBench/<n>", "instruction": str}, ...]}   148 rows
  sidecar.json.gz  {task_id: {"test", "entry_point", "canonical_solution", "code_prompt", "complete_prompt"}}
                   148 rows (evaluation only)
  pool.json.gz     [{"id": "humanevalplus_HumanEval/<n>" | "mbppplus_<n>", "instruction": str}, ...]   542 rows

Processed outputs (<data_dir>, default data/processed/bigcodebench/):
  tasks.jsonl           148 rows in official order: {"i", "task_id", "task", "messages": null, "domain": null}
  calibration.jsonl     pool prompts, content-deduplicated and disjoint from deployment:
                        {"source_id", "task", "messages": null, "domain": null}
  eval_sidecar.json.gz  the sidecar mapping above, unchanged (opened only by src/eval)
  manifest.json          pins, counts and file hashes

Usage:
  python -m data.prepare_bigcodebench --config configs/bigcodebench_qwen8b.yaml [--from-frozen | --from-source]
``--from-frozen`` is the default whenever the frozen bundle exists. ``--from-source`` downloads the
pinned parquet files from the Hugging Face hub (needs ``huggingface_hub`` and ``pyarrow`` or
``pandas``) and verifies the task order against the pinned id list.
"""
from __future__ import annotations

import argparse
import json
import os

from src.utils.config import cfg_get, load_config
from src.utils.io import atomic_write, ensure_dir, file_sha, read_gz_json, sha, write_gz_json, write_json

BENCH = "bigcodebench"
BCB_REPO = "bigcode/bigcodebench-hard"
BCB_REV_FILE = "data/v0.1.4-00000-of-00001.parquet"
N_SRC = 148
IDS_SHA256 = "4e7b4e64c640d310121ed27d15e50f06578db2004d3ae7a3ed7d94d26784ee0d"
POOL_REPOS = (("evalplus/humanevalplus", "test-00000-of-00001.parquet"),
              ("evalplus/mbppplus", "test-00000-of-00001.parquet"))
SIDECAR_FIELDS = ("test", "entry_point", "canonical_solution", "code_prompt", "complete_prompt")
FROZEN_SHA256 = {
    "public.json.gz": "132d0fbc4b247a3352d33ce912b95e3c7151d2e112f2aeb1212315220b700b34",
    "sidecar.json.gz": "f9c8cbba4cf170f473c2451cd2bd1a07517f83a5084fd74e79cfa5ba132751db",
    "pool.json.gz": "f7eb118413c1472c1747f152aedcf08ff9d81359c190a5ac7daeb5628b8e89c3",
}
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


def _read_parquet(path):
    try:
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    except ImportError:
        import pandas as pd
        return pd.read_parquet(path).to_dict("records")


def _hub_file(repo, fname):
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(repo, fname, repo_type="dataset")
    except Exception:  # noqa: BLE001  file name drift across revisions
        from huggingface_hub import list_repo_files
        cand = [f for f in list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet")]
        if not cand:
            raise
        return hf_hub_download(repo, cand[0], repo_type="dataset")


def load_frozen():
    pub_p = _frozen_file("public.json.gz")
    side_p = _frozen_file("sidecar.json.gz")
    pool_p = _frozen_file("pool.json.gz")
    if pub_p is None or side_p is None or pool_p is None:
        raise FileNotFoundError(f"frozen BigCodeBench bundle missing under {frozen_dir()}; use --from-source")
    return read_gz_json(pub_p)["items"], read_gz_json(side_p), read_gz_json(pool_p)


def load_source():
    from huggingface_hub import hf_hub_download
    print(f"{TAG} downloading {BCB_REPO}/{BCB_REV_FILE}", flush=True)
    rows = _read_parquet(hf_hub_download(BCB_REPO, BCB_REV_FILE, repo_type="dataset"))
    if len(rows) != N_SRC:
        raise RuntimeError(f"BigCodeBench-Hard rows {len(rows)} != {N_SRC}")
    items = [{"i": i, "task_id": r["task_id"], "instruction": r["instruct_prompt"]} for i, r in enumerate(rows)]
    sidecar = {r["task_id"]: {k: r.get(k, "") for k in SIDECAR_FIELDS} for r in rows}
    pool = []
    for repo, fname in POOL_REPOS:
        print(f"{TAG} downloading {repo}", flush=True)
        for r in _read_parquet(_hub_file(repo, fname)):
            txt = r.get("prompt") or r.get("text") or ""
            if txt.strip():
                pool.append({"id": f"{repo.split('/')[1]}_{r.get('task_id', len(pool))}",
                             "instruction": txt.strip()})
    return items, sidecar, pool


def build(items, sidecar, pool, out):
    if len(items) != N_SRC or len({r["task_id"] for r in items}) != N_SRC:
        raise RuntimeError(f"expected {N_SRC} unique BigCodeBench-Hard tasks, got {len(items)}")
    ids_sha = sha(json.dumps([r["task_id"] for r in items]))
    if ids_sha != IDS_SHA256:
        raise RuntimeError("task id list/order differs from the pinned official order")
    missing = [r["task_id"] for r in items if r["task_id"] not in sidecar]
    if missing:
        raise RuntimeError(f"sidecar lacks {len(missing)} task ids (e.g. {missing[:3]})")
    ensure_dir(out)
    tasks = [{"i": int(r["i"]), "task_id": r["task_id"], "task": r["instruction"], "messages": None,
              "domain": None} for r in items]
    dep_shas = {sha(r["instruction"]) for r in items}
    calib, seen = [], set()
    for r in pool:
        h = sha(r["instruction"])
        if h in dep_shas or h in seen:
            continue
        seen.add(h)
        calib.append({"source_id": str(r["id"]), "task": r["instruction"], "messages": None, "domain": None})
    _write_jsonl(os.path.join(out, "tasks.jsonl"), tasks)
    _write_jsonl(os.path.join(out, "calibration.jsonl"), calib)
    write_gz_json(os.path.join(out, "eval_sidecar.json.gz"), {r["task_id"]: sidecar[r["task_id"]] for r in items})
    manifest = {
        "benchmark": BENCH,
        "source": {"repo": BCB_REPO, "file": BCB_REV_FILE, "split": "instruct", "ids_sha256": ids_sha},
        "calibration_sources": [repo for repo, _ in POOL_REPOS],
        "counts": {"tasks": len(tasks), "calibration": len(calib), "pool_raw": len(pool), "sidecar": len(tasks)},
        "files": {n: file_sha(os.path.join(out, n)) for n in sorted(os.listdir(out)) if n != "manifest.json"},
    }
    write_json(os.path.join(out, "manifest.json"), manifest, indent=2)
    print(f"{TAG} wrote {len(tasks)} tasks, {len(calib)} calibration prompts (from {len(pool)}) and the test "
          f"sidecar to {out}", flush=True)
    return manifest


def main(cfg, mode="auto", out=None):
    """Build the processed BigCodeBench-Hard files; ``mode`` is 'auto', 'frozen' or 'source'."""
    out = out_dir(cfg, out)
    have_frozen = all(os.path.exists(os.path.join(frozen_dir(), n)) for n in FROZEN_SHA256)
    if mode == "auto":
        mode = "frozen" if have_frozen else "source"
    print(f"{TAG} mode={mode} out={out}", flush=True)
    items, sidecar, pool = load_frozen() if mode == "frozen" else load_source()
    return build(items, sidecar, pool, out)


def _cli():
    ap = argparse.ArgumentParser(description="Prepare BigCodeBench-Hard (Instruct) for UCoE")
    ap.add_argument("--config", default="config.yaml", help="pipeline config (benchmark.data_dir is the output)")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--from-frozen", action="store_true", help="build from data/frozen/bigcodebench (default)")
    group.add_argument("--from-source", action="store_true", help="download the pinned parquet files")
    ap.add_argument("--out_dir", default=None, help="override the output directory")
    args = ap.parse_args()
    cfg = load_config(args.config) if os.path.exists(args.config) else {}
    mode = "frozen" if args.from_frozen else "source" if args.from_source else "auto"
    main(cfg, mode, args.out_dir)


if __name__ == "__main__":
    _cli()
