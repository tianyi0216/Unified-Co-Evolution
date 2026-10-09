"""Prepare IFBench for the UCoE pipeline (paper App. "Benchmarks and dataset splits": IFBench, 150 evaluated tasks).

Official source: github.com/allenai/IFBench, file data/IFBench_test.jsonl at the pinned commit
(300 rows with a unique ``key``, 58 constraint types). The public view of a task is its key and
the rendered user prompt; the instruction ids and kwargs that drive the official verifiers are
evaluation-only and are written to a separate sidecar that the method path never opens.

Frozen bundles (data/frozen/ifbench/, pinned by sha256):
  public.json.gz   {"commit": str, "jsonl_sha16": str,
                    "items": [{"i": int, "task_id": str, "prompt": str}, ...]}        300 rows, official order
  sidecar.json.gz  {task_id: {"instruction_id_list": [str, ...], "kwargs": [dict, ...]}}   300 rows (evaluation only)
  pool.json.gz     [{"id": "code_<HumanEval+/MBPP+ id>" | "math_<MATH id>", "prompt": str}, ...]
                   2,940 prompt-only Arena battery sources (used by ``benchmark.protocol: full``)

Processed outputs (<data_dir>, default data/processed/ifbench/):
  tasks.jsonl               150 deployment tasks of the matched split, in split order:
                            {"i", "task_id", "task", "messages": null, "domain": null}
  calibration.jsonl         150 calibration prompts of the matched split:
                            {"source_id", "task", "messages": null, "domain": null}
  tasks_full.jsonl          all 300 prompts in official order (``benchmark.protocol: full``)
  calibration_arena.jsonl   the Arena battery prompts, deployment-disjoint (``benchmark.protocol: full``)
  eval_sidecar.json.gz      the sidecar mapping above, unchanged (opened only by src/eval)
  manifest.json              pins, split rule, counts and file hashes

Matched split rule (the paper setting): sort the 300 keys by sha256("<jsonl_sha16>|<SPLIT_SALT>|<key>");
the first 150 keys are calibration sources, the remaining 150 the deployment stream. Both id lists
are pinned by hash so the split cannot drift.

Usage:
  python -m data.prepare_ifbench --config configs/ifbench_qwen8b.yaml [--from-frozen | --from-source]
``--from-frozen`` is the default whenever the frozen bundle exists. ``--from-source`` downloads the
pinned jsonl from GitHub and verifies its hash; the Arena pool is still taken from the frozen copy
when present.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request

from src.utils.config import cfg_get, load_config
from src.utils.io import atomic_write, ensure_dir, file_sha, read_gz_json, sha, write_gz_json, write_json

BENCH = "ifbench"
IF_REPO = "allenai/IFBench"
IF_COMMIT = "fcd289db21d43aaa96c6d9291d32561cd6e19305"
IF_FILE = "data/IFBench_test.jsonl"
IF_SHA16 = "d2ada7da94a38cfe"
N_SRC = 300
N_CAL = 150
SPLIT_SALT = "matchedbank_v2"
CAL_IDS_SHA256 = "a67ede830aada5b10ab9bd1d0799f252f4a4c7607267d3916dcaa9a9a8980223"
DEP_IDS_SHA256 = "17ddcea8bfc3137a8b062d7cc7e70908974a80952e68c60574fad696dc893c10"
FROZEN_SHA256 = {
    "public.json.gz": "c895017b767086b59b653a9f9cceaccb1c3fa99c8c3914b2e6ff3adb93f550fc",
    "sidecar.json.gz": "c9c2f6a4362ff6e5527ef8aabb0eaa21996b9165f437a37e8b671b4703fbaa13",
    "pool.json.gz": "e89fe39cd29566b173dec8c52c2d2f654bce958b5368097e9bff1ea29eaea857",
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


def _check_public(public):
    items = public.get("items", [])
    if public.get("commit") != IF_COMMIT or public.get("jsonl_sha16") != IF_SHA16 \
            or len(items) != N_SRC or len({str(r["task_id"]) for r in items}) != N_SRC:
        raise RuntimeError("IFBench public bundle does not match the pinned commit, hash or row count")


def load_frozen():
    pub_p = _frozen_file("public.json.gz")
    side_p = _frozen_file("sidecar.json.gz")
    pool_p = _frozen_file("pool.json.gz")
    if pub_p is None or side_p is None:
        raise FileNotFoundError(f"frozen IFBench bundle missing under {frozen_dir()}; use --from-source")
    pool = read_gz_json(pool_p) if pool_p else None
    return read_gz_json(pub_p), read_gz_json(side_p), pool


def load_source():
    url = f"https://raw.githubusercontent.com/{IF_REPO}/{IF_COMMIT}/{IF_FILE}"
    print(f"{TAG} downloading {url}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as resp:
        raw = resp.read()
    got = hashlib.sha256(raw).hexdigest()[:16]
    if got != IF_SHA16:
        raise RuntimeError(f"IFBench_test.jsonl sha {got} != pinned {IF_SHA16}")
    rows = [json.loads(ln) for ln in raw.decode("utf-8").split("\n") if ln.strip()]
    if len(rows) != N_SRC or len({str(r["key"]) for r in rows}) != N_SRC:
        raise RuntimeError("IFBench row count / unique key mismatch")
    public = {"commit": IF_COMMIT, "jsonl_sha16": IF_SHA16,
              "items": [{"i": i, "task_id": str(r["key"]), "prompt": r["prompt"]}
                        for i, r in enumerate(rows)]}
    sidecar = {str(r["key"]): {"instruction_id_list": r["instruction_id_list"], "kwargs": r["kwargs"]}
               for r in rows}
    pool_p = _frozen_file("pool.json.gz")
    pool = read_gz_json(pool_p) if pool_p else None
    return public, sidecar, pool


def compute_split(public):
    """Deterministic matched split: (calibration ids, deployment ids), verified against the pins."""
    salt = public["jsonl_sha16"]
    ordered = sorted(public["items"], key=lambda t: sha(f"{salt}|{SPLIT_SALT}|{t['task_id']}"))
    cal = [str(t["task_id"]) for t in ordered[:N_CAL]]
    dep = [str(t["task_id"]) for t in ordered[N_CAL:]]
    if sha(json.dumps(cal)) != CAL_IDS_SHA256 or sha(json.dumps(dep)) != DEP_IDS_SHA256:
        raise RuntimeError("matched split differs from the pinned calibration/deployment id lists")
    return cal, dep


def build(public, sidecar, pool, out):
    _check_public(public)
    items = public["items"]
    by_id = {str(t["task_id"]): t for t in items}
    missing = [k for k in by_id if k not in sidecar]
    if missing:
        raise RuntimeError(f"sidecar lacks {len(missing)} task ids (e.g. {missing[:3]})")
    cal, dep = compute_split(public)
    ensure_dir(out)
    tasks = [{"i": i, "task_id": k, "task": by_id[k]["prompt"], "messages": None, "domain": None}
             for i, k in enumerate(dep)]
    calib = [{"source_id": k, "task": by_id[k]["prompt"], "messages": None, "domain": None} for k in cal]
    full = [{"i": int(t["i"]), "task_id": str(t["task_id"]), "task": t["prompt"], "messages": None,
             "domain": None} for t in items]
    _write_jsonl(os.path.join(out, "tasks.jsonl"), tasks)
    _write_jsonl(os.path.join(out, "calibration.jsonl"), calib)
    _write_jsonl(os.path.join(out, "tasks_full.jsonl"), full)
    n_arena = 0
    if pool is not None:
        dep_shas = {sha(t["prompt"]) for t in items}
        arena = [{"source_id": str(r["id"]), "task": r["prompt"], "messages": None, "domain": None}
                 for r in pool if sha(r["prompt"]) not in dep_shas]
        _write_jsonl(os.path.join(out, "calibration_arena.jsonl"), arena)
        n_arena = len(arena)
    write_gz_json(os.path.join(out, "eval_sidecar.json.gz"), {k: sidecar[k] for k in by_id})
    manifest = {
        "benchmark": BENCH,
        "source": {"repo": IF_REPO, "commit": IF_COMMIT, "file": IF_FILE, "jsonl_sha16": IF_SHA16},
        "split_rule": f"sort keys by sha256('<jsonl_sha16>|{SPLIT_SALT}|<key>'); first {N_CAL} calibration, "
                      f"remaining {N_SRC - N_CAL} deployment",
        "calibration_ids_sha256": CAL_IDS_SHA256,
        "deployment_ids_sha256": DEP_IDS_SHA256,
        "counts": {"tasks": len(tasks), "calibration": len(calib), "tasks_full": len(full),
                   "calibration_arena": n_arena, "sidecar": len(by_id)},
        "files": {n: file_sha(os.path.join(out, n)) for n in sorted(os.listdir(out)) if n != "manifest.json"},
    }
    write_json(os.path.join(out, "manifest.json"), manifest, indent=2)
    print(f"{TAG} wrote {len(tasks)} deployment tasks, {len(calib)} calibration prompts, "
          f"{len(full)} full-order tasks, {n_arena} Arena pool prompts and the sidecar to {out}", flush=True)
    return manifest


def main(cfg, mode="auto", out=None):
    """Build the processed IFBench files; ``mode`` is 'auto', 'frozen' or 'source'."""
    out = out_dir(cfg, out)
    have_frozen = all(os.path.exists(os.path.join(frozen_dir(), n)) for n in ("public.json.gz", "sidecar.json.gz"))
    if mode == "auto":
        mode = "frozen" if have_frozen else "source"
    print(f"{TAG} mode={mode} out={out}", flush=True)
    public, sidecar, pool = load_frozen() if mode == "frozen" else load_source()
    return build(public, sidecar, pool, out)


def _cli():
    ap = argparse.ArgumentParser(description="Prepare IFBench for UCoE")
    ap.add_argument("--config", default="config.yaml", help="pipeline config (benchmark.data_dir is the output)")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--from-frozen", action="store_true", help="build from data/frozen/ifbench (default)")
    group.add_argument("--from-source", action="store_true", help="download the pinned official file")
    ap.add_argument("--out_dir", default=None, help="override the output directory")
    args = ap.parse_args()
    cfg = load_config(args.config) if os.path.exists(args.config) else {}
    mode = "frozen" if args.from_frozen else "source" if args.from_source else "auto"
    main(cfg, mode, args.out_dir)


if __name__ == "__main__":
    _cli()
