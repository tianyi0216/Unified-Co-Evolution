"""Data preparation dispatcher and shared helpers (paper App. "Benchmarks and dataset splits").

Usage:
    python -m data.prepare --config config.yaml [--from-frozen | --from-source] [--benchmark NAME|all]
    (``all`` = the nine benchmarks with frozen bundles; ``--benchmark ssb2`` or configs/ssb2_codex.yaml prepares the
    SpreadsheetBench-2 agent-harness data, which needs the network)

The dispatcher reads ``benchmark.name`` from the config and runs ``data.prepare_<name>``. Every
prepare script writes the canonical processed layout under ``benchmark.data_dir`` (default
``data/processed/<name>``):

    tasks.jsonl            frozen deployment stream (Task rows: i, task_id, task, messages, domain)
    calibration.jsonl      prompt-only calibration pool (Source rows: source_id, task, messages, domain)
    discovery.jsonl        optional development prompts for relation discovery
    eval_sidecar.json.gz   evaluation-only fields keyed by task_id (opened only by src/eval)
    manifest.json          counts, dataset pins, split rules and sha256 of the id lists

``--from-frozen`` builds the files from the frozen bundles under ``data/frozen/<name>/`` and is the
default whenever that bundle exists; ``--from-source`` downloads the pinned official release and
must produce the same files.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import inspect
import json
import os
import sys
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREPARE_MODULES = {
    "arena_hard": "prepare_arena_hard",
    "writingprompts": "prepare_writingprompts",
    "wildbench": "prepare_wildbench",
    "writingbench": "prepare_writingbench",
    "multichallenge": "prepare_multichallenge",
    "refinebench": "prepare_refinebench",
    "ifbench": "prepare_ifbench",
    "hard_math": "prepare_hard_math",
    "bigcodebench_hard": "prepare_bigcodebench",
    "ssb2": "prepare_ssb2",
}
_ALIASES = {"arena-hard": "arena_hard", "arenahard": "arena_hard", "wp": "writingprompts",
            "multi_challenge": "multichallenge", "refinebench_ff": "refinebench", "hardmath": "hard_math",
            "math": "hard_math", "bigcodebench": "bigcodebench_hard", "bcb": "bigcodebench_hard",
            "bcb_hard": "bigcodebench_hard", "spreadsheetbench2": "ssb2", "spreadsheetbench-2": "ssb2"}


# ----------------------------------------------------------------------------- config

def load_config(path):
    """Config loader: the pipeline's own loader when importable, else plain YAML."""
    try:
        from src.utils.config import load_config as _load  # type: ignore
        return _load(path)
    except ImportError:
        pass
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("PyYAML is required to read the config; install it with `pip install pyyaml`") from e
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def cfg_get(cfg, dotted, default=None):
    cur = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            return default
        cur = cur[k]
    return cur


def resolve_name(name):
    key = str(name or "").strip().lower()
    key = _ALIASES.get(key, key)
    if key not in PREPARE_MODULES:
        raise ValueError(f"unknown benchmark {name!r}; known: {', '.join(sorted(PREPARE_MODULES))}")
    return key


def processed_dir(cfg, name, out_dir=None):
    """``benchmark.data_dir`` applies to the configured benchmark; others use the default layout."""
    if out_dir:
        return out_dir
    if str(cfg_get(cfg, "benchmark.name", "")).strip().lower() == name and cfg_get(cfg, "benchmark.data_dir"):
        return cfg_get(cfg, "benchmark.data_dir")
    return os.path.join("data", "processed", name)


def frozen_dir(cfg, name):
    if str(cfg_get(cfg, "benchmark.name", "")).strip().lower() == name and cfg_get(cfg, "benchmark.frozen_dir"):
        return cfg_get(cfg, "benchmark.frozen_dir")
    return os.path.join(REPO_ROOT, "data", "frozen", name)


def choose_mode(mode, frozen_public):
    """``frozen`` when requested or when the frozen bundle exists, else ``source``."""
    if mode in ("frozen", "source"):
        if mode == "frozen" and not os.path.exists(frozen_public):
            raise FileNotFoundError(f"frozen bundle missing: {frozen_public}")
        return mode
    return "frozen" if os.path.exists(frozen_public) else "source"


# ----------------------------------------------------------------------------- io

def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ids_sha(ids):
    return sha(json.dumps(list(ids)))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def read_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def write_gz(path, obj):
    """Gzip JSON without a timestamp or file name in the header, so reruns produce identical bytes."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:
        f.write(json.dumps(obj, ensure_ascii=False).encode("utf-8"))


def write_json(path, obj):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def write_jsonl(path, rows):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def http_bytes(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "ucoe-data-prep"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def hf_download(repo_id, filename, revision, repo_type="dataset"):
    """Download one file of a Hugging Face repository at a pinned revision."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError("huggingface_hub is required for --from-source; `pip install huggingface_hub`") from e
    return hf_hub_download(repo_id, filename, repo_type=repo_type, revision=revision)


def read_parquet_rows(path):
    """Rows of a parquet file as plain dicts (pyarrow, with a pandas fallback)."""
    try:
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    except ImportError:
        pass
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError("pyarrow (or pandas) is required to read parquet for --from-source") from e
    return pd.read_parquet(path).to_dict(orient="records")


def task_row(i, task_id, task, messages=None, domain=None, **extra):
    row = {"i": i, "task_id": str(task_id), "task": task, "messages": messages, "domain": domain}
    row.update(extra)
    return row


def source_row(source_id, task, messages=None, domain=None, **extra):
    row = {"source_id": str(source_id), "task": task, "messages": messages, "domain": domain}
    row.update(extra)
    return row


def write_outputs(out_dir, name, tasks, sources, sidecar, manifest, discovery=None):
    """Write the canonical processed layout and the manifest; returns the manifest."""
    ensure_dir(out_dir)
    dep = {sha(t["task"]) for t in tasks}
    overlap = [s["source_id"] for s in sources if sha(s["task"]) in dep]
    if overlap:
        raise RuntimeError(f"[{name}] {len(overlap)} calibration sources overlap the deployment stream")
    write_jsonl(os.path.join(out_dir, "tasks.jsonl"), tasks)
    write_jsonl(os.path.join(out_dir, "calibration.jsonl"), sources)
    if discovery is not None:
        write_jsonl(os.path.join(out_dir, "discovery.jsonl"), discovery)
    write_gz(os.path.join(out_dir, "eval_sidecar.json.gz"), sidecar)
    man = dict(manifest)
    man["benchmark"] = name
    counts = {"tasks": len(tasks), "calibration": len(sources), "sidecar": len(sidecar)}
    if discovery is not None:
        counts["discovery"] = len(discovery)
    man["counts"] = counts
    man["ids_sha256"] = {"tasks": ids_sha([t["task_id"] for t in tasks]),
                         "calibration": ids_sha([s["source_id"] for s in sources])}
    if discovery is not None:
        man["ids_sha256"]["discovery"] = ids_sha([s["source_id"] for s in discovery])
    write_json(os.path.join(out_dir, "manifest.json"), man)
    print(f"[prepare:{name}] tasks {len(tasks)} | calibration {len(sources)} | sidecar {len(sidecar)}"
          f"{' | discovery ' + str(len(discovery)) if discovery is not None else ''} -> {out_dir}", flush=True)
    return man


def run_cli(main_fn, name):
    """Standard command line of one prepare script."""
    ap = argparse.ArgumentParser(description=f"Prepare {name} for UCoE")
    ap.add_argument("--config", default="config.yaml", help="pipeline config (benchmark.data_dir is honoured)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--from-frozen", action="store_true",
                   help="build from data/frozen/<benchmark>/ (default if present)")
    g.add_argument("--from-source", action="store_true", help="download the pinned official release")
    ap.add_argument("--out_dir", default=None, help="override the processed directory")
    args = ap.parse_args()
    cfg = load_config(args.config) if os.path.exists(args.config) else {}
    mode = "frozen" if args.from_frozen else ("source" if args.from_source else None)
    try:
        main_fn(cfg, mode=mode, out_dir=args.out_dir)
    except Exception as e:  # noqa: BLE001
        print(f"[prepare:{name}] ERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(1)


# ----------------------------------------------------------------------------- dispatcher

def dispatch(cfg, name, mode=None, out_dir=None):
    name = resolve_name(name)
    mod = importlib.import_module(f"data.{PREPARE_MODULES[name]}")
    params = inspect.signature(mod.main).parameters
    kwargs = {}
    if "mode" in params and mode is not None:
        kwargs["mode"] = mode
    if out_dir is not None:
        for key in ("out_dir", "out"):
            if key in params:
                kwargs[key] = out_dir
                break
    print(f"[prepare] benchmark = {name} (mode = {mode or 'auto'})", flush=True)
    return mod.main(cfg, **kwargs)


def main():
    ap = argparse.ArgumentParser(description="UCoE data preparation dispatcher")
    ap.add_argument("--config", default="config.yaml", help="path to the pipeline config")
    ap.add_argument("--benchmark", default=None,
                    help="benchmark name, or `all` for every benchmark with a frozen bundle (SpreadsheetBench-2 is "
                         "prepared on its own); default: benchmark.name of the config")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--from-frozen", action="store_true")
    g.add_argument("--from-source", action="store_true")
    ap.add_argument("--out_dir", default=None, help="override the processed directory (single benchmark only)")
    args = ap.parse_args()
    cfg = load_config(args.config) if os.path.exists(args.config) else {}
    mode = "frozen" if args.from_frozen else ("source" if args.from_source else None)
    name = args.benchmark or cfg_get(cfg, "benchmark.name")
    if not name:
        print(f"[prepare] ERROR: config {args.config} has no benchmark.name and --benchmark was not given",
              file=sys.stderr, flush=True)
        sys.exit(1)
    # `all` = the frozen-bundle benchmarks; SpreadsheetBench-2 (network, agent-harness layout) is prepared alone
    names = sorted(n for n in PREPARE_MODULES if n != "ssb2") if str(name).lower() == "all" else [name]
    for nm in names:
        try:
            dispatch(cfg, nm, mode=mode, out_dir=args.out_dir if len(names) == 1 else None)
        except Exception as e:  # noqa: BLE001
            print(f"[prepare] ERROR ({nm}): {e}", file=sys.stderr, flush=True)
            if len(names) == 1:
                sys.exit(1)
    print("[prepare] done", flush=True)


if __name__ == "__main__":
    main()
