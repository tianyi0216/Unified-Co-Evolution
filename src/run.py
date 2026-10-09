"""Command-line entry point of the UCoE pipeline (paper Algorithm 1 and App. "Experimental settings").

Usage:
    python -m src.run --config config.yaml --stage {all,raw,discover,banks,run,status} [--n N] [--workers W]

Stages:
    raw       Direct Generation drafts for the deployment stream (sealed raw pool; the draft of every arm)
    discover  relation discovery and mining on the development prompts (only if ucoe.relation_discovery.enabled)
    banks     response-contrast banks for stage 0 and every checkpoint (selection and held-out splits)
    run       the UCoE arm: H0 admission, per-task revision lifecycle, checkpoint verifier evolution
    status    what exists on disk for this config
    all       raw -> discover -> banks -> run
``--n`` and ``--workers`` override ``benchmark.n_tasks`` and ``output.workers``.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from src.benchmarks.registry import get_benchmark
from src.ucoe.lifecycle import UCoERun
from src.utils.config import cfg_get, load_config

STAGES = ("all", "raw", "discover", "banks", "run", "status")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml", help="YAML config (default: config.yaml)")
    ap.add_argument("--stage", default="all", choices=STAGES)
    ap.add_argument("--n", type=int, default=None, help="number of deployment tasks (default: benchmark.n_tasks)")
    ap.add_argument("--workers", type=int, default=None, help="parallel LLM calls (default: output.workers)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    n = args.n if args.n is not None else cfg_get(cfg, "benchmark.n_tasks")
    n = int(n) if n else None
    workers = int(args.workers or cfg_get(cfg, "output.workers", 8) or 8)
    work_dir = str(cfg_get(cfg, "output.work_dir", "runs/default"))
    bench = get_benchmark(cfg)
    print(
        f"[ucoe] benchmark {cfg_get(cfg, 'benchmark.name')} | backend {cfg_get(cfg, 'models.backend')} | "
        f"actor {cfg_get(cfg, 'models.actor')} | verifier {cfg_get(cfg, 'models.verifier')} | "
        f"work_dir {work_dir} | stage {args.stage} | n {n or 'all'} | workers {workers}",
        flush=True,
    )
    run = UCoERun(cfg, bench, work_dir)
    if args.stage == "status":
        run.status()
        return 0
    run.transport  # noqa: B018 - backend prerequisites (package, API key) are checked before any stage starts
    if args.stage in ("all", "raw"):
        run.run_raw(n, workers)
    if args.stage in ("all", "discover"):
        run.run_discovery(workers)
    if args.stage in ("all", "banks"):
        run.build_banks(workers, n)
    if args.stage in ("all", "run"):
        run.run(n, workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
