"""Evaluation CLI (paper App. "Evaluation"):
``python -m src.eval.run --config C --mode {preference,native,all} [--arms ucoe] [--baseline raw] [--workers W]``.

``preference`` scores every method arm against the baseline with the blinded both-order judge
(``src.eval.preference``); ``native`` runs the benchmark's own metric (``src.eval.native``) on the
baseline and the method arms; ``all`` runs both. Arm outputs are read from
``<output.work_dir>/<arm>/rows.jsonl`` and every result lands in ``<output.work_dir>/eval``.
"""
from __future__ import annotations

import argparse

from src.benchmarks.registry import get_benchmark
from src.eval.native import get_native_evaluator, native_for_benchmark
from src.eval.preference import evaluate_preference
from src.utils.config import cfg_get, load_config


def native_evaluator(cfg, bench):
    """Native metric chosen by, in order: ``evaluation.native.evaluator`` in the config, the adapter's
    ``native_evaluator()`` (a metric name or an object with ``evaluate``), then the paper's default for the
    benchmark; None when the benchmark has no native metric in this release."""
    name = cfg_get(cfg, "evaluation.native.evaluator")
    if name:
        ev = get_native_evaluator(name)
        if ev is None:
            raise KeyError(f"unknown native evaluator {name!r} (evaluation.native.evaluator)")
        return ev
    own = bench.native_evaluator()
    if isinstance(own, str):
        return get_native_evaluator(own) or native_for_benchmark(bench.name)
    if own is not None and hasattr(own, "evaluate"):
        return own
    return native_for_benchmark(bench.name)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate UCoE arms: preference vs Direct Generation and native metrics")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--mode", choices=("preference", "native", "all"), default="all")
    ap.add_argument("--arms", default="ucoe", help="comma-separated method arms (directories under output.work_dir)")
    ap.add_argument("--baseline", default="raw", help="Direct Generation arm directory")
    ap.add_argument("--workers", type=int, default=None, help="override output.workers")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.workers:
        cfg.setdefault("output", {})["workers"] = a.workers
    workers = int(cfg_get(cfg, "output.workers", 16))
    bench = get_benchmark(cfg)
    work_dir = cfg_get(cfg, "output.work_dir")
    arms = [x.strip() for x in a.arms.split(",") if x.strip() and x.strip() != a.baseline]
    if not arms:
        raise SystemExit("--arms must name at least one method arm other than the baseline")
    if a.mode in ("preference", "all"):
        evaluate_preference(cfg, bench, work_dir, arms=arms, baseline=a.baseline, workers=workers)
    if a.mode in ("native", "all"):
        ev = native_evaluator(cfg, bench)
        if ev is None:
            print(f"[eval] {bench.name}: no native metric in this release; the preference score is the primary "
                  f"endpoint", flush=True)
        else:
            ev.evaluate(cfg, bench, work_dir, [a.baseline] + arms)


if __name__ == "__main__":
    main()
