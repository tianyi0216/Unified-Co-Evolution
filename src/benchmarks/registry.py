"""Benchmark registry: resolves ``cfg["benchmark"]["name"]`` to an adapter class (paper App.
"Benchmarks and dataset splits", Table "Benchmarks, task types, domains, and evaluated tasks").

Adapter modules are imported lazily inside ``get_benchmark`` so that an unused benchmark's optional
dependencies never load.
"""
from __future__ import annotations

import importlib

BENCHMARKS = {
    "arena_hard": ("src.benchmarks.arena_hard", "ArenaHardBenchmark"),
    "writingprompts": ("src.benchmarks.writingprompts", "WritingPromptsBenchmark"),
    "wildbench": ("src.benchmarks.wildbench", "WildBenchBenchmark"),
    "writingbench": ("src.benchmarks.writingbench", "WritingBenchBenchmark"),
    "multichallenge": ("src.benchmarks.multichallenge", "MultiChallengeBenchmark"),
    "refinebench": ("src.benchmarks.refinebench", "RefineBenchBenchmark"),
    "ifbench": ("src.benchmarks.ifbench", "IFBenchBenchmark"),
    "hard_math": ("src.benchmarks.hard_math", "HardMathBenchmark"),
    "bigcodebench_hard": ("src.benchmarks.bigcodebench", "BigCodeBenchBenchmark"),
}
AGENT_HARNESS_NAMES = ("spreadsheetbench2", "spreadsheetbench-2", "ssb2")
_ALIASES = {"arena-hard": "arena_hard", "arenahard": "arena_hard", "wp": "writingprompts",
            "multi_challenge": "multichallenge", "refinebench_ff": "refinebench",
            "hardmath": "hard_math", "math": "hard_math", "bigcodebench": "bigcodebench_hard",
            "bcb": "bigcodebench_hard", "bcb_hard": "bigcodebench_hard"}


def benchmark_names():
    return sorted(BENCHMARKS)


def resolve_name(name):
    key = str(name or "").strip().lower()
    if key in AGENT_HARNESS_NAMES:
        raise SystemExit(f"benchmark {name!r} is the agent-harness study: run it with run_agent_harness.sh (python -m "
                         f"src.agent_harness.run), not through src.run or src.eval.run")
    key = _ALIASES.get(key, key)
    if key not in BENCHMARKS:
        raise KeyError(f"unknown benchmark {name!r}; known: {', '.join(benchmark_names())}")
    return key


def benchmark_class(name):
    module_name, class_name = BENCHMARKS[resolve_name(name)]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def get_benchmark(cfg):
    """Instantiate the adapter named by ``cfg["benchmark"]["name"]`` with the full config."""
    bench = (cfg or {}).get("benchmark") or {}
    name = bench.get("name") if isinstance(bench, dict) else None
    if not name:
        raise KeyError("config is missing benchmark.name")
    return benchmark_class(name)(cfg)
