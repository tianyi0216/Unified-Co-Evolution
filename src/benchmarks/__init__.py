"""Benchmark adapters: one ``Benchmark`` subclass per evaluated benchmark (paper App. "Benchmarks and
dataset splits"). ``get_benchmark(cfg)`` in ``registry`` resolves ``cfg["benchmark"]["name"]`` to an adapter."""
from src.benchmarks.base import Benchmark, render_convo
from src.benchmarks.registry import BENCHMARKS, benchmark_names, get_benchmark

__all__ = ["BENCHMARKS", "Benchmark", "benchmark_names", "get_benchmark", "render_convo"]
