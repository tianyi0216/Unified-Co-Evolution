"""Shared utilities: configuration loading, durable filesystem primitives, statistics and text helpers.

These modules underpin every stage of the pipeline (paper Algorithm 1 and App. "Implementation details"):
resumable artifacts, deterministic seeding, the Clopper-Pearson bounds of the conservative update rule and
the evaluation statistics.
"""
from .config import cfg_get, cfg_set, load_config
from .io import (append_jsonl, atomic_write, ensure_dir, file_sha, read_gz_json, read_json, read_jsonl,
                 repair_jsonl_tail, seed_from, sha, sha_json, write_gz_json, write_json)
from .stats import block_bootstrap_ci, cp_lower, holm, sign_test_p
from .text import extract_json, jaccard, sentences, tokens

__all__ = [
    "cfg_get", "cfg_set", "load_config",
    "append_jsonl", "atomic_write", "ensure_dir", "file_sha", "read_gz_json", "read_json", "read_jsonl",
    "repair_jsonl_tail", "seed_from", "sha", "sha_json", "write_gz_json", "write_json",
    "block_bootstrap_ci", "cp_lower", "holm", "sign_test_p",
    "extract_json", "jaccard", "sentences", "tokens",
]
