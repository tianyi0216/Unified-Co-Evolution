"""Benchmark adapter interface shared by every benchmark of the release.

A ``Benchmark`` adapter owns everything the method needs that is benchmark-specific: the frozen
deployment stream and the disjoint calibration pool (paper App. "Benchmarks and dataset splits"),
the base criterion catalogue and the quality dimensions used by verifier synthesis and by the
procedural memory (paper Sec. 3.2, App. "Harness-state instantiation"), the reference contrast
families with their deterministic corrupters (paper Sec. 3.2 "Response-contrast construction",
App. "Contrast-set construction"), the actor and verifier-model prompts of the per-task lifecycle
(paper Sec. 3.3, Algorithm 1) and the blinded pairwise prompt of the preference evaluation
(paper App. "Evaluation").

Adapters only open ``tasks.jsonl``, ``calibration.jsonl`` and the optional ``discovery.jsonl``
under ``cfg["benchmark"]["data_dir"]``; evaluation sidecars are never read here.

Request objects are plain dicts: ``{"system": str, "user": str}`` for single-turn prompts and
``{"messages": [{"role", "content"}, ...]}`` for native multi-turn calls.
"""
from __future__ import annotations

import json
import os
import re

from src.utils.config import cfg_get
from src.utils.io import read_json, read_jsonl, sha

DEFAULT_TASK_CAP = 60000   # above the longest deployment prompt of every benchmark: the verifier sees the whole task
_CONVO_LINE = re.compile(r"^\[(user|assistant|system)\] ", re.M)


def render_convo(msgs):
    """Canonical role-tagged serialization of a conversation (never drops turns)."""
    return "\n".join(f"[{m['role']}] {m['content']}" for m in msgs)


def parse_convo(text):
    """Inverse of ``render_convo`` for callers that only hold the rendered text."""
    hits = list(_CONVO_LINE.finditer(text))
    if not hits:
        return [{"role": "user", "content": text}]
    msgs = []
    for k, h in enumerate(hits):
        end = hits[k + 1].start() if k + 1 < len(hits) else len(text)
        msgs.append({"role": h.group(1), "content": text[h.end():end].rstrip("\n")})
    return msgs


def task_text(task):
    """The model-visible text of a Task/Source dict (or a bare string)."""
    if isinstance(task, dict):
        return task.get("task") or ""
    return str(task)


def task_messages(task):
    """Native messages of a Task/Source dict; rebuilt from the rendering when absent."""
    if isinstance(task, dict) and task.get("messages"):
        return [dict(m) for m in task["messages"]]
    return parse_convo(task_text(task))


def task_id_of(task):
    if isinstance(task, dict):
        return task.get("task_id") or task.get("source_id") or ""
    return sha(str(task))[:16]


def with_guidelines(text, guidelines):
    """Retrieved procedures are appended after the task text, as in the paper's prompts."""
    g = (guidelines or "").strip()
    return text + "\n\n" + g if g else text


def messages_with_guidelines(msgs, guidelines):
    """Multi-turn variant: the guidelines are appended to the final user turn only."""
    out = [dict(m) for m in msgs]
    g = (guidelines or "").strip()
    if g and out:
        out[-1]["content"] = out[-1]["content"] + "\n\n" + g
    return out


def format_scores(scores):
    """Frozen criterion scores as rendered in the localization prompts."""
    if isinstance(scores, str):
        return scores
    return json.dumps({k: round(float(v), 3) for k, v in sorted(scores.items())})


def reward_view(text, cap=DEFAULT_TASK_CAP):
    """Head-preserving view of a long single-turn task for reward prompts."""
    if len(text) <= cap:
        return text
    return text[:cap - 60] + f"\n[... truncated {len(text) - cap + 60} chars ...]"


def reward_view_messages(msgs, cap):
    """Tail-preserving rendering of a conversation: the whole conversation if it fits, else the
    latest turns that fit under ``cap`` (the final user turn always present) with an omission
    header."""
    full = render_convo(msgs)
    if len(full) <= cap:
        return full
    keep, used = [], 0
    for m in reversed(msgs):
        line = f"[{m['role']}] {m['content']}"
        if keep and used + len(line) + 1 > cap - 80:
            break
        keep.append(line)
        used += len(line) + 1
    k = len(msgs) - len(keep)
    return (f"[earlier turns omitted: {k}]\n" + "\n".join(reversed(keep)))[-cap:]


def decoding_kw(cfg, section, defaults):
    """Suite decoding defaults overridden by ``decoding.<section>`` when present in the config."""
    out = dict(defaults)
    over = cfg_get(cfg, f"decoding.{section}", None) or {}
    if isinstance(over, dict):
        for k in ("max_tokens", "temperature", "thinking"):
            if over.get(k) is not None:
                out[k] = over[k]
    return out


def ngram_nonredundancy(text, token_re=r"[a-z0-9']+", n=8):
    """Unique n-gram ratio over lowercased tokens (1.0 when there are fewer than n tokens)."""
    toks = re.findall(token_re, text.lower())
    grams = [tuple(toks[j:j + n]) for j in range(len(toks) - n + 1)]
    return round(len(set(grams)) / len(grams), 4) if grams else 1.0


def drop_densest(units, k=2, word_re=r"[a-zA-Z]{6,}"):
    """Indices of the ``k`` most information-dense units (content-word and digit density)."""
    dens = sorted(range(len(units)),
                  key=lambda j: -(len(re.findall(word_re, units[j])) + 3 * len(re.findall(r"\d", units[j]))))
    return set(dens[:k])


class Benchmark:
    """Adapter contract. Subclasses set the class attributes to the values of the paper's runs and
    implement the request builders, the deterministic criteria, the corrupters and the duel prompt."""

    name = "base"
    families = ()
    module_ids = ()
    module_defs = {}
    det_crit = ()
    dims = ()
    crit2dim = {}
    scope_tags = ("general",)
    synth_system = ""
    block_system = ""
    multi_turn = False
    TASK_CAP = DEFAULT_TASK_CAP
    GEN_KW = {"max_tokens": 4096, "temperature": 0.7}
    REV_KW = {"max_tokens": 4096, "temperature": 0.3}
    SCORE_KW = {"max_tokens": 640, "temperature": 0.0}
    LOC_KW = {"max_tokens": 640, "temperature": 0.0}
    DISTILL_KW = {"max_tokens": 256, "temperature": 0.0}
    SYNTH_KW = {"max_tokens": 6000, "temperature": 0.0, "thinking": True}
    BANK_KW = None

    def __init__(self, cfg):
        """Subclasses may set any of these attributes (or class-level ``gen_kw``-style constants) before
        calling this initializer; values already present are kept."""
        self.cfg = cfg or {}
        self._default("data_dir", lambda: cfg_get(self.cfg, "benchmark.data_dir", None)
                      or os.path.join("data", "processed", self.name))
        self._default("eps", lambda: float(cfg_get(self.cfg, "ucoe.eps", 0.01)))
        for attr, section, const in (("gen_kw", "generation", "GEN_KW"), ("rev_kw", "revision", "REV_KW"),
                                     ("score_kw", "scoring", "SCORE_KW"), ("loc_kw", "localize", "LOC_KW"),
                                     ("distill_kw", "distill", "DISTILL_KW"), ("synth_kw", "synthesis", "SYNTH_KW")):
            suite = getattr(type(self), attr, None) or getattr(self, const)
            self._default(attr, lambda s=section, d=suite: decoding_kw(self.cfg, s, d))
        self._default("bank_kw", lambda: decoding_kw(self.cfg, "bank", getattr(type(self), "bank_kw", None)
                                                     or self.BANK_KW or self.gen_kw))
        self._default("_tasks", lambda: None)
        self._default("_sources", lambda: None)
        self._default("_eval_range", lambda: None)

    def _default(self, attr, make):
        if attr not in self.__dict__:
            setattr(self, attr, make())

    # ------------------------------------------------------------------ data
    def _path(self, name):
        return os.path.join(self.data_dir, name)

    def manifest(self):
        return read_json(self._path("manifest.json"), default=None)

    def _task_row(self, row, i):
        row = dict(row)
        row.setdefault("i", i)
        row.setdefault("messages", None)
        row.setdefault("domain", None)
        if row.get("task") is None and row.get("messages"):
            row["task"] = render_convo(row["messages"])
        return row

    def _source_row(self, row):
        row = dict(row)
        row.setdefault("messages", None)
        row.setdefault("domain", None)
        if row.get("task") is None and row.get("messages"):
            row["task"] = render_convo(row["messages"])
        return row

    def load_tasks(self):
        """Frozen deployment stream in its sealed order."""
        if self._tasks is None:
            p = self._path("tasks.jsonl")
            if not os.path.exists(p):
                raise FileNotFoundError(f"[{self.name}] missing {p}: run `python -m data.prepare --config <config>`")
            self._tasks = [self._task_row(r, i) for i, r in enumerate(read_jsonl(p))]
        return list(self._tasks)

    def calibration_sources(self):
        """Prompt-only calibration pool, sha-checked to be disjoint from the deployment stream."""
        if self._sources is None:
            p = self._path("calibration.jsonl")
            if not os.path.exists(p):
                raise FileNotFoundError(f"[{self.name}] missing {p}: run `python -m data.prepare --config <config>`")
            dep = {sha(t["task"]) for t in self.load_tasks()}
            rows = [self._source_row(r) for r in read_jsonl(p)]
            keep = [r for r in rows if sha(r["task"]) not in dep]
            if len(keep) != len(rows):
                print(f"[{self.name}] dropped {len(rows) - len(keep)} calibration sources overlapping "
                      f"deployment", flush=True)
            self._sources = keep
        return list(self._sources)

    def discovery_sources(self):
        """Development prompts for relation discovery: ``discovery.jsonl`` when the paper used a separate
        split, else the first ``n_discovery_prompts`` calibration sources in hash order."""
        p = self._path("discovery.jsonl")
        if os.path.exists(p):
            return [self._source_row(r) for r in read_jsonl(p)]
        n = int(cfg_get(self.cfg, "ucoe.relation_discovery.n_discovery_prompts", 32))
        pool = sorted(self.calibration_sources(), key=lambda s: sha(s["task"]))
        return pool[:n]

    @property
    def eval_range(self):
        """``(start, end)`` slice of the stream that the final evaluation scores (default: all)."""
        if self._eval_range is None:
            return (0, len(self.load_tasks()))
        return self._eval_range

    @eval_range.setter
    def eval_range(self, value):
        self._eval_range = tuple(value) if value is not None else None

    def stage_families(self, stage):
        return tuple(self.families)

    def task_cap(self):
        return self.TASK_CAP

    def reward_view(self, task):
        """Capped view of the task text for reward-model prompts of long tasks."""
        if self.multi_turn:
            return reward_view_messages(task_messages(task), self.task_cap())
        return reward_view(task_text(task), self.task_cap())

    # -------------------------------------------------------------- requests
    def draft_request(self, task, guidelines=""):
        raise NotImplementedError

    def revision_request(self, task, draft, witness, repair, guidelines=""):
        raise NotImplementedError

    def score_request(self, task, response, criteria):
        raise NotImplementedError

    def localize_request(self, task, response, scores, weakest, weakdef):
        raise NotImplementedError

    def distill_request(self, task, y0, yr, witness, weakest):
        raise NotImplementedError

    def duel_request(self, task, a, b):
        raise NotImplementedError

    # ---------------------------------------------------------- deterministic
    def det_prims(self, response):
        return {}

    def valid_base(self, response):
        return bool(response) and self.det_prims(response).get("fc", 1) == 1

    def corrupt(self, family, source, response, rng):
        raise NotImplementedError

    def contrast_pair(self, family, source, response, rng):
        """The (y+, y-) texts of one contrast: the base response and its degraded copy, or None when the source
        is ineligible. Adapters whose verifier scores an extracted span (BigCodeBench: the solution code) return
        that span and its edited copy instead."""
        neg = self.corrupt(family, source, response, rng)
        return (response, neg) if neg else None

    # ------------------------------------------------------------ evaluation
    def duel_parse(self, obj):
        if not isinstance(obj, dict):
            return None
        w = str(obj.get("winner", "")).strip().upper()
        return w if w in ("A", "B", "TIE") else None

    def prompt_tags(self, task):
        return {"general"}

    def native_evaluator(self):
        return None

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _defs_lines(criteria):
        return "\n".join(f"- {c['definition']}" for c in criteria)

    @staticmethod
    def _ids_json(criteria):
        return ", ".join(f'"{c["id"]}": x' for c in criteria)

    @staticmethod
    def _apply(fn, response):
        """Run one corrupter and discard empty or unchanged outputs."""
        neg = fn(response)
        if not neg or neg == response:
            return None
        return neg
