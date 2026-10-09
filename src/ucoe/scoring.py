"""Criterion scoring of a response under a verifier (paper Sec. 3.2 Eq. reward-interface; App. "Scoring and
diagnosis").

Every block of a verifier is scored independently and cached under ``block_sha | task_sha | response_sha``, so
criteria shared by several candidates (the base block, an inherited extension) are measured once and reused across
candidates and checkpoints. Semantic criteria are scored by the verifier model through the benchmark's scoring
prompt (one JSON object with every id in [0, 1]); deterministic criteria come from the adapter's ``det_prims``.
A missing or invalid score makes the whole evaluation unusable (None) rather than being clamped or imputed.
Relation modules that do not apply to a task score 1.0 (neutral) when an applicability oracle is supplied.
"""

from __future__ import annotations

import json
import math
import os
import threading
from typing import Any, Callable, Dict, Optional

from src.llm.client import ContentInvalid, json_call
from src.ucoe import relations, verifier
from src.utils.io import atomic_write, read_json, sha

FLUSH_EVERY = 32


class ScoreStore:
    """Thread-safe JSON dictionary of block scores (and other cached verdicts) persisted every 32 puts."""

    def __init__(self, path: str):
        self.path = path
        self.lock = threading.Lock()
        self.data: Dict[str, Any] = dict(read_json(path, None) or {}) if path else {}
        self._n = 0
        self._dirty = False

    def get(self, key: str):
        with self.lock:
            return self.data.get(key)

    def put(self, key: str, vals) -> None:
        with self.lock:
            self.data[key] = vals
            self._n += 1
            self._dirty = True
            if self._n % FLUSH_EVERY == 0:
                self._write()

    def flush(self) -> None:
        with self.lock:
            if self._dirty:
                self._write()

    def _write(self) -> None:
        if self.path:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            atomic_write(self.path, json.dumps(self.data, ensure_ascii=False))
        self._dirty = False

    def __len__(self) -> int:
        return len(self.data)


def _valid_scores(ids):
    def validate(obj):
        out = {}
        for cid in ids:
            x = float(obj[cid])
            if not (math.isfinite(x) and 0.0 <= x <= 1.0):
                return None
            out[cid] = x
        return out
    return validate


def score_block(client, meter, store: ScoreStore, bench, block: dict, task_text: str, response: str) -> Optional[dict]:
    """Scores of one block for (task, response), from the store or freshly measured; None when unusable."""
    bsha, psha, ysha = verifier.block_sha(block), sha(task_text), sha(response)
    key = f"{bsha}|{psha}|{ysha}"
    got = store.get(key)
    if got is not None:
        return dict(got)
    det_crit = set(getattr(bench, "det_crit", ()) or ())
    llm = [c for c in block["criteria"] if c["id"] not in det_crit]
    got = {}
    if llm:
        req = bench.score_request(task_text, response, llm)
        try:
            scores = json_call(client, meter, "score", relations.model_for(bench, "verifier"), req["system"],
                               req["user"], _valid_scores([c["id"] for c in llm]), dict(bench.score_kw),
                               f"block|{bsha[:12]}|{psha[:12]}|{ysha[:12]}")
        except ContentInvalid:
            scores = None
        if scores is None:
            return None
        got.update(scores)
    det_ids = [c["id"] for c in block["criteria"] if c["id"] in det_crit]
    if det_ids:
        det = bench.det_prims(response)
        for d in det_ids:
            if d not in det:
                return None
            got[d] = float(det[d])
    store.put(key, got)
    return dict(got)


def score_version(client, meter, store: ScoreStore, bench, V: dict, task_text: str, response: str,
                  rd_applicability: Optional[Callable[[dict, str], bool]] = None) -> Optional[Dict[str, float]]:
    """All criterion scores of ``V`` for (task, response), or None when any criterion could not be measured."""
    vals: Dict[str, float] = {}
    for block in V["blocks"]:
        got = score_block(client, meter, store, bench, block, task_text, response)
        if got is None:
            return None
        vals.update(got)
    if any(c not in vals for c in verifier.crit_ids(V)):
        return None
    if rd_applicability is not None:
        for m in V.get("rd_modules") or []:
            if m["id"] in vals and not rd_applicability(m, task_text):
                vals[m["id"]] = 1.0
    return vals
