"""Cache-first, metered LLM client with a durable request ledger, JSON calls and a parallel map.

Every model call of the pipeline goes through ``Client`` (paper App. "Implementation details"): a request is
identified by the sha256 of (model, system, user, kw, nonce); identical requests are served from an append-only
JSONL ledger, so a run is resumable and byte-for-byte replayable, and concurrent identical requests are
de-duplicated (single flight). ``Meter`` counts fresh calls per model role against optional ceilings so that a
run can never exceed its budget. ``json_call`` wraps a structured call with the seed convention
``seed_from(namespace, "call", seed_key)`` and at most one syntax retry (a fresh nonce). ``pmap`` is the
order-preserving thread pool used for batched scoring and bank construction.
"""
from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..utils.io import append_jsonl, atomic_write, read_json, repair_jsonl_tail, seed_from, sha
from ..utils.text import extract_json

ROLE_ORDER = ("actor", "verifier", "judge", "comparator")
SINGLE_FLIGHT_WAIT_S = 1800
TRANSPORT_ATTEMPTS = 2          # one physical submission plus at most one transport retry per logical call


class ContentInvalid(Exception):
    """The model returned empty output for a structured call."""


class CostCeilingExceeded(RuntimeError):
    """A Meter ceiling would be exceeded; raised before the request is submitted."""


class TransportFailure(RuntimeError):
    """The transport failed after its retry allowance (or a single-flight wait timed out)."""


class LedgerIntegrityError(RuntimeError):
    """The ledger holds a corrupted row that is not a crash-truncated tail; refusing to resume from it."""


def request_hash(model: str, system: str, user: str, kw: Dict[str, Any], nonce: int = 0) -> str:
    """sha256 of the sorted JSON {model, system, user, kw, nonce}. For message-list calls the caller passes
    ``system=""`` and ``user=json.dumps(messages)``."""
    body = json.dumps({"model": model, "system": system, "user": user, "kw": kw, "nonce": nonce},
                      sort_keys=True, ensure_ascii=False)
    return sha(body)


# ------------------------------------------------------------------ metering

class Meter:
    """Per-role call accounting with optional ceilings, persisted as JSON.

    ``ceilings`` maps a role class ("actor", "verifier", "judge", "comparator", ...) to the maximum number
    of fresh calls, or None for unlimited; a missing role is unlimited. ``max_transport_retries`` optionally
    bounds the total number of transport retries of the run.
    """

    def __init__(self, path: Optional[str], ceilings: Optional[Dict[str, Optional[int]]] = None,
                 max_transport_retries: Optional[int] = None):
        self.path = path
        self.ceilings: Dict[str, Optional[int]] = {k: (int(v) if v is not None else None)
                                                   for k, v in (ceilings or {}).items()}
        self.max_transport_retries = max_transport_retries
        self.lock = threading.Lock()
        d = read_json(path, {}) if path else {}
        self.new: Dict[str, int] = {k: int(v) for k, v in dict(d.get("new", {})).items()}
        self.hits: Dict[str, int] = {k: int(v) for k, v in dict(d.get("hits", {})).items()}
        for k in self.ceilings:
            self.new.setdefault(k, 0)
            self.hits.setdefault(k, 0)
        self.transport_retries = int(d.get("transport_retries", 0))
        self.syntax_retries = int(d.get("syntax_retries", 0))

    def _persist(self) -> None:
        if self.path:
            atomic_write(self.path, json.dumps(self.state(), sort_keys=True))

    def state(self) -> Dict[str, Any]:
        return {"new": dict(self.new), "hits": dict(self.hits),
                "transport_retries": self.transport_retries, "syntax_retries": self.syntax_retries}

    def charge(self, role_class: str) -> None:
        """Count one fresh submission for ``role_class``; raises CostCeilingExceeded before exceeding."""
        with self.lock:
            cap = self.ceilings.get(role_class)
            n = self.new.get(role_class, 0)
            if cap is not None and n + 1 > cap:
                raise CostCeilingExceeded(f"cost ceiling for {role_class}: {n} >= {cap}; aborting before submission")
            self.new[role_class] = n + 1
            self._persist()

    def hit(self, role_class: str) -> None:
        with self.lock:
            self.hits[role_class] = self.hits.get(role_class, 0) + 1
            self._persist()

    def transport_retry(self) -> None:
        with self.lock:
            cap = self.max_transport_retries
            if cap is not None and self.transport_retries + 1 > cap:
                raise CostCeilingExceeded("transport-retry allowance exhausted")
            self.transport_retries += 1
            self._persist()

    def syntax_retry(self) -> None:
        with self.lock:
            self.syntax_retries += 1
            self._persist()

    def snapshot(self, label: str) -> None:
        print(f"[ucoe-cost] {label}: new={self.new} hits={self.hits} "
              f"t_retries={self.transport_retries} s_retries={self.syntax_retries}", flush=True)


# ------------------------------------------------------------------ client

class Client:
    """Single-flight, cache-first client over a durable JSONL ledger.

    ``role_models`` maps role names to model names ({"actor": ..., "verifier": ..., "judge": ...,
    "comparator": ...}); ``role_class(model)`` charges the first role (in ROLE_ORDER, then declaration order)
    that uses the model, or "other". ``ns`` (the ledger's basename without extension) is the namespace used for
    seeding by ``json_call``.
    """

    def __init__(self, ledger_path: str, transport: Callable[..., Dict[str, Any]], meter: Meter,
                 role_models: Optional[Dict[str, str]] = None, role_classes: Optional[Dict[str, str]] = None):
        self.ledger_path = ledger_path
        self.ns = os.path.splitext(os.path.basename(ledger_path))[0]
        self.transport = transport
        self.meter = meter
        self.role_models: Dict[str, str] = dict(role_models or role_classes or {})
        self.lock = threading.Lock()
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.inflight: Dict[str, threading.Event] = {}
        self._load()

    def _load(self) -> None:
        """Load the ledger, repairing a crash-truncated final line before anything is appended."""
        if not os.path.exists(self.ledger_path):
            return
        if repair_jsonl_tail(self.ledger_path) == "truncated":
            print(f"[ucoe-llm] repaired a crash-truncated tail of {self.ledger_path}", flush=True)
        with open(self.ledger_path, "rb") as f:
            raw = f.read()
        for i, line in enumerate(raw.split(b"\n")):
            if not line.strip():
                continue
            try:
                row = json.loads(line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise LedgerIntegrityError(f"{self.ledger_path}: line {i + 1} is corrupted and is not the final "
                                           f"line; refusing to resume from a damaged ledger") from None
            h = row.get("request_hash") if isinstance(row, dict) else None
            if not (isinstance(h, str) and len(h) == 64 and "text" in row):
                raise LedgerIntegrityError(f"{self.ledger_path}: line {i + 1} is not a ledger row")
            prev = self.cache.get(h)
            if prev is not None and prev.get("text") != row.get("text"):
                raise LedgerIntegrityError(f"{self.ledger_path}: conflicting rows for request {h[:12]}..")
            self.cache[h] = row

    def role_class(self, model: str) -> str:
        for role in ROLE_ORDER:
            if self.role_models.get(role) == model:
                return role
        for role, m in self.role_models.items():
            if m == model:
                return role
        return "other"

    def __len__(self) -> int:
        return len(self.cache)

    def call(self, role: str, model: str, system: str, user: str, kw: Dict[str, Any], nonce: int = 0) -> Dict[str, Any]:
        """Cached single-prompt call -> {"request_hash","role","model","text","finish_reason","usage","model_meta"}."""
        kw = dict(kw or {})
        h = request_hash(model, system, user, kw, nonce)
        return self._serve(h, role, model, lambda: self.transport(model, system, user, kw))

    def call_messages(self, role: str, model: str, messages: List[Dict[str, Any]], kw: Dict[str, Any],
                      nonce: int = 0) -> Dict[str, Any]:
        """Cached native multi-turn call; hashed with system "" and user = json.dumps(messages)."""
        kw = dict(kw or {})
        messages = list(messages)
        h = request_hash(model, "", json.dumps(messages), kw, nonce)
        return self._serve(h, role, model, lambda: self.transport.call_messages(model, messages, kw))

    def _serve(self, h: str, role: str, model: str, submit: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        cls = self.role_class(model)
        with self.lock:
            row = self.cache.get(h)
            if row is not None:
                self.meter.hit(cls)
                return row
            ev = self.inflight.get(h)
            if ev is None:
                self.inflight[h] = threading.Event()
        if ev is not None:                       # another thread owns this request: wait for its row
            ev.wait(SINGLE_FLIGHT_WAIT_S)
            with self.lock:
                row = self.cache.get(h)
                if row is not None:
                    self.meter.hit(cls)
                    return row
            raise TransportFailure(f"{role}: single-flight wait failed for request {h[:12]}..")
        try:
            last: Optional[BaseException] = None
            out: Optional[Dict[str, Any]] = None
            for attempt in range(TRANSPORT_ATTEMPTS):
                if attempt:
                    self.meter.transport_retry()
                self.meter.charge(cls)
                try:
                    out = submit()
                    break
                except Exception as e:  # noqa: BLE001 - transport failures are retried once, then surfaced
                    last = e
                    out = None
                    print(f"[ucoe-llm] {role}/{model}: attempt {attempt + 1} failed: "
                          f"{type(e).__name__}: {str(e)[:160]}", flush=True)
            if out is None:
                raise TransportFailure(f"{role}: transport failed after retry: {last}")
            row = {"request_hash": h, "role": role, "model": model,
                   "text": out.get("text") or "",
                   "finish_reason": out.get("finish_reason"),
                   "usage": out.get("usage") or {},
                   "model_meta": out.get("model_meta") or {}}
            with self.lock:
                self.cache[h] = row
                append_jsonl(self.ledger_path, row)
            return row
        finally:
            with self.lock:
                done = self.inflight.pop(h, None)
                if done is not None:
                    done.set()


# ------------------------------------------------------------------ structured calls

def json_call(client: Client, meter: Meter, role: str, model: str, system: str, user: str,
              validate: Callable[[Any], Any], kw: Dict[str, Any], seed_key: str, nonce_retries: int = 1) -> Any:
    """Structured call: seeded by ``seed_from(client.ns, "call", seed_key)``; a reply that holds no JSON is
    retried with a fresh nonce at most ``nonce_retries`` times; JSON that fails ``validate`` (None or a
    TypeError/KeyError/ValueError) is final -> None; empty output raises ContentInvalid."""
    kw = dict(kw or {})
    kw["seed"] = seed_from(client.ns, "call", seed_key)
    for nonce in range(int(nonce_retries) + 1):
        if nonce:
            meter.syntax_retry()
        row = client.call(role, model, system, user, kw, nonce)
        text = row.get("text") or ""
        if not text.strip():
            raise ContentInvalid(role)
        obj = extract_json(text)
        if obj is None:
            continue
        try:
            return validate(obj)
        except (TypeError, KeyError, ValueError, AttributeError, IndexError):
            return None
    return None


def pmap(fn: Callable[[Any], Any], items: Sequence[Any], workers: int = 8, label: str = "pmap",
         every: int = 50) -> List[Any]:
    """Ordered parallel map over ``items`` with progress prints every ``every`` completions; exceptions
    propagate to the caller."""
    items = list(items)
    results: List[Any] = [None] * len(items)
    if not items:
        return results
    done = [0]
    lock = threading.Lock()
    every = max(1, int(every))

    def wrap(i: int) -> None:
        results[i] = fn(items[i])
        with lock:
            done[0] += 1
            if done[0] % every == 0 or done[0] == len(items):
                print(f"  [{label}] {done[0]}/{len(items)}", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = [ex.submit(wrap, i) for i in range(len(items))]
        for f in futs:
            f.result()
    return results
