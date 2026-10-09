"""Filesystem and hashing primitives shared by every stage of the pipeline.

They implement the resumability conventions behind Algorithm 1 of the paper (the deployment stream is processed
once, in order, and every artifact must survive an interruption): sha256 helpers, atomic JSON writes, durable
JSONL appends with crash-tail repair, gzip JSON bundles and the deterministic seed derivation used for every
sampled model call (App. "Implementation details").
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import threading
import time
from typing import Any, List, Optional


# ------------------------------------------------------------------ hashing / seeds

def sha(text: str) -> str:
    """sha256 hex digest of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha_json(obj: Any) -> str:
    """sha256 of the canonical (sorted-keys) JSON rendering of ``obj``."""
    return sha(json.dumps(obj, sort_keys=True, ensure_ascii=False))


def file_sha(path: str) -> str:
    """sha256 hex digest of a file's bytes (streamed)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_from(ns: str, kind: str, key: str) -> int:
    """Deterministic 31-bit seed: int(sha256(f"{ns}-{kind}|{key}")) % 2**31."""
    return int(hashlib.sha256(f"{ns}-{kind}|{key}".encode("utf-8")).hexdigest(), 16) % (2 ** 31)


# ------------------------------------------------------------------ directories / atomic writes

def ensure_dir(path: str) -> str:
    """Create ``path`` (and parents) if missing; returns ``path``."""
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def _parent(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _replace(tmp: str, path: str) -> None:
    for attempt in range(8):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.25 * (attempt + 1))


def atomic_write(path: str, body: str) -> None:
    """Temp file + flush + fsync + atomic replace: an interrupted write never leaves a partial file."""
    _parent(path)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(body)
        f.flush()
        os.fsync(f.fileno())
    _replace(tmp, path)


# ------------------------------------------------------------------ JSON

def read_json(path: str, default: Any = None) -> Any:
    """Load a JSON file, or return ``default`` when it does not exist."""
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any, indent: Optional[int] = None) -> None:
    """Atomically write ``obj`` as JSON."""
    atomic_write(path, json.dumps(obj, ensure_ascii=False, indent=indent))


def read_gz_json(path: str) -> Any:
    """Load a gzip-compressed JSON file."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def write_gz_json(path: str, obj: Any) -> None:
    """Atomically write ``obj`` as gzip-compressed JSON; the gzip header carries no timestamp or file name, so
    identical content gives identical bytes on every run."""
    _parent(path)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:
        f.write(json.dumps(obj, ensure_ascii=False).encode("utf-8"))
    _replace(tmp, path)


# ------------------------------------------------------------------ JSONL (append-only, crash-safe)

def _last_newline(f, size: int) -> int:
    """Byte offset of the last b"\\n" in an open binary file, or -1 (scans backwards in chunks)."""
    chunk = 1 << 16
    end = size
    while end > 0:
        start = max(0, end - chunk)
        f.seek(start)
        buf = f.read(end - start)
        idx = buf.rfind(b"\n")
        if idx >= 0:
            return start + idx
        end = start
    return -1


def repair_jsonl_tail(path: str) -> str:
    """Make a JSONL file end on a newline so that an append can never corrupt an earlier row.

    A file that ends on a newline (or is empty / missing) is left alone. An unterminated tail that parses as
    JSON is completed with a newline (a hand-written file without a final newline); an unterminated tail that
    does not parse is a crash-truncated row and is cut back to the last complete line.
    Returns one of "missing", "ok", "completed", "truncated".
    """
    if not os.path.exists(path):
        return "missing"
    with open(path, "r+b") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        if size == 0:
            return "ok"
        f.seek(size - 1)
        if f.read(1) == b"\n":
            return "ok"
        cut = _last_newline(f, size) + 1
        f.seek(cut)
        tail = f.read()
        try:
            json.loads(tail.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            f.truncate(cut)
            f.flush()
            os.fsync(f.fileno())
            return "truncated"
        f.seek(0, os.SEEK_END)
        f.write(b"\n")
        f.flush()
        os.fsync(f.fileno())
        return "completed"


def read_jsonl(path: str) -> List[Any]:
    """Read every row of a JSONL file (missing file -> []).

    A malformed final line without a trailing newline is a crash-truncated row: it is skipped with a warning
    (``append_jsonl`` repairs it before the next write). A malformed line anywhere else raises ``ValueError``.
    """
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        raw = f.read()
    rows: List[Any] = []
    pieces = raw.split(b"\n")
    last = len(pieces) - 1
    terminated = raw.endswith(b"\n")
    for i, line in enumerate(pieces):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line.decode("utf-8")))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if i == last and not terminated:
                print(f"[ucoe-io] {path}: dropping a crash-truncated final line", flush=True)
                break
            raise ValueError(f"{path}: line {i + 1} is not valid JSON") from None
    return rows


def append_jsonl(path: str, row: Any) -> None:
    """Append one JSON row durably (newline-terminated, flushed and fsynced), repairing a damaged tail first."""
    _parent(path)
    repair_jsonl_tail(path)
    line = json.dumps(row, ensure_ascii=False) + "\n"
    for attempt in range(6):
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            return
        except (PermissionError, OSError):
            if attempt == 5:
                raise
            time.sleep(0.5 * (attempt + 1))
