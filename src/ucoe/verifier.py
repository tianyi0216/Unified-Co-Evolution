"""Verifier programs: criterion blocks, bounded aggregation expressions and proposal parsing (paper Sec. 3.2
"Initial verifier synthesis", Eq. reward-interface; App. "Implementation details").

A verifier is a program over typed criterion scorers: an ordered list of blocks (the base block ``B0`` with the
benchmark's catalogue, an optional ``X_rd`` block with mined relation modules, and one appended extension block per
synthesis step) plus one aggregation expression over criterion ids that must stay within [0, 1]. Verifiers are
immutable and content-addressed (``sha256`` over version, parent, blocks and aggregation); a child keeps every
block of its parent and may append at most ``max_new`` criteria (static validator: defined ids, every new id used,
finite probe values within the unit interval).
"""

from __future__ import annotations

import ast
import json
import math
import random
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.utils.io import sha

BASE_BLOCK_ID = "B0"
ID_RE = re.compile(r"[a-z][a-z0-9_]{1,24}")
MAX_QUESTION_CHARS = 220
MAX_ANCHOR_CHARS = 160
ALLOWED_FUNCS = ("min", "max", "sorted", "abs")
MAX_CONSTANT = 1e6          # numeric literals beyond this magnitude (or non-numeric literals) fail the validator
MAX_EXPR_CHARS = 4000
PROBE_SEED = 7
N_RANDOM_PROBES = 12
_ALLOWED_NODES = tuple(
    n
    for n in (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call, ast.Name, ast.Load, ast.Constant, ast.IfExp,
        ast.BoolOp, ast.And, ast.Or, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd, ast.Lt, ast.LtE,
        ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.List, ast.Tuple, ast.Subscript, ast.Slice,
        getattr(ast, "Index", None),
    )
    if n is not None
)


# ---------------------------------------------------------------- blocks and versions
def base_block(bench) -> dict:
    """The benchmark's base criterion block (ids and definitions verbatim from its suite)."""
    return {"block_id": BASE_BLOCK_ID,
            "criteria": [{"id": c, "definition": bench.module_defs[c]} for c in bench.module_ids]}


def block_sha(block: dict) -> str:
    return sha(json.dumps(block, sort_keys=True, ensure_ascii=False))


def verifier_sha(V: dict) -> str:
    body = {k: V[k] for k in ("version", "parent_sha256", "blocks", "aggregation")}
    return sha(json.dumps(body, sort_keys=True, ensure_ascii=False))


def make_verifier(version: str, blocks: Sequence[dict], aggregation: str, parent_sha: str = "root",
                  origin: str = "synthesized", rd_modules: Optional[List[dict]] = None) -> dict:
    """A frozen verifier; ``rd_modules`` carries the relation modules whose criteria sit in the ``X_rd`` block."""
    V = {"version": version, "parent_sha256": parent_sha,
         "blocks": [json.loads(json.dumps(b)) for b in blocks], "aggregation": str(aggregation).strip(),
         "origin": origin}
    if rd_modules:
        V["rd_modules"] = [json.loads(json.dumps(m)) for m in rd_modules]
    V["sha256"] = verifier_sha(V)
    return V


def make_child(version: str, parent: dict, ext_block: Optional[dict], aggregation: str) -> dict:
    """Cumulative archive: the child keeps every block of its parent and appends ``ext_block`` when given."""
    blocks = list(parent["blocks"]) + ([ext_block] if ext_block else [])
    return make_verifier(version, blocks, aggregation, parent_sha=parent["sha256"],
                         origin=parent.get("origin", "synthesized"), rd_modules=parent.get("rd_modules"))


def crit_ids(V: dict) -> List[str]:
    return [c["id"] for b in V["blocks"] for c in b["criteria"]]


def ext_ids(V: dict) -> List[str]:
    return [c["id"] for b in V["blocks"] if b["block_id"] != BASE_BLOCK_ID for c in b["criteria"]]


def defs(V: dict) -> Dict[str, str]:
    return {c["id"]: c["definition"] for b in V["blocks"] for c in b["criteria"]}


def active_ids(V: dict) -> List[str]:
    """Criteria that participate in the aggregation (word-bounded occurrence); all criteria when none is named."""
    expr = V.get("aggregation") or ""
    act = [c for c in crit_ids(V) if re.search(rf"\b{re.escape(c)}\b", expr)]
    return act or crit_ids(V)


# ---------------------------------------------------------------- aggregation expressions
def check_expr(expr: str, names: Iterable[str]) -> bool:
    """AST whitelist: arithmetic, comparisons, conditionals, min/max/sorted/abs calls, list indexing, known names and
    finite numeric literals of bounded magnitude (strings and huge numbers could make evaluation allocate or
    overflow)."""
    if len(expr) > MAX_EXPR_CHARS:
        return False
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError):
        return False
    allowed = set(names) | set(ALLOWED_FUNCS)
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return False
        if isinstance(node, ast.Constant) and not isinstance(node.value, bool):
            v = node.value
            if not isinstance(v, (int, float)) or abs(v) > MAX_CONSTANT:
                return False
            if isinstance(v, float) and not math.isfinite(v):
                return False
        if isinstance(node, ast.Name) and node.id not in allowed:
            return False
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCS):
            return False
    return True


def eval_expr(expr: str, vals: Dict[str, float]) -> float:
    env: Dict[str, Any] = dict(vals)
    env.update({"min": min, "max": max, "sorted": sorted, "abs": abs, "__builtins__": {}})
    return float(eval(expr, env))  # noqa: S307 - the expression passed check_expr's whitelist


def agg_ok(expr: str, ids: Sequence[str], must_use: Iterable[str]) -> bool:
    """Safe, executable, finite, bounded to [0, 1] on 15 probe vectors, and referencing every ``must_use`` id."""
    if not check_expr(expr, ids):
        return False
    for mid in must_use:
        if not re.search(rf"\b{re.escape(mid)}\b", expr):
            return False
    rng = random.Random(PROBE_SEED)
    probes = [{i: 0.0 for i in ids}, {i: 1.0 for i in ids}, {i: 0.5 for i in ids}]
    probes += [{i: rng.random() for i in ids} for _ in range(N_RANDOM_PROBES)]
    for p in probes:
        try:
            v = eval_expr(expr, p)
        except (ArithmeticError, ValueError, TypeError, IndexError, KeyError, RecursionError, MemoryError):
            return False
        if not (isinstance(v, float) and math.isfinite(v) and -1e-9 <= v <= 1 + 1e-9):
            return False
    return True


def agg_score(V: dict, vals: Dict[str, float]) -> float:
    """The verifier's reward: the aggregation evaluated on the criterion scores (missing criteria score 0)."""
    return eval_expr(V["aggregation"], {k: float(vals.get(k, 0.0)) for k in crit_ids(V)})


# ---------------------------------------------------------------- proposals
def parse_proposal(obj: Any, retained_ids: Sequence[str], max_new: int = 2, min_new: int = 0,
                   leak_terms: Iterable[str] = ()) -> Optional[Tuple[Optional[dict], str]]:
    """Parse a synthesis reply into ``({"criteria": [...]} | None, aggregation)`` or None when invalid.

    The listed criteria must equal the retained ids plus the proposal's own new ids; new ids match
    ``[a-z][a-z0-9_]{1,24}`` and are not retained; questions (<= 220 chars) and anchors (<= 160) are non-empty and
    free of leak terms and long digit runs; the aggregation passes ``agg_ok`` over all ids and uses every new id.
    """
    if not isinstance(obj, dict):
        return None
    crit, new = obj.get("criteria"), obj.get("new_criteria", [])
    agg = str(obj.get("aggregation", "")).strip()
    if new is None:
        new = []
    if not isinstance(crit, list) or not agg or not isinstance(new, list) or not min_new <= len(new) <= max_new:
        return None
    retained = set(retained_ids)
    own_new = {str(c.get("id", "")).strip() for c in new if isinstance(c, dict)}
    kept = {str(c) for c in crit} - own_new
    if kept != retained:
        return None
    exts, seen = [], set()
    leaks = tuple(str(t).lower() for t in leak_terms)
    for c in new:
        if not isinstance(c, dict):
            return None
        cid = str(c.get("id", "")).strip()
        q = str(c.get("question", "")).strip()
        a0 = str(c.get("anchor0", "")).strip()
        a1 = str(c.get("anchor1", "")).strip()
        if (not ID_RE.fullmatch(cid) or cid in retained or cid in seen or not q or not a0 or not a1
                or len(q) > MAX_QUESTION_CHARS or len(a0) > MAX_ANCHOR_CHARS or len(a1) > MAX_ANCHOR_CHARS):
            return None
        blob = (cid + " " + q + " " + a0 + " " + a1).lower()
        if any(t in blob for t in leaks) or re.search(r"\d{3,}", blob):
            return None
        seen.add(cid)
        exts.append({"id": cid, "definition": f"{cid}: {q} (0: {a0}; 1: {a1})"})
    ids = sorted(retained | {e["id"] for e in exts})
    if not agg_ok(agg, ids, [e["id"] for e in exts]):
        return None
    return ({"criteria": exts} if exts else None), agg
