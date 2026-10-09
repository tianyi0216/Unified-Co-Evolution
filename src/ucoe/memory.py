"""Bounded procedural memory, the harness state of the main experiments (paper Sec. 3.3 "Harness update";
App. "Harness-state instantiation"; Table "Default UCoE hyperparameters").

A procedure is one generalized trigger/action pair distilled by the verifier model from an accepted repair
(trigger <= 10 words, action <= 14 words, no task-specific numbers, quotes, links or copied 5-grams). Retrieval
ranks eligible procedures by criterion match, dimension match and record (wins - losses), returning at most
``max_retrieved`` procedures within a word budget; commits de-duplicate within a dimension by token-set Jaccard and
evict the weakest record (ties by age) beyond the per-dimension and global capacities. Rejected revisions update
usage statistics only.
"""

from __future__ import annotations

import inspect
import json
import re
from typing import Iterable, List, Optional, Set, Tuple

from src.llm.client import ContentInvalid, json_call
from src.ucoe import relations
from src.utils.io import sha
from src.utils.text import jaccard, tokens

MEM_CAP = 12
PER_DIM_CAP = 2
MAX_RETRIEVE = 2
TOKEN_CAP = 64
DEDUP_J = 0.75
TRIGGER_WORDS = 10
ACTION_WORDS = 14
LEAK_NGRAM = 5
_FORBIDDEN = re.compile(r"\d|[\"“”]")


def rule_line(r: dict) -> str:
    return f"- ({r['dimension']}) when {r['trigger']}: {r['corrective_action']}"


def _rule_tokens(r: dict) -> Set[str]:
    return set(tokens(r["trigger"] + " " + r["corrective_action"]))


def mem_sha(rules: List[dict]) -> str:
    return sha(json.dumps(rules, sort_keys=True, ensure_ascii=False))


def retrieve(rules: List[dict], weakest: Optional[str], dim: Optional[str], active_crit: Iterable[str],
             ptags: Iterable[str], k: int = MAX_RETRIEVE, token_cap: int = TOKEN_CAP) -> List[dict]:
    """Eligible procedures (record not worse than two net losses, criterion still defined, scope tag matching the
    task) ranked by criterion match, dimension match, record, age and id; at most ``k`` within ``token_cap`` words."""
    active, tags = set(active_crit), set(ptags)
    elig = [r for r in rules
            if r["losses"] <= r["wins"] + 2 and r["criterion_id"] in active
            and ("general" in r["scope_tags"] or set(r["scope_tags"]) & tags)]
    pri = sorted(elig, key=lambda r: (
        0 if (weakest is not None and r["criterion_id"] == weakest) else
        1 if (dim is not None and r["dimension"] == dim) else 2,
        -(r["wins"] - r["losses"]), r["created_at"], r["rule_id"]))
    out, used = [], 0
    for r in pri:
        t = len(rule_line(r).split())
        if len(out) >= k or used + t > token_cap:
            break
        out.append(r)
        used += t
    return out


def commit_rule(rules: List[dict], rule: dict, i: int, reward_version: str, cap: int = MEM_CAP,
                per_dim_cap: int = PER_DIM_CAP, dedup_j: float = DEDUP_J) -> Tuple[List[dict], str]:
    """Append one distilled procedure (statuses: duplicate_skipped | evicted_dim | evicted_cap | appended |
    disabled when the capacity is zero)."""
    if cap <= 0:
        return rules, "disabled"
    toks = _rule_tokens(rule)
    for r in rules:
        if r["dimension"] != rule["dimension"]:
            continue
        rt = _rule_tokens(r)
        if toks == rt or jaccard(toks, rt) >= dedup_j:
            return rules, "duplicate_skipped"
    new = {**rule, "rule_id": f"r{i:05d}_{sha(json.dumps(rule, sort_keys=True))[:6]}", "created_at": i,
           "source_reward_version": reward_version, "wins": 1, "losses": 0, "retrievals": 0}
    rules = list(rules) + [new]
    weakest_first = lambda r: (r["wins"] - r["losses"], r["created_at"], r["rule_id"])  # noqa: E731
    same_dim = [r for r in rules if r["dimension"] == rule["dimension"]]
    if len(same_dim) > per_dim_cap:
        ev = sorted(same_dim, key=weakest_first)[0]
        return [r for r in rules if r["rule_id"] != ev["rule_id"]], "evicted_dim"
    if len(rules) > cap:
        ev = sorted(rules, key=weakest_first)[0]
        return [r for r in rules if r["rule_id"] != ev["rule_id"]], "evicted_cap"
    return rules, "appended"


def account(rules: List[dict], rule_ids: Iterable[str], won: Optional[bool]) -> List[dict]:
    """Usage statistics of the retrieved procedures: every use counts a retrieval; a revision-time use also adds a
    win when the revision was accepted and a loss otherwise. ``won=None`` marks a draft-time use, which has no
    accept/reject outcome and therefore leaves the record untouched."""
    ids = set(rule_ids)
    for r in rules:
        if r["rule_id"] in ids:
            r["retrievals"] += 1
            if won is None:
                continue
            if won:
                r["wins"] += 1
            else:
                r["losses"] += 1
    return rules


def _source_text(task, y0: str, yr: str) -> str:
    text = (task.get("prompt") or task.get("task") or "") if isinstance(task, dict) else str(task)
    return text + " " + (y0 or "") + " " + (yr or "")


def validate_rule(obj, bench, task, y0: str, yr: str, weakest: str, active_crit: Iterable[str]) -> Optional[dict]:
    """Schema and generality checks of a distilled procedure (None when it must not enter the memory)."""
    if not isinstance(obj, dict) or obj.get("dimension") not in tuple(bench.dims):
        return None
    cid = str(obj.get("criterion_id") or weakest)[:24]
    if cid not in set(active_crit):
        return None
    tr = str(obj.get("trigger", "")).strip()
    ca = str(obj.get("corrective_action", "")).strip()
    if not tr or not ca or len(tr.split()) > TRIGGER_WORDS or len(ca.split()) > ACTION_WORDS:
        return None
    text = tr + " " + ca
    if _FORBIDDEN.search(text) or "http" in text.lower():
        return None
    src = tokens(_source_text(task, y0, yr))
    grams = {tuple(src[j:j + LEAK_NGRAM]) for j in range(len(src) - LEAK_NGRAM + 1)}
    tk = tokens(text)
    if any(tuple(tk[j:j + LEAK_NGRAM]) in grams for j in range(len(tk) - LEAK_NGRAM + 1)):
        return None
    allowed = set(getattr(bench, "scope_tags", ("general",)) or ("general",))
    tags = [str(t).strip().lower() for t in (obj.get("scope_tags") or [])]
    tags = [t for t in tags if t in allowed][:2] or ["general"]
    return {"criterion_id": cid, "dimension": obj["dimension"], "scope_tags": tags, "trigger": tr,
            "corrective_action": ca}


def distill(client, meter, bench, task, y0: str, yr: str, witness: str, weakest: str,
            active_crit: Iterable[str], rids: Optional[Iterable[str]] = None) -> Optional[dict]:
    """One procedure distilled from an accepted repair by the verifier model (None when invalid). ``rids`` are
    the ids of the procedures retrieved for the revision, shown by adapters whose distillation prompt lists them."""
    if "rids" in inspect.signature(bench.distill_request).parameters:
        req = bench.distill_request(task, y0, yr, witness, weakest, rids=list(rids or []))
    else:
        req = bench.distill_request(task, y0, yr, witness, weakest)
    tid = task.get("task_id") if isinstance(task, dict) else sha(str(task))[:16]
    active = set(active_crit)
    try:
        return json_call(client, meter, "distill", relations.model_for(bench, "verifier"), req["system"],
                         req["user"], lambda o: validate_rule(o, bench, task, y0, yr, weakest, active),
                         dict(bench.distill_kw), f"distill|{tid}")
    except ContentInvalid:
        return None
