"""Relation discovery and relation-family contrasts (paper Sec. 3.2 "relation discovery"; App. "Relation families").

The actor and a larger frozen comparison model answer the same development prompts.  The verifier model screens each
pair for completeness, reads it in a seeded random order with the system identities hidden, and names atomic,
task-general differences.  The proposals are consolidated into a few recurring modules, each kept only if it recurs
in at least ``min_recurrence`` development prompts and can be written as one binary criterion with one edit that
breaks that property alone.  Every frozen module later serves two purposes: its definition is offered to verifier
synthesis as a retained criterion, and its break recipe builds relation-family contrasts (one validated local patch
of the actor's own base response).

Public functions:
  discover(client, meter, bench, cfg, work_dir, workers)   pairs.json + proposals.json
  mine(client, meter, bench, cfg, work_dir)                frozen relations/modules.json (one repair retry)
  applicable(client, meter, store, bench, module, task)    keyword fast path, else a cached verifier-model verdict
  patch_contrast(client, meter, bench, module, source, base_response, split, nonce_budget=3)
  null_contrast(client, meter, bench, source, base_response, split)   paraphrase control (never a selection family)
  constructibility_check(bank_select, bank_heldout, modules, quotas)   usable module ids
  request_text(client, role, model, request, kw, seed_key)             actor-side request helper (shared)

Prompts, regular expressions and text limits are those of the paper's runs.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

from src.llm.client import ContentInvalid, json_call, pmap
from src.utils.config import cfg_get
from src.utils.io import ensure_dir, read_json, seed_from, sha, write_json

# ---------------------------------------------------------------- prompts (verbatim)
SCREEN_S = "You check whether two anonymous responses basically complete a public task and whether either is cut off. Respond with exactly one JSON object."
SCREEN_U = (
    "Task:\n{task}\n\n<RESPONSE_A>\n{a}\n</RESPONSE_A>\n\n<RESPONSE_B>\n{b}\n</RESPONSE_B>\n\n"
    'JSON: {{"a_complete": true|false, "b_complete": true|false, "a_truncated": true|false, "b_truncated": true|false}}. Nothing else.'
)
MINE_S = (
    "You compare two anonymous responses to the same real task and extract ATOMIC, checkable differences, each framed as a RELATION a strict "
    "grader can verify (e.g. 'each proposed measure addresses a cause the response itself stated', 'the register matches the named audience', "
    "'an abstract recommendation is instantiated by a concrete case'). You do not know which system wrote which response and you must not "
    "judge which whole response is better. Exclude length/verbosity, wording polish, unsupported extra detail, a system's habitual style, and "
    "anything specific to this one task. Respond with exactly one JSON object."
)
MINE_U = (
    "Task:\n{task}\n\n<RESPONSE_A>\n{a}\n</RESPONSE_A>\n\n<RESPONSE_B>\n{b}\n</RESPONSE_B>\n\n"
    'JSON: {{"modules": [{{"name": "<3-6 words>", "relation": "<the specific relation checked>", "question": "<operational 0-1 grader question>", '
    '"anchor0": "<what a 0 looks like>", "anchor1": "<what a 1 looks like>", "applicability": "<which tasks this applies to>", '
    '"evidence_a": "<verbatim span from A, <=200 chars>", "evidence_b": "<verbatim span from B, <=200 chars>", '
    '"break_recipe": "<how to break ONLY this relation inside a short span>", "preserve": "<what must stay unchanged>"}}]}} (1 to 4 items). Nothing else.'
)
CLUSTER_S = (
    "You consolidate atomic difference modules proposed independently from several real tasks into at most {k} RECURRING atomic modules. Keep each "
    "module atomic (one checkable relation); do not collapse them into generic 'logic', 'specificity' or 'writing quality'. Keep a module only if "
    "proposals from at least 3 different source ids describe the same relation. Reject modules rewarding length, wording, unsupported detail or a "
    "system's style. Respond with exactly one JSON object."
)
CLUSTER_S3 = CLUSTER_S.replace(
    "Respond with exactly one JSON object.",
    "Both 0/1 anchors must be opposite outcomes on that SAME relation, not different task examples. The break recipe must replace or delete one short "
    "span to degrade the relation, and preserve must protect only non-target attributes. Use task-general wording without source-specific names, "
    "facts, passage labels or numbers. Respond with exactly one JSON object.",
)
CLUSTER_U = (
    "Proposals (source :: name :: relation :: question :: applicability :: break recipe :: preserve):\n{items}\n\n"
    'JSON: {{"modules": [{{"id": "<[a-z][a-z0-9_]{{2,8}}>", "name": "...", "relation": "...", "question": "<0-1 grader question>", "anchor0": "...", '
    '"anchor1": "...", "applicability_keywords": ["<lowercase task cues>"], "applicability": "...", "break_recipe": "...", "preserve": "...", '
    '"member_sources": ["<source ids>"]}}], "rejected": [{{"name": "...", "reason": "..."}}]}}. Nothing else.'
)
MINE_S2 = MINE_S + (
    " State every relation in TASK-GENERAL terms: no names, entities, numbers, passage or material ids from this task; the same "
    "relation must be recognizable in a different task of this benchmark."
)
MINE_U2 = MINE_U.replace(
    '"relation": "<the specific relation checked>"', '"relation": "<the relation checked, task-general wording>"'
)
CLUSTER_U2 = CLUSTER_U.replace(
    '"member_sources": ["<source ids>"]',
    '"member_sources": ["<EVERY source id whose proposal instantiates this module -- required>"]',
)
ASSIGN_S = (
    "You decide which proposed atomic modules instantiate one general module. Respond with exactly one JSON object."
)
ASSIGN_U = (
    "General module: {name} :: {relation} :: {question}\n\nProposals (source :: name :: relation):\n{items}\n\n"
    'JSON: {{"member_sources": ["<source ids whose proposal checks this same relation>"]}}. Nothing else.'
)
MODULE_AUDIT_S = (
    "You are a skeptical pre-freeze auditor of an atomic response-quality module. Reject ambiguous or task-specific modules. "
    "Judge the proposed module itself, not whether a particular answer is good. Respond with exactly one JSON object."
)
MODULE_AUDIT_U = (
    "Candidate module:\n{module}\n\n"
    "Mark each condition true ONLY when it is unambiguously satisfied:\n"
    "same_axis: anchor0 and anchor1 are opposite quality outcomes for the SAME checkable relation; neither is a grading instruction "
    "or an example from a different task.\n"
    "localized_degradation: break_recipe tells an editor how to REPLACE or DELETE a short response span so the relation goes from "
    "holding to failing; it is not a checking procedure, an append-only change, or a broad rewrite.\n"
    "preserve_separable: preserve protects only non-target facts/qualities and does NOT require keeping the relation, link, "
    "content, or evidence that the recipe must break.\n"
    "task_general: relation, question, BOTH anchors, recipe, and preserve contain no names, quoted facts, numbers, passage "
    "labels, or details tied to a particular source task.\n"
    "atomic: there is just one checkable relation, not a bundle of independent criteria.\n"
    "substantive: the relation is about the correctness, faithfulness, completeness or task-fit of the content -- NOT merely the "
    "presence, location, length, formatting or verbosity of explanation, comments or prose (unless the task explicitly requires it).\n\n"
    'JSON: {{"same_axis": true|false, "localized_degradation": true|false, "preserve_separable": true|false, '
    '"task_general": true|false, "atomic": true|false, "substantive": true|false, "reason": "<specific reason if any condition is false>"}}. Nothing else.'
)
APPLIC_S = "You decide whether a quality module is checkable for a task. Respond with exactly one JSON object."
APPLIC_U = (
    "Task:\n{task}\n\nModule: {definition}\nApplies to: {appl}\n\nIs this module checkable for a response to this task? "
    'JSON: {{"applies": true|false}}. Nothing else.'
)
PATCH_S = (
    "You build a minimal controlled contrast on ONE SHORT span of a response for exactly one atomic relation. Copy the span verbatim from the "
    "response (40-900 characters, appearing exactly once). Give a POSITIVE version of that span in which the relation clearly holds (keep the "
    "original if it already does; otherwise repair it minimally without adding facts from outside the task and response) and a NEGATIVE version "
    "of similar length and fluency in which ONLY this relation is broken (replace the relation; do not merely append a filler sentence). "
    "Everything outside the span stays untouched. Respond with exactly one JSON object."
)
PATCH_U = (
    "Task:\n{task}\n\n<RESPONSE>\n{resp}\n</RESPONSE>\n\nRelation: {relation}\nModule: {definition}\nHow to break only it: {recipe}\nPreserve: {preserve}\n\n"
    'JSON: {{"target_quote": "<verbatim span>", "positive_span": "<span where the relation holds>", "replacement": "<span where only this relation is broken>", '
    '"what_breaks": "<one clause>"}}. Nothing else.'
)
VALID_S = (
    "Two anonymous versions of one response differ only inside the marked ⟦...⟧ span. Judge the marked place in the context of the whole task and "
    "response. Respond with exactly one JSON object."
)
VALID_U = (
    "Task:\n{task}\n\nRelation to check: {relation}\nModule: {definition}\n\n<VERSION_X>\n{x}\n</VERSION_X>\n\n<VERSION_Y>\n{y}\n</VERSION_Y>\n\n"
    'JSON: {{"relation_holds_in": "X"|"Y"|"both"|"neither", "other_attributes_damaged_in": true|false}}. '
    'Set other_attributes_damaged_in to false only if neither version damages any other attribute. Nothing else.'
)
NULL_S = (
    "You rewrite ONE span of a response so that its meaning, information content and quality are unchanged (a faithful paraphrase of equal length "
    "and register). Copy the span verbatim (40-900 chars, appearing once). Respond with exactly one JSON object."
)
NULL_U = 'Task:\n{task}\n\n<RESPONSE>\n{resp}\n</RESPONSE>\n\nJSON: {{"target_quote": "<verbatim span>", "paraphrase": "<equivalent rewrite>"}}. Nothing else.'

# ---------------------------------------------------------------- decoding defaults and gates (verbatim limits)
JUDGE_KW = {"max_tokens": 300, "temperature": 0.0}
MINE_KW = {"max_tokens": 2500, "temperature": 0.0}
CLUSTER_KW = {"max_tokens": 4000, "temperature": 0.0}
PATCH_KW = {"max_tokens": 1500, "temperature": 0.4}
REJECT_RE = re.compile(
    r"\b(verbos|word count|length|longer|wording|eloquen|polish|formatting|markdown|bullet|headings?|style)\b", re.I
)
# The mere presence / location / length / formatting / verbosity of explanatory text, comments or docstrings is not a quality
# relation unless the task explicitly requires it; such modules are rejected before the semantic audit.
PRESENCE_RE = re.compile(
    r"\b(presence of|includes?|contains?|provides?|has|adds?|accompan\w*|placement|position(ed)?|located|before|after|outside|inside|beyond)\b"
    r".{0,40}\b(explanation|explanatory|explains?|commentary|comments?|docstrings?|narrative|prose|natural.language|description|summary)\b"
    r"|\b(explanation|explanatory|commentary|comments?|docstrings?)\b.{0,40}\b(present|included|provided|exists?|appears?|placement|position|located|outside|inside|length|amount|verbos)\b",
    re.I | re.S,
)
MODULE_AUDIT_KEYS = (
    "same_axis",
    "localized_degradation",
    "preserve_separable",
    "task_general",
    "atomic",
    "substantive",
)
MODULE_TEXT_LIMITS = {
    "name": 80,
    "relation": 300,
    "question": 300,
    "anchor0": 250,
    "anchor1": 250,
    "applicability": 300,
    "break_recipe": 400,
    "preserve": 300,
}
TASK_DETAIL_RE = re.compile(
    r"\b(?:response\s+[ab]|passage\s+\[?[a-z]\]?|group\s+\d+|q[1-4]|(?:19|20)\d{2}|(?:refinebench|writingbench)[-_]\d+)\b",
    re.I,
)
TRUNCATION_RE = re.compile(r"(?:\.{3,}|…|\[\s*(?:truncated|ellipsis)\s*\]|<\s*truncated\s*>)", re.I)
CHECK_ONLY_RE = re.compile(
    r"^(?:check|verify|assess|evaluate|compare|ensure|look\s+for|determine|judge|score|test|identify|see\s+whether)\b",
    re.I,
)
MODULE_ID_RE = re.compile(r"[a-z][a-z0-9_]{2,8}")
RELATIONS_DIR = "relations"


# ---------------------------------------------------------------- small helpers
def _norm(t: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip().lower()


def _words(t: Optional[str]) -> List[str]:
    return re.findall(r"[a-z0-9一-鿿]{2,}", (t or "").lower())


def _ns(client) -> str:
    return str(getattr(client, "ns", "ucoe"))


def model_for(bench, role: str, cfg: Optional[dict] = None) -> str:
    """Model name for a role ("actor", "verifier", "comparator", "judge") from the config carried by the adapter."""
    cfg = cfg if cfg is not None else getattr(bench, "cfg", None)
    model = cfg_get(cfg or {}, f"models.{role}")
    if not model:
        raise RuntimeError(f"models.{role} is not configured")
    return str(model)


def relations_dir(work_dir: str) -> str:
    return os.path.join(work_dir, RELATIONS_DIR)


def request_text(client, role: str, model: str, request: dict, kw: dict, seed_key: str, nonce: int = 0) -> str:
    """Send a Request ({system, user} or {messages}) through the cache-first client; returns the stripped text."""
    kw = dict(kw or {})
    kw["seed"] = seed_from(_ns(client), "call", seed_key)
    if "messages" in request:
        row = client.call_messages(role, model, request["messages"], kw, nonce)
    else:
        row = client.call(role, model, request["system"], request["user"], kw, nonce)
    return str(row.get("text") or "").strip()


def _json(client, meter, role, model, system, user, validate, kw, seed_key):
    """json_call with the invalid-path fallback: empty output counts as no answer."""
    try:
        return json_call(client, meter, role, model, system, user, validate, kw, seed_key)
    except ContentInvalid:
        return None


def _locate(base: str, quote: str) -> Optional[str]:
    if base.count(quote) == 1:
        return quote
    pat = r"\s*".join(re.escape(tok) for tok in quote.split())
    hits = list(re.finditer(pat, base)) if quote.strip() else []
    return hits[0].group(0) if len(hits) == 1 else None


def _patch_ok(base: str, quote: str, pos: str, neg: str) -> Optional[str]:
    if not quote or not neg or len(quote) < 40 or len(quote) > 900 or base.count(quote) != 1:
        return "span length/uniqueness"
    if _norm(pos) == _norm(neg):
        return "pos == neg"
    if len(set(_words(pos)) ^ set(_words(neg))) < 3:
        return "trivial pos/neg delta"
    if _norm(neg).startswith(_norm(pos)) and len(_norm(neg)) > len(_norm(pos)):
        return "negative merely appends to the positive (filler, not a replaced relation)"
    return None


def _mark(text: str, span: str) -> str:
    return text.replace(span, "⟦" + span + "⟧", 1)


def _single_factor_valid(v, pos_label: str) -> bool:
    return (
        isinstance(v, dict)
        and v.get("relation_holds_in") == pos_label
        and v.get("other_attributes_damaged_in") is False
    )


def module_static_issue(m: dict, mid: str) -> Optional[str]:
    """Cheap, conservative gates before the semantic audit; never silently truncate a frozen definition."""
    for key, limit in MODULE_TEXT_LIMITS.items():
        value = m.get(key)
        if not isinstance(value, str) or not value.strip():
            return f"missing/non-text {key}"
        if len(value.strip()) > limit:
            return f"{key} exceeds {limit} characters (would be truncated)"
    if not isinstance(m.get("applicability_keywords"), list) or any(
        not isinstance(k, str) or not k.strip() or len(k) > 80 for k in m["applicability_keywords"]
    ):
        return "invalid applicability keywords"
    if len(m["applicability_keywords"]) > 12:
        return "too many applicability keywords (would be truncated)"
    fields = " ".join(m[k] for k in MODULE_TEXT_LIMITS)
    if TRUNCATION_RE.search(fields):
        return "truncation/ellipsis in module text"
    if TASK_DETAIL_RE.search(fields):
        return "task-specific response/passage/date identifier"
    if _norm(m["anchor0"]) == _norm(m["anchor1"]):
        return "identical 0/1 anchors"
    if CHECK_ONLY_RE.match(m["break_recipe"].strip()):
        return "break recipe is a check, not an edit"
    if CHECK_ONLY_RE.match(m["anchor0"].strip()) or CHECK_ONLY_RE.match(m["anchor1"].strip()):
        return "anchor is a grading instruction, not a quality outcome"
    if _norm(m["preserve"]) in (_norm(m["relation"]), _norm(m["question"])):
        return "preserve restates the targeted relation"
    definition = f"{mid}: {m['question'].strip()} (0: {m['anchor0'].strip()}; 1: {m['anchor1'].strip()})"
    if len(definition) > 900:
        return "definition exceeds 900 characters (would be truncated)"
    return None


def module_audit_issue(audit) -> Optional[str]:
    if not isinstance(audit, dict):
        return "semantic audit returned no valid JSON"
    failed = [key for key in MODULE_AUDIT_KEYS if audit.get(key) is not True]
    return (
        ("semantic audit failed " + ", ".join(failed) + ": " + str(audit.get("reason", ""))[:200]) if failed else None
    )


def _module_id(m: dict, used: set, taken: set) -> str:
    """A frozen module id: the proposal's own id when well-formed and free, else a slug of its name."""
    mid = str(m.get("id", "")).strip()
    if not MODULE_ID_RE.fullmatch(mid) or mid in used or mid in taken:
        base = [
            w
            for w in re.sub(r"[^a-z0-9]+", "_", str(m.get("name", "mod")).lower()).split("_")
            if w and w not in ("and", "the", "of", "to", "a")
        ]
        mid = ("_".join(w[:4] for w in base[:2])[:8] or "mod").ljust(3, "x")
        k = 2
        while mid in used or mid in taken:
            mid, k = mid[:6] + str(k), k + 1
    return mid


# ---------------------------------------------------------------- discovery
def discover(client, meter, bench, cfg, work_dir, workers) -> dict:
    """Blinded development pairs (actor vs comparator), completeness screen, and per-pair module proposals."""
    rd_dir = relations_dir(work_dir)
    ensure_dir(rd_dir)
    n = int(cfg_get(cfg, "ucoe.relation_discovery.n_discovery_prompts", 32))
    max_pairs = int(cfg_get(cfg, "ucoe.relation_discovery.max_pairs", 24))
    sources = list(bench.discovery_sources())[:n]
    by_id = {s["source_id"]: s for s in sources}
    order = [s["source_id"] for s in sources]
    actor, comparator = model_for(bench, "actor", cfg), model_for(bench, "comparator", cfg)
    judge = model_for(bench, "verifier", cfg)
    ns = _ns(client)
    pairs_p = os.path.join(rd_dir, "pairs.json")
    pairs = read_json(pairs_p, None) or {}

    def one(sid):
        if sid in pairs:
            return sid, pairs[sid]
        src = by_id[sid]
        req = bench.draft_request(src, "")
        ya = request_text(client, "discovery_actor", actor, req, bench.gen_kw, f"disc|{sid}")
        yb = request_text(client, "discovery_comparator", comparator, req, bench.gen_kw, f"disc|{sid}")
        flip = int(sha(f"{ns}|ab|{sid}"), 16) % 2 == 1
        a, b = (yb, ya) if flip else (ya, yb)
        scr = _json(
            client,
            meter,
            "screen",
            judge,
            SCREEN_S,
            SCREEN_U.format(task=src["task"], a=a, b=b),
            lambda o: o if isinstance(o, dict) and "a_complete" in o else None,
            JUDGE_KW,
            f"screen|{sid}",
        )
        ok = bool(
            scr
            and scr.get("a_complete")
            and scr.get("b_complete")
            and not scr.get("a_truncated")
            and not scr.get("b_truncated")
            and ya
            and yb
        )
        return sid, {
            "A": a,
            "B": b,
            "flip": flip,
            "target_sha256": sha(ya),
            "comparator_sha256": sha(yb),
            "screen": scr,
            "qualifies": ok,
        }

    pairs.update(dict(pmap(one, order, workers, "relations_discover", every=8)))
    write_json(pairs_p, pairs)
    qual = [sid for sid in order if pairs.get(sid, {}).get("qualifies")][:max_pairs]
    props_p = os.path.join(rd_dir, "proposals.json")
    props = read_json(props_p, None) or {}

    def mine_one(sid):
        if sid in props:
            return sid, props[sid]
        pr = pairs[sid]

        def val(o):
            if not isinstance(o, dict):
                return None
            return [
                m for m in (o.get("modules") or []) if isinstance(m, dict) and m.get("relation") and m.get("question")
            ][:4]

        got = _json(
            client,
            meter,
            "mine",
            judge,
            MINE_S2,
            MINE_U2.format(task=by_id[sid]["task"], a=pr["A"], b=pr["B"]),
            val,
            MINE_KW,
            f"mine|{sid}",
        )
        return sid, got or []

    props.update(dict(pmap(mine_one, qual, workers, "relations_mine", every=8)))
    write_json(props_p, props, indent=1)
    n_q = sum(1 for v in pairs.values() if v.get("qualifies"))
    print(
        f"[ucoe-relations] discover: pairs {len(pairs)} (flipped {sum(1 for v in pairs.values() if v.get('flip'))}), "
        f"qualifying {n_q} -> used {len(qual)} (cap {max_pairs}); proposals {sum(len(v) for v in props.values())}",
        flush=True,
    )
    for sid in qual[:2]:
        for m in props.get(sid, [])[:2]:
            print(f"    {sid}: {m.get('name')} :: {str(m.get('relation', ''))[:150]}", flush=True)
    return {"pairs": pairs, "proposals": props, "qualifying": qual}


def _cluster_once(client, meter, bench, props, judge, max_modules, min_recurrence, attempt, prior):
    """One clustering pass: consolidation, assignment fallback, lexical and static gates, semantic audit."""
    items = [
        f"{sid} :: {m['name']} :: {m['relation']} :: {m['question']} :: {m.get('applicability', '')} :: "
        f"{m.get('break_recipe', '')} :: {m.get('preserve', '')}"
        for sid in sorted(props)
        for m in props[sid]
    ]
    short = [f"{sid} :: {m['name']} :: {m['relation']}" for sid in sorted(props) for m in props[sid]]
    cluster_user = CLUSTER_U2.format(items="\n".join(items))
    if prior:
        cluster_user += (
            "\n\nRepair the first cluster attempt. Every candidate was rejected for these reasons; propose corrected, "
            "task-general modules from the SAME proposals (no new sources or selection/outcome feedback):\n"
            + json.dumps(prior, ensure_ascii=False)[:4000]
        )
    got = _json(
        client,
        meter,
        "cluster",
        judge,
        CLUSTER_S3.format(k=max_modules + 2),
        cluster_user,
        lambda o: o if isinstance(o, dict) and isinstance(o.get("modules"), list) else None,
        CLUSTER_KW,
        "cluster" if attempt == 0 else "cluster|repair1",
    )
    if not got:
        return [], [{"name": "<cluster>", "reason": "clustering produced no valid JSON"}], {}
    mods, rejected, used, assigned = [], list(got.get("rejected") or []), set(), {}
    taken = set(bench.module_ids)
    for m in got["modules"]:
        if not isinstance(m, dict):
            rejected.append({"name": "<non-object>", "reason": "cluster item is not an object"})
            continue
        srcs = sorted(set(str(x) for x in (m.get("member_sources") or []) if str(x) in props))
        if len(srcs) < min_recurrence and m.get("relation") and m.get("question"):
            a = _json(
                client,
                meter,
                "assign",
                judge,
                ASSIGN_S,
                ASSIGN_U.format(
                    name=m.get("name", ""), relation=m["relation"], question=m["question"], items="\n".join(short)
                ),
                lambda o: [str(x) for x in (o.get("member_sources") or [])] if isinstance(o, dict) else None,
                CLUSTER_KW,
                f"assign|{_norm(str(m.get('name', '')))[:40]}",
            )
            if a:
                srcs = sorted(set(srcs) | {x for x in a if x in props})
                assigned[str(m.get("name", ""))] = srcs
        mid = _module_id(m, used, taken)
        text = " ".join(str(m.get(k, "")) for k in ("name", "relation", "question"))
        if len(srcs) < min_recurrence:
            why = f"recurs on {len(srcs)} < {min_recurrence} sources"
        elif REJECT_RE.search(text):
            why = "length/wording/style-type module"
        elif PRESENCE_RE.search(text + " " + str(m.get("anchor1", "")) + " " + str(m.get("break_recipe", ""))):
            why = "presence/placement/verbosity-of-explanation module (not a quality relation)"
        else:
            why = module_static_issue(m, mid)
        audit = None
        if not why:
            audit = _json(
                client,
                meter,
                "module_audit",
                judge,
                MODULE_AUDIT_S,
                MODULE_AUDIT_U.format(
                    module=json.dumps({k: m[k].strip() for k in MODULE_TEXT_LIMITS}, ensure_ascii=False)
                ),
                lambda o: o if isinstance(o, dict) and all(k in o for k in MODULE_AUDIT_KEYS) else None,
                JUDGE_KW,
                f"module_audit|{mid}|{sha(json.dumps(m, sort_keys=True, ensure_ascii=False))[:16]}",
            )
            why = module_audit_issue(audit)
        if why:
            rejected.append({"name": m.get("name"), "reason": why, "sources": srcs, "audit": audit})
            continue
        used.add(mid)
        ev = [
            {
                "source": sid,
                "name": pm.get("name"),
                "evidence_a": str(pm.get("evidence_a", ""))[:200],
                "evidence_b": str(pm.get("evidence_b", ""))[:200],
            }
            for sid in srcs
            for pm in props[sid]
        ][:8]
        mods.append(
            {
                "id": mid,
                "name": m["name"].strip(),
                "relation": m["relation"].strip(),
                "definition": f"{mid}: {m['question'].strip()} (0: {m['anchor0'].strip()}; 1: {m['anchor1'].strip()})",
                "applicability": m["applicability"].strip(),
                "applicability_keywords": [x.strip().lower() for x in m["applicability_keywords"]],
                "break_recipe": m["break_recipe"].strip(),
                "preserve": m["preserve"].strip(),
                "member_sources": srcs,
                "evidence": ev,
                "module_audit": audit,
            }
        )
    mods = sorted(mods, key=lambda x: -len(x["member_sources"]))[:max_modules]
    return mods, rejected, assigned


def mine(client, meter, bench, cfg, work_dir) -> List[dict]:
    """Consolidate the proposals into frozen atomic modules (relations/modules.json); one repair retry."""
    rd_dir = relations_dir(work_dir)
    mp = os.path.join(rd_dir, "modules.json")
    frozen = read_json(mp, None)
    if frozen is not None:
        return list(frozen.get("modules") or [])
    props = read_json(os.path.join(rd_dir, "proposals.json"), None)
    if props is None:
        raise RuntimeError("relations/proposals.json missing: run discovery first")
    max_modules = int(cfg_get(cfg, "ucoe.relation_discovery.max_modules", 2))
    min_recurrence = int(cfg_get(cfg, "ucoe.relation_discovery.min_recurrence", 3))
    judge = model_for(bench, "verifier", cfg)
    attempts, prior, mods, rejected, assigned = [], None, [], [], {}
    for attempt in (0, 1):
        mods, rejected, assigned = _cluster_once(
            client, meter, bench, props, judge, max_modules, min_recurrence, attempt, prior
        )
        attempts.append({"attempt": attempt, "accepted": [m["id"] for m in mods], "rejected": rejected})
        if mods:
            break
        prior = rejected
    n_props = sum(len(v) for v in props.values())
    man = {
        "modules": mods,
        "rejected": rejected,
        "assignment_fallback": assigned,
        "n_proposals": n_props,
        "mining_attempts": attempts,
        "sha256": sha(json.dumps(mods, sort_keys=True, ensure_ascii=False)),
    }
    write_json(mp, man, indent=1)
    if mods:
        print(
            f"[ucoe-relations] mine: frozen {len(mods)} atomic modules: "
            + "; ".join(
                f"{m['id']} [{m['name']}] {len(m['member_sources'])} src :: {m['relation'][:100]}" for m in mods
            )
            + f"; rejected {len(rejected)}",
            flush=True,
        )
    else:
        print(
            f"[ucoe-relations] mine: every candidate module was rejected after one repair retry; no relation family "
            f"(reasons: {json.dumps(rejected, ensure_ascii=False)[:600]})",
            flush=True,
        )
    return mods


# ---------------------------------------------------------------- applicability and contrasts
def applicable(client, meter, store, bench, module: dict, task_text: str) -> bool:
    """Keyword fast path, else a cached verifier-model verdict on whether the module is checkable for the task."""
    tl = task_text.lower()
    if any(k in tl for k in module.get("applicability_keywords", [])):
        return True
    key = f"applic|{module['id']}|{sha(task_text)}"
    got = store.get(key)
    if got is None:
        verdict = _json(
            client,
            meter,
            "applicable",
            model_for(bench, "verifier"),
            APPLIC_S,
            APPLIC_U.format(task=task_text, definition=module["definition"], appl=module.get("applicability", "")),
            lambda o: bool(o.get("applies")) if isinstance(o, dict) else None,
            JUDGE_KW,
            f"applic|{module['id']}|{sha(task_text)[:12]}",
        )
        got = {"applies": bool(verdict)}
        store.put(key, got)
    return bool(got.get("applies"))


def patch_contrast(
    client, meter, bench, module: dict, source: dict, base_response: str, split: str, nonce_budget: int = 3
) -> Optional[dict]:
    """One validated local patch: PATCH -> locate/validate the span -> anonymized single-factor VALID (seeded flip)."""
    task, sid, mid, y = source["task"], source["source_id"], module["id"], base_response
    judge, ns = model_for(bench, "verifier"), _ns(client)
    for nonce in range(nonce_budget):
        got = _json(
            client,
            meter,
            "patch",
            judge,
            PATCH_S,
            PATCH_U.format(
                task=task,
                resp=y,
                relation=module["relation"],
                definition=module["definition"],
                recipe=module["break_recipe"],
                preserve=module["preserve"],
            ),
            lambda o: o if isinstance(o, dict) and o.get("target_quote") and o.get("replacement") else None,
            PATCH_KW,
            f"patch|{split}|{sid}|{mid}|{nonce}",
        )
        if not got:
            continue
        quote = _locate(y, str(got["target_quote"]).strip()) or str(got["target_quote"])
        posq = str(got.get("positive_span") or quote).strip() or quote
        rep = str(got["replacement"]).strip()
        if _patch_ok(y, quote, posq, rep):
            continue
        pos, neg = y.replace(quote, posq, 1), y.replace(quote, rep, 1)
        flip = int(sha(f"{ns}|val|{split}|{sid}|{mid}|{nonce}"), 16) % 2 == 1
        x, yv = (_mark(neg, rep), _mark(pos, posq)) if flip else (_mark(pos, posq), _mark(neg, rep))
        v = _json(
            client,
            meter,
            "validate",
            judge,
            VALID_S,
            VALID_U.format(task=task, relation=module["relation"], definition=module["definition"], x=x, y=yv),
            lambda o: (
                o if isinstance(o, dict) and "relation_holds_in" in o and "other_attributes_damaged_in" in o else None
            ),
            JUDGE_KW,
            f"val|{split}|{sid}|{mid}|{nonce}",
        )
        if not _single_factor_valid(v, "Y" if flip else "X"):
            continue
        return {
            "cluster_id": f"{split}-{sid}-{mid}",
            "family": mid,
            "group": "relation",
            "source_id": sid,
            "domain": source.get("domain"),
            "task": task,
            "messages": source.get("messages"),
            "pos": pos,
            "neg": neg,
            "edit": {
                "kind": "patch",
                "target_quote": quote,
                "positive_span": posq,
                "replacement": rep,
                "positive_repaired": int(posq != quote),
                "what_breaks": got.get("what_breaks", ""),
                "validator": v,
                "validator_flip": flip,
            },
        }
    return None


def null_contrast(client, meter, bench, source: dict, base_response: str, split: str) -> Optional[dict]:
    """Paraphrase control: one span rewritten with unchanged meaning (a diagnostic pair, never a selection family)."""
    task, sid, y = source["task"], source["source_id"], base_response
    got = _json(
        client,
        meter,
        "null",
        model_for(bench, "verifier"),
        NULL_S,
        NULL_U.format(task=task, resp=y),
        lambda o: o if isinstance(o, dict) and o.get("target_quote") and o.get("paraphrase") else None,
        PATCH_KW,
        f"null|{split}|{sid}",
    )
    if not got:
        return None
    q = _locate(y, str(got["target_quote"]).strip())
    if not q or not 40 <= len(q) <= 900 or _norm(got["paraphrase"]) == _norm(q):
        return None
    return {
        "cluster_id": f"{split}-{sid}-null",
        "family": "null",
        "group": "null",
        "source_id": sid,
        "domain": source.get("domain"),
        "task": task,
        "messages": source.get("messages"),
        "pos": y,
        "neg": y.replace(q, str(got["paraphrase"]).strip(), 1),
        "edit": {"kind": "paraphrase", "target_quote": q, "paraphrase": str(got["paraphrase"]).strip()},
    }


def relation_counts(clusters: List[dict]) -> Dict[str, int]:
    """Number of relation-family contrasts per module id in a bank."""
    counts: Dict[str, int] = {}
    for c in clusters:
        if c.get("group") == "relation":
            counts[c["family"]] = counts.get(c["family"], 0) + 1
    return counts


def constructibility_check(
    bank_select: List[dict], bank_heldout: List[dict], modules: List[dict], quotas: dict
) -> List[str]:
    """Usable module ids: relation families that reached both the selection and the held-out quota at stage 0."""
    sel, ho = relation_counts(bank_select), relation_counts(bank_heldout)
    return [
        m["id"]
        for m in modules
        if sel.get(m["id"], 0) >= int(quotas["select"]) and ho.get(m["id"], 0) >= int(quotas["heldout"])
    ]
