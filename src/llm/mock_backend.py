"""Deterministic offline transport for the tests and the smoke preset.

It never contacts a model: every reply is a pure function of the prompt text (via sha256), so runs are
reproducible and the whole lifecycle of paper Algorithm 1 (verifier synthesis, criterion scoring, weakness
localization, revision, procedure distillation, relation discovery, pairwise judging) can be exercised without
credentials.

Detection rules, applied to system + user text in this order:
  1. synthesis      - the prompt contains '"new_criteria"'              -> proposal JSON: the retained ids
                      (parsed from 'in "criteria": [...]'), one or two new criteria with letter-only ids,
                      and a bounded aggregation (mean, or a min/mean mix) over every id
  2. scoring        - 'mapping each criterion id'                       -> {id: score} for every id rendered
                      as '"id": x' (fallback: 'id: definition' lines under 'Criteria:')
  3. localize       - '"witness"' and '"repair"'                        -> {"witness", "repair"}
  4. distill        - '"corrective_action"'                             -> rule JSON; criterion id and the
                      dimension alternatives ('"dimension": "a|b|c"') are parsed from the prompt
  5. duel           - '"winner"' or '"overall"' with <TAG_A>/<TAG_B> blocks -> {"winner", "overall"} chosen by
                      the quality heuristic below (symmetric, so both presentation orders agree)
  6. relations      - '"a_complete"' (screen), '"evidence_a"' (mine), '"rejected"' (cluster),
                      '"member_sources"' (assign), '"same_axis"' (audit), '"applies"' (applicability),
                      '"replacement"' (patch contrast), '"relation_holds_in"' (validate), '"paraphrase"'
                      (null rewrite)
  7. any other prompt that asks for a JSON object                       -> "{}"
  8. revision       - a <DRAFT> block, or a prompt asking for a revised answer -> the draft plus one sentence
  9. otherwise                                                          -> a multi-sentence answer echoing the task

Quality heuristic (used for scores and duels): a response's quality rises with its number of long words and
sentences, drops by a fixed penalty for every negative marker (the sentences injected by the reference-family
corrupters, the relation-break span inserted by rule 6, duplicated sentences, placeholder words), and rises
slightly for the end-of-answer markers that the requirement families remove. A tiny per-criterion jitter from
sha(prompt) keeps scores distinct. Hence every contrast is separable: the positive member of a pair scores
above the negative by more than the margin eps, and a revision (draft + one dense sentence) is accepted.
Markers can be extended through the config (``models.mock.negative_markers`` / ``positive_markers``).
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .base import Transport, make_result

NEGATIVE_MARKERS = (
    "external review of",          # unsupported-claim injection
    "opposite conclusion",
    "unrelated aside",             # off-topic graft
    "unrelated checklist",
    "opposite is true",            # appended contradiction
    "opposite holds",
    "disregard",
    "the opposite of all this",    # story contradiction
    "it was what it was",          # cliche filler
    "redhawk",                     # conflicting fact substitution
    "[relation broken]",           # relation-break span inserted by this mock
)
POSITIVE_MARKERS = ("end of answer.", "end of draft.", "end of document")
BREAK_SPAN = ("This recommendation rests on an unrelated preference that the response never tied to any "
              "identified need [relation broken].")
REVISION_SENTENCE = ("In addition, the revised answer verifies each requirement against the original request "
                     "and corrects the identified weakness.")
DRAFT_SENTENCES = (
    "The essential requirement is identified first and addressed explicitly.",
    "Each step is explained with concrete reasoning and verified before moving on.",
    "Relevant assumptions are stated clearly, and alternatives are considered where appropriate.",
    "Supporting details are organized so the main conclusion follows naturally.",
    "Potential pitfalls are flagged together with practical mitigations.",
    "The conclusion summarizes the result and notes remaining limitations.",
    "Overall the response remains focused, accurate, and complete.",
)
NEW_CRITERIA = (
    ("Does the response make its central claim explicit and support it with reasoning stated in the "
     "response itself?",
     "the central claim is missing or asserted without support",
     "the central claim is explicit and supported by stated reasoning"),
    ("Does each step of the response follow from the preceding material without a gap a careful reader "
     "would notice?",
     "at least one step relies on an unstated jump",
     "every step follows from the stated preceding material"),
    ("Does the response address the request exactly as posed rather than a nearby easier version of it?",
     "the response answers a different or narrower question",
     "the response answers precisely what was asked"),
)
PROCEDURES = (
    ("the request lists several explicit requirements", "restate every requirement and confirm each is met"),
    ("a claim is made without supporting reasoning", "add the reasoning that supports the claim in plain terms"),
    ("the reasoning skips an intermediate step", "write out the missing step before stating the conclusion"),
    ("the conclusion drifts from the question asked", "return to the original question and answer it directly"),
    ("the response ends without a clear result", "close with one sentence that states the final result plainly"),
    ("a key term is used without definition", "define each key term the first time it appears"),
)
LOCALIZATIONS = (
    ("The answer asserts its main point without grounding it in the specifics of the request.",
     "Add one sentence that grounds the main point in the specific request."),
    ("The middle of the answer skips the step that connects the premise to the conclusion.",
     "Insert the missing connecting step before the conclusion."),
    ("The answer never states which requirement of the request it satisfies.",
     "Name the requirement being satisfied at the start of the answer."),
)
MOCK_MODULES = (
    {"id": "needtie", "name": "Recommendation tied to stated need",
     "relation": "every recommendation in the response is tied to a need the response itself identified",
     "question": "Is each recommendation tied to a need that the response itself identified?",
     "anchor0": "a recommendation appears with no identified need that it serves",
     "anchor1": "every recommendation names the identified need that it serves",
     "applicability_keywords": ["recommend", "should", "propose", "plan", "steps", "advice", "improve"],
     "applicability": "tasks whose answer proposes actions, steps or recommendations",
     "break_recipe": "replace the clause tying one recommendation to its stated need with an unrelated "
                     "justification",
     "preserve": "keep every other recommendation and the stated needs unchanged"},
    {"id": "claimsup", "name": "Claim supported by given material",
     "relation": "each factual claim in the response is supported by material given in the task or derived "
                 "in the response",
     "question": "Is each factual claim supported by material given in the task or derived in the response?",
     "anchor0": "a claim is asserted with no support from the task or the response",
     "anchor1": "every claim is traced to the task material or to reasoning in the response",
     "applicability_keywords": ["explain", "why", "because", "analy", "argue", "evidence", "compare"],
     "applicability": "tasks whose answer makes factual or analytical claims",
     "break_recipe": "replace the support for one claim with an assertion that nothing in the task or "
                     "response backs",
     "preserve": "keep all other claims and their support unchanged"},
)
AUDIT_KEYS = ("same_axis", "localized_degradation", "preserve_separable", "task_general", "atomic", "substantive")

_SENT = re.compile(r"[^.!?\n]+[.!?]+(?:\s|$)|[^.!?\n]+$")
_BLOCK = re.compile(r"<([A-Z][A-Z_]*)>\n?(.*?)\n?</\1>", re.DOTALL)
_PLACEHOLDER = re.compile(r"\bthing\b")
_CONTEXT_TAGS = {"TASK", "PROMPT", "REQUEST", "USER_REQUEST", "CONVERSATION", "HISTORY", "CONTEXT", "CRITERIA",
                 "RULES", "GUIDELINES", "INSTRUCTIONS", "MESSAGES"}


# ------------------------------------------------------------------ helpers

def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _h(text: str) -> int:
    return int(_digest(text)[:14], 16)


def _blocks(text: str) -> List[Tuple[str, str]]:
    return [(m.group(1), m.group(2)) for m in _BLOCK.finditer(text or "")]


def _response_block(text: str, prefer: Tuple[str, ...] = ()) -> Optional[str]:
    """The response text inside the last tagged block that is not task context (or a preferred tag)."""
    blocks = _blocks(text)
    for tag in prefer:
        for t, body in blocks:
            if t == tag:
                return body
    cands = [body for t, body in blocks if t not in _CONTEXT_TAGS and not t.endswith(("_A", "_B", "_X", "_Y"))]
    return cands[-1] if cands else None


def _tagged(text: str, suffix: str) -> Optional[str]:
    m = re.search(r"<([A-Z][A-Z_]*)" + suffix + r">\n?(.*?)\n?</\1" + suffix + ">", text or "", re.DOTALL)
    return m.group(2) if m else None


def _sents(text: str) -> List[str]:
    return [s.strip() for s in _SENT.findall(text or "") if s.strip()]


def quality(text: str) -> float:
    """Deterministic quality of a response text in [0, 1] (see the module docstring)."""
    t = text or ""
    low = t.lower()
    long_words = len(re.findall(r"[A-Za-z]{6,}", t))
    sents = _sents(t)
    q = 0.25 + 0.4 * min(1.0, long_words / 80.0) + 0.15 * min(1.0, len(sents) / 10.0)
    q -= 0.3 * sum(1 for m in NEGATIVE_MARKERS if m in low)
    q += 0.05 * sum(1 for m in POSITIVE_MARKERS if m in low)
    q -= 0.04 * min(5, len(_PLACEHOLDER.findall(low)))
    lowered = [s.lower() for s in sents]
    if len(set(lowered)) < len(lowered):
        q -= 0.15
    return min(1.0, max(0.0, q))


def _ids_after(text: str, anchor: str) -> List[str]:
    tail = text.split(anchor, 1)[1] if anchor in text else text
    return list(dict.fromkeys(re.findall(r'"([a-z][a-z0-9_]{0,24})":\s*x\b', tail)))


def _ids_under(text: str, header: str) -> List[str]:
    m = re.search(re.escape(header) + r"\n(.*?)(?:\n\n|$)", text, re.DOTALL)
    if not m:
        return []
    return list(dict.fromkeys(re.findall(r"^\s*([a-z][a-z0-9_]{0,24}):", m.group(1), re.MULTILINE)))


def _letters(hexs: str) -> str:
    return "".join(chr(ord("a") + int(c, 16)) for c in hexs)


def _echo(user: str) -> str:
    t = _BLOCK.sub(" ", user or "")
    for label in ("User request:", "Task:", "Prompt:", "Request:"):
        if label in t:
            t = t.split(label, 1)[1]
    t = re.sub(r"\s+", " ", t).strip()
    return t[:100].rstrip(" ,;:.!?")


def _span(resp: Optional[str]) -> Optional[str]:
    """A sentence of 40-900 characters that occurs exactly once in the response (deterministic: the middle one)."""
    if not resp:
        return None
    cands = [s for s in _sents(resp) if 40 <= len(s) <= 900 and resp.count(s) == 1]
    return cands[len(cands) // 2] if cands else None


def _evidence(text: Optional[str]) -> str:
    t = (text or "").strip()
    if len(t) <= 150:
        return t
    cut = t[:150]
    return cut[:cut.rfind(" ")] if " " in cut else cut


def _source_ids(text: str) -> List[str]:
    m = re.search(r"Proposals \([^)]*\):\n(.*?)(?:\n\n|$)", text, re.DOTALL)
    body = m.group(1) if m else text
    return list(dict.fromkeys(re.findall(r"^\s*(\S+)\s*::", body, re.MULTILINE)))


# ------------------------------------------------------------------ replies per role

def _synthesis(prompt: str, h: int) -> str:
    m = re.search(r'in "criteria": \[(.*?)\]', prompt, re.DOTALL)
    retained = list(dict.fromkeys(re.findall(r'"([^"]+)"', m.group(1)))) if m else []
    if not retained:
        retained = _ids_under(prompt, "(id: definition):")
    if not retained:
        return "{}"
    digest = _digest(prompt)
    variant = h % 3
    new: List[Dict[str, str]] = []
    for k in range(2 if variant == 2 else 1):
        cid = ("mk" if k == 0 else "mx") + _letters(digest[5 * k:5 * k + 5])
        while cid in retained or any(c["id"] == cid for c in new):
            cid += "x"
        q, a0, a1 = NEW_CRITERIA[(h // 7 + k) % len(NEW_CRITERIA)]
        new.append({"id": cid, "question": q, "anchor0": a0, "anchor1": a1})
    ids = retained + [c["id"] for c in new]
    mean = "(" + " + ".join(ids) + f") / {len(ids)}"
    agg = mean if variant != 1 else f"0.5 * min({', '.join(ids)}) + 0.5 * {mean}"
    return json.dumps({"criteria": ids, "new_criteria": new, "aggregation": agg})


def _scores(prompt: str) -> str:
    ids = _ids_after(prompt, "mapping each criterion id") or _ids_under(prompt, "Criteria:")
    resp = _response_block(prompt)
    q = quality(resp if resp is not None else prompt)
    out: Dict[str, float] = {}
    for cid in ids:
        jitter = (_h(prompt + "|" + cid) % 1000) / 1000.0 * 0.008 - 0.004
        out[cid] = round(min(1.0, max(0.0, q + jitter)), 4)
    return json.dumps(out)


def _localize(h: int) -> str:
    witness, repair = LOCALIZATIONS[(h // 17) % len(LOCALIZATIONS)]
    return json.dumps({"witness": witness, "repair": repair})


def _distill(prompt: str, h: int) -> str:
    m = re.search(r'"criterion_id":\s*"([^"]*)"', prompt)
    cid = m.group(1) if m else ""
    m = re.search(r'"dimension":\s*"([^"]+)"', prompt)
    dims = [d.strip() for d in (m.group(1).split("|") if m else ["general"]) if d.strip()]
    trigger, action = PROCEDURES[(h // 13) % len(PROCEDURES)]
    return json.dumps({"criterion_id": cid, "dimension": dims[(h // 11) % len(dims)], "scope_tags": ["general"],
                       "trigger": trigger, "corrective_action": action})


def _duel(prompt: str) -> str:
    a, b = _tagged(prompt, "_A"), _tagged(prompt, "_B")
    if a is None or b is None:
        verdict = "TIE"
    else:
        qa, qb = quality(a), quality(b)
        verdict = "A" if qa > qb + 0.005 else "B" if qb > qa + 0.005 else "TIE"
    return json.dumps({"winner": verdict, "overall": verdict})


def _relations(prompt: str, low: str) -> Optional[str]:
    if '"a_complete"' in low:
        return json.dumps({"a_complete": True, "b_complete": True, "a_truncated": False, "b_truncated": False})
    if '"evidence_a"' in low:
        ea, eb = _evidence(_tagged(prompt, "_A")), _evidence(_tagged(prompt, "_B"))
        mods = [{"name": m["name"], "relation": m["relation"], "question": m["question"], "anchor0": m["anchor0"],
                 "anchor1": m["anchor1"], "applicability": m["applicability"], "evidence_a": ea, "evidence_b": eb,
                 "break_recipe": m["break_recipe"], "preserve": m["preserve"]} for m in MOCK_MODULES]
        return json.dumps({"modules": mods})
    if '"rejected"' in low:
        ids = _source_ids(prompt)
        return json.dumps({"modules": [dict(m, member_sources=ids) for m in MOCK_MODULES], "rejected": []})
    if '"member_sources"' in low:
        return json.dumps({"member_sources": _source_ids(prompt)})
    if '"same_axis"' in low:
        out: Dict[str, Any] = {k: True for k in AUDIT_KEYS}
        out["reason"] = ""
        return json.dumps(out)
    if '"applies"' in low:
        return json.dumps({"applies": True})
    if '"replacement"' in low:
        span = _span(_response_block(prompt, prefer=("RESPONSE",)))
        if span is None:
            return json.dumps({"target_quote": "", "positive_span": "", "replacement": "", "what_breaks": ""})
        return json.dumps({"target_quote": span, "positive_span": span, "replacement": BREAK_SPAN,
                           "what_breaks": "the recommendation is no longer tied to a stated need"})
    if '"relation_holds_in"' in low:
        marker = "[relation broken]"
        x_broken = marker in (_tagged(prompt, "_X") or "").lower()
        y_broken = marker in (_tagged(prompt, "_Y") or "").lower()
        holds = ("Y" if x_broken and not y_broken else "X" if y_broken and not x_broken
                 else "both" if not (x_broken or y_broken) else "neither")
        return json.dumps({"relation_holds_in": holds, "other_attributes_damaged_in": False})
    if '"paraphrase"' in low:
        span = _span(_response_block(prompt, prefer=("RESPONSE",)))
        if span is None:
            return json.dumps({"target_quote": "", "paraphrase": ""})
        return json.dumps({"target_quote": span, "paraphrase": "Put differently, " + span[0].lower() + span[1:]})
    return None


def _revision(prompt: str) -> Optional[str]:
    draft = _response_block(prompt, prefer=("DRAFT",))
    if draft is None:
        return None
    return draft.rstrip() + "\n\n" + REVISION_SENTENCE


def _answer(user: str, h: int) -> str:
    echo = _echo(user)
    lead = f"This answer addresses the request: {echo}." if echo else "This answer addresses the request directly."
    k = h % len(DRAFT_SENTENCES)
    body = list(DRAFT_SENTENCES[k:] + DRAFT_SENTENCES[:k])
    del body[(h // 7) % len(body)]
    return lead + " " + " ".join(body)


# ------------------------------------------------------------------ transport

class MockTransport(Transport):
    """Deterministic offline transport; config keys (optional): negative_markers, positive_markers (lists)."""

    name = "mock"

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        global NEGATIVE_MARKERS, POSITIVE_MARKERS
        cfg = cfg or {}
        extra_neg = tuple(str(m).lower() for m in (cfg.get("negative_markers") or []))
        extra_pos = tuple(str(m).lower() for m in (cfg.get("positive_markers") or []))
        NEGATIVE_MARKERS = tuple(dict.fromkeys(NEGATIVE_MARKERS + extra_neg))
        POSITIVE_MARKERS = tuple(dict.fromkeys(POSITIVE_MARKERS + extra_pos))

    def reply(self, system: str, user: str) -> str:
        prompt = (system or "") + "\n\n" + (user or "")
        low = prompt.lower()
        h = _h(prompt)
        if '"new_criteria"' in low:
            return _synthesis(prompt, h)
        if "mapping each criterion id" in low:
            return _scores(prompt)
        if '"witness"' in low and '"repair"' in low:
            return _localize(h)
        if '"corrective_action"' in low:
            return _distill(prompt, h)
        if ('"winner"' in low or '"overall"' in low) and _tagged(prompt, "_A") is not None:
            return _duel(prompt)
        rel = _relations(prompt, low)
        if rel is not None:
            return rel
        if "json" in low[-300:] and "nothing else" in low[-300:]:
            return "{}"
        if any(t == "DRAFT" for t, _ in _blocks(prompt)) or "revised answer" in low or "revised" in low:
            rev = _revision(prompt)
            if rev is not None:
                return rev
        return _answer(user, h)

    def __call__(self, model: str, system: str, user: str, kw: Dict[str, Any]) -> Dict[str, Any]:
        text = self.reply(system, user)
        n_in = (len(system or "") + len(user or "")) // 4
        return make_result(text, finish_reason="stop", prompt_tokens=n_in, completion_tokens=len(text) // 4,
                           model=model, sdk="mock", request_id=_digest((system or "") + "\n\n" + (user or ""))[:12])
