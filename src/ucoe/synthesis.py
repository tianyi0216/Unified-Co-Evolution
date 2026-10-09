"""Verifier synthesis by the verifier model (paper Sec. 3.2 "Initial verifier synthesis"; Sec. 3.3 "Verifier
evolution": a checkpoint candidate is synthesized from the current definitions and an unlabeled summary of the
completed tasks).

``synthesize`` renders the available criterion definitions (the benchmark catalogue, plus mined relation modules
when offered) into the synthesis prompt, asks for a scoring program that lists every retained id, may append at
most ``ucoe.max_new_criteria`` new criteria and gives one bounded aggregation expression, and returns a frozen
verifier (``S_<tag>`` for an initial candidate, ``V_<tag>`` for a child of the current verifier) or None when the
proposal fails the static validator. ``checkpoint_residual`` is the unlabeled summary (repair acceptance rate,
weakest-criterion histogram, memory size over the last tasks) and ``failure_residual`` the diagnostic shown by the
conservative update rule after a failed certification.
"""

from __future__ import annotations

import json
from typing import List, Optional

from src.llm.client import ContentInvalid, json_call
from src.ucoe import relations, verifier
from src.utils.config import cfg_get

SYNTH_USER = ("Available typed 0-1 scoring modules (id: definition):\n"
              "{defs}\n\n{residual}"
              "Design a reward controller. List exactly these "
              "available module ids in \"criteria\": [{retained}]. You "
              "may APPEND at most two NEW task-general response-"
              "quality modules (short scoring question judged 0-1 with "
              "anchors; ids match [a-z][a-z0-9_]{{1,24}}); appending "
              "none is allowed. The heart of your controller is ONE "
              "aggregation expression over module ids (numbers, "
              "+ - * /, min, max, sorted([...])[i], comparisons, "
              "conditionals) that stays within [0,1], uses EVERY "
              "module you append, and encodes your judgement of what "
              "matters most.\nProposal slot: {slot}.\n\nEmit exactly "
              'one JSON object:\n{{"criteria": [{retained}], '
              '"new_criteria": [{{"id": "...", "question": "...", '
              '"anchor0": "...", "anchor1": "..."}}], '
              '"aggregation": "<expression>"}}. Nothing else.')
RD_NOTE = ("Modules {ids} were discovered from real differences between two systems' answers on this benchmark "
           "(mined, task-general relations); they are part of the retained module set.\n\n")
RD_BLOCK_ID = "X_rd"
RESIDUAL_WINDOW = 128


def synthesize(client, meter, bench, cfg, tag: str, parent: Optional[dict] = None, residual: str = "",
               rd_modules: Optional[List[dict]] = None) -> Optional[dict]:
    """One synthesis call for slot ``tag``; returns a frozen verifier or None (invalid or empty proposal)."""
    max_new = int(cfg_get(cfg, "ucoe.max_new_criteria", 2))
    leak_terms = tuple(getattr(bench, "leak_terms", ()) or ())
    rd = list(rd_modules or []) if parent is None else []
    if parent is None:
        defs = {c: bench.module_defs[c] for c in bench.module_ids}
        for m in rd:
            defs[m["id"]] = m["definition"]
        if rd:
            residual = RD_NOTE.format(ids=", ".join(m["id"] for m in rd)) + (residual or "")
    else:
        defs = verifier.defs(parent)
    retained = list(defs)
    user = SYNTH_USER.format(defs="\n".join(f"  {v}" for v in defs.values()), residual=residual or "",
                             retained=", ".join(f'"{c}"' for c in retained), slot=tag)

    def validate(obj):
        got = verifier.parse_proposal(obj, retained, max_new=max_new, leak_terms=leak_terms)
        if got is None:
            return None
        ext, agg = got
        if parent is not None and ext is None and agg == str(parent["aggregation"]).strip():
            return None  # a checkpoint candidate must append a criterion or change the aggregation
        return got

    try:
        got = json_call(client, meter, "synth", relations.model_for(bench, "verifier", cfg), bench.synth_system,
                        user, validate, dict(bench.synth_kw), f"synth|{tag}")
    except ContentInvalid:
        got = None
    if not got:
        print(f"[ucoe-synth] slot {tag}: no valid proposal", flush=True)
        return None
    ext, agg = got
    block = {"block_id": f"X_{tag}", "criteria": ext["criteria"]} if ext else None
    if parent is None:
        blocks = [verifier.base_block(bench)]
        if rd:
            blocks.append({"block_id": RD_BLOCK_ID,
                           "criteria": [{"id": m["id"], "definition": m["definition"]} for m in rd]})
        if block:
            blocks.append(block)
        V = verifier.make_verifier(f"S_{tag}", blocks, agg, rd_modules=rd or None)
    else:
        V = verifier.make_child(f"V_{tag}", parent, block, agg)
    print(f"[ucoe-synth] slot {tag}: {V['version']} criteria {verifier.crit_ids(V)} aggregation {V['aggregation']}",
          flush=True)
    return V


def checkpoint_residual(rows_prefix: List[dict], memory: List[dict], window: int = RESIDUAL_WINDOW) -> str:
    """Unlabeled summary of the last ``window`` completed tasks shown to the checkpoint synthesis."""
    recent = list(rows_prefix)[-window:]
    win = sum(1 for r in recent if r.get("b"))
    hist = {}
    for r in recent:
        wk = r.get("weakest")
        if wk:
            hist[wk] = hist.get(wk, 0) + 1
    return (f"Aggregate prefix diagnostics: repair win rate {win}/{max(1, len(recent))}; weakest-criterion counts "
            f"{json.dumps(hist, sort_keys=True)}; memory rules {len(memory)}.\n\n")


def failure_residual(reports: List[dict]) -> str:
    """Diagnostic of the last failed certification (conservative rule only): the weakest family and its counts."""
    if not reports:
        return ""
    rep = reports[-1]
    bounds, per = rep.get("bounds") or {}, rep.get("per_family") or {}
    if not bounds:
        return ""
    weakest = min(bounds, key=lambda f: bounds[f])
    kn = per.get(weakest, (0, 0))
    lines = ["Previous certification attempts on constructed probe banks failed. THE WEAKEST FAILED FAMILY IS "
             f"'{weakest}' (CP lower bound {bounds[weakest]}, wins {kn[0]}/{kn[1]}, exact ties "
             f"{(rep.get('diag') or {}).get(weakest, {}).get('exact_ties')})."]
    for f, kn in per.items():
        lines.append(f"  {f}: {kn[0]}/{kn[1]}, bound {bounds.get(f)}")
    lines.append(f"The controller must separate a clean answer from the '{weakest}' corruption by a clear score "
                 "margin at one fixed threshold, without regressing the other families.")
    return "\n".join(lines) + "\n\n"
