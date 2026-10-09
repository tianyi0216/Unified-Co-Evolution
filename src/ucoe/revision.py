"""The per-task revision step (paper Sec. 3.3 "Response revision and selection"; Eq. revision-acceptance).

The verifier scores the draft; the weakest participating criterion is localized by the verifier model into a
witness (the grounded weakness) and one repair instruction; the actor makes one local repair of the draft with the
retrieved guidelines; the revision is served only when the same verifier scores it above the draft by more than
the margin. Unusable feedback, a missing revision or an invalid score leaves the draft in place.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional

from src.llm.client import ContentInvalid, json_call
from src.ucoe import relations, verifier
from src.utils.io import sha

WITNESS_CHARS = 400
REPAIR_CHARS = 250


def weakest_criterion(V: dict, vals: dict, exclude: Iterable[str] = ()) -> str:
    """Lowest-scoring criterion among those that participate in the aggregation, ties broken by catalogue order
    (paper App. "Scoring and diagnosis"); deterministic criteria take part like any other. ``exclude`` is an
    opt-in restriction for harnesses that diagnose semantic criteria only; it never empties the pool."""
    order = {c: k for k, c in enumerate(verifier.crit_ids(V))}
    act = verifier.active_ids(V)
    skip = set(exclude or ())
    pool = [c for c in act if c not in skip] or act
    return min(pool, key=lambda c: (float(vals.get(c, 0.0)), order[c]))


def _task_id(task) -> str:
    if isinstance(task, dict):
        return task.get("task_id") or task.get("source_id") or sha(task.get("task") or "")[:16]
    return sha(str(task))[:16]


def _view(bench, task) -> str:
    if isinstance(task, dict) and hasattr(bench, "reward_view"):
        return bench.reward_view(task)
    return task["task"] if isinstance(task, dict) else str(task)


def localize(client, meter, bench, V: dict, task, response: str, vals: dict, weakest: str) -> Optional[dict]:
    """Witness (<= 400 chars) and one repair instruction (<= 250 chars) for the selected weakness, or None."""
    weakdef = verifier.defs(V).get(weakest, weakest)
    req = bench.localize_request(_view(bench, task), response, vals, weakest, weakdef)

    def validate(obj):
        wit = str(obj.get("witness", "")).strip()
        rep = str(obj.get("repair", "")).strip()
        return {"witness": wit[:WITNESS_CHARS], "repair": rep[:REPAIR_CHARS]} if wit and rep else None

    try:
        return json_call(client, meter, "localize", relations.model_for(bench, "verifier"), req["system"],
                         req["user"], validate, dict(bench.loc_kw), f"loc|{V['sha256'][:8]}|{_task_id(task)}")
    except ContentInvalid:
        return None


def revise(client, bench, task, y0: str, witness: str, repair: str, guidelines: str) -> str:
    """One local repair by the actor (the revision prompt of the benchmark suite with the retrieved guidelines)."""
    req = bench.revision_request(task, y0, witness, repair, guidelines or "")
    return relations.request_text(client, "revise", relations.model_for(bench, "actor"), req, dict(bench.rev_kw),
                                  f"rev|{_task_id(task)}|{sha(guidelines or '')[:8]}")


def accept(s0: float, sR: Optional[float], eps: float) -> bool:
    """Eq. revision-acceptance: the revision replaces the draft iff r(yR) > r(y0) + eps."""
    return sR is not None and math.isfinite(sR) and sR > s0 + eps


def guidelines_text(rules: List[dict]) -> str:
    """Retrieved procedures rendered for the actor ("" when nothing was retrieved)."""
    if not rules:
        return ""
    return "\nGeneral guidelines:\n" + "\n".join(
        f"- ({r['dimension']}) when {r['trigger']}: {r['corrective_action']}" for r in rules)
