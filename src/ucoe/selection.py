"""Driving capacity, maximin selection and the verifier-update decisions (paper Sec. 3.2 "Driving capacity and
selection"; Sec. 3.3 "Verifier evolution"; App. B.3 Eq. empirical-admission; App. "Scoring candidates").

A candidate verifier is measured on response contrasts: a contrast is a success when the candidate's reward for
the base response exceeds the reward for its degraded copy by more than the margin ``eps``; the family accuracy
``J_j = 2 p_j - 1`` is centered at chance, and the (signed) driving capacity is a quarter of the worst family.
Selection takes the maximin candidate on the selection split, rounded to four decimals, with ties broken by mean
accuracy, aggregation simplicity and candidate order. Initialization (admission) adopts the selection winner.
At a later checkpoint (one challenger by default, ``ucoe.candidates_per_checkpoint`` more) the winner is fixed
before the held-out comparison and adopted only under the empirical rule:
a strict worst-family gain on the selection split, more held-out contrasts won than lost against the current
verifier over the same clusters, and no family regression beyond ``kappa``; every failed or undefined comparison
retains the current verifier. The conservative ``hard`` rule additionally requires Clopper-Pearson lower bounds
above chance (recorded as diagnostics under the relative rule).
"""

from __future__ import annotations

import json
import math
import string
from typing import Callable, Dict, List, Optional, Tuple

from src.llm.client import pmap
from src.ucoe import scoring, synthesis, verifier
from src.utils.config import cfg_get
from src.utils.stats import cp_lower

ALPHA_TOTAL = 0.05
RULES = ("relative", "hard")


def _view(bench, cluster: dict) -> str:
    """Task text as the verifier sees it (the adapter's capped reward view when it defines one)."""
    if hasattr(bench, "reward_view"):
        return bench.reward_view(cluster)
    return cluster["task"]


def _slot(k: int) -> str:
    return f"h0{string.ascii_lowercase[k]}" if k < len(string.ascii_lowercase) else f"h0_{k}"


# ---------------------------------------------------------------- contrast accuracy
def head_J(client, meter, store, bench, V: dict, clusters: List[dict], eps: float, workers: int,
           rd_applicability: Optional[Callable[[dict, str], bool]] = None) -> Tuple[dict, dict, dict]:
    """Per-family (k, n), margin diagnostics and per-cluster wins of ``V`` on ``clusters``.

    Success iff r(y+) > r(y-) + eps; a cluster whose scores are missing or non-finite counts as a failure.
    """

    def one(c):
        view = _view(bench, c)
        vp = scoring.score_version(client, meter, store, bench, V, view, c["pos"], rd_applicability)
        vn = scoring.score_version(client, meter, store, bench, V, view, c["neg"], rd_applicability)
        if vp is None or vn is None:
            return c["cluster_id"], c["family"], None
        sp, sn = verifier.agg_score(V, vp), verifier.agg_score(V, vn)
        if not (math.isfinite(sp) and math.isfinite(sn)):
            return c["cluster_id"], c["family"], None
        return c["cluster_id"], c["family"], round(sp - sn, 6)

    per: Dict[str, Tuple[int, int]] = {}
    diag: Dict[str, dict] = {}
    wins: Dict[str, int] = {}
    for cid, fam, margin in pmap(one, clusters, workers, f"headJ_{V['version']}", every=50):
        ok = int(margin is not None and margin > eps)
        k, n = per.get(fam, (0, 0))
        per[fam] = (k + ok, n + 1)
        wins[cid] = ok
        d = diag.setdefault(fam, {"exact_ties": 0, "sub_margin": 0, "unscored": 0})
        if margin is None:
            d["unscored"] += 1
        elif margin == 0:
            d["exact_ties"] += 1
        elif 0 < margin <= eps:
            d["sub_margin"] += 1
    return per, diag, wins


def family_J(per: dict) -> Dict[str, float]:
    return {f: 2 * k / n - 1 for f, (k, n) in per.items() if n}


def driving_capacity(J: dict) -> float:
    """Signed driving capacity: a quarter of the worst family accuracy (its nonnegative part is max(0, .))."""
    return 0.25 * min(J.values()) if J else 0.0


def heldout_diagnostic(per: dict, families, alpha_total: float = ALPHA_TOTAL) -> dict:
    """Clopper-Pearson lower bounds per family (alpha split three ways over the families); recorded, never a
    gate under the relative rule."""
    fams = list(families)
    alpha_f = alpha_total / (3 * max(1, len(fams)))
    bounds = {f: round(cp_lower(k, n, alpha_f), 4) for f, (k, n) in per.items() if n}
    complete = all(f in per for f in fams)
    return {"bounds": bounds, "alpha_per_family": alpha_f, "complete": complete,
            "all_above_chance": complete and all(b > 0.5 for b in bounds.values())}


# ---------------------------------------------------------------- selection
def select_rows(client, meter, store, bench, cands: List[dict], sel_bank: List[dict], eps: float, workers: int,
                families, rd_applicability=None) -> List[dict]:
    """Selection-split rows of every candidate that has a measurement for every family."""
    rows = []
    for V in cands:
        per, _, _ = head_J(client, meter, store, bench, V, sel_bank, eps, workers, rd_applicability)
        J = family_J(per)
        missing = [f for f in families if f not in J]
        if missing:
            print(f"[ucoe-select] {V['version']}: dropped (no measurement for families {missing})", flush=True)
            continue
        J = {f: J[f] for f in families}
        rows.append({"candidate_id": V["version"], "V": V, "per_family": {f: round(x, 4) for f, x in J.items()},
                     "worst_j": round(min(J.values()), 4), "mean_j": round(sum(J.values()) / len(J), 4),
                     "n_ext": len(verifier.ext_ids(V))})
    return rows


def pick(rows: List[dict]) -> dict:
    """Maximin winner: worst family (rounded), then mean accuracy, shorter aggregation, candidate order."""
    return sorted(rows, key=lambda r: (-round(r["worst_j"], 4), -r["mean_j"], len(r["V"]["aggregation"]),
                                       r["candidate_id"]))[0]


def _families(bench, stage: int, bank: List[dict]) -> List[str]:
    """Family inventory of a stage: the reference families plus the relation families present in the bank."""
    fams = list(bench.stage_families(stage))
    extra = sorted({c["family"] for c in bank if c.get("group") == "relation" and c["family"] not in fams})
    return fams + extra


def _public(rows: List[dict]) -> List[dict]:
    return [{k: r[k] for k in ("candidate_id", "per_family", "worst_j", "mean_j", "n_ext")} for r in rows]


def _per_list(per: dict) -> dict:
    return {f: [k, n] for f, (k, n) in per.items()}


# ---------------------------------------------------------------- admission (no verifier deployed)
def admission(client, meter, store, bench, cfg, state: dict, stage: int, banks_fn, workers: int,
              rule: str = "relative", rd_modules: Optional[List[dict]] = None,
              rd_applicability=None) -> Tuple[Optional[dict], dict]:
    """Synthesize the candidate pool (K slots at stage 0, one slot later), select the maximin winner and deploy it.

    The relative rule deploys the selection winner; the conservative rule requires every held-out family bound
    above chance and otherwise stops (the failure diagnostic feeds the next synthesis).
    """
    eps = float(cfg_get(cfg, "ucoe.eps", 0.01))
    arch = state.setdefault("extra", {}).setdefault("archive", {})
    if stage == 0:
        slots = [_slot(k) for k in range(int(cfg_get(cfg, "ucoe.initial_candidates", 2)))]
        residual = ""
    else:
        slots = [f"ck{stage}"]
        residual = synthesis.failure_residual(state["extra"].get("fail_reports", [])) if rule == "hard" else ""
    for slot in slots:
        V = synthesis.synthesize(client, meter, bench, cfg, slot, residual=residual, rd_modules=rd_modules)
        if V is not None:
            arch[V["sha256"]] = V
    report = {"stage": stage, "rule": rule, "candidates": [v["version"] for v in arch.values()]}

    def retain(msg):
        d = f"RETAIN null ({stage}: {msg})"
        report.update({"decision": d, "decision_rel": d, "decision_hard": d})
        return None, report

    if not arch:
        return retain("no valid candidates")
    sel = banks_fn(stage, "select")
    families = _families(bench, stage, sel)
    rows = select_rows(client, meter, store, bench, list(arch.values()), sel, eps, workers, families,
                       rd_applicability)
    report["families"] = families
    report["selection"] = _public(rows)
    if not rows:
        return retain("selection scoring failed")
    w = pick(rows)
    heldout = banks_fn(stage, "heldout")
    per, diag, _ = head_J(client, meter, store, bench, w["V"], heldout, eps, workers, rd_applicability)
    cp = heldout_diagnostic(per, families)
    hard_ok = cp["all_above_chance"]
    report.update({"winner": w["candidate_id"],
                   "selection_point": {"worst_j": w["worst_j"], "mean_j": w["mean_j"]},
                   "heldout_per_family": _per_list(per),
                   "heldout_J": {f: round(x, 4) for f, x in family_J(per).items()},
                   "cp_bounds": cp["bounds"], "diag": diag, "hard_ok": hard_ok,
                   "decision_rel": f"DEPLOY {w['candidate_id']} (selection winner)",
                   "decision_hard": f"ADMIT {w['candidate_id']}" if hard_ok else "STOP_H0_CERT"})
    report["decision"] = report["decision_rel" if rule != "hard" else "decision_hard"]
    deploy = True if rule != "hard" else hard_ok
    if not hard_ok:
        state["extra"].setdefault("fail_reports", []).append(
            {"stage": stage, "winner": w["candidate_id"], "per_family": _per_list(per), "bounds": cp["bounds"],
             "diag": diag})
    return (w["V"] if deploy else None), report


# ---------------------------------------------------------------- promotion (a verifier is deployed)
def promotion(client, meter, store, bench, cfg, state: dict, stage: int, rows_prefix: List[dict], banks_fn,
              workers: int, rule: str = "relative", rd_modules: Optional[List[dict]] = None,
              rd_applicability=None) -> Tuple[dict, dict]:
    """One challenger against the incumbent under the empirical update rule; returns (verifier to deploy, report)."""
    eps = float(cfg_get(cfg, "ucoe.eps", 0.01))
    kappa = float(cfg_get(cfg, "ucoe.regression_tolerance", 0.10))
    inc = state["V"]
    report: Dict[str, object] = {"stage": stage, "rule": rule, "incumbent": inc["version"]}

    def abstain(msg):
        d = f"ABSTAIN ({msg})"
        report.update({"decision": d, "decision_rel": d, "decision_hard": d})
        return inc, report

    residual = synthesis.checkpoint_residual(rows_prefix, state.get("memory", []))
    n_ch = max(1, int(cfg_get(cfg, "ucoe.candidates_per_checkpoint", 1)))
    slots = [f"ck{stage}"] if n_ch == 1 else [f"ck{stage}{string.ascii_lowercase[k]}" for k in range(min(n_ch, 26))]
    challengers = []
    for slot in slots:
        V = synthesis.synthesize(client, meter, bench, cfg, slot, parent=inc, residual=residual, rd_modules=rd_modules)
        if V is not None:
            state.setdefault("extra", {}).setdefault("archive", {})[V["sha256"]] = V
            challengers.append(V)
    if not challengers:
        return abstain("no valid challenger")
    report["challenger"] = challengers[0]["version"]
    if len(challengers) > 1:
        report["challengers"] = [V["version"] for V in challengers]
    sel = banks_fn(stage, "select")
    families = _families(bench, stage, sel)
    rows = select_rows(client, meter, store, bench, [inc] + challengers, sel, eps, workers, families,
                       rd_applicability)
    report["families"] = families
    report["selection"] = _public(rows)
    inc_row = next((r for r in rows if r["candidate_id"] == inc["version"]), None)
    if not rows or inc_row is None:
        return abstain("selection scoring failed")
    w = pick(rows)
    others = [r for r in rows if r["candidate_id"] != inc["version"]]
    ch_row = w if w["candidate_id"] != inc["version"] else (pick(others) if others else None)
    if ch_row is not None:
        report["challenger"] = ch_row["candidate_id"]
    report["selection_point"] = {
        "winner": w["candidate_id"],
        "challenger": {"worst_j": ch_row["worst_j"], "mean_j": ch_row["mean_j"]} if ch_row else None,
        "incumbent": {"worst_j": inc_row["worst_j"], "mean_j": inc_row["mean_j"]}}
    if w["candidate_id"] == inc["version"] or round(w["worst_j"], 4) <= round(inc_row["worst_j"], 4):
        return abstain("no strict selection gain")
    heldout = banks_fn(stage, "heldout")
    perW, diagW, winsW = head_J(client, meter, store, bench, w["V"], heldout, eps, workers, rd_applicability)
    perI, diagI, winsI = head_J(client, meter, store, bench, inc, heldout, eps, workers, rd_applicability)
    jw, ji = family_J(perW), family_J(perI)
    report["heldout_J"] = {"challenger": {f: round(x, 4) for f, x in jw.items()},
                           "incumbent": {f: round(x, 4) for f, x in ji.items()}}
    report["heldout_per_family"] = {"challenger": _per_list(perW), "incumbent": _per_list(perI)}
    report["diag"] = {"challenger": diagW, "incumbent": diagI}
    if any(f not in jw or f not in ji for f in families):
        return abstain("held-out scoring failed")
    ids = [c["cluster_id"] for c in heldout]
    complete = bool(ids) and all(k in winsW and k in winsI for k in ids)
    n10 = sum(1 for k in ids if winsW.get(k) and not winsI.get(k))
    n01 = sum(1 for k in ids if winsI.get(k) and not winsW.get(k))
    regress = [f for f in families if jw[f] < ji[f] - kappa]
    cp = heldout_diagnostic(perW, families)
    paired_lb = cp_lower(n10, max(1, n10 + n01), cp["alpha_per_family"])
    gate_rel = {"paired_wins_gt_losses": n10 > n01, f"no_regression_{kappa:g}": not regress,
                "records_complete": complete}
    gate_hard = {"family_lb": cp["all_above_chance"], "paired_lb": paired_lb > 0.5, "no_regression": not regress}
    report.update({"paired": {"n10": n10, "n01": n01, "lb": round(paired_lb, 4)}, "regressions": regress,
                   "cp_bounds": cp["bounds"], "gate_rel": gate_rel, "gate_hard": gate_hard,
                   "decision_rel": (f"PROMOTE {w['candidate_id']} (paired {n10}>{n01}, no regression)"
                                    if all(gate_rel.values())
                                    else f"ABSTAIN (relative rule: {json.dumps(gate_rel)})"),
                   "decision_hard": (f"PROMOTE {w['candidate_id']}" if all(gate_hard.values())
                                     else "ABSTAIN (certification gate)")})
    report["decision"] = report["decision_rel" if rule != "hard" else "decision_hard"]
    promote = all((gate_rel if rule != "hard" else gate_hard).values())
    return (w["V"] if promote else inc), report
