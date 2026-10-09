"""Tie-adjusted pairwise preference against Direct Generation (paper App. "Evaluation", paragraph
"Preference score", Eq. preference).

For each evaluated task the judge model compares the method's full response with the matched
Direct Generation response of the same actor in both presentation orders, blinded to method
identities and with deterministic decoding. The two verdicts are merged by sign into one task-level
outcome: s = sgn(order 1) - sgn(order 2) is a win when positive, a loss when negative and a tie
otherwise; identical or empty response pairs are mechanical ties and are never judged. With W, T
and L the numbers of won, tied and lost tasks, Pref = 100 (W + T/2) / (W + T + L), reported with a
contiguous-block bootstrap interval, an exact two-sided sign test over the decisive tasks and Holm
correction across the family of method-vs-baseline contrasts. The judge prompt and its parser come
from the benchmark adapter (``duel_request`` / ``duel_parse``).
"""
from __future__ import annotations

import json
import os

from src.eval import eval_dir, judge_client, load_arms
from src.llm.client import ContentInvalid, json_call, pmap
from src.utils.config import cfg_get
from src.utils.io import seed_from, sha, write_json
from src.utils.stats import block_bootstrap_ci, holm, sign_test_p
from src.utils.text import extract_json

SIGN = {"A": 1, "B": -1, "TIE": 0}
ATTEMPTS = 3                                    # judge calls per order before a task is reported unresolved
JUDGE_KW = {"max_tokens": 512, "temperature": 0.0}


def _verdict(client, meter, bench, judge, kw, task, a, b, key):
    """One blinded verdict for the pair (a, b): "A", "B" or "TIE", or None when no attempt yielded a valid
    JSON object. Later attempts vary the seed so deterministic decoding can change its output."""
    req = bench.duel_request(task, a, b)
    for attempt in range(ATTEMPTS):
        seed_key = key if attempt == 0 else f"{key}|retry{attempt}"
        try:
            if "messages" in req:
                k = dict(kw, seed=seed_from("eval", "call", seed_key))
                row = client.call_messages("duel", judge, req["messages"], k, nonce=attempt)
                v = bench.duel_parse(extract_json(row.get("text") or ""))
            else:
                v = json_call(client, meter, "duel", judge, req["system"], req["user"], bench.duel_parse, kw,
                              seed_key, nonce_retries=1)
        except ContentInvalid:
            v = None
        if v in SIGN:
            return v
    return None


def outcome_seq(verdicts, ids):
    """Per-task credit: 1 for a win, 0 for a loss, 0.5 for a tie (unjudged tasks are ties)."""
    return [0.5 if t not in verdicts else (1.0 if verdicts[t] > 0 else 0.0 if verdicts[t] < 0 else 0.5)
            for t in ids]


def duel(client, meter, bench, tasks_by_id, texts_a, texts_b, ids, judge, kw, workers, both_orders, boot, tag):
    """Judge arm A against arm B on ``ids``. Returns the summary row and the merged per-task verdicts
    (sign sums in {-2, ..., 2}; tasks with identical or empty texts are absent and count as ties)."""
    jobs = [t for t in ids if texts_a[t] != texts_b[t] and texts_a[t] and texts_b[t]]

    def one(tid):
        a, b = texts_a[tid], texts_b[tid]
        v1 = _verdict(client, meter, bench, judge, kw, tasks_by_id[tid], a, b,
                      f"duel|{tid}|{sha(a)[:12]}|{sha(b)[:12]}")
        if v1 is None:
            return tid, None
        if not both_orders:
            return tid, SIGN[v1]
        v2 = _verdict(client, meter, bench, judge, kw, tasks_by_id[tid], b, a,
                      f"duel|{tid}|{sha(b)[:12]}|{sha(a)[:12]}")
        return tid, None if v2 is None else SIGN[v1] - SIGN[v2]

    verdicts = dict(pmap(one, jobs, workers, f"duel_{tag}", every=50))
    unresolved = sorted(t for t, v in verdicts.items() if v is None)
    if unresolved:
        raise RuntimeError(f"[eval] {tag}: {len(unresolved)} tasks have no usable judge verdict after {ATTEMPTS} "
                           f"attempts per order (resolved calls are cached; re-run to retry): {unresolved[:5]}")
    seq = outcome_seq(verdicts, ids)
    wins = sum(1 for x in seq if x == 1.0)
    losses = sum(1 for x in seq if x == 0.0)
    mean, ci = block_bootstrap_ci(seq, block=boot["block"], reps=boot["reps"], seed=boot["seed"])
    row = {"n": len(ids), "judged": len(jobs), "W_L_T": [wins, losses, len(ids) - wins - losses],
           "pref": round(100 * mean, 2), "ci95": [round(100 * ci[0], 2), round(100 * ci[1], 2)],
           "sign_p": round(sign_test_p(wins, losses), 6)}
    return row, verdicts


def evaluate_preference(cfg, bench, work_dir, arms=("ucoe",), baseline="raw", workers=16):
    """Preference of every arm against ``baseline``. Writes ``<work_dir>/eval/preference.json`` and the
    per-task verdicts to ``preference_verdicts.json``; returns the results dict."""
    arms = [a for a in arms if a != baseline]
    if not arms:
        raise ValueError("[eval] no method arm to compare against the baseline")
    tasks, texts, missing = load_arms(cfg, bench, work_dir, list(arms) + [baseline])
    ids = [t["task_id"] for t in tasks]
    by_id = {t["task_id"]: t for t in tasks}
    judge = cfg_get(cfg, "models.judge")
    kw = {"max_tokens": int(cfg_get(cfg, "decoding.judge.max_tokens", JUDGE_KW["max_tokens"])),
          "temperature": float(cfg_get(cfg, "decoding.judge.temperature", JUDGE_KW["temperature"]))}
    both = bool(cfg_get(cfg, "evaluation.judge_both_orders", True))
    boot = {"reps": int(cfg_get(cfg, "evaluation.bootstrap.reps", 50000)),
            "block": int(cfg_get(cfg, "evaluation.bootstrap.block", 64)),
            "seed": int(cfg_get(cfg, "evaluation.bootstrap.seed", 42))}
    client, meter = judge_client(cfg, work_dir, "preference")
    results = {"label": f"tie-adjusted pairwise preference vs {baseline}: judge {judge}, blinded, "
                        f"{'both orders merged by sign' if both else 'one order'}, deterministic decoding; "
                        f"identical or empty outputs are mechanical ties",
               "n": len(ids), "judge": judge, "baseline": baseline, "arms": list(arms), "both_orders": both,
               "bootstrap": boot, "missing_rows": missing,
               "empty_baseline": sum(1 for t in ids if not texts[baseline][t])}
    domains = sorted({t["domain"] for t in tasks if t.get("domain")})
    verdicts, pvals = {}, {}
    for arm in arms:
        name = f"{arm}_vs_{baseline}"
        row, verd = duel(client, meter, bench, by_id, texts[arm], texts[baseline], ids, judge, kw, workers, both,
                         boot, name)
        row["changed_n"] = sum(1 for t in ids if texts[arm][t] != texts[baseline][t])
        if domains:
            row["pref_by_domain"] = {}
            for d in domains:
                seq = outcome_seq(verd, [t for t in ids if by_id[t].get("domain") == d])
                row["pref_by_domain"][d] = round(100 * sum(seq) / max(1, len(seq)), 2)
        results[name], verdicts[name], pvals[name] = row, verd, row["sign_p"]
        print(f"[eval] {name}: {json.dumps(row)}", flush=True)
    for name, p in holm(pvals).items():
        results[name]["holm_p"] = round(p, 6)
    meter.snapshot("preference evaluation")
    out = eval_dir(work_dir)
    write_json(os.path.join(out, "preference.json"), results, indent=1)
    write_json(os.path.join(out, "preference_verdicts.json"), verdicts, indent=1)
    _table(results, arms, baseline)
    return results


def _table(results, arms, baseline):
    print(f"\n[eval] preference vs {baseline} (n={results['n']}, judge {results['judge']})", flush=True)
    print(f"{'arm':<24}{'pref':>8}{'CI95':>20}{'W:L:T':>14}{'sign_p':>10}{'holm_p':>10}{'changed':>9}", flush=True)
    print(f"{baseline:<24}{'50.00':>8}", flush=True)
    for arm in arms:
        r = results[f"{arm}_vs_{baseline}"]
        print(f"{arm:<24}{r['pref']:>8}{str(r['ci95']):>20}{':'.join(map(str, r['W_L_T'])):>14}"
              f"{r['sign_p']:>10}{r.get('holm_p', '-'):>10}{r['changed_n']:>9}", flush=True)
