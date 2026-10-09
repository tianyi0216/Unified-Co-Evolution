"""WildBench native metric: WB-Score under the official ``score.v2`` template (paper Table "Native
metric and scorer": mean rating from 1 to 10 with the full-response template, judged by the configured
judge model).

Protocol: the vendored ``wildbench_score_v2_template.md`` is filled with the full conversation history,
the current user query, the served response and the task's checklist without any truncation; the request
uses the official message form (system "You are a helpful AI assistant." plus the filled template as the
user turn), JSON mode, temperature 0 and 1024 output tokens. An empty response receives the official
placeholder score 1 without a call, while a non-empty response whose score cannot be parsed after one
retry is recorded as missing, never as 1. Aggregation follows the official viewer: raw mean on the 1-10
scale, adjusted score (mean - 5) * 20, and the task-macro adjusted score over primary tags when the
sidecar provides them. Paired deltas against Direct Generation use the tasks scored in both arms; among
the unscored pairs, identical texts count as zero differences and the rest are bounded by +-9 points.
"""
from __future__ import annotations

import json
import os
import re

from src.eval import eval_dir, judge_client, load_arms
from src.eval.native import (baseline_of, boot_cfg, boot_ci, judge_rows, load_sidecar, slice_means,
                             write_results)
from src.llm.client import pmap
from src.utils.config import cfg_get
from src.utils.io import read_json, sha, write_json
from src.utils.stats import holm, sign_test_p
from src.utils.text import extract_json

TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wildbench_score_v2_template.md")
SYSTEM = "You are a helpful AI assistant."          # official chat-request form
KW = {"max_tokens": 1024, "temperature": 0.0, "json_mode": True}
MAX_WORDS = -1                                        # no truncation (the official default truncates at 1000 words)
ATTEMPTS = 2                                          # one retry on a length or parse failure
RECORDS_FILE = "wildbench_score_requests.json"
BATCH = 256


def template():
    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        return f.read()


def shorten(text, K):
    """Official ``shorten``; the cap is passed explicitly at every call and ``MAX_WORDS = -1`` makes it a no-op."""
    if K > 0 and len(text.split(" ")) > K:
        text = " ".join(text.split(" ")[:K]) + "... (truncated)"
    return text


def compose(conv):
    """Official split of a conversation into the rendered history and the current user query."""
    history = ""
    for x in conv[:-1]:
        if x["role"] == "user":
            history += "USER: " + x["content"] + "\n\n"
        elif x["role"] == "assistant":
            history += "ASSISTANT: " + x["content"] + "\n\n"
    return history, conv[-1]["content"]


def build_prompt(tpl, conv, checklist, output, K=MAX_WORDS):
    history, query = compose(conv)
    p = tpl.replace("{$history}", shorten(history, K)).replace("{$user_query}", shorten(query, K))
    p = p.replace("{$model_output}", shorten(output, K))
    return p.replace("{$checklist}", "".join(f"- {c}\n" for c in checklist))


def parse_score(text):
    """Score in [1, 10] from the judge's JSON object, with a regex fallback on the raw text; None otherwise."""
    o = extract_json(text or "")
    v = o.get("score") if isinstance(o, dict) else None
    if v is None:
        m = re.search(r'"score"\s*:\s*"?\s*(\d+(?:\.\d+)?)', text or "")
        v = m.group(1) if m else None
    try:
        s = float(v)
    except (TypeError, ValueError):
        return None
    return s if 1 <= s <= 10 else None


def conversation(task):
    return task.get("messages") or [{"role": "user", "content": task["task"]}]


def requests(tpl, tasks, side, texts, arms):
    """Unique judge requests keyed by content hash, and the ``(arm, task_id) -> key`` slots (None marks an
    empty output, which takes the placeholder score without a call)."""
    uniq, slots = {}, {}
    for arm in arms:
        for t in tasks:
            tid = t["task_id"]
            y = texts[arm][tid]
            if not y.strip():
                slots[(arm, tid)] = None
                continue
            prompt = build_prompt(tpl, conversation(t), side[tid]["checklist"], y)
            h = sha(json.dumps([SYSTEM, prompt, KW], sort_keys=True))
            uniq.setdefault(h, (tid, prompt))
            slots[(arm, tid)] = h
    return uniq, slots


def score_request(client, meter, judge, key, prompt):
    """Judge one request: the record carries the score or the reason it is missing (content filter, length or
    parse failure after the retry, context length, transport)."""
    rec = {"request_hash": key, "retries": 0, "score": None, "error": None}
    try:
        for attempt, row in judge_rows(client, meter, "wb_score", judge, SYSTEM, prompt, KW, key, ATTEMPTS):
            rec.update({"judge_served": (row.get("model_meta") or {}).get("model"),
                        "finish_reason": row.get("finish_reason"), "usage": row.get("usage"), "nonce": attempt})
            if row.get("finish_reason") == "content_filter":
                rec["error"] = "content_filter"
                return rec
            s = parse_score(row.get("text") or "")
            if s is not None:
                rec.update({"score": s, "error": None})
                return rec
            rec["error"] = "length" if row.get("finish_reason") == "length" else "parse"
            rec["retries"] = attempt + 1
    except Exception as e:  # noqa: BLE001 - a failed transport is recorded as missing, never imputed
        msg = str(e)
        rec.update({"error": "context_length" if "context" in msg.lower() else "transport", "detail": msg[-200:]})
    return rec


def aggregate(scores, side):
    """Official aggregation: raw mean, adjusted (mean - 5) * 20, task-macro over primary tags when present."""
    vals = list(scores.values())
    if not vals:
        return {"n_scored": 0}
    mean = sum(vals) / len(vals)
    out = {"n_scored": len(vals), "raw_score": round(mean, 4), "adjusted_score": round((mean - 5) * 20, 2)}
    tags = {u: (side[u].get("primary_tag") if isinstance(side[u], dict) else None) for u in scores}
    if all(tags.values()):
        adj = {c: round((v - 5) * 20, 2) for c, v in slice_means(scores, list(scores), lambda u: tags[u]).items()}
        out["task_macro_score"] = round(sum(adj.values()) / len(adj), 2)
        out["per_tag_adjusted"] = adj
    return out


def paired(pa, pb, ids, ta, tb, boot):
    """Paired comparison on the tasks scored in both arms, with the missing pairs accounted for."""
    common = [u for u in ids if pa[u]["score"] is not None and pb[u]["score"] is not None]
    cs = set(common)
    miss = [u for u in ids if u not in cs]
    d = [pa[u]["score"] - pb[u]["score"] for u in common]
    wins = sum(1 for x in d if x > 0)
    losses = sum(1 for x in d if x < 0)
    zero = sum(1 for u in miss if ta[u] == tb[u])
    unk = len(miss) - zero
    tot = sum(d)
    return {"n": len(common), "n_missing": len(miss), "missing_identical_text_zero": zero, "missing_unknown": unk,
            "delta_raw10": round(tot / len(d), 4) if d else None,
            "delta_adjusted": round(tot / len(d) * 20, 2) if d else None,
            "ci95_raw10": boot_ci(d, boot["reps"], boot["seed"]) if d else None,
            "W_L_T": [wins, losses, len(d) - wins - losses], "sign_p": round(sign_test_p(wins, losses), 6),
            "missing_bound_delta_raw10": ([round((tot - 9 * unk) / len(ids), 4), round((tot + 9 * unk) / len(ids), 4)]
                                          if unk else None)}


def evaluate(cfg, bench, work_dir, arms):
    tasks, texts, missing = load_arms(cfg, bench, work_dir, arms)
    side = load_sidecar(cfg)
    tpl = template()
    ids = [t["task_id"] for t in tasks]
    baseline = baseline_of(arms)
    judge = cfg_get(cfg, "models.judge")
    workers = int(cfg_get(cfg, "output.workers", 8))
    uniq, slots = requests(tpl, tasks, side, texts, arms)
    rec_path = os.path.join(eval_dir(work_dir), RECORDS_FILE)
    recs = read_json(rec_path, default={}) or {}
    client, meter = judge_client(cfg, work_dir, "wildbench_score")

    def one(h):
        r = recs.get(h)
        if r and (r.get("score") is not None or r.get("error") in ("content_filter", "context_length")):
            return h, r
        tid, prompt = uniq[h]
        return h, dict(score_request(client, meter, judge, h, prompt), task_id=tid)

    keys = sorted(uniq)
    print(f"[eval] wildbench score.v2: {len(keys)} unique requests over {len(arms)} arms; "
          f"{sum(1 for v in slots.values() if v is None)} empty outputs take the placeholder score 1", flush=True)
    for j in range(0, len(keys), BATCH):
        recs.update(dict(pmap(one, keys[j:j + BATCH], workers, "wb_score", every=64)))
        write_json(rec_path, recs)
    meter.snapshot("wildbench score.v2")
    per = {}
    for arm in arms:
        per[arm] = {}
        for tid in ids:
            h = slots[(arm, tid)]
            if h is None:
                per[arm][tid] = {"score": 1.0, "placeholder": "empty", "error": None}
            else:
                per[arm][tid] = {"score": recs[h].get("score"), "error": recs[h].get("error"), "request_hash": h}
    common = [u for u in ids if all(per[a][u]["score"] is not None for a in arms)]
    results = {"label": f"WildBench WB-Score, official score.v2 template, full text, judge {judge}",
               "n": len(ids), "judge": judge, "baseline": baseline, "missing_rows": missing,
               "n_common_complete": len(common), "scores": {}, "scores_common_complete": {}, "missing": {}}
    for arm in arms:
        sc = {u: per[arm][u]["score"] for u in ids if per[arm][u]["score"] is not None}
        results["scores"][arm] = aggregate(sc, side)
        results["scores_common_complete"][arm] = aggregate({u: sc[u] for u in common}, side)
        miss = {u: per[arm][u].get("error") for u in ids if per[arm][u]["score"] is None}
        results["missing"][arm] = {"n": len(miss),
                                   "by_error": {e: sum(1 for x in miss.values() if x == e) for e in set(miss.values())},
                                   "ids": sorted(miss),
                                   "empty_placeholders": sum(1 for u in ids if per[arm][u].get("placeholder"))}
        print(f"[eval] wildbench {arm}: raw {results['scores'][arm].get('raw_score')} adjusted "
              f"{results['scores'][arm].get('adjusted_score')} missing {len(miss)}", flush=True)
    boot = boot_cfg(cfg)
    pvals = {}
    for arm in arms:
        if arm == baseline:
            continue
        name = f"{arm}_vs_{baseline}"
        results[name] = paired(per[arm], per[baseline], ids, texts[arm], texts[baseline], boot)
        pvals[name] = results[name]["sign_p"]
    if pvals:
        for name, p in holm(pvals).items():
            results[name]["holm_p"] = round(p, 6)
    results["per_task"] = per
    write_results(work_dir, bench.name, results)
    return results
