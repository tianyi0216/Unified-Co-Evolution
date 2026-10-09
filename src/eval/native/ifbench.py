"""IFBench native metric (paper Table "Native metric and scorer": programmatic verifiers; Table
"Verifiable tasks with Qwen3-8B").

Runs the official IFBench evaluator, vendored unmodified under ``instruction_following_eval/``, on the
served responses of every arm: strict and loose prompt-level and instruction-level accuracy, loose
instruction accuracy per constraint category, and paired comparisons against Direct Generation on the
primary accuracy. The primary is prompt-level loose accuracy by default, the primary figure of the research
evaluation (the 34.67 / 30.00 of the paper's Table 2); ``evaluation.native.ifbench_primary: strict_prompt_acc``
selects the strict variant instead.
Constraint ids and keyword arguments come from the evaluation sidecar. Zero model calls.
"""
from __future__ import annotations

import os
import sys

from src.eval import load_arms
from src.eval.native import baseline_of, boot_cfg, load_sidecar, paired_vs_baseline, write_results
from src.utils.config import cfg_get

VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "instruction_following_eval")
PRIMARY = "loose_prompt_acc"
LEVELS = ("strict", "loose")


def official():
    """The vendored evaluator module. Its package uses absolute ``ifbench`` imports, so the vendored
    directory joins ``sys.path`` once."""
    if VENDOR_DIR not in sys.path:
        sys.path.insert(0, VENDOR_DIR)
    import evaluation_lib
    return evaluation_lib


def score_arm(ev, inputs, prompt_to_response):
    """Strict and loose outputs of the official evaluator for one arm."""
    out = {}
    for level, fn in (("strict", ev.test_instruction_following_strict),
                      ("loose", ev.test_instruction_following_loose)):
        outs = [fn(inp, prompt_to_response) for inp in inputs]
        n_instr = sum(len(o.follow_instruction_list) for o in outs)
        out[level] = {
            "prompt_acc": round(100 * sum(o.follow_all_instructions for o in outs) / len(outs), 2),
            "instr_acc": round(100 * sum(sum(o.follow_instruction_list) for o in outs) / max(1, n_instr), 2),
            "per_prompt": {str(inp.key): bool(o.follow_all_instructions) for inp, o in zip(inputs, outs)},
            "per_instr": {str(inp.key): [bool(x) for x in o.follow_instruction_list]
                          for inp, o in zip(inputs, outs)}}
    return out


def evaluate(cfg, bench, work_dir, arms):
    tasks, texts, missing = load_arms(cfg, bench, work_dir, arms)
    side = load_sidecar(cfg)
    ev = official()
    ids = [t["task_id"] for t in tasks]
    inputs = [ev.InputExample(key=t["task_id"], instruction_id_list=list(side[t["task_id"]]["instruction_id_list"]),
                              prompt=t["task"], kwargs=[dict(k) for k in side[t["task_id"]]["kwargs"]])
              for t in tasks]
    primary = str(cfg_get(cfg, "evaluation.native.ifbench_primary", PRIMARY))
    level = primary.split("_", 1)[0]
    if level not in LEVELS or not primary.endswith("_prompt_acc"):
        raise ValueError(f"evaluation.native.ifbench_primary must be loose_prompt_acc or strict_prompt_acc, "
                         f"got {primary!r}")
    baseline = baseline_of(arms)
    per = {arm: score_arm(ev, inputs, {t["task"]: texts[arm][t["task_id"]] for t in tasks}) for arm in arms}
    cats = sorted({iid.split(":")[0] for t in ids for iid in side[t]["instruction_id_list"]})
    results = {"label": "IFBench official programmatic evaluation (vendored evaluator; strict and loose, "
                        "prompt and instruction level; paired statistics on the primary accuracy)",
               "n": len(ids), "primary": primary, "baseline": baseline, "missing_rows": missing}
    scores = {}
    for arm in arms:
        for lv in LEVELS:
            results[f"{arm}_{lv}_prompt_acc"] = per[arm][lv]["prompt_acc"]
            results[f"{arm}_{lv}_instr_acc"] = per[arm][lv]["instr_acc"]
        per_instr = per[arm]["loose"]["per_instr"]
        by_cat = {}
        for cat in cats:
            hit = tot = 0
            for tid in ids:
                for j, iid in enumerate(side[tid]["instruction_id_list"]):
                    if iid.split(":")[0] == cat:
                        tot += 1
                        hit += int(per_instr[tid][j])
            by_cat[cat] = round(100 * hit / max(1, tot), 2)
        results[f"{arm}_loose_instr_by_category"] = by_cat
        scores[arm] = {tid: float(per[arm][level]["per_prompt"][tid]) for tid in ids}
        print(f"[eval] ifbench {arm}: loose prompt {per[arm]['loose']['prompt_acc']} / strict prompt "
              f"{per[arm]['strict']['prompt_acc']} / loose instr {per[arm]['loose']['instr_acc']}", flush=True)
    results.update(paired_vs_baseline(scores, texts, ids, arms, baseline, boot_cfg(cfg), scale=100.0, digits=3))
    results["per_prompt"] = {arm: {lv: per[arm][lv]["per_prompt"] for lv in LEVELS} for arm in arms}
    write_results(work_dir, bench.name, results)
    return results
