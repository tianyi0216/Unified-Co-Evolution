"""WritingBench native metric: Critic score (paper Table "Native metric and scorer": critic score from
1 to 10 by Critic-7B).

Official contract (WritingBench evaluation): the critic model receives ``evaluate_system`` and
``evaluate_prompt`` verbatim with the task's query, the served response and one checklist criterion
injected as the string form of the criterion dictionary; decoding uses temperature 1.0 and top_p 0.95
with one evaluation per criterion; scores are integers from 1 to 10 and a task's score is the mean over
its criteria. The critic is served through the OpenAI-compatible endpoint named in
``evaluation.native.critic`` (the paper serves WritingBench-Critic-Model-Qwen-7B locally). Scores are
keyed by (task id, criterion index, response hash) so byte-identical responses share one score; an output
that stays unparseable after bounded seed-varied retries aborts the run after persisting the resolved
scores rather than filling a value. Reports the mean score per arm, per-domain and per-subset slices, and
paired comparisons against Direct Generation.
"""
from __future__ import annotations

import json
import os
import re

from src.eval import eval_dir, load_arms
from src.eval.native import (baseline_of, boot_cfg, load_sidecar, paired_vs_baseline, slice_means,
                             write_results)
from src.llm.client import pmap
from src.utils.config import cfg_get
from src.utils.io import read_json, seed_from, sha, write_json

CRITIC_MODEL = "AQuarterMile/WritingBench-Critic-Model-Qwen-7B"
CRITIC_KW = {"max_tokens": 1024, "temperature": 1.0}
TOP_P = 0.95
MAX_ATTEMPTS = 8
SCORES_FILE = "writingbench_critic_scores.json"

# --- official prompts, verbatim from WritingBench ---
EVALUATE_SYSTEM = ("You are an expert evaluator with extensive "
                   "experience in evaluating response of given "
                   "query.")
EVALUATE_PROMPT = """Evaluate the Response based on the Query and Criteria provided following the Scoring Rules.

** Scoring Rules **

"1-2": "Low score description: Critical deficiencies and major issues that prevent adequate functionality.",
"3-4": "Below average score description: Lacking with noticeable shortcomings that impact overall effectiveness and require improvement.",
"5-6": "Average score description: Adequate but not exemplary, Baseline performance that meets essential requirements. Most models may achieve this score.",
"7-8": "Above average score description: Strong performance characterized by competent execution, though minor refinements are needed to achieve excellence.",
"9-10": "High score description: Exceptional performance with all aspects optimally addressed, demonstrating superior effectiveness and quality without any flaws."

-Provide reasons for each score by indicating specific strengths or deficiencies within the Response. Reference exact text passages to justify the score, ensuring that each reason is concrete and aligns with the criteria requirements while highlighting key gaps from the ideal answer.

-Be very STRICT and do not be misled by format or length; ensure that the Response is thoroughly evaluated beyond superficial appearances.

-Carefully discern whether the content of the Response is an illusion, appearing substantial but actually entirely fabricated.

-Sometimes the model may only provide an introduction or an overview without truly completing the query, which should be considered a failed response. Carefully discern this.

-Scoring Range: Assign an integer score between 1 to 10

** Output format **
(Remove symbols that interfere with JSON parsing, don't use " inside reason)
Return the results in the following JSON format, Only output the following JSON format and nothing else:
```json
{{
    "score": an integer score between 1 to 10,
    "reason": "Specific and detailed justification for the score using text elements."
}}

** Criteria **
```{criteria}```

** Query **
```{query}```

** Response **
```{response}```

Provide your evaluation based on the criteria restated below:

```{criteria}```

** Output format **
(Remove symbols that interfere with JSON parsing, don't use " inside reason)
Return the results in the following JSON format, Only output the following JSON format and nothing else:
```json
{{
    "score": an integer score between 1 to 10,
    "reason": "Specific and detailed justification for the score using text elements."
}}
```"""


def parse_score(text):
    """Official parse: ``json.loads(response.strip('json|```'))``, plus a fence-regex fallback; the score
    must be an integer from 1 to 10."""
    for cand in (text.strip().strip("json|`"),
                 re.sub(r"^.*?```json\s*|\s*```.*$", "", text.strip(), flags=re.S)):
        try:
            o = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        try:
            sc = int(o.get("score"))
        except (TypeError, ValueError, AttributeError):
            continue
        if 1 <= sc <= 10:
            return sc
    m = re.search(r'"score"\s*:\s*(\d+)', text)
    if m and 1 <= int(m.group(1)) <= 10:
        return int(m.group(1))
    return None


def critic_client(cfg, work_dir):
    """Cache-first client for the critic endpoint (OpenAI-compatible, ``top_p`` carried in ``extra_body``);
    returns ``(client, meter, model)``."""
    from src.llm.base import make_transport
    from src.llm.client import Client, Meter
    crit = cfg_get(cfg, "evaluation.native.critic") or {}
    models_cfg = {"backend": "openai",
                  "openai": {"base_url": crit.get("base_url"), "api_key_env": crit.get("api_key_env", "OPENAI_API_KEY"),
                             "extra_body": {"top_p": TOP_P}}}
    d = eval_dir(work_dir)
    meter = Meter(os.path.join(d, "cost_meter_writingbench_critic.json"), None)
    client = Client(os.path.join(d, "llm_cache_writingbench_critic.jsonl"), make_transport(models_cfg), meter)
    return client, meter, str(crit.get("model") or CRITIC_MODEL)


def evaluate(cfg, bench, work_dir, arms):
    tasks, texts, missing = load_arms(cfg, bench, work_dir, arms)
    side = load_sidecar(cfg)
    ids = [t["task_id"] for t in tasks]
    queries = {t["task_id"]: t["task"] for t in tasks}
    baseline = baseline_of(arms)
    workers = int(cfg_get(cfg, "output.workers", 8))
    scores_path = os.path.join(eval_dir(work_dir), SCORES_FILE)
    scores = read_json(scores_path, default={}) or {}
    jobs = {}
    for arm in arms:
        for tid in ids:
            rsha = sha(texts[arm][tid])[:16]
            for k in range(len(side[tid]["checklist"])):
                gkey = f"{tid}|{k}|{rsha}"
                if gkey not in scores:
                    jobs[gkey] = (tid, k, texts[arm][tid])
    groups = {f"{tid}|{sha(texts[a][tid])[:16]}" for tid in ids for a in arms}
    print(f"[eval] writingbench critic: {len(groups)} unique task-response groups; {len(jobs)} criterion scores to "
          f"grade ({len(scores)} already in the ledger)", flush=True)
    client, meter, model = critic_client(cfg, work_dir)

    def one(entry):
        gkey, (tid, k, response) = entry
        user = EVALUATE_PROMPT.format(criteria=str(side[tid]["checklist"][k]), query=queries[tid], response=response)
        base = seed_from("eval", "critic", gkey)
        for nonce in range(MAX_ATTEMPTS):
            if nonce:
                meter.syntax_retry()
            row = client.call("wb_critic", model, EVALUATE_SYSTEM, user, dict(CRITIC_KW, seed=base + nonce), nonce)
            sc = parse_score(row.get("text") or "")
            if sc is not None:
                return gkey, sc
        return gkey, None

    got = dict(pmap(one, sorted(jobs.items()), workers, "wb_critic", every=200))
    scores.update({k: v for k, v in got.items() if v is not None})
    write_json(scores_path, scores)
    unresolved = sorted(k for k, v in got.items() if v is None)
    if unresolved:
        raise RuntimeError(f"[eval] writingbench critic: {len(unresolved)} criterion scores unresolved after "
                           f"{MAX_ATTEMPTS} attempts (resolved scores saved): {unresolved[:3]}")
    meter.snapshot("writingbench critic")
    per = {}
    for arm in arms:
        per[arm] = {}
        for tid in ids:
            rsha = sha(texts[arm][tid])[:16]
            vals = [scores[f"{tid}|{k}|{rsha}"] for k in range(len(side[tid]["checklist"]))]
            per[arm][tid] = sum(vals) / len(vals)
    results = {"label": f"WritingBench critic score ({model}; official prompt and decoding, one evaluation per "
                        f"criterion; task score = mean over criteria)",
               "n": len(ids), "critic": model, "baseline": baseline, "missing_rows": missing}
    for arm in arms:
        mean = sum(per[arm].values()) / len(ids)
        results[f"score_{arm}"] = round(mean, 3)
        results[f"score_{arm}_x10"] = round(10 * mean, 1)
        by_domain = slice_means(per[arm], ids, lambda t: str(side[t].get("domain1") or "all"), digits=3)
        results[f"score_{arm}_by_domain"] = by_domain
        results[f"score_{arm}_domain_macro"] = round(sum(by_domain.values()) / len(by_domain), 3)
        subsets = sorted({s for t in ids for s in (side[t].get("subsets") or {}) if not s.endswith("_C")})
        results[f"score_{arm}_by_subset"] = {
            ss: round(sum(per[arm][t] for t in ids if (side[t].get("subsets") or {}).get(ss))
                      / max(1, sum(1 for t in ids if (side[t].get("subsets") or {}).get(ss))), 3)
            for ss in subsets}
        print(f"[eval] writingbench {arm}: score {results[f'score_{arm}']} (domain macro "
              f"{results[f'score_{arm}_domain_macro']})", flush=True)
    results.update(paired_vs_baseline(per, texts, ids, arms, baseline, boot_cfg(cfg), digits=4))
    write_results(work_dir, bench.name, results)
    return results
