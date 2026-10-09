"""Arena-Hard native metric: export for the official ``arena-hard-auto`` toolkit (paper Table "Native metric and
scorer": style-adjusted win rate against the o3-mini-2025-01-31 anchor, both orders, five-level w3 weighting,
judged by gpt-4.1-2025-04-14).

The toolkit is not part of this release. This wrapper writes each arm's served answers in the toolkit's
``model_answer`` JSONL layout (one file per arm, the arm name as the model id; uid, ans_id, model, the user turn
with the prompt, the assistant turn with ``{"answer": text}``, tstamp and metadata) under
``<work_dir>/eval/arena_hard_official/model_answer/`` and prints the commands that produce the score. Zero model
calls; ``metadata.token_len`` is filled with the toolkit's tiktoken count when ``tiktoken`` is importable, and the
toolkit's ``utils/add_markdown_info.py`` completes the markdown features used by style control.
"""
from __future__ import annotations

import json
import os

from src.eval import eval_dir, load_arms
from src.eval.native import write_results
from src.utils.io import atomic_write

BENCH_DIR = "arena-hard-v2.0"
JUDGE = "gpt-4.1-2025-04-14"
BASELINE = "o3-mini-2025-01-31"
CATEGORY = "hard_prompt"
TOKENIZER_MODEL = "gpt-4o"


def _token_counter():
    """The toolkit's token length (tiktoken, gpt-4o encoding) when the package is installed, else None."""
    try:
        import tiktoken
    except ImportError:
        return None
    enc = tiktoken.encoding_for_model(TOKENIZER_MODEL)
    return lambda text: len(enc.encode(text, disallowed_special=()))


def export_answers(tasks, texts, out_dir, count_tokens=None):
    """One ``model_answer/<arm>.jsonl`` per arm in the toolkit's record layout; returns {arm: path}."""
    paths = {}
    os.makedirs(os.path.join(out_dir, "model_answer"), exist_ok=True)
    for arm, by_id in texts.items():
        lines = []
        for t in tasks:
            uid, text = t["task_id"], by_id[t["task_id"]]
            rec = {"uid": uid, "ans_id": f"{arm}-{uid}", "model": arm,
                   "messages": [{"role": "user", "content": t["task"]},
                                {"role": "assistant", "content": {"answer": text}}],
                   "tstamp": 0.0,
                   "metadata": {"token_len": count_tokens(text)} if count_tokens else {}}
            lines.append(json.dumps(rec, ensure_ascii=False))
        path = os.path.join(out_dir, "model_answer", f"{arm}.jsonl")
        atomic_write(path, "\n".join(lines) + "\n")
        paths[arm] = path
    return paths


def commands(out_dir, arms):
    """The toolkit steps, with the paper's judge and anchor, for answers exported under ``out_dir``."""
    models = " ".join(arms)
    return [
        f"# copy {out_dir}/model_answer/*.jsonl into <arena-hard-auto>/data/{BENCH_DIR}/model_answer/ next to the "
        f"anchor {BASELINE}.jsonl and the official question.jsonl",
        f"python utils/add_markdown_info.py --dir data/{BENCH_DIR}/model_answer "
        f"--output-dir data/{BENCH_DIR}/model_answer",
        f"# config/judge_config.yaml: bench_name {BENCH_DIR}, judge_model {JUDGE}, baseline {BASELINE}, category "
        f"{CATEGORY}, model_list [{models}]",
        "python gen_judgment.py --config-file config/judge_config.yaml",
        f"python show_result.py --judge-names {JUDGE} --control-features markdown length  # style-controlled score",
    ]


def evaluate(cfg, bench, work_dir, arms):
    """Export the served answers of every arm and record the toolkit commands in ``native_arena_hard.json``."""
    tasks, texts, missing = load_arms(cfg, bench, work_dir, arms)
    out_dir = os.path.join(eval_dir(work_dir), "arena_hard_official")
    count_tokens = _token_counter()
    paths = export_answers(tasks, texts, out_dir, count_tokens)
    cmds = commands(out_dir, list(texts))
    results = {"benchmark": bench.name, "metric": "style-controlled win rate against the o3-mini anchor (official "
               "arena-hard-auto toolkit, not part of this release)", "judge": JUDGE, "baseline": BASELINE,
               "category": CATEGORY, "n": len(tasks), "missing": missing, "exported": paths,
               "token_len": "tiktoken gpt-4o" if count_tokens else "absent: run utils/add_markdown_info.py",
               "commands": cmds}
    write_results(work_dir, bench.name, results)
    print(f"[eval] arena_hard: exported {len(tasks)} answers per arm for {sorted(texts)} to {out_dir}", flush=True)
    for c in cmds:
        print(f"  {c}", flush=True)
    return results
