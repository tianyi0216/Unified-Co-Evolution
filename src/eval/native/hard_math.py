"""Hard-MATH native metric (paper Table "Native metric and scorer": answer match against the reference;
Table "Verifiable tasks with Qwen3-8B").

Answer scorer: only the last brace-balanced ``\\boxed{...}`` of a response is parsed, so reasoning text
never reaches the matcher and a missing box is incorrect without last-line guessing. MATH-500 items use
math-verify (parse + verify) when it is installed and otherwise a conservative normalized comparison that
keeps ``\\text{...}`` content; AIME items are judged as integers in 0..999 with leading zeros allowed.
Reference answers come from the evaluation sidecar. Reports accuracy per arm (overall, MATH-500 and AIME
slices, per domain) and paired comparisons against Direct Generation with the exact McNemar sign test.
"""
from __future__ import annotations

import os
import re

from src.eval import load_arms
from src.eval.native import (baseline_of, boot_cfg, load_sidecar, paired_vs_baseline, slice_means,
                             write_results)
from src.utils.config import cfg_get

_MV_NO_TIMEOUT = os.name == "nt"      # math-verify's timeout wrapper needs signal.alarm, absent on Windows
_MV = None


def math_verify_api():
    """``(parse, verify)`` from math-verify when importable, else None; resolved once per process."""
    global _MV
    if _MV is None:
        try:
            from math_verify import parse, verify
            _MV = (parse, verify)
        except ImportError:
            _MV = ()
    return _MV or None


def extract_boxed(text):
    """Content of the last ``\\boxed{...}`` with balanced braces, or None."""
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
    i = idx + len("\\boxed{")
    depth, out = 1, []
    while i < len(text) and depth:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
        i += 1
    return "".join(out) if depth == 0 else None


def _norm(a):
    a = a.strip().strip("$").strip()
    a = re.sub(r"\\text\{([^{}]*)\}", r"\1", a)      # keep the content
    a = re.sub(r"\\mbox\{([^{}]*)\}", r"\1", a)
    a = a.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    a = a.replace("\\left", "").replace("\\right", "").replace("\\!", "")
    a = a.replace("\\,", "").replace("\\;", "")
    a = re.sub(r"(?<!\\)\\ ", "", a)      # latex escaped space; never eat one backslash of a row separator
    a = re.sub(r"\s+", "", a)
    a = re.sub(r"^[a-zA-Z]\\?=", "", a)
    a = re.sub(r"^[a-zA-Z]+\\in", "", a)          # "x \in [2,5]" -> "[2,5]"
    a = re.sub(r"\\(?:big|Big|bigg|Bigg)[lr]?", "", a)
    a = a.rstrip(".").rstrip("\\")
    a = re.sub(r"\^\{?\\circ\}?$", "", a)
    a = a.replace("^{\\circ}", "").replace("%", "")
    return a


def _as_num(a):
    try:
        return float(a)
    except ValueError:
        m = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", a)
        if m and int(m.group(2)):
            return int(m.group(1)) / int(m.group(2))
        m = re.fullmatch(r"(-?\d+)/(-?\d+)", a)
        if m and int(m.group(2)):
            return int(m.group(1)) / int(m.group(2))
        return None


def _aime_int(a):
    a = _norm(a)
    if re.fullmatch(r"0*\d{1,4}", a):
        v = int(a)
        return v if 0 <= v <= 999 else None
    return None


class ScorerUnavailable(RuntimeError):
    """math-verify was required for a MATH-500 item but is not installed."""


def correct(solution_text, gold, domain="math", require_mv=False):
    """True when the response's final boxed answer matches ``gold`` under the domain's rule. ``require_mv``
    refuses to fall back to the normalized comparison on MATH-500 items."""
    if require_mv and not domain.startswith("aime") and math_verify_api() is None:
        raise ScorerUnavailable("math-verify is required for MATH-500 scoring "
                                "(evaluation.native.require_math_verify) but is not installed")
    got = extract_boxed(solution_text)
    if got is None:
        return False
    if domain.startswith("aime"):
        g, h = _aime_int(got), _aime_int(gold)
        return g is not None and h is not None and g == h
    mv = math_verify_api()
    if mv is not None:
        parse, verify = mv
        try:
            if _MV_NO_TIMEOUT:
                gp = parse("$" + gold + "$", parsing_timeout=None)
                ap = parse("$" + got + "$", parsing_timeout=None)
                if gp and ap and verify(gp, ap, timeout_seconds=None):
                    return True
            else:
                gp = parse("$" + gold + "$")
                ap = parse("$" + got + "$")
                if gp and ap and verify(gp, ap):
                    return True
        except Exception:  # noqa: BLE001 - any parser failure falls through to the normalized comparison
            pass
    g1, g2 = _norm(got), _norm(gold)
    if g1 == g2:
        return True
    n1, n2 = _as_num(g1), _as_num(g2)
    return n1 is not None and n2 is not None and abs(n1 - n2) < 1e-6


def reference(side, task):
    """``(gold, domain)`` for a task: the sidecar entry is the gold string or a record with ``gold`` (or
    ``answer``) and optionally ``domain``; the task's own domain wins when present."""
    entry = side[task["task_id"]]
    if isinstance(entry, dict):
        gold = entry.get("gold", entry.get("answer"))
        domain = task.get("domain") or entry.get("domain")
    else:
        gold, domain = entry, task.get("domain")
    if gold is None:
        raise KeyError(f"[eval] task {task['task_id']}: no reference answer in the sidecar")
    return str(gold), str(domain or "math")


def _pct(values, ids):
    return round(100 * sum(values[t] for t in ids) / len(ids), 2) if ids else None


def evaluate(cfg, bench, work_dir, arms):
    tasks, texts, missing = load_arms(cfg, bench, work_dir, arms)
    side = load_sidecar(cfg)
    require_mv = bool(cfg_get(cfg, "evaluation.native.require_math_verify", False))
    ids = [t["task_id"] for t in tasks]
    ref = {t["task_id"]: reference(side, t) for t in tasks}
    aime = [t for t in ids if ref[t][1].startswith("aime")]
    math500 = [t for t in ids if not ref[t][1].startswith("aime")]
    baseline = baseline_of(arms)
    results = {"label": "Hard-MATH answer accuracy: last boxed answer vs the reference (math-verify when "
                        "installed, else normalized match; AIME answers as integers)",
               "n": len(ids), "slices": {"math500": len(math500), "aime": len(aime)},
               "math_verify": math_verify_api() is not None, "baseline": baseline, "missing_rows": missing}
    scores = {}
    for arm in arms:
        scores[arm] = {t: float(correct(texts[arm][t], ref[t][0], ref[t][1], require_mv)) for t in ids}
        acc = {"all": _pct(scores[arm], ids), "math500": _pct(scores[arm], math500), "aime": _pct(scores[arm], aime),
               "by_domain": slice_means(scores[arm], ids, lambda t: ref[t][1], scale=100.0, digits=2)}
        results[f"acc_{arm}"] = acc
        results[f"boxed_{arm}"] = round(100 * sum(1 for t in ids if extract_boxed(texts[arm][t]) is not None)
                                        / len(ids), 2)
        print(f"[eval] hard_math {arm}: acc {acc['all']} (math500 {acc['math500']}, aime {acc['aime']}; "
              f"boxed {results[f'boxed_{arm}']}%)", flush=True)
    results.update(paired_vs_baseline(scores, texts, ids, arms, baseline, boot_cfg(cfg), scale=100.0, digits=2))
    write_results(work_dir, bench.name, results)
    return results
