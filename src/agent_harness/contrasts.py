"""Response-contrast pairs for the agent-harness study.

Implements the contrast construction of paper §3.2 ("Response-contrast construction") for
workbooks: the positive is the actor's own draft on a deployment-disjoint calibration source,
the negative applies ONE declared realistic perturbation, and the pair is admitted only when
the checks of `core.make_contrast` hold (same base, non-empty declared-only diff, direction
rule). Families (App. "Agent-harness study on SpreadsheetBench-2"):
  formula_error      a range shortened by one row or an operator swapped in a new formula
  broken_reference   one cell reference moved to a neighbouring cell of the same kind
  regression_damage  one untouched numeric input cell outside the formula precedents altered
"""

from __future__ import annotations

import re

from src.agent_harness import core as C
from src.agent_harness.evidence import canon_formula

_REF = re.compile(r"(?<![A-Z$!:])\$?([A-Z]{1,3})\$?(\d+)(?![\d:(])")
_RANGE = re.compile(r"\$?([A-Z]{1,3})\$?(\d+):\$?([A-Z]{1,3})\$?(\d+)")
_CELLS = {}


def clear_cache():
    _CELLS.clear()


def cells_of(path):
    """{(sheet, coord): value} of every non-empty cell, cached by (path, sha); array formulas canonicalized."""
    import openpyxl

    key = (path, C.sha_file(path))
    if key not in _CELLS:
        wb = openpyxl.load_workbook(path)
        _CELLS[key] = {
            (n, c.coordinate): (canon_formula(c.value) if hasattr(c.value, "text") else c.value)
            for n in wb.sheetnames
            for row in wb[n].iter_rows()
            for c in row
            if c.value is not None
        }
    return _CELLS[key]


def _kind(v):
    if isinstance(v, dict) and v.get("array"):
        return "array_formula"
    if isinstance(v, str) and v.startswith("="):
        return "formula"
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return "number"
    return "text" if v is not None else None


def _changed_formula_cells(inp, pos):
    return sorted(k for k, v in pos.items() if _kind(v) == "formula" and inp.get(k) != v)


def perturb_formula(inp, pos):
    for k in _changed_formula_cells(inp, pos):
        f = pos[k]
        m = _RANGE.search(f)
        if m and (m.group(1) == m.group(3)) and int(m.group(4)) - int(m.group(2)) >= 2:
            return (
                k,
                f[: m.start()] + "%s%s:%s%d" % (m.group(1), m.group(2), m.group(3), int(m.group(4)) - 1) + f[m.end() :],
            )
        for a, b in (("*", "+"), ("-", "+"), ("/", "*")):
            if a in f:
                return k, f.replace(a, b, 1)
    return None


def perturb_reference(inp, pos):
    import openpyxl.utils as U

    for k in _changed_formula_cells(inp, pos):
        sheet, f = k[0], pos[k]
        for m in _REF.finditer(f):
            col, row = m.group(1), int(m.group(2))
            ci = U.column_index_from_string(col)
            src = pos.get((sheet, "%s%d" % (col, row)))
            if src is None or (sheet, "%s%d" % (col, row)) == k:
                continue
            for dc, dr in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                r2, c2 = row + dr, ci + dc
                if r2 < 1 or c2 < 1:
                    continue
                coord = "%s%d" % (U.get_column_letter(c2), r2)
                v2 = pos.get((sheet, coord))
                if v2 is not None and (sheet, coord) != k and _kind(v2) == _kind(src) and v2 != src:
                    return k, f[: m.start()] + coord + f[m.end() :]
    return None


def _referenced_cells(sheet, formula):
    import openpyxl.utils as U

    out = set()
    for m in _RANGE.finditer(formula):
        c1, r1, c2, r2 = (
            U.column_index_from_string(m.group(1)),
            int(m.group(2)),
            U.column_index_from_string(m.group(3)),
            int(m.group(4)),
        )
        for r in range(min(r1, r2), max(r1, r2) + 1):
            for c in range(min(c1, c2), max(c1, c2) + 1):
                out.add((sheet, "%s%d" % (U.get_column_letter(c), r)))
    for m in _REF.finditer(formula):
        out.add((sheet, "%s%s" % (m.group(1), m.group(2))))
    return out


def perturb_regression(inp, pos, instruction):
    prec = set()
    for k in _changed_formula_cells(inp, pos):
        prec |= _referenced_cells(k[0], pos[k])
    for k in sorted(inp, key=lambda x: (x[0], int(re.match(r"[A-Z]+(\d+)", x[1]).group(1)), x[1])):
        v = inp[k]
        if _kind(v) != "number" or pos.get(k) != v or k in prec or re.search(r"\b%s\b" % k[1], instruction):
            continue
        return k, (v + 1 if isinstance(v, int) else round(v * 1.01 + 0.01, 6))
    return None


def make_pair(src, draft_path, family, out_dir, audit=None):
    """One contrast of `family` from a calibration source {"task_id", "category", "instruction", "input_path"} and the
    actor's draft.
    Returns the pair manifest, or None when no perturbation of that family is constructible (recorded through `audit`
    when given).
    """
    import openpyxl

    inp, pos = cells_of(src["input_path"]), cells_of(draft_path)
    plan = {
        "formula_error": lambda: perturb_formula(inp, pos),
        "broken_reference": lambda: perturb_reference(inp, pos),
        "regression_damage": lambda: perturb_regression(inp, pos, src["instruction"]),
    }[family]()
    if not plan:
        return None
    (sheet, coord), new = plan
    old = pos[(sheet, coord)]

    def fn(path):
        wb = openpyxl.load_workbook(path)
        wb[sheet][coord] = new
        wb.save(path)
        return [(sheet, coord)]

    def rule(path):
        wb = openpyxl.load_workbook(path)
        return wb[sheet][coord].value == old

    try:
        man = C.make_contrast(src["input_path"], draft_path, family, out_dir, fn, rule, protected=set())
    except ValueError as e:
        if audit:
            audit("pair_rejected", source=src["task_id"], family=family, reason=str(e)[:200])
        return None
    man.update(
        {
            "source": src["task_id"],
            "category": src["category"],
            "instruction": src["instruction"],
            "perturbation": {"cell": "%s!%s" % (sheet, coord), "from": old, "to": new},
            "dir": out_dir,
        }
    )
    return man
