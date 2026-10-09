# UCoE module catalogue (SpreadsheetBench 2 stream)

Base modules (every candidate keeps all of them; ids are fixed):

In `candidate.json` the `range` field is a JSON ARRAY of two numbers, `[0, 1]`, for EVERY criterion (deterministic modules take values in {0,1}, which lies in [0,1]); never the string "[0,1]" or "{0,1}".

| id  | range | check         | definition |
|-----|-------|---------------|------------|
| req | [0,1] | llm+tool      | every explicit requirement of the instruction is implemented (right sheet, cells, structure, units) |
| frm | [0,1] | llm+tool      | formulas are correct and internally consistent across rows/columns; no hard-coded numbers where a formula is expected |
| ref | [0,1] | llm+tool      | references point at the intended cells/sheets; nothing points at blanks or wrong ranges |
| prs | [0,1] | tool          | unrelated existing content is preserved (no collateral edits, deletions, renamed sheets) |
| num | [0,1] | llm+tool      | computed values are numerically plausible and consistent with the stated assumptions and labels |
| err | {0,1} | deterministic | 1 iff no recalculated error value (#REF!, #DIV/0!, #VALUE!, #NAME?, #N/A) in the changed region |
| chg | {0,1} | deterministic | 1 iff the candidate differs from the input |

Allowed checks inside a scoring session: open the workbook with `openpyxl` (formulas and cached values), recalculate an evaluation copy with the runner-provided `recalc` helper, diff against the input copy, list precedents of a formula, count error cells. Not allowed: the golden workbook, `answer_position`, the official evaluator, other candidates' scores.

A candidate may append at most two new modules of the same shape (id `[a-z][a-z0-9_]{1,24}`, a scoring question with 0/1 anchors, a declared check).
