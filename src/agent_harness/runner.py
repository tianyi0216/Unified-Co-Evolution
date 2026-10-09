"""Lifecycle runner of UCoE inside the Codex agent harness on SpreadsheetBench-2.

Implements Algorithm 1 of the paper for the agent-harness study (Results, "UCoE transfers to a
Codex agent harness"; App. "Agent-harness study on SpreadsheetBench-2") with three arms on one
frozen task stream:
  native  Direct Generation: draft only, no feedback, no harness state
  judge   LLM as a Judge: a fixed holistic verifier guides one revision
  ucoe    K synthesized verifier candidates, deployment-disjoint calibration banks built from the
          actor's own drafts, maximin selection with predeclared bank gates, one localized revision
          per task accepted iff r(yR) > r(y0) + eps, a bounded experience block updated only after
          accepted repairs, and checkpoints that admit a challenger under the empirical relative rule.
Engineering invariants: crash-safe stage journal, copy-on-write task trees, fail-closed
recalculation and evidence binding (echo of the evidence sha), read-only prepared inputs rehashed
before scoring and sealing, reward-ceiling skip, immutable served archive, exactly-once sealing.
Offline seams: `tasks_fn`, `calib_fn`, `recalc_fn` and `transport_factory` replace the dataset,
LibreOffice and the Codex app-server in tests.
"""

from __future__ import annotations

import collections
import json
import os
import random
import shutil
import stat
import time

from src.agent_harness import HardStop
from src.agent_harness import core as C
from src.agent_harness import contrasts as X
from src.agent_harness import evaluate as EV
from src.agent_harness import evidence as E
from src.agent_harness.codex_backend import DEFAULT_LIMITS, CodexBackend, audit, model_matches, transport_ref
from src.utils.io import atomic_write, read_json, sha

HERE = os.path.dirname(os.path.abspath(__file__))
PROTOCOL_VERSION = "v2"
EVIDENCE_VERSION = E.EVIDENCE_VERSION
ARMS = ("native", "judge", "ucoe")
CATS = ("Template", "Financial_Model", "Debugging")
DEFAULT_K = 2
DEFAULT_TARGETS = {"Template": 30, "Financial_Model": 40, "Debugging": 30}
DEFAULT_EXCLUDE = {"Financial_Model": ["06"]}  # workbook family whose inputs cannot be read
STREAM_SEED = "cc100"  # fixed namespace of the frozen stream order; changing it changes the task order
CALIB_SEED = "v2-calib"
PROFILE_HINTS = {
    "h0a": "balanced weighted aggregation",
    "h0b": "correctness/formula-gated aggregation",
    "h0c": "bottleneck/min-sensitive aggregation",
    "h0d": "preservation/recalculation-conservative aggregation",
}
FORBIDDEN_KEYS = ("golden_response_path", "answer_position")
STATUSES = (
    "accepted",
    "reward_ceiling_skip",
    "score_abstain",
    "localize_abstain",
    "c1_abstain",
    "proposal_noop",
    "no_strict_gain",
    "draft_missing",
)
HARD = (
    "recalc_error",
    "invalid_xlsx",
    "provenance_mismatch",
    "missing_effort",
    "timeout",
    "malformed_output",
    "evidence_mismatch",
    "model_fallback",
    "transport_error",
)

V_FIXED = {
    "version": "V_fixed_judge",
    "criteria": [
        {
            "id": "req",
            "definition": "every explicit requirement of the instruction is implemented (right sheet, cells, "
            "structure, units)",
            "range": [0, 1],
            "check": "llm+tool",
        },
        {
            "id": "frm",
            "definition": "formulas are correct and internally consistent across rows/columns; no hard-coded numbers "
            "where a formula is expected",
            "range": [0, 1],
            "check": "llm+tool",
        },
        {
            "id": "ref",
            "definition": "references point at the intended cells/sheets; nothing points at blanks or wrong ranges",
            "range": [0, 1],
            "check": "llm+tool",
        },
        {
            "id": "prs",
            "definition": "unrelated existing content is preserved (no collateral edits, deletions, renamed sheets)",
            "range": [0, 1],
            "check": "tool",
        },
        {
            "id": "num",
            "definition": "computed values are numerically plausible and consistent with the stated assumptions and "
            "labels",
            "range": [0, 1],
            "check": "llm+tool",
        },
        {
            "id": "err",
            "definition": "1 iff no recalculated error value in the changed region",
            "range": [0, 1],
            "check": "deterministic",
        },
        {
            "id": "chg",
            "definition": "1 iff the candidate differs from the input",
            "range": [0, 1],
            "check": "deterministic",
        },
    ],
    "aggregation": "err * chg * (req + frm + ref + prs + num) / 5",
    "feedback_rule": "weakest active non-deterministic criterion; one local repair",
}
V_FIXED = C.validate_candidate(V_FIXED)  # frozen the core way (sha over criteria/aggregation/feedback_rule/parent)
V_FIXED["version"] = "V_fixed_judge"

# ------------------------------------------------- prompts (verbatim)
SYNTH = (
    'Read ./modules.md. Write ./candidate.json = a complete reward controller for spreadsheet tasks: {{"criteria": '
    "[every base module with its id/definition/"
    'range/check, plus at most two NEW modules], "aggregation": "<bounded [0,1] expression over the ids, executed as '
    'written>", "feedback_rule": "<how the weakest '
    'criterion is chosen and how one local repair is reported>"}}. Slot {slot}. {residual}No task drafts, no official '
    "outcomes. Then stop."
)
SYNTH_SCHEMA = (
    ' SCHEMA (strict): each criterion = {"id": <[a-z][a-z0-9_]{1,24}>, "definition": <text>, "range": [0, 1] (JSON '
    'array), "check": one of '
    "llm|tool|llm+tool|deterministic}; every base module id must appear; at most two new modules, each used by the "
    "aggregation."
)
SCORE_TMPL = (
    "You are the verifier (role: scorer). Score the candidate workbook against the task using the bound EVIDENCE below "
    "(recalculated by the runner; "
    "version {evv}, sha256 {evsha}). ./input.xlsx and ./candidate.xlsx (recalculated copies) and `python recalc.py "
    "<file>` are available for extra checks; "
    "anything not in the evidence and not checked is unverified, not correct.\n\nTask:\n{instruction}\n\nEVIDENCE "
    "(version {evv}, sha256 {evsha}):\n{evtext}\n\n"
    'Verifier {vver} (sha256 {vsha}). Criteria:\n{defs}\n\nWrite ./scores.json = {{"scores": {{<id>: 0..1 for EVERY '
    'criterion}}, "evidence_sha256": "{evsha}"}} '
    'or {{"abstain": true, "reason": "...", "evidence_sha256": "{evsha}"}} if you cannot score reliably. Deterministic '
    "criteria (err, chg) are computed by the runner. Nothing else; then stop."
)
LOC_TMPL = (
    "You are the verifier (role: localizer). The scores are FROZEN inputs; do not rescore. Bound EVIDENCE (version "
    "{evv}, sha256 {evsha}) follows; ./input.xlsx and "
    "./candidate.xlsx are recalculated copies.\n\nTask:\n{instruction}\n\nEVIDENCE (version {evv}, sha256 "
    "{evsha}):\n{evtext}\n\nFrozen criterion scores: {scores}\n"
    "Selected weakest criterion (runner-computed): {weakest} -- {weakdef}\n\nGround this weakness in specific cells of "
    "the evidence and give ONE actionable local repair. "
    'Write ./localize.json = {{"witness": "<sheet!cell-grounded weakness, <=60 words>", "repair": "<one concrete '
    'repair instruction, <=40 words>", '
    '"evidence_sha256": "{evsha}"}} or {{"abstain": true, "reason": "...", "evidence_sha256": "{evsha}"}}. Nothing '
    "else; then stop."
)
AGENTS_MD = (
    "# SpreadsheetBench task (single task, fresh session)\nSolve the task in ./task.json on ./input.xlsx (never modify "
    "it); write the completed workbook to ./work/output.xlsx "
    "with python/openpyxl (real formulas where calculations are implied); `python recalc.py <file>` recalculates a "
    "copy. No network. Nothing outside this directory is "
    "part of the task; the only experience available is the EXPERIENCE JSON in the prompt.\n"
)
CAL_AGENTS_MD = (
    "# Calibration workbook task (single task, fresh session)\nSolve ./task.json on ./input.xlsx (never modify it); "
    "write ./work/output.xlsx with python/openpyxl "
    "(real formulas where calculations are implied); `python recalc.py <file>` recalculates a copy. No network. "
    "Nothing outside this directory is part of the task.\n"
)
HPROP_PROMPT = (
    "Your revision was accepted. Write ./work/h_proposal.json with ONE general entry for future unrelated workbooks: "
    '{"trigger": "<general situation, <=12 words>", '
    '"procedure": "<general procedure, <=20 words>", "dimension": '
    '"requirements|formulas|references|preservation|numbers", "evidence": {"cells_changed": [...], '
    '"check_that_confirmed": "<the check you actually ran>"}}. No sheet names, cell addresses, numbers, entities or '
    "file names in trigger/procedure. Then stop."
)
PIN_PROMPT = "Reply with exactly the text PIN-OK and nothing else. Do not read or write any file."

with open(os.path.join(HERE, "modules.md"), encoding="utf-8") as _f:
    MODULES_MD = _f.read()
PROTO_SHA = sha(
    json.dumps(
        {
            "protocol": PROTOCOL_VERSION,
            "evidence": EVIDENCE_VERSION,
            "score": SCORE_TMPL,
            "loc": LOC_TMPL,
            "synth": SYNTH + SYNTH_SCHEMA,
            "modules_md": MODULES_MD,
        }
    )
)


def draft_prompt(t, hjson):
    return (
        "Task %s (%s): read ./task.json (instruction) and work on ./input.xlsx (never modify it). Produce the "
        "completed workbook at ./work/output.xlsx "
        "with python/openpyxl (real formulas where calculations are implied); `python recalc.py <file>` recalculates a "
        "copy. No network.\n"
        "EXPERIENCE (runner-curated general procedures from earlier accepted revisions; JSON, may be empty): "
        "%s\nFinish with a short summary of what you changed." % (t["id"], t["category"], hjson)
    )


def revise_prompt(t, fb, hjson):
    return (
        "Task %s (%s): ./task.json is the instruction, ./input.xlsx the original input, and ./work/output.xlsx a DRAFT "
        "solution. A verifier reviewed the draft and "
        "reports ONE weakness to repair now:\n- criterion: %s\n- witness: %s\n- repair: %s\n\nApply this repair in "
        "place in ./work/output.xlsx, "
        "preserving every other satisfied requirement and all unrelated content. Do not shorten or remove requested "
        "elements.\nEXPERIENCE (JSON, may be empty): %s\n"
        "Reply with a short summary of the exact cells you changed."
        % (t["id"], t["category"], fb["weakest"], fb["witness"], fb["repair"], hjson)
    )


# ------------------------------------------------- small helpers
def h0_slots(K):
    return ["h0%s" % chr(97 + i) for i in range(int(K))]


def _save(p, o):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    atomic_write(p, json.dumps(o, indent=1, default=str))


def _load(p, d=None):
    return read_json(p, d) if os.path.exists(p) else d


def _cfg(cfg, path, default=None):
    cur = cfg
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return default if cur is None else cur


def _tree_sha(root):
    out = []
    for dp, _, fs in os.walk(root):
        for f in sorted(fs):
            p = os.path.join(dp, f)
            out.append((os.path.relpath(p, root), C.sha_file(p)))
    return sha(json.dumps(sorted(out)))


def max_reward(V):
    return C.aggregate(V, {c["id"]: 1.0 for c in V["criteria"]})


def schema_repair(o):
    """Declared schema normalization: a `range` given as the STRING "[0,1]" becomes [0, 1]; "{0,1}" becomes [0, 1] only
    for deterministic criteria."""
    if not isinstance(o, dict) or not isinstance(o.get("criteria"), list):
        return o, []
    fixes = []
    for c in o["criteria"]:
        if isinstance(c, dict) and isinstance(c.get("range"), str) and c["range"].replace(" ", "") == "[0,1]":
            c["range"] = [0, 1]
            fixes.append("%s.range string->array" % c.get("id"))
        elif (
            isinstance(c, dict)
            and isinstance(c.get("range"), str)
            and c["range"].replace(" ", "") == "{0,1}"
            and c.get("check") == "deterministic"
        ):
            c["range"] = [0, 1]
            fixes.append("%s.range '{0,1}'->[0,1]" % c.get("id"))
        for k in ("id", "definition", "check"):
            if isinstance(c, dict) and isinstance(c.get(k), str) and c[k] != c[k].strip():
                c[k] = c[k].strip()
                fixes.append("%s.%s trimmed" % (c.get("id"), k))
    return o, fixes


def validate_strict(o):
    o2, fixes = schema_repair(json.loads(json.dumps(o)))
    V = C.validate_candidate(o2)
    V["schema_repairs"] = fixes
    return V


def load_task(data_dir, cat, tid):
    """Agent-visible task record from the prepared tree (id, category, instruction, input copy): never the reference
    workbook."""
    d = os.path.join(data_dir, "tasks", cat, str(tid))
    tj = _load(os.path.join(d, "task.json"))
    if not tj:
        raise SystemExit(
            "[ucoe-agent] task %s|%s missing under %s; run `python -m data.prepare_ssb2 --config <cfg>` first"
            % (cat, tid, os.path.join(data_dir, "tasks"))
        )
    return {
        "id": str(tid),
        "category": cat,
        "instruction": tj["instruction"],
        "input_path": os.path.join(d, tj.get("input_file", "input.xlsx")),
    }


def build_stream(index, pilot_ids, targets, exclude=None):
    """Frozen family-stratified, interleaved deployment stream: per category the readable tasks outside the development
    pilot and the
    excluded families, families hash-ordered, at most one task per family per round (<= 4 rounds); categories
    interleaved by a seeded shuffle.
    """
    exclude = exclude or {}
    per_cat = {}
    for c in CATS:
        fam, bad = collections.defaultdict(list), set()
        for t in index.get(c, []):
            fam[t["family"]].append(str(t["id"]))
            if not t.get("readable", True):
                bad.add(str(t["id"]))
        pil = {str(x) for x in pilot_ids.get(c, [])}
        order = sorted(fam, key=lambda f: sha("%s|%s|%s" % (STREAM_SEED, c, f)))
        chosen, per_fam, rounds = [], collections.Counter(), 0
        while len(chosen) < targets[c] and rounds < 4:
            for f in order:
                if len(chosen) >= targets[c]:
                    break
                if any(f.startswith(x) for x in exclude.get(c, [])):
                    continue
                taken = {x["id"] for x in chosen}
                cand = [
                    i
                    for i in sorted(fam[f], key=lambda i: sha("%s|%s|%s" % (STREAM_SEED, c, i)))
                    if i not in bad and i not in pil and i not in taken
                ]
                if len(cand) > per_fam[f]:
                    chosen.append({"id": cand[per_fam[f]], "family": f, "category": c})
                    per_fam[f] += 1
            rounds += 1
        if len(chosen) < targets[c]:
            raise SystemExit(
                "%s: only %d readable tasks outside the pilot and excluded families (target %d)"
                % (c, len(chosen), targets[c])
            )
        per_cat[c] = chosen
    stream, ptr = [], {c: 0 for c in CATS}
    slots = [c for c in CATS for _ in range(targets[c])]
    random.Random(int(sha(STREAM_SEED + "|interleave"), 16) % (2**31)).shuffle(slots)
    for c in slots:
        stream.append(per_cat[c][ptr[c]])
        ptr[c] += 1
    return stream


class Harness:
    """One work directory, one backend, three arms. `cfg` is the loaded YAML config (configs/ssb2_codex.yaml)."""

    def __init__(
        self,
        cfg,
        root=None,
        tasks_fn=None,
        calib_fn=None,
        recalc_fn=None,
        transport_factory=None,
        initial_candidates=None,
    ):
        self.cfg = cfg
        self.root = os.path.abspath(root or _cfg(cfg, "output.work_dir", "runs/ssb2_codex"))
        self.data_dir = os.path.abspath(_cfg(cfg, "benchmark.data_dir", "data/raw/ssb2"))
        self.backend_name = _cfg(cfg, "models.backend", "codex")
        if self.backend_name != "codex":
            raise SystemExit("[ucoe-agent] models.backend must be 'codex' (got %r)" % self.backend_name)
        self.roles = {
            r: {"model": str(_cfg(cfg, "models.%s.model" % r)), "effort": str(_cfg(cfg, "models.%s.effort" % r))}
            for r in ("actor", "verifier")
        }
        for r, v in self.roles.items():
            if v["model"] in ("None", "") or v["effort"] not in ("low", "medium", "high", "xhigh"):
                raise SystemExit("[ucoe-agent] models.%s needs a model id and an effort in low|medium|high|xhigh" % r)
        self.limits = {k: int(_cfg(cfg, "limits.%s" % k, v)) for k, v in DEFAULT_LIMITS.items()}
        self.K = int(initial_candidates or _cfg(cfg, "ucoe.initial_candidates", DEFAULT_K))
        self.eps = float(_cfg(cfg, "ucoe.eps", C.EPS))
        self.ckpts = tuple(int(x) for x in _cfg(cfg, "ucoe.checkpoints", [34, 67]))
        self.sel_per = int(_cfg(cfg, "ucoe.contrasts.selection_per_family", 4))
        self.cert_per = int(_cfg(cfg, "ucoe.contrasts.heldout_per_family", 2))
        self.h_cap = int(_cfg(cfg, "ucoe.memory.capacity", C.H_CAP))
        self.h_per_dim = int(_cfg(cfg, "ucoe.memory.per_dimension", C.H_PER_DIM_CAP))
        self.h_chars = int(_cfg(cfg, "ucoe.memory.max_chars", 2400))
        self.gate = {
            "min_family_coverage": float(_cfg(cfg, "ucoe.gates.min_family_coverage", 0.75)),
            "min_worst_J": float(_cfg(cfg, "ucoe.gates.min_worst_J", 0.0)),
        }
        self.calib_chunk = {
            "select": int(_cfg(cfg, "ucoe.calibration.chunk_select", 26)),
            "cert": int(_cfg(cfg, "ucoe.calibration.chunk_heldout", 13)),
        }
        self.smoke_reserve = int(_cfg(cfg, "ucoe.calibration.smoke_reserve", 4))
        self.n_tasks = int(_cfg(cfg, "benchmark.n_tasks", sum(DEFAULT_TARGETS.values())))
        self.targets = {c: int(v) for c, v in (_cfg(cfg, "benchmark.category_targets", DEFAULT_TARGETS)).items()}
        self.exclude = {c: list(v) for c, v in (_cfg(cfg, "benchmark.exclude_families", DEFAULT_EXCLUDE)).items()}
        self.arms = ARMS
        self.tasks_fn, self.calib_fn = tasks_fn, calib_fn
        self.recalc_fn = recalc_fn or EV.make_recalc_fn(cfg, self.root)
        self.recalc_settings = None if recalc_fn else EV.recalc_settings(cfg, self.root)
        self.backend = CodexBackend(self.root, self.roles, self.limits, transport_factory)
        self.task_limit = None  # offline tests only
        self.pin_required = True  # `run` needs a passing pin probe and live smoke

    # ------------------------------------------------- paths, audit, transport
    def p(self, *a):
        return os.path.join(self.root, *a)

    def audit(self, event, **kw):
        audit(self.root, event, **kw)

    def roles_dict(self):
        return {r: dict(self.roles[r]) for r in ("actor", "verifier")}

    def turn(self, arm, role, kind, cwd, text, log, thread_id=None, keep=False, served=None):
        return self.backend.turn(arm, role, kind, cwd, text, log, thread_id, keep, served)

    def evidence(self, input_xlsx, cand_xlsx, instruction, label):
        return E.evidence(self.root, input_xlsx, cand_xlsx, instruction, label, self.recalc_fn)

    def _write_recalc_files(self, d):
        shutil.copyfile(os.path.join(HERE, "recalc_helper.py"), os.path.join(d, "recalc.py"))
        if self.recalc_settings:
            settings = dict(self.recalc_settings)
            if settings.get("profile"):  # relative to the helper's directory: no machine path in the run tree
                settings["profile"] = os.path.relpath(settings["profile"], d)
            _save(os.path.join(d, "recalc.json"), settings)

    # ------------------------------------------------- verifier requests (score, localize, synthesize)
    def _stage_dir(self, arm, tag, evd):
        d = self.p("private", "verify", PROTOCOL_VERSION, arm, tag)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        E.valid_xlsx(evd["input"], evd["input_recalc_sha"])
        E.valid_xlsx(evd["candidate"], evd["candidate_recalc_sha"])
        shutil.copyfile(evd["input"], os.path.join(d, "input.xlsx"))
        shutil.copyfile(evd["candidate"], os.path.join(d, "candidate.xlsx"))
        E.valid_xlsx(os.path.join(d, "input.xlsx"), evd["input_recalc_sha"])
        E.valid_xlsx(os.path.join(d, "candidate.xlsx"), evd["candidate_recalc_sha"])
        self._write_recalc_files(d)
        with open(os.path.join(d, "evidence.txt"), "w", encoding="utf-8") as f:
            f.write("version %s sha256 %s\n%s" % (evd["version"], evd["sha"], evd["text"]))
        return d

    @staticmethod
    def _vread(d, name):
        p = os.path.join(d, name)
        if not os.path.exists(p):
            return None, "missing"
        try:
            with open(p, encoding="utf-8") as f:
                o = json.load(f)
            return (o, None) if isinstance(o, dict) else (None, "not an object")
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return None, "malformed: %s" % str(e)[:80]

    def _verifier_json(self, arm, kind, tag, d, prompt, name, validate):
        """Two attempts; malformed output or a wrong evidence echo on both => HardStop (never an abstention, never
        sealed)."""
        errors = []
        r = None
        for nonce in (0, 1):
            retry = (
                ""
                if not nonce
                else "\n(retry: previous ./%s was %s; write a valid JSON object with the exact evidence_sha256)"
                % (name, errors[-1])
            )
            r = self.turn(
                arm, "verifier", kind, d, prompt + retry, self.p("logs", arm, "%s_%s_%d.json" % (kind, tag, nonce))
            )
            o, err = self._vread(d, name)
            if err:
                errors.append(err)
                continue
            ok, why = validate(o)
            if ok:
                return o, r, nonce + 1
            errors.append(why)
        code = "evidence_mismatch" if any("evidence" in e for e in errors) else "malformed_output"
        self.audit(code, arm=arm, kind=kind, tag=tag, errors=errors)
        raise HardStop(code, "%s %s: %s" % (kind, tag, errors))

    def score(self, arm, tag, V, instruction, evd):
        key = sha(
            "|".join(
                [
                    PROTOCOL_VERSION,
                    V["sha256"],
                    self.roles["verifier"]["model"],
                    self.roles["verifier"]["effort"],
                    PROTO_SHA,
                    evd["sha"],
                ]
            )
        )[:16]
        cp = self.p("private", "verify_cache", PROTOCOL_VERSION, arm, key + ".json")
        prior = _load(cp)
        if prior and prior.get("evidence_sha") == evd["sha"]:
            return {**prior, "resumed": True}
        d = self._stage_dir(arm, tag, evd)
        defs = "\n".join("- %s [%s]: %s" % (c["id"], c["check"], c["definition"]) for c in V["criteria"])
        prompt = SCORE_TMPL.format(
            evv=evd["version"],
            evsha=evd["sha"],
            instruction=instruction,
            evtext=evd["text"],
            vver=V.get("version"),
            vsha=V["sha256"],
            defs=defs,
        )

        def validate(o):
            if o.get("evidence_sha256") != evd["sha"]:
                return False, "evidence echo %s != bound sha" % str(o.get("evidence_sha256"))[:12]
            if o.get("abstain"):
                return True, None
            try:
                sc = {
                    c["id"]: (evd["det"][c["id"]] if c["id"] in C.DET_IDS else float(o["scores"][c["id"]]))
                    for c in V["criteria"]
                }
            except (KeyError, TypeError, ValueError, AttributeError) as e:
                return False, "schema: %s %s" % (type(e).__name__, str(e)[:60])
            return (True, None) if all(0 <= v <= 1 for v in sc.values()) else (False, "score out of range")

        o, r, attempts = self._verifier_json(arm, "score", tag, d, prompt, "scores.json", validate)
        if o.get("abstain"):
            out = {"status": "score_abstain", "reason": str(o.get("reason"))[:300]}
        else:
            sc = {
                c["id"]: (evd["det"][c["id"]] if c["id"] in C.DET_IDS else float(o["scores"][c["id"]]))
                for c in V["criteria"]
            }
            out = {"status": "scored", "scores": sc, "model_weakest_ignored": o.get("weakest")}
        out.update(
            {
                "evidence_sha": evd["sha"],
                "evidence_version": evd["version"],
                "V_sha": V["sha256"],
                "verifier": dict(self.roles["verifier"]),
                "protocol_sha": PROTO_SHA,
                "cache_key": key,
                "attempts": attempts,
                "transport": transport_ref(r),
                "echo_ok": True,
            }
        )
        _save(cp, out)
        return out

    def localize(self, arm, tag, V, instruction, evd, scores, weakest):
        d = self._stage_dir(arm, tag, evd)
        weakdef = next(c["definition"] for c in V["criteria"] if c["id"] == weakest)
        prompt = LOC_TMPL.format(
            evv=evd["version"],
            evsha=evd["sha"],
            instruction=instruction,
            evtext=evd["text"],
            scores=json.dumps(scores, sort_keys=True),
            weakest=weakest,
            weakdef=weakdef,
        )

        def validate(o):
            if o.get("evidence_sha256") != evd["sha"]:
                return False, "evidence echo mismatch"
            if o.get("abstain"):
                return True, None
            return (
                (True, None)
                if (str(o.get("witness", "")).strip() and str(o.get("repair", "")).strip())
                else (False, "schema: witness/repair missing")
            )

        o, r, attempts = self._verifier_json(arm, "localize", tag, d, prompt, "localize.json", validate)
        if o.get("abstain"):
            return {
                "status": "localize_abstain",
                "reason": str(o.get("reason"))[:300],
                "evidence_sha": evd["sha"],
                "attempts": attempts,
                "transport": transport_ref(r),
            }
        return {
            "status": "localized",
            "witness": str(o["witness"])[:600],
            "repair": str(o["repair"])[:400],
            "weakest": weakest,
            "evidence_sha": evd["sha"],
            "attempts": attempts,
            "transport": transport_ref(r),
        }

    def synthesize(self, slot, residual="", hint=""):
        cached = _load(self.p("ucoe", "state", "candidate_%s.json" % slot))
        if cached:
            return cached
        d = self.p("private", "synth", slot)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        shutil.copyfile(os.path.join(HERE, "modules.md"), os.path.join(d, "modules.md"))
        errors = []
        for nonce in (0, 1):
            extra = "" if not nonce else " (retry: previous candidate invalid: %s)" % errors[-1][:300]
            prompt = (
                SYNTH.format(slot=slot, residual=residual)
                + SYNTH_SCHEMA
                + (" Profile hint for this slot: %s." % hint if hint else "")
                + extra
            )
            r = self.turn(
                "ucoe", "verifier", "synth", d, prompt, self.p("logs", "ucoe", "synth_%s_%d.json" % (slot, nonce))
            )
            o, err = self._vread(d, "candidate.json")
            if err:
                errors.append("candidate.json %s" % err)
                continue
            try:
                V = validate_strict(o)
                V["version"], V["slot"], V["synthesized_by"], V["transport"] = (
                    ("S_%s" % slot if not residual else "V_%s" % slot),
                    slot,
                    dict(self.roles["verifier"]),
                    transport_ref(r),
                )
                V["profile_hint"] = hint or None
                _save(self.p("ucoe", "state", "candidate_%s.json" % slot), V)
                return V
            except (ValueError, json.JSONDecodeError) as e:
                errors.append(str(e))
                _save(
                    self.p("ucoe", "state", "candidate_%s_invalid_%d.json" % (slot, nonce)),
                    {"reason": str(e), "raw": o, "transport": transport_ref(r)},
                )
        return None

    # ------------------------------------------------- frozen protocol: deployment tasks + deployment-disjoint calibrat
    def _index(self):
        index = _load(os.path.join(self.data_dir, "tasks", "index.json"))
        if not index:
            raise SystemExit(
                "[ucoe-agent] task index missing: %s (run `python -m data.prepare_ssb2 --config <cfg>`)"
                % os.path.join(self.data_dir, "tasks", "index.json")
            )
        return index

    def _frozen_mismatch(self, man):
        """Protocol-defining config values that differ from the ones frozen in the manifest: {field: (live, frozen)}."""
        live = {
            "initial_candidates": int(self.K),
            "checkpoints": list(self.ckpts),
            "eps": float(self.eps),
            "H": {"cap": self.h_cap, "per_dim": self.h_per_dim, "chars": self.h_chars},
            "banks.select_per_family": self.sel_per,
            "banks.cert_per_family": self.cert_per,
            "banks.gates": self.gate,
            "category_targets": self.targets,
            "roles": self.roles_dict(),
        }
        if not self.tasks_fn:
            live["n"] = self.n_tasks
        frozen = {
            "initial_candidates": int(man.get("initial_candidates", DEFAULT_K)),
            "checkpoints": list(man.get("checkpoints", [])),
            "eps": man.get("eps"),
            "H": man.get("H"),
            "banks.select_per_family": (man.get("banks") or {}).get("select_per_family"),
            "banks.cert_per_family": (man.get("banks") or {}).get("cert_per_family"),
            "banks.gates": (man.get("banks") or {}).get("gates"),
            "category_targets": man.get("category_targets"),
            "roles": man.get("roles"),
            "n": man.get("n"),
        }
        return {k: (v, frozen.get(k)) for k, v in live.items() if v != frozen.get(k)}

    def manifest(self):
        mp = self.p("manifest.json")
        if os.path.exists(mp):
            man = _load(mp)
            bad = self._frozen_mismatch(man)
            if bad:
                raise SystemExit(
                    "[ucoe-agent] the config differs from the protocol frozen in %s (live != frozen): %s; restore the "
                    "frozen values or use a fresh work_dir" % (mp, json.dumps(bad, sort_keys=True, default=str))
                )
            return man
        K = int(self.K)
        assert 1 <= K <= 26, K
        if self.tasks_fn:
            tasks = self.tasks_fn()
            dep = [
                {"i": k, "id": t["id"], "category": t["category"], "input_sha256": C.sha_file(t["input_path"])}
                for k, t in enumerate(tasks)
            ]
            calib = [
                {"task_id": s["task_id"], "category": s["category"], "input_sha256": C.sha_file(s["input_path"])}
                for s in self.calib_fn()
            ]
            src = "injected (offline tests)"
        else:
            index = self._index()
            pilot = (_load(os.path.join(self.data_dir, "pilot.json"), {}) or {}).get("pilot", {})
            pilot_ids = {c: [t["id"] for t in pilot.get(c, [])] for c in CATS}
            stream = build_stream(index, pilot_ids, self.targets, self.exclude)
            dep = []
            for k, t in enumerate(stream[: self.n_tasks]):
                ti = load_task(self.data_dir, t["category"], t["id"])
                dep.append(
                    {
                        "i": k,
                        "id": t["id"],
                        "category": t["category"],
                        "family": t.get("family"),
                        "input_sha256": C.sha_file(ti["input_path"]),
                    }
                )
            stream_ids = {"%s|%s" % (t["category"], t["id"]) for t in stream}
            calib = []
            for c in CATS:
                for t in index.get(c, []):
                    tid = str(t["id"])
                    if (
                        "%s|%s" % (c, tid) in stream_ids
                        or not t.get("readable", True)
                        or tid in {str(x) for x in pilot_ids.get(c, [])}
                        or any(tid.startswith(x) for x in self.exclude.get(c, []))
                    ):
                        continue
                    calib.append(
                        {
                            "task_id": "%s|%s" % (c, tid),
                            "category": c,
                            "input_sha256": C.sha_file(os.path.join(self.data_dir, "tasks", c, tid, "input.xlsx")),
                        }
                    )
            src = os.path.join(self.data_dir, "tasks", "index.json")
        dep_ids = {"%s|%s" % (t["category"], t["id"]) for t in dep}
        calib = sorted(calib, key=lambda s: sha(CALIB_SEED + "|" + s["task_id"]))
        assert all(s["task_id"] not in dep_ids for s in calib), "calibration overlaps deployment"
        # frozen chunks: smoke reserve first, then per stage select/cert chunks (source-disjoint by construction)
        chunks, pos = {}, 0
        chunks["smoke"] = [s["task_id"] for s in calib[pos : pos + self.smoke_reserve]]
        pos += self.smoke_reserve
        for stage in (0,) + self.ckpts:
            for kind in ("select", "cert"):
                n = self.calib_chunk[kind]
                chunks["%d_%s" % (stage, kind)] = [s["task_id"] for s in calib[pos : pos + n]]
                pos += n
        allc = [x for v in chunks.values() for x in v]
        assert len(allc) == len(set(allc)) and len(allc) <= len(calib), (len(allc), len(calib))
        man = {
            "label": "UCoE agent harness (%s, %d initial candidates): deployment-disjoint model-native calibration, "
            "evidence %s" % (self.backend_name, K, EVIDENCE_VERSION),
            "protocol_version": PROTOCOL_VERSION,
            "evidence_version": EVIDENCE_VERSION,
            "protocol_sha256": PROTO_SHA,
            "backend": self.backend_name,
            "roles": self.roles_dict(),
            "n": len(dep),
            "deployment_tasks": dep,
            "deployment_tasks_sha256": sha(json.dumps(["%s|%s" % (t["category"], t["id"]) for t in dep])),
            "deployment_inputs_sha256": sha(json.dumps([t["input_sha256"] for t in dep])),
            "source_manifest": src,
            "checkpoints": list(self.ckpts),
            "arms": list(self.arms),
            "category_targets": self.targets,
            "calibration": {
                "n": len(calib),
                "sources": calib,
                "sha256": sha(json.dumps([s["task_id"] for s in calib])),
                "inputs_sha256": sha(json.dumps([s["input_sha256"] for s in calib])),
                "chunks": chunks,
                "rule": "public dataset tasks outside the frozen stream, excluding unreadable inputs, excluded "
                "families and the development pilot ids; hash-ordered; "
                "fixed chunks per stage and kind; smoke reserve never enters a bank",
                "overlap_audit": {"deployment_x_calibration": 0, "chunks_pairwise_disjoint": True},
            },
            "banks": {
                "select_per_family": self.sel_per,
                "cert_per_family": self.cert_per,
                "families": list(C.FAMS),
                "gates": self.gate,
                "source": "model-native drafts by the actor on calibration sources; pos = draft, neg = one declared "
                "realistic perturbation",
            },
            "initial_candidates": K,
            "profile_hints": {sl: PROFILE_HINTS.get(sl) for sl in h0_slots(K)},
            "eps": self.eps,
            "H": {"cap": self.h_cap, "per_dim": self.h_per_dim, "chars": self.h_chars},
            "V_fixed_sha256": V_FIXED["sha256"],
            "statuses": list(STATUSES),
            "hard_stops": list(HARD),
            "forbidden_inputs": list(FORBIDDEN_KEYS) + ["official.json", "final-evaluator feedback"],
            "limits": self.limits,
            "frozen_at": time.time(),
        }
        _save(mp, man)
        print(
            "[ucoe-agent manifest] %d deployment tasks (sha %s); calibration %d sources (sha %s), chunks {%s}; roles %s"
            % (
                man["n"],
                man["deployment_tasks_sha256"][:16],
                len(calib),
                man["calibration"]["sha256"][:16],
                ", ".join("%s:%d" % (k, len(v)) for k, v in chunks.items()),
                man["roles"],
            ),
            flush=True,
        )
        return man

    def task_inputs(self, t):
        if "input_path" in t and "instruction" in t:
            return t
        if self.tasks_fn:
            return {**t, **next(x for x in self.tasks_fn() if x["id"] == t["id"] and x["category"] == t["category"])}
        return {**t, **load_task(self.data_dir, t["category"], t["id"])}

    def calib_source(self, task_id):
        if self.calib_fn:
            return next(s for s in self.calib_fn() if s["task_id"] == task_id)
        c, tid = task_id.split("|")
        t = load_task(self.data_dir, c, tid)
        return {"task_id": task_id, "category": c, "instruction": t["instruction"], "input_path": t["input_path"]}

    # ------------------------------------------------- model-native calibration drafts
    def calib_draft(self, task_id):
        """One fresh actor session per calibration source; the draft is archived read-only under private/calib_drafts
        (never a deployment task)."""
        ad = self.p("private", "calib_drafts", task_id.replace("|", "_"))
        rec = _load(os.path.join(ad, "record.json"))
        if rec and rec.get("status") == "draft_ok":
            E.valid_xlsx(os.path.join(ad, "draft.xlsx"), rec["draft_sha256"])
            return rec
        if rec and rec.get("status") == "draft_missing":
            return rec
        s = self.calib_source(task_id)
        d = self.p("calib_work", task_id.replace("|", "_"))
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(os.path.join(d, "work"))
        shutil.copyfile(s["input_path"], os.path.join(d, "input.xlsx"))
        _save(os.path.join(d, "task.json"), {"id": task_id, "category": s["category"], "instruction": s["instruction"]})
        with open(os.path.join(d, "AGENTS.md"), "w", encoding="utf-8") as f:
            f.write(CAL_AGENTS_MD)
        self._write_recalc_files(d)
        r = self.turn(
            "calib",
            "actor",
            "calib_draft",
            d,
            draft_prompt({"id": task_id, "category": s["category"]}, "[]"),
            self.p("logs", "calib", task_id.replace("|", "_") + ".json"),
        )
        out = os.path.join(d, "work", "output.xlsx")
        os.makedirs(ad, exist_ok=True)
        if os.path.exists(out):
            try:
                E.valid_xlsx(out)
            except HardStop:
                rec = {
                    "task_id": task_id,
                    "status": "draft_missing",
                    "why": "invalid xlsx",
                    "transport": transport_ref(r),
                }
            else:
                shutil.copyfile(out, os.path.join(ad, "draft.xlsx"))
                os.chmod(os.path.join(ad, "draft.xlsx"), 0o444)
                rec = {
                    "task_id": task_id,
                    "status": "draft_ok",
                    "draft_sha256": C.sha_file(os.path.join(ad, "draft.xlsx")),
                    "input_sha256": C.sha_file(s["input_path"]),
                    "transport": transport_ref(r),
                }
        else:
            rec = {"task_id": task_id, "status": "draft_missing", "transport": transport_ref(r)}
        _save(os.path.join(ad, "record.json"), rec)
        shutil.rmtree(d, ignore_errors=True)
        return rec

    # ------------------------------------------------- banks per stage
    def build_bank(self, stage, kind):
        """Bank (stage, kind) from the frozen chunk: drafts generated lazily source by source until every family has its
        quota; a source enters at most
        one pair anywhere (registry); every pair's recalculated evidence is computed and validated; shortfall is
        recorded, never backfilled.
        """
        p = self.p("ucoe", "banks", "%d_%s" % (stage, kind), "manifest.json")
        if os.path.exists(p):
            return _load(p)
        man = self.manifest()
        chunk = man["calibration"]["chunks"].get("%d_%s" % (stage, kind))
        if chunk is None:
            raise SystemExit(
                "[ucoe-agent] no calibration chunk %d_%s in the frozen manifest (checkpoints %s); the checkpoint "
                "schedule cannot change after `manifest`" % (stage, kind, man.get("checkpoints"))
            )
        reg = _load(self.p("ucoe", "banks", "registry.json"), {"used": []})
        used = set(reg["used"])
        per = self.sel_per if kind == "select" else self.cert_per
        got = {f: 0 for f in C.FAMS}
        items = []
        for tid in chunk:
            if all(got[f] >= per for f in C.FAMS):
                break
            if tid in used:
                continue
            rec = self.calib_draft(tid)
            if rec.get("status") != "draft_ok":
                continue
            src = self.calib_source(tid)
            dp = self.p("private", "calib_drafts", tid.replace("|", "_"), "draft.xlsx")
            for fam in C.FAMS:
                if got[fam] >= per:
                    continue
                cid = "%d-%s-%s-%d" % (stage, kind, fam, got[fam])
                d = self.p("ucoe", "banks", "%d_%s" % (stage, kind), cid)
                pm = X.make_pair(src, dp, fam, d, audit=self.audit)
                if pm is None:
                    shutil.rmtree(d, ignore_errors=True)
                    continue
                pm.update(
                    {
                        "cluster_id": cid,
                        "family": fam,
                        "stage": stage,
                        "kind": kind,
                        "draft_sha256": rec["draft_sha256"],
                    }
                )
                items.append(pm)
                used.add(tid)
                got[fam] += 1
                break
        for it in items:
            evp = self.evidence(
                os.path.join(it["dir"], "input.xlsx"),
                os.path.join(it["dir"], "pos.xlsx"),
                it["instruction"],
                "bank_%d_%s_%s_pos" % (stage, kind, it["cluster_id"]),
            )
            evn = self.evidence(
                os.path.join(it["dir"], "input.xlsx"),
                os.path.join(it["dir"], "neg.xlsx"),
                it["instruction"],
                "bank_%d_%s_%s_neg" % (stage, kind, it["cluster_id"]),
            )
            it["evidence_sha"] = {"pos": evp["sha"], "neg": evn["sha"]}
        reg["used"] = sorted(used)
        _save(self.p("ucoe", "banks", "registry.json"), reg)
        _save(p, items)
        _save(
            self.p("ucoe", "banks", "%d_%s" % (stage, kind), "counts.json"),
            {
                "quota_per_family": per,
                "got": got,
                "shortfall": {f: max(0, per - got[f]) for f in C.FAMS},
                "chunk_size": len(chunk),
                "sources_used": sorted({it["source"] for it in items}),
            },
        )
        print(
            "[ucoe-agent banks] %d/%s: %d pairs %s (quota %d/family) from chunk of %d; registry %d"
            % (stage, kind, len(items), got, per, len(chunk), len(used)),
            flush=True,
        )
        return items

    def score_bank(self, V, bank, tag):
        recs = []
        for it in bank:
            out = {}
            for side in ("pos", "neg"):
                evd = self.evidence(
                    os.path.join(it["dir"], "input.xlsx"),
                    os.path.join(it["dir"], "%s.xlsx" % side),
                    it["instruction"],
                    "bank_%s_%s_%s" % (tag, it["cluster_id"], side),
                )
                r = self.score(
                    "ucoe", "%s_%s_%s_%s" % (tag, V["version"], it["cluster_id"], side), V, it["instruction"], evd
                )
                out[side] = r.get("scores") if r["status"] == "scored" else None
                out[side + "_status"] = r["status"]
            rec = C.pair_record(V, it["cluster_id"], it["family"], out["pos"], out["neg"], eps=self.eps)
            rec.update({"pos_status": out["pos_status"], "neg_status": out["neg_status"], "source": it["source"]})
            recs.append(rec)
        _save(self.p("ucoe", "records", "%s_%s.json" % (tag, V["version"])), recs)
        return recs

    def bank_gate(self, cands, records, pick=None):
        """Hard failures = a family without pairs, coverage below threshold, winner worst-family J not positive.
        Saturation (a candidate sweeps a family or every pair) and identical win vectors across candidates are
        WARNINGS (recorded, never fatal).
        """
        rep, fails, warns, vecs, worst = {}, [], [], {}, {}
        for v in cands:
            recs = records[v["version"]]
            per = {f: [r for r in recs if r["family"] == f] for f in C.FAMS}
            cov = {
                f: (sum(1 for r in per[f] if r["status"] == "scored") / len(per[f]) if per[f] else 0.0) for f in C.FAMS
            }
            wins = {f: sum(int(r["correct"]) for r in per[f]) for f in C.FAMS}
            J, _ = C.family_J(recs)
            vecs[v["version"]] = [int(r["correct"]) for r in sorted(recs, key=lambda r: r["cluster_id"])]
            worst[v["version"]] = min(J.get(f, -1) for f in C.FAMS)
            rep[v["version"]] = {
                "coverage": cov,
                "wins": wins,
                "n": {f: len(per[f]) for f in C.FAMS},
                "J": J,
                "worst_J": worst[v["version"]],
                "win_vector": vecs[v["version"]],
            }
            for f in C.FAMS:
                if not per[f]:
                    fails.append("%s: family %s has no pairs (bank_uninformative)" % (v["version"], f))
                elif cov[f] < self.gate["min_family_coverage"]:
                    fails.append(
                        "%s: %s coverage %.2f < %s (bank_uninformative)"
                        % (v["version"], f, cov[f], self.gate["min_family_coverage"])
                    )
                if per[f] and wins[f] == len(per[f]):
                    warns.append("%s: %s has no non-win (bank_saturated)" % (v["version"], f))
            if recs and sum(wins.values()) == len(recs):
                warns.append("%s: wins every pair (bank_saturated)" % v["version"])
        if (
            len(cands) > 1
            and len({tuple(x) for x in vecs.values()}) == 1
            and len({round(x, 6) for x in worst.values()}) == 1
        ):
            warns.append("candidates identical in pair-win vector and worst-family J (bank_uninformative)")
        if pick and pick.get("winner") and worst.get(pick["winner"], -1) <= self.gate["min_worst_J"]:
            fails.append(
                "winner %s worst-family J %.4f is not positive (no driving capacity)"
                % (pick["winner"], worst[pick["winner"]])
            )
        return {"pass": not fails, "failures": fails, "warnings": warns, "per_candidate": rep}

    def _h0_stop(self, st, rep, flag, msg):
        rep["decision"] = msg
        st.update({"h0_done": True, flag: True, "h0": rep})
        _save(self.p("state", "ucoe.json"), st)
        _save(self.p("ucoe", "state", "h0_report.json"), rep)

    def ucoe_h0(self, st):
        """Initial selection: K candidates (slots h0a, h0b, ... with fixed profile hints), stage-0 selection bank,
        maximin pick under the bank gate,
        held-out diagnostic of the winner; the winner is deployed (Algorithm 1, t = 0)."""
        if st.get("h0_done"):
            return
        K = int(self.manifest().get("initial_candidates", DEFAULT_K))
        cands = [v for v in (self.synthesize(sl, hint=PROFILE_HINTS.get(sl, "")) for sl in h0_slots(K)) if v]
        rep = {
            "candidates": [v["version"] for v in cands],
            "initial_candidates": K,
            "profile_hints": {sl: PROFILE_HINTS.get(sl) for sl in h0_slots(K)},
            "schema_repairs": {v["version"]: v.get("schema_repairs") for v in cands},
        }
        if not cands:
            self._h0_stop(st, rep, "initialization_failed", "initialization_failed: zero valid candidates")
            raise SystemExit("[ucoe-agent ucoe] INITIALIZATION FAILED: no valid verifier candidate")
        sel = self.build_bank(0, "select")
        counts = _load(self.p("ucoe", "banks", "0_select", "counts.json"))
        if any(v for v in counts["shortfall"].values()):
            self._h0_stop(
                st,
                rep,
                "bank_gate_failed",
                "stopped: selection bank shortfall %s (unavailable deployment-disjoint calibration data)"
                % counts["shortfall"],
            )
            raise SystemExit("[ucoe-agent ucoe] BANK SHORTFALL before task 0: %s" % counts)
        records = {v["version"]: self.score_bank(v, sel, "sel0") for v in cands}
        pick = C.select({v["version"]: (v, records[v["version"]]) for v in cands})
        gate = self.bank_gate(cands, records, pick)
        rep.update(
            {"bank_gate": gate, "selection": pick, "overlap_audit": self.manifest()["calibration"]["overlap_audit"]}
        )
        if not gate["pass"] or not pick.get("winner"):
            self._h0_stop(
                st,
                rep,
                "bank_gate_failed",
                "bank_gate_failed: " + "; ".join(gate["failures"] or [str(pick.get("reason"))]),
            )
            raise SystemExit("[ucoe-agent ucoe] BANK GATE FAILED before task 0: %s" % rep["decision"])
        w = next(v for v in cands if v["version"] == pick["winner"])
        cert = self.build_bank(0, "cert")
        rep["heldout_diagnostic"] = C.heldout_diagnostic(self.score_bank(w, cert, "cert0"))
        rep["cert_counts"] = _load(self.p("ucoe", "banks", "0_cert", "counts.json"))
        st["V"] = w
        st["lineage_V"] = ["null", w["version"]]
        rep["decision"] = "DEPLOY %s (maximin winner; gates passed; worst J %.4f)" % (
            w["version"],
            gate["per_candidate"][w["version"]]["worst_J"],
        ) + ("; warnings: %s" % "; ".join(gate["warnings"]) if gate.get("warnings") else "")
        st["h0_done"], st["h0"] = True, rep
        _save(self.p("ucoe", "state", "h0_report.json"), rep)
        _save(self.p("state", "ucoe.json"), st)
        print(
            "[ucoe-agent ucoe] H0: %s; gate %s"
            % (
                rep["decision"],
                json.dumps({k: {"wins": v["wins"], "worst_J": v["worst_J"]} for k, v in gate["per_candidate"].items()}),
            ),
            flush=True,
        )

    def ucoe_checkpoint(self, st, k, rows):
        """Checkpoint k (Algorithm 1, t in T): one challenger synthesized from the incumbent and an unlabeled summary of
        the prefix; adopt iff strict
        worst-family gain on the fresh selection bank, more held-out wins than losses, no family regression beyond
        the tolerance (empirical relative rule).
        """
        inc = st["V"]
        rep = {"stage": k, "incumbent": inc["version"]}
        recent = rows[-128:]
        hist = collections.Counter(r.get("weakest") for r in recent if r.get("weakest"))
        residual = (
            "Aggregate prefix diagnostics: repair win rate %d/%d; weakest-criterion counts %s; reward-ceiling skips "
            "%d; memory rules %d.\n"
            % (
                sum(1 for r in recent if r.get("status") == "accepted"),
                len(recent),
                json.dumps(dict(sorted(hist.items()))),
                sum(1 for r in recent if r.get("status") == "reward_ceiling_skip"),
                len(self.h_load("ucoe")),
            )
            + "Incumbent verifier (keep every criterion id; adjust definitions/aggregation/feedback or add <=2 "
            "modules): "
            + json.dumps({x: inc[x] for x in ("criteria", "aggregation", "feedback_rule")})
            + "\nPropose ONE challenger. "
        )
        ch = self.synthesize("ck%d" % k, residual=residual)
        if ch is None:
            rep["decision"] = "RETAIN (no valid challenger)"
            return rep
        ch["parent_sha256"] = inc["sha256"]
        sel = self.build_bank(k, "select")
        counts = _load(self.p("ucoe", "banks", "%d_select" % k, "counts.json"))
        if any(v for v in counts["shortfall"].values()):
            rep["decision"] = "RETAIN (selection bank shortfall %s)" % counts["shortfall"]
            return rep
        Ji, _ = C.family_J(self.score_bank(inc, sel, "sel%d" % k))
        Jc, _ = C.family_J(self.score_bank(ch, sel, "sel%d" % k))
        rep["selection_J"] = {"incumbent": Ji, "challenger": Jc}
        if any(f not in Jc for f in C.FAMS) or min(Jc[f] for f in C.FAMS) <= min(Ji.get(f, -1) for f in C.FAMS):
            rep["decision"] = "RETAIN (no strict selection gain)"
            return rep
        ho = self.build_bank(k, "cert")
        dec = C.evolve_decision(Ji, Jc, self.score_bank(inc, ho, "cert%d" % k), self.score_bank(ch, ho, "cert%d" % k))
        rep["evolve"] = dec
        if dec["promote"]:
            st["V"] = ch
            st["lineage_V"].append(ch["version"])
            rep["decision"] = "PROMOTE %s (empirical relative rule; not statistically certified)" % ch["version"]
        else:
            rep["decision"] = "RETAIN (%s)" % (dec.get("reason") or "tests failed")
        return rep

    # ------------------------------------------------- bounded experience block H
    def h_load(self, arm):
        return _load(self.p("private", "H", "%s.json" % arm), [])

    def h_serialize(self, H):
        ents = [
            {"dimension": e["dimension"], "trigger": e["trigger"], "procedure": e["procedure"]}
            for e in sorted(H, key=lambda e: (e["created_at"], e["id"]))
        ]
        while ents and len(json.dumps(ents, sort_keys=True, separators=(",", ":"))) > self.h_chars:
            ents = ents[1:]
        return json.dumps(ents, sort_keys=True, separators=(",", ":"))

    def h_commit(self, arm, prop, task_id, v_sha, i):
        H = self.h_load(arm)
        H2, status = C.h_update(H, prop, True, task_id, v_sha, i, cap=self.h_cap, per_dim_cap=self.h_per_dim)
        assert len(H2) <= self.h_cap and all(
            sum(1 for e in H2 if e["dimension"] == d) <= self.h_per_dim for d in C.DIMS
        )
        _save(self.p("private", "H", "%s.json" % arm), H2)
        with open(self.p("private", "H", "%s_changelog.jsonl" % arm), "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "task": task_id,
                        "status": status,
                        "size": len(H2),
                        "chars": len(self.h_serialize(H2)),
                        "t": time.time(),
                    }
                )
                + "\n"
            )
        return status, C.sha(H2)

    # ------------------------------------------------- task workspaces and sealing
    def _task_root(self, arm, i, t):
        return self.p("tasks", arm, "t%03d_%s_%s" % (i, t["category"], t["id"]))

    def prepare_task(self, arm, i, t):
        d = os.path.join(self._task_root(arm, i, t), "draft")
        if not os.path.exists(os.path.join(d, "task.json")):
            ti = self.task_inputs(t)
            os.makedirs(os.path.join(d, "work"), exist_ok=True)
            shutil.copyfile(ti["input_path"], os.path.join(d, "input.xlsx"))
            _save(
                os.path.join(d, "task.json"),
                {"id": t["id"], "category": t["category"], "instruction": ti["instruction"]},
            )
            with open(os.path.join(d, "AGENTS.md"), "w", encoding="utf-8") as f:
                f.write(AGENTS_MD)
            self._write_recalc_files(d)
            for f in ("input.xlsx", "task.json"):
                os.chmod(
                    os.path.join(d, f), 0o444
                )  # prepared inputs are read-only; rehashed before scoring and sealing
        return d

    @staticmethod
    def prepared_hashes(d):
        return {"input": C.sha_file(os.path.join(d, "input.xlsx")), "task": C.sha_file(os.path.join(d, "task.json"))}

    def check_prepared(self, d, rec, where):
        now = self.prepared_hashes(d)
        if now != rec["prepared_sha"]:
            raise HardStop(
                "provenance_mismatch", "%s: prepared inputs changed %s -> %s" % (where, rec["prepared_sha"], now)
            )

    def _lock_task(self, arm, i, t):
        r = self._task_root(arm, i, t)
        if os.path.isdir(r):
            os.chmod(r, 0)

    def seal(self, arm, i, t, served_path, rec):
        ad = self.p("private", "archive", arm, "t%03d_%s_%s" % (i, t["category"], t["id"]))
        os.makedirs(ad, exist_ok=True)
        dst = os.path.join(ad, "served.xlsx")
        if os.path.exists(dst):
            os.chmod(dst, stat.S_IWUSR | stat.S_IRUSR)
            os.remove(dst)
        if served_path and os.path.exists(served_path):
            E.valid_xlsx(served_path)
            shutil.copyfile(served_path, dst)
            os.chmod(dst, stat.S_IRUSR | stat.S_IRGRP)
            rec["served_sha"] = C.sha_file(dst)
        else:
            rec["served_sha"] = None
        rec["sealed_at"] = time.time()
        _save(os.path.join(ad, "record.json"), rec)
        os.chmod(os.path.join(ad, "record.json"), stat.S_IRUSR | stat.S_IRGRP)
        return rec

    def _pending(self, arm):
        return self.p("state", "%s_pending.json" % arm)

    @staticmethod
    def _row(rec):
        keys = (
            "i",
            "task_id",
            "category",
            "status",
            "served",
            "served_sha",
            "sealed_at",
            "decision",
            "weakest",
            "h_update",
            "H_sha_before",
            "H_sha_after",
            "draft_tree_sha",
            "draft_tree_sha_after",
            "actor_threads",
            "verifier_calls",
            "evidence_sha",
            "roles",
            "r0",
            "max_reward",
            "skipped_stages",
        )
        return {k: rec.get(k) for k in keys}

    # ------------------------------------------------- one task (Algorithm 1, lines 5-18)
    def run_task(self, arm, i, t, st, V):
        pend = _load(self._pending(arm))
        rec = pend["rec"] if pend and pend.get("i") == i else None
        if rec and rec.get("stage") == "sealed":
            return rec
        arec = _load(
            os.path.join(self.p("private", "archive", arm, "t%03d_%s_%s" % (i, t["category"], t["id"])), "record.json")
        )
        if arec and arec.get("sealed_at"):
            return arec
        d = self.prepare_task(arm, i, t)
        tj = _load(os.path.join(d, "task.json"))
        hjson = self.h_serialize(self.h_load(arm))
        if rec is None:
            rec = {
                "i": i,
                "task_id": "%s|%s" % (t["category"], t["id"]),
                "category": t["category"],
                "arm": arm,
                "backend": self.backend_name,
                "V_version": V.get("version") if V else None,
                "H_sha_before": C.sha(self.h_load(arm)),
                "H_chars": len(hjson),
                "roles": self.roles_dict(),
                "actor_threads": [],
                "verifier_calls": 0,
                "evidence_sha": {},
                "prepared_sha": self.prepared_hashes(d),
                "stage": "start",
            }
        out = os.path.join(d, "work", "output.xlsx")

        def save(stage):
            rec["stage"] = stage
            _save(self._pending(arm), {"i": i, "rec": rec})

        if rec["stage"] == "start":
            if os.path.exists(out):
                os.remove(out)
            dr = self.turn(
                arm, "actor", "draft", d, draft_prompt(t, hjson), self.p("logs", arm, "task%03d_draft.json" % i)
            )
            rec["draft_turn"] = {
                **{
                    k: dr.get(k)
                    for k in ("turn_id", "status", "seconds", "token_usage", "served", "requested", "cost_usd")
                },
                "transport": transport_ref(dr),
            }
            rec["actor_threads"].append(dr["thread_id"])
            self.check_prepared(d, rec, "after draft")
            if not os.path.exists(out):
                rec.update({"status": "draft_missing", "served": None})
                save("drafted_missing")
            else:
                E.valid_xlsx(out)
                rec["draft_tree_sha"] = _tree_sha(d)
                save("drafted")
        if rec["stage"] == "drafted_missing":
            return self._finish(arm, i, t, rec, None, save)
        if V is None:
            rec.update({"status": "native", "served": "candidate_0"})
            save("native")
            return self._finish(arm, i, t, rec, out, save)
        if rec["stage"] == "drafted":
            self.check_prepared(d, rec, "before c0")
            ev0 = self.evidence(os.path.join(d, "input.xlsx"), out, tj["instruction"], "%s_t%03d_c0" % (arm, i))
            rec["evidence_sha"]["c0"] = {
                "sha": ev0["sha"],
                "version": ev0["version"],
                "errors": {k: ev0["ev"]["errors"][k] for k in ("global", "inside", "outside")},
                "det": ev0["det"],
            }
            s0 = self.score(arm, "task%03d_c0" % i, V, tj["instruction"], ev0)
            rec["verifier_calls"] += s0.get("attempts", 1)
            rec["score_c0"] = s0
            if s0["status"] != "scored":
                rec["status"] = s0["status"]
                save("c0_done")
                return self._finish(arm, i, t, rec, out, save)
            r0, mr = C.aggregate(V, s0["scores"]), max_reward(V)
            rec.update({"r0": r0, "max_reward": mr})
            if r0 + self.eps >= mr:  # reward ceiling: nothing to repair under this verifier
                rec.update(
                    {
                        "status": "reward_ceiling_skip",
                        "reason": "r0 %.4f + eps %s >= max reward %.4f" % (r0, self.eps, mr),
                        "skipped_stages": ["weakest", "localize", "revise", "c1", "h_update"],
                    }
                )
                save("c0_done")
                return self._finish(arm, i, t, rec, out, save)
            rec["weakest"] = C.weakest(V, s0["scores"])
            loc = self.localize(arm, "task%03d_loc" % i, V, tj["instruction"], ev0, s0["scores"], rec["weakest"])
            rec["verifier_calls"] += loc.get("attempts", 1)
            rec["localize"] = loc
            if loc["status"] != "localized":
                rec["status"] = loc["status"]
                save("c0_done")
                return self._finish(arm, i, t, rec, out, save)
            save("localized")
        rd = os.path.join(self._task_root(arm, i, t), "rev")
        if rec["stage"] == "localized":
            shutil.rmtree(rd, ignore_errors=True)
            shutil.copytree(d, rd)
            assert _tree_sha(d) == rec["draft_tree_sha"]
            rr = self.turn(
                arm,
                "actor",
                "revise",
                rd,
                revise_prompt(t, rec["localize"], hjson),
                self.p("logs", arm, "task%03d_revise.json" % i),
                keep=True,
            )
            rec["revise_turn"] = {
                **{
                    k: rr.get(k)
                    for k in ("turn_id", "status", "seconds", "token_usage", "served", "requested", "cost_usd")
                },
                "transport": transport_ref(rr),
            }
            rec["actor_threads"].append(rr["thread_id"])
            rec["rev_thread"] = rr["thread_id"]
            self.check_prepared(d, rec, "after revise")
            rout = os.path.join(rd, "work", "output.xlsx")
            if not os.path.exists(rout) or C.sha_file(rout) == C.sha_file(out):
                rec.update({"status": "proposal_noop"})
                save("revised_noop")
                return self._finish(arm, i, t, rec, out, save)
            E.valid_xlsx(rout)
            save("revised")
        if rec["stage"] == "revised":
            rout = os.path.join(rd, "work", "output.xlsx")
            ev1 = self.evidence(os.path.join(rd, "input.xlsx"), rout, tj["instruction"], "%s_t%03d_c1" % (arm, i))
            rec["evidence_sha"]["c1"] = {
                "sha": ev1["sha"],
                "version": ev1["version"],
                "errors": {k: ev1["ev"]["errors"][k] for k in ("global", "inside", "outside")},
                "det": ev1["det"],
            }
            if ev1["sha"] == rec["evidence_sha"]["c0"]["sha"]:
                rec.update({"status": "proposal_noop", "noop_kind": "evidence_identical"})
                save("revised_noop")
                return self._finish(arm, i, t, rec, out, save)
            s1 = self.score(arm, "task%03d_c1" % i, V, tj["instruction"], ev1)
            rec["verifier_calls"] += s1.get("attempts", 1)
            rec["score_c1"] = s1
            if s1["status"] == "score_abstain":
                rec["status"] = "c1_abstain"
            else:
                dec = C.accept(V, rec["score_c0"]["scores"], s1["scores"], eps=self.eps)
                rec["decision"] = dec
                rec["status"] = "accepted" if dec["accepted"] else "no_strict_gain"
            save("c1_done")
        if rec["stage"] == "c1_done":
            if rec["status"] == "accepted":
                hp = os.path.join(rd, "work", "h_proposal.json")
                hr = self.turn(
                    arm,
                    "actor",
                    "hproposal",
                    rd,
                    HPROP_PROMPT,
                    self.p("logs", arm, "task%03d_hproposal.json" % i),
                    thread_id=rec.get("rev_thread"),
                    served=rec["revise_turn"].get("served"),
                )
                rec["hproposal_turn"] = {
                    **{k: hr.get(k) for k in ("turn_id", "status", "seconds", "served", "cost_usd")},
                    "transport": transport_ref(hr),
                }
                prop = _load(hp)
                rec["h_update"], rec["H_sha_after"] = self.h_commit(arm, prop, rec["task_id"], V["sha256"], i)
                rec["served"] = "candidate_1"
                return self._finish(arm, i, t, rec, os.path.join(rd, "work", "output.xlsx"), save)
            return self._finish(arm, i, t, rec, out, save)
        raise RuntimeError("unknown stage %s" % rec.get("stage"))

    def _finish(self, arm, i, t, rec, served_path, save):
        d = os.path.join(self._task_root(arm, i, t), "draft")
        if rec.get("status") not in ("accepted",):
            rec["served"] = "candidate_0" if served_path else None
            if rec.get("draft_tree_sha"):
                rec["draft_tree_sha_after"] = _tree_sha(d)
                assert (
                    rec["draft_tree_sha_after"] == rec["draft_tree_sha"]
                ), "draft tree changed by a rejected or skipped revision"
            rec.setdefault("h_update", "no_update")
            rec["H_sha_after"] = C.sha(self.h_load(arm))
            assert rec["H_sha_after"] == rec["H_sha_before"], "H changed on a rejected or skipped task"
        if rec.get("prepared_sha"):
            self.check_prepared(d, rec, "before seal")
        rec["stage"] = "sealed"
        rec = self.seal(arm, i, t, served_path, rec)
        save("sealed")
        rd = os.path.join(self._task_root(arm, i, t), "rev")
        if rec.get("rev_thread") and "actor" in self.backend.inst:
            try:
                self.backend.server("actor").thread_delete(rec["rev_thread"])
            except Exception:  # noqa: BLE001
                pass
        if os.path.isdir(rd):
            shutil.rmtree(rd, ignore_errors=True)
        self._lock_task(arm, i, t)
        return rec

    # ------------------------------------------------- the stream (Algorithm 1)
    def _preflight(self):
        pin = _load(self.p("private", "pin_probe.json"))
        if not pin or not pin.get("ok"):
            raise SystemExit("[ucoe-agent] pin-probe missing or failed: run `pin-probe` first")
        smk = _load(self.p("private", "smoke_report.json"))
        if not smk or not smk.get("ok"):
            raise SystemExit("[ucoe-agent] live smoke missing or failed: run `smoke` first")
        want = {
            "backend": self.backend_name,
            "protocol_version": PROTOCOL_VERSION,
            "evidence_version": EVIDENCE_VERSION,
            "protocol_sha256": PROTO_SHA,
            "roles": self.roles_dict(),
        }
        stale = {k: (smk.get(k), v) for k, v in want.items() if smk.get(k) != v}
        if stale or not all(g in smk.get("checks", {}) and smk["checks"][g] is True for g in smk.get("gates", [])):
            raise SystemExit(
                "[ucoe-agent] smoke report is stale or incomplete (mismatch %s); rerun `smoke` under the current "
                "protocol before `run`" % (stale or "gate set")
            )

    def run(self, arm):
        man = self.manifest()
        if arm not in self.arms:
            raise SystemExit("unknown arm %s" % arm)
        if self.pin_required:
            self._preflight()
        sp = self.p("state", "%s.json" % arm)
        st = _load(
            sp, {"arm": arm, "rows": [], "V": None, "lineage_V": ["null"], "roles": self.roles_dict(), "ckpt": {}}
        )
        if st["roles"] != self.roles_dict():
            raise SystemExit(
                "[ucoe-agent] %s state was created under roles %s; refusing to continue with %s"
                % (arm, st["roles"], self.roles)
            )
        if st.get("initialization_failed") or st.get("bank_gate_failed"):
            raise SystemExit("[ucoe-agent %s] initialization/bank gate failed earlier; not consuming tasks" % arm)
        if arm == "ucoe":
            self.ucoe_h0(st)
            _save(sp, st)
        tasks = man["deployment_tasks"][: self.task_limit] if self.task_limit else man["deployment_tasks"]
        for i, t in enumerate(tasks):
            if i < len(st["rows"]):
                continue
            V = st["V"] if arm == "ucoe" else (V_FIXED if arm == "judge" else None)
            if arm == "ucoe" and i in self.ckpts and str(i) not in st["ckpt"]:
                try:
                    st["ckpt"][str(i)] = self.ucoe_checkpoint(st, i, st["rows"])
                except HardStop as e:
                    st.setdefault("incidents", []).append(
                        {"i": i, "where": "checkpoint", "code": e.code, "err": str(e)[:300], "t": time.time()}
                    )
                    _save(sp, st)
                    raise SystemExit(
                        "[ucoe-agent %s] %s at checkpoint %d: %s. Nothing sealed; rerun the same command to resume"
                        % (arm, e.code, i, e)
                    )
                _save(sp, st)
                print("[ucoe-agent ucoe] checkpoint %d: %s" % (i, st["ckpt"][str(i)]["decision"]), flush=True)
                V = st["V"]
            try:
                rec = self.run_task(arm, i, t, st, V)
            except HardStop as e:
                self.audit(e.code, arm=arm, i=i, err=str(e)[:300])
                st.setdefault("incidents", []).append({"i": i, "code": e.code, "err": str(e)[:300], "t": time.time()})
                _save(sp, st)
                raise SystemExit(
                    "[ucoe-agent %s] %s on task %d: %s. NOT sealed (fail closed); rerun the same command to resume"
                    % (arm, e.code, i, e)
                )
            st["rows"].append(self._row(rec))
            _save(sp, st)
            if os.path.exists(self._pending(arm)):
                os.remove(self._pending(arm))
            u = _load(self.p("state", "usage_%s.json" % arm), {})
            print(
                "[ucoe-agent %s] %d/%d %s status=%s served=%s r0=%s weakest=%s h=%s H=%d turns=%s usd=%s"
                % (
                    arm,
                    i + 1,
                    len(tasks),
                    rec["task_id"],
                    rec.get("status"),
                    rec.get("served"),
                    rec.get("r0") if rec.get("r0") is None else round(rec["r0"], 3),
                    rec.get("weakest"),
                    rec.get("h_update"),
                    len(self.h_load(arm)),
                    u.get("turns"),
                    round(u.get("usd", 0), 2),
                ),
                flush=True,
            )
        print(
            "[ucoe-agent %s] COMPLETE %d tasks; statuses %s"
            % (arm, len(st["rows"]), dict(collections.Counter(r["status"] for r in st["rows"]))),
            flush=True,
        )

    # ------------------------------------------------- pin probe, live smoke, status, audit
    def pin_probe(self):
        """One probe turn per role: the served model and effort must equal the requested ones before any arm runs."""
        out = {"ok": True, "backend": self.backend_name, "roles": {}, "t": time.time()}
        for role in ("actor", "verifier"):
            d = self.p("private", "pin", role)
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d)
            try:
                r = self.turn("pin", role, "pin", d, PIN_PROMPT, self.p("logs", "pin", "%s.json" % role))
                out["roles"][role] = {
                    "requested": dict(self.roles[role]),
                    "served": r.get("served"),
                    "reply": (r.get("text") or "")[-40:],
                    "status": r["status"],
                    "seconds": r["seconds"],
                }
                s = r.get("served") or {}
                if (
                    not model_matches(s.get("model"), self.roles[role]["model"])
                    or s.get("effort") != self.roles[role]["effort"]
                ):
                    out["ok"] = False
            except (SystemExit, HardStop) as e:
                out["roles"][role] = {"requested": dict(self.roles[role]), "error": str(e)[:300]}
                out["ok"] = False
        _save(self.p("private", "pin_probe.json"), out)
        print(json.dumps(out, indent=1, default=str))
        print("[ucoe-agent pin-probe]", "OK" if out["ok"] else "FAILED: do not run", flush=True)
        return out

    def smoke(self):
        """Exactly two live tasks from the frozen smoke reserve (never banks, never deployment), judge arm, fresh smoke
        root: provenance, effort,
        recalculation, evidence binding + echo, valid staged XLSX, immutable sealing, resume/exactly-once, no
        forbidden input.
        """
        man = self.manifest()
        ids = man["calibration"]["chunks"]["smoke"][:2]
        tasks = []
        for tid in ids:
            s = self.calib_source(tid)
            tasks.append(
                {
                    "id": s["task_id"].split("|")[1],
                    "category": s["category"],
                    "instruction": s["instruction"],
                    "input_path": s["input_path"],
                }
            )
        rep = {"ok": True, "backend": self.backend_name, "tasks": ids, "checks": {}, "t": time.time()}
        sp = self.p("state", "smoke_judge.json")
        st = _load(sp, {"arm": "smoke_judge", "rows": [], "roles": self.roles_dict()})
        for i, t in enumerate(tasks):
            if i < len(st["rows"]):
                continue
            try:
                rec = self.run_task("smoke_judge", i, t, st, V_FIXED)
            except HardStop as e:
                rep.update({"ok": False, "error": "%s: %s" % (e.code, e)})
                _save(self.p("private", "smoke_report.json"), rep)
                raise SystemExit("[ucoe-agent smoke] FAILED %s: %s" % (e.code, e))
            st["rows"].append(self._row(rec))
            _save(sp, st)
            if os.path.exists(self._pending("smoke_judge")):
                os.remove(self._pending("smoke_judge"))
            if i == 0:  # exactly-once: re-entering the same task must return the sealed record without any new turn
                n_turns = sum(1 for _ in open(self.p("state", "turns_smoke_judge.jsonl"), encoding="utf-8"))
                again = self.run_task("smoke_judge", 0, t, st, V_FIXED)
                rep["checks"]["exactly_once"] = (
                    again.get("sealed_at") == rec.get("sealed_at")
                    and sum(1 for _ in open(self.p("state", "turns_smoke_judge.jsonl"), encoding="utf-8")) == n_turns
                )
        recs = [
            _load(
                os.path.join(
                    self.p("private", "archive", "smoke_judge", "t%03d_%s_%s" % (i, t["category"], t["id"])),
                    "record.json",
                )
            )
            for i, t in enumerate(tasks)
        ]
        with open(self.p("state", "turns_smoke_judge.jsonl"), encoding="utf-8") as f:
            turns = [json.loads(ln) for ln in f]
        ch = rep["checks"]
        ch["actor_provenance"] = all(
            model_matches((x.get("served") or {}).get("model"), self.roles["actor"]["model"])
            for x in turns
            if x["role"] == "actor"
        )
        ch["verifier_provenance"] = all(
            model_matches((x.get("served") or {}).get("model"), self.roles["verifier"]["model"])
            for x in turns
            if x["role"] == "verifier"
        )
        done = [x for x in turns if x.get("status") == "completed"]
        ch["all_match"] = bool(done) and all(
            (x.get("served") or {}).get("effort") == self.roles[x["role"]]["effort"] for x in done
        )
        ch["effective_effort"] = ch["all_match"]
        ch["resolved_effort_values"] = sorted({str((x.get("served") or {}).get("effort")) for x in done})
        ch["requested_effort"] = {r: self.roles[r]["effort"] for r in ("actor", "verifier")}
        ch["recalculation"] = all(r.get("evidence_sha", {}).get("c0") for r in recs if r["status"] != "draft_missing")
        ch["evidence_binding_echo"] = all((r.get("score_c0") or {}).get("echo_ok") for r in recs if r.get("score_c0"))
        ch["staged_xlsx_valid"] = True
        for r in recs:
            for k in ("c0", "c1"):
                if r.get("score_%s" % k):
                    sd = self.p("private", "verify", PROTOCOL_VERSION, "smoke_judge", "task%03d_%s" % (r["i"], k))
                    for f in ("input.xlsx", "candidate.xlsx"):
                        try:
                            E.valid_xlsx(os.path.join(sd, f))
                        except HardStop:
                            ch["staged_xlsx_valid"] = False
        ch["immutable_seal"] = all(
            not os.access(
                os.path.join(
                    self.p("private", "archive", "smoke_judge", "t%03d_%s_%s" % (i, t["category"], t["id"])),
                    "served.xlsx",
                ),
                os.W_OK,
            )
            and r["served_sha"]
            == C.sha_file(
                os.path.join(
                    self.p("private", "archive", "smoke_judge", "t%03d_%s_%s" % (i, t["category"], t["id"])),
                    "served.xlsx",
                )
            )
            for i, (t, r) in enumerate(zip(tasks, recs))
            if r.get("served_sha")
        )
        ch["no_forbidden_input"] = not self._forbidden_hits()
        ch["statuses"] = [r["status"] for r in recs]
        gates = (
            "exactly_once",
            "actor_provenance",
            "verifier_provenance",
            "effective_effort",
            "all_match",
            "recalculation",
            "evidence_binding_echo",
            "staged_xlsx_valid",
            "immutable_seal",
            "no_forbidden_input",
        )
        rep["gates"] = list(gates)
        rep["ok"] = all(ch.get(g) is True for g in gates)
        rep.update(
            {
                "protocol_version": PROTOCOL_VERSION,
                "evidence_version": EVIDENCE_VERSION,
                "protocol_sha256": PROTO_SHA,
                "roles": self.roles_dict(),
                "n_completed_turns": len(done),
            }
        )
        rep["usage"] = _load(self.p("state", "usage_smoke_judge.json"), {})
        _save(self.p("private", "smoke_report.json"), rep)
        print(json.dumps(rep, indent=1, default=str))
        print(
            "[ucoe-agent smoke]",
            "PASS: all live checks passed" if rep["ok"] else "FAIL: do not launch this backend",
            flush=True,
        )
        return rep

    def _forbidden_hits(self):
        forb = []
        for dp, _, fs in os.walk(self.p("logs")):
            for f in fs:
                with open(os.path.join(dp, f), "rb") as fh:
                    txt = fh.read().decode("utf-8", "ignore")
                if any(k in txt for k in FORBIDDEN_KEYS) or "golden" in txt.lower():
                    forb.append(os.path.relpath(os.path.join(dp, f), self.root))
        return forb

    def status(self):
        for arm in self.arms:
            st = _load(self.p("state", "%s.json" % arm))
            if st:
                v = (st.get("V") or {}).get("version") if arm == "ucoe" else ("V_fixed" if arm == "judge" else "-")
                print(
                    "%s: %d rows; statuses %s; V %s; H %d; incidents %d; usage %s"
                    % (
                        arm,
                        len(st["rows"]),
                        dict(collections.Counter(r["status"] for r in st["rows"])),
                        v,
                        len(self.h_load(arm)),
                        len(st.get("incidents", [])),
                        _load(self.p("state", "usage_%s.json" % arm), {}),
                    ),
                    flush=True,
                )
        for k in ("pin_probe", "smoke_report"):
            r = _load(self.p("private", k + ".json"))
            print(k, (r or {}).get("ok"), flush=True)

    def audit_arm(self, arm="ucoe"):
        """Post-hoc integrity audit of one arm: exactly-once rows, archive hashes, role provenance, evidence binding,
        unchanged trees on rejection,
        clean reward-ceiling skips, bounded H, locked task directories, no forbidden inputs in logs, calibration
        disjoint from deployment.
        """
        man, st = self.manifest(), _load(self.p("state", "%s.json" % arm), {"rows": []})
        rows, checks = st["rows"], {}
        n = len(man["deployment_tasks"][: self.task_limit] if self.task_limit else man["deployment_tasks"])
        checks["rows_exactly_once"] = (
            len(rows) == n and [r["i"] for r in rows] == list(range(n)) and len({r["task_id"] for r in rows}) == n
        )
        arch_ok, prov_ok, ev_ok, rb_ok, ceil_ok = [], [], [], [], []
        for r in rows:
            ad = self.p("private", "archive", arm, "t%03d_%s_%s" % (r["i"], r["category"], r["task_id"].split("|")[1]))
            arec = _load(os.path.join(ad, "record.json"))
            arch_ok.append(
                bool(arec)
                and (
                    (
                        arec.get("served_sha") == C.sha_file(os.path.join(ad, "served.xlsx"))
                        and not os.access(os.path.join(ad, "served.xlsx"), os.W_OK)
                    )
                    if arec and arec.get("served_sha")
                    else True
                )
            )
            turns = [arec.get(k) for k in ("draft_turn", "revise_turn", "hproposal_turn") if arec and arec.get(k)]
            prov_ok.append(
                all(model_matches((tr.get("served") or {}).get("model"), self.roles["actor"]["model"]) for tr in turns)
                and (arec or {}).get("roles") == self.roles_dict()
            )
            for k in ("score_c0", "score_c1"):
                s = (arec or {}).get(k)
                if s and s.get("status") == "scored":
                    ev_ok.append(
                        s.get("evidence_sha") == ((arec.get("evidence_sha") or {}).get(k[-2:]) or {}).get("sha")
                        and s.get("echo_ok") is True
                        and model_matches(
                            ((s.get("transport") or {}).get("served") or {}).get("model"),
                            self.roles["verifier"]["model"],
                        )
                    )
            if r["status"] not in ("accepted",) and r.get("draft_tree_sha"):
                rb_ok.append(
                    r.get("draft_tree_sha_after") == r["draft_tree_sha"]
                    and r.get("H_sha_after") == r.get("H_sha_before")
                )
            if r["status"] == "reward_ceiling_skip":
                ceil_ok.append(
                    not (arec or {}).get("localize")
                    and not (arec or {}).get("score_c1")
                    and not (arec or {}).get("weakest")
                    and (arec or {}).get("skipped_stages") == ["weakest", "localize", "revise", "c1", "h_update"]
                )
        checks.update(
            {
                "archive_integrity": all(arch_ok) and len(arch_ok) == n,
                "role_provenance": all(prov_ok) and len(prov_ok) == n,
                "evidence_bound_and_echoed": all(ev_ok),
                "rejected_or_skipped_unchanged": all(rb_ok),
                "reward_ceiling_skips_clean": all(ceil_ok),
            }
        )
        H = self.h_load(arm)
        checks["H_bounded"] = (
            len(H) <= self.h_cap
            and all(sum(1 for e in H if e["dimension"] == d) <= self.h_per_dim for d in C.DIMS)
            and len(self.h_serialize(H)) <= self.h_chars
        )
        checks["no_incidents_open"] = not st.get("incidents") or len(rows) == n
        roots = [
            self._task_root(arm, r["i"], {"category": r["category"], "id": r["task_id"].split("|")[1]}) for r in rows
        ]
        checks["task_dirs_locked"] = all((os.stat(p).st_mode & 0o777) == 0 for p in roots if os.path.exists(p))
        forb = self._forbidden_hits()
        checks["no_forbidden_inputs_in_logs"] = not forb
        checks["calibration_disjoint"] = man["calibration"]["overlap_audit"]["deployment_x_calibration"] == 0 and set(
            _load(self.p("ucoe", "banks", "registry.json"), {"used": []})["used"]
        ).isdisjoint({"%s|%s" % (t["category"], t["id"]) for t in man["deployment_tasks"]})
        counts = {"n_scored_requests": len(ev_ok), "n_rejections_checked": len(rb_ok), "n_ceiling_skips": len(ceil_ok)}
        summary = {
            "arm": arm,
            "backend": self.backend_name,
            "n": len(rows),
            "statuses": dict(collections.Counter(r["status"] for r in rows)),
            "incidents": st.get("incidents", []),
            "H_size": len(H),
            "h0": (st.get("h0") or {}).get("decision"),
            "checkpoints": {k: v.get("decision") for k, v in (st.get("ckpt") or {}).items()},
            "checks": checks,
            "counts": counts,
            "all_gates_pass": all(v is True for v in checks.values()),
            "usage": _load(self.p("state", "usage_%s.json" % arm), {}),
            "forbidden_hits": forb,
        }
        _save(self.p("audit_%s.json" % arm), summary)
        print(json.dumps(summary, indent=1, default=str), flush=True)
        return summary
