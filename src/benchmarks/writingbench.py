"""WritingBench adapter (paper App. "Benchmarks and dataset splits": the 1,000 official queries are
split by hash into a 500-query calibration pool and a 500-task deployment stream).

Implements the base criteria of the writing verifier (rq/gr/cv/wq + deterministic fc/nr with
CJK-aware tokenization), the four memory dimensions, the four response-side contrast families built
from the benchmark's own calibration queries (salient omission, claim conflict, off-topic graft,
truncation; paper App. "Contrast-set construction"), the lifecycle prompts (paper Sec. 3.3) and the
pairwise prompt of the preference evaluation (paper App. "Evaluation"). The native metric is the
Critic score (paper Table "Native metric and scorer").
"""
from __future__ import annotations

import re

from src.benchmarks.base import Benchmark, format_scores, ngram_nonredundancy, task_text, with_guidelines
from src.benchmarks.arena_hard import DUEL_S, DUEL_U

RAW_S = ("You are a capable, careful writing assistant. Complete "
         "the writing task thoroughly and directly, following "
         "every stated requirement.")
REV_U = ("Writing task:\n{task}\n\n<DRAFT>\n{draft}\n</DRAFT>\n\n"
         "{instr}")
REV_INSTR_S = ("You revise a draft for a writing task. Improve the "
               "draft where it falls short; keep everything "
               "already right. Output only the revised text.")
MSCORE_S = ("You are a strict evaluator executing one scoring "
            "criterion on a response. Respond with exactly one "
            "JSON object.")
BLOCK_U = ("Writing task:\n{task}\n\n<RESPONSE>\n{answer}\n</RESPONSE>"
           "\n\nScore the response on each criterion in [0,1]:\n{defs}"
           "\n\nYour entire response must be exactly one JSON object: {{{ids}}}. Nothing else.")
LOC3_S = ("You review a response against typed criteria scores. "
          "Identify the single most damaging concrete weakness on "
          "the weakest criterion and a minimal repair. Respond "
          "with exactly one JSON object.")
LOC3_U = ("Writing task:\n{task}\n\n<RESPONSE>\n{answer}\n"
          "</RESPONSE>\n\nCriterion scores: {scores}\nWeakest "
          'criterion: {weakest} -- {weakdef}\n\nJSON: {{"witness":'
          ' "<what concretely is wrong, max 60 words>", "repair": '
          '"<minimal fix instruction, max 35 words>"}}. '
          "Nothing else.")
UREV_INSTR = ("Reviewer note: {witness}\nApply this minimal fix: "
              "{repair}{skills}\nKeep everything already right. "
              "Output only the revised text.")
UDIST_S = ("You maintain short transferable writing procedures. "
           "From one successful revision, distill ONE general rule "
           "for future unrelated tasks. No topic-specific words or "
           "numbers. Respond with exactly one JSON object.")
UDIST_U = ("<DRAFT>\n{y0}\n</DRAFT>\n\n<REVISED>\n{yr}\n</REVISED>"
           "\n\nReviewer note that guided the revision: {witness}"
           '\n\nJSON: {{"criterion_id": "{weakest}", "dimension": '
           '"requirements|grounding|coverage|quality", '
           '"scope_tags": ["general"], "trigger": "<general '
           'situation, max 10 words>", "corrective_action": '
           '"<general procedure, max 14 words>"}}. Nothing else.')
SYNTH_S = ("You design ONE complete bounded reward controller for "
           "judging responses to writing tasks WITHOUT references, "
           "as a scoring program over typed measurement modules. "
           "Respond with exactly one JSON object and nothing else. "
           "No topic-specific words.")

MODULE_IDS = ("rq", "gr", "cv", "wq", "fc", "nr")
DET_CRIT = ("fc", "nr")
MODULE_DEFS = {
    "rq": "rq: every explicit requirement of the task (format, "
          "structure, style, length, content) is satisfied",
    "gr": "gr: faithful to any materials provided in the task; no "
          "fabricated facts or citations",
    "cv": "cv: every part of the task is addressed with adequate "
          "depth",
    "wq": "wq: writing quality -- coherent, well-organized, "
          "appropriate tone",
    "fc": "fc: deterministic form check (nonempty, bounded length, "
          "sentence-structured)",
    "nr": "nr: deterministic non-redundancy (unique 8-gram ratio)",
}
DIMS = ("requirements", "grounding", "coverage", "quality")
CRIT2DIM = {"rq": "requirements", "gr": "grounding",
            "cv": "coverage", "wq": "quality",
            "fc": "requirements", "nr": "quality"}
FAMS = ("salient_omission", "claim_conflict", "offtopic_graft", "truncation")

CONTRA = ("On reflection, the central claim above is wrong and the "
          "opposite is true; the preceding draft should be "
          "disregarded.")
OFF_PREFIX = ("Before the draft, an unrelated aside: the capital "
              "of France is Paris, croissants are popular there, "
              "and none of this bears on the task.\n\n"
              "## Unrelated Checklist\n- item one\n- item two\n\n")
_SENT = re.compile(r"[^.!?。！？\n]+[.!?。！？]+(?:\s|$)?"
                   r"|[^.!?。！？\n]+$")
_TOKEN = r"[a-z0-9一-鿿']+"


def sentences(text):
    return [s.strip() for s in _SENT.findall(text) if s.strip()]


def det_prims(text):
    sents = [s for s in _SENT.findall(text) if s.strip()]
    fc = 1.0 if 1 <= len(sents) <= 800 and len(text) < 120000 else 0.0
    return {"fc": fc, "nr": ngram_nonredundancy(text, token_re=_TOKEN)}


def corrupt_omission(text):
    """Drop the two densest sentences, or the two longest non-empty lines of a line-structured text."""
    ss = sentences(text)
    if len(ss) >= 5:
        dens = sorted(range(len(ss)),
                      key=lambda j: -(len(re.findall(r"[a-zA-Z一-鿿]{2,}", ss[j]))
                                      + 3 * len(re.findall(r"\d", ss[j]))))
        drop = set(dens[:2])
        return " ".join(s for j, s in enumerate(ss) if j not in drop)
    lines = text.split("\n")
    idx = [j for j, l in enumerate(lines) if l.strip()]
    if len(idx) < 5:
        return None
    dens = sorted(idx, key=lambda j: -len(lines[j]))
    drop = set(dens[:2])
    return "\n".join(l for j, l in enumerate(lines) if j not in drop)


def corrupt_claim(text):
    if len(text) < 200:
        return None
    return text + "\n\n" + CONTRA


def corrupt_offtopic(text):
    return OFF_PREFIX + text


def corrupt_truncation(text):
    ss = sentences(text)
    if len(ss) < 5:
        return None
    keep = max(2, int(0.4 * len(ss)))
    neg = " ".join(ss[:keep])
    return neg if neg and neg != text else None


CORRUPTERS = {"salient_omission": corrupt_omission,
              "claim_conflict": corrupt_claim,
              "offtopic_graft": corrupt_offtopic,
              "truncation": corrupt_truncation}


class WritingBenchBenchmark(Benchmark):
    name = "writingbench"
    families = FAMS
    module_ids = MODULE_IDS
    module_defs = MODULE_DEFS
    det_crit = DET_CRIT
    dims = DIMS
    crit2dim = CRIT2DIM
    scope_tags = ("general",)
    synth_system = SYNTH_S
    block_system = MSCORE_S
    GEN_KW = {"max_tokens": 8192, "temperature": 0.0}
    REV_KW = {"max_tokens": 8192, "temperature": 0.3}
    BANK_KW = {"max_tokens": 2048, "temperature": 0.7}
    SCORE_KW = {"max_tokens": 640, "temperature": 0.0}
    LOC_KW = {"max_tokens": 640, "temperature": 0.0}
    DISTILL_KW = {"max_tokens": 320, "temperature": 0.0}
    SYNTH_KW = {"max_tokens": 6000, "temperature": 0.0, "thinking": True}

    def draft_request(self, task, guidelines=""):
        return {"system": RAW_S, "user": with_guidelines(task_text(task), guidelines)}

    def revision_request(self, task, draft, witness, repair, guidelines=""):
        instr = UREV_INSTR.format(witness=witness, repair=repair, skills=guidelines or "")
        return {"system": REV_INSTR_S, "user": REV_U.format(task=task_text(task), draft=draft, instr=instr)}

    def score_request(self, task, response, criteria):
        return {"system": MSCORE_S,
                "user": BLOCK_U.format(task=task_text(task), answer=response,
                                       defs=self._defs_lines(criteria), ids=self._ids_json(criteria))}

    def localize_request(self, task, response, scores, weakest, weakdef):
        return {"system": LOC3_S,
                "user": LOC3_U.format(task=task_text(task), answer=response, scores=format_scores(scores),
                                      weakest=weakest, weakdef=weakdef)}

    def distill_request(self, task, y0, yr, witness, weakest):
        return {"system": UDIST_S,
                "user": UDIST_U.format(y0=y0[:2500], yr=yr[:2500], witness=witness, weakest=weakest)}

    def det_prims(self, response):
        return det_prims(response)

    def corrupt(self, family, source, response, rng):
        """One response-side edit; the negative must keep a sane form and must not hand the positive a
        free deterministic win (its non-redundancy may not drop by more than the acceptance margin)."""
        fn = CORRUPTERS.get(family)
        neg = self._apply(fn, response) if fn else None
        if neg is None:
            return None
        dp, dn = det_prims(response), det_prims(neg)
        if dn["fc"] != 1.0 or dp["nr"] - dn["nr"] > self.eps:
            return None
        return neg

    def duel_request(self, task, a, b):
        return {"system": DUEL_S, "user": DUEL_U.format(prompt=task_text(task), a=a, b=b)}

    def native_evaluator(self):
        return "writingbench_critic"
