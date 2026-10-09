"""RefineBench Free-Form adapter (paper App. "Benchmarks and dataset splits": the 478 Free-Form exam
tasks are split, stratified by field, into a 200-task calibration pool and a 278-task deployment
stream that keeps the official order).

Implements the base criteria of the exam-answer verifier (co/cv/gr/cl + deterministic fc/nr), the
four memory dimensions, the three response-side contrast families built from calibration tasks
(salient omission, claim flip, off-topic graft; paper App. "Contrast-set construction") with their
structural validators, the lifecycle prompts (paper Sec. 3.3) and the pairwise prompt of the
preference evaluation (paper App. "Evaluation"). The native metric is the checklist accuracy
(paper Table "Native metric and scorer").
"""
from __future__ import annotations

import re

from src.benchmarks.base import Benchmark, drop_densest, format_scores, ngram_nonredundancy, task_text, with_guidelines
from src.benchmarks.arena_hard import DUEL_S, DUEL_U

RAW_S = ("You are a capable, careful assistant. Answer the exam "
         "task thoroughly and directly.")
REV_INSTR = ("You revise a draft answer to an exam task. Improve "
             "the draft where it falls short of fully answering "
             "the task; keep everything already right. Output only "
             "the revised answer.")
REV_U = ("Task:\n{task}\n\n<DRAFT>\n{draft}\n</DRAFT>\n\n"
         "{instr}")
MSCORE_S = ("You are a strict evaluator executing one scoring "
            "criterion on an answer. Respond with exactly one JSON "
            "object.")
BLOCK_U = ("Task:\n{task}\n\n<ANSWER>\n{answer}\n</ANSWER>\n\nScore the answer on "
           "each criterion in [0,1]:\n{defs}"
           "\n\nYour entire response must be exactly one JSON object: {{{ids}}}. Nothing else.")
LOC3_S = ("You review an answer against typed criteria scores. "
          "Identify the single most damaging concrete weakness on "
          "the weakest criterion and a minimal repair. Respond "
          "with exactly one JSON object.")
LOC3_U = ("Task:\n{task}\n\n<ANSWER>\n{answer}\n</ANSWER>\n\n"
          "Criterion scores: {scores}\nWeakest criterion: "
          '{weakest} -- {weakdef}\n\nJSON: {{"witness": "<what '
          'concretely is wrong, max 60 words>", "repair": '
          '"<minimal fix instruction, max 35 words>"}}. '
          "Nothing else.")
UREV_INSTR = ("Reviewer note: {witness}\nApply this minimal fix: "
              "{repair}{skills}\nKeep everything already right. "
              "Output only the revised answer.")
UDIST_S = ("You maintain short transferable answering procedures. "
           "From one successful revision, distill ONE general rule "
           "for future unrelated tasks. No topic-specific words or "
           "numbers. Respond with exactly one JSON object.")
UDIST_U = ("<DRAFT>\n{y0}\n</DRAFT>\n\n<REVISED>\n{yr}\n</REVISED>"
           "\n\nReviewer note that guided the revision: {witness}"
           '\n\nJSON: {{"criterion_id": "{weakest}", "dimension": '
           '"correctness|coverage|grounding|clarity", "scope_tags":'
           ' ["general"], "trigger": "<general situation, max 10 '
           'words>", "corrective_action": "<general procedure, max '
           '14 words>"}}. Nothing else.')
SYNTH_S = ("You design ONE complete bounded reward controller for "
           "judging answers to open-ended exam tasks WITHOUT "
           "references, as a scoring program over typed "
           "measurement modules. Respond with exactly one JSON "
           "object and nothing else. No topic-specific words.")

MODULE_IDS = ("co", "cv", "gr", "cl", "fc", "nr")
DET_CRIT = ("fc", "nr")
MODULE_DEFS = {
    "co": "co: content and reasoning are correct and internally "
          "consistent",
    "cv": "cv: every part and sub-question of the task is "
          "addressed",
    "gr": "gr: claims are grounded in the provided passages/"
          "materials when present; no fabricated citations",
    "cl": "cl: clear, well-organized, appropriately concise",
    "fc": "fc: deterministic form check (nonempty, bounded length, "
          "sentence-structured)",
    "nr": "nr: deterministic non-redundancy (unique 8-gram ratio)",
}
DIMS = ("correctness", "coverage", "grounding", "clarity")
CRIT2DIM = {"co": "correctness", "cv": "coverage",
            "gr": "grounding", "cl": "clarity",
            "fc": "correctness", "nr": "clarity"}
FAMS = ("salient_omission", "claim_flip", "offtopic_graft")

CONTRA = ("Actually, on reflection the key conclusion above is "
          "wrong and the opposite holds; disregard the previous "
          "analysis.")
OFF_PREFIX = ("Before answering, an unrelated aside: the capital "
              "of France is Paris, croissants are popular there, "
              "and none of this bears on the task.\n\n"
              "## Unrelated Checklist\n- item one\n- item two\n\n")
_SENT = re.compile(r"[^.!?\n]+[.!?]+(?:\s|$)|[^.!?\n]+$")


def bundle(question, passages, materials):
    """Model-visible task text: question plus any provided passages and materials."""
    parts = [f"Task:\n{question}"]
    if passages:
        parts.append("Passages:\n" + "\n\n".join(passages))
    if materials:
        parts.append("Materials:\n" + "\n\n".join(materials))
    return "\n\n".join(parts)


def sentences(text):
    return [s.strip() for s in _SENT.findall(text) if s.strip()]


def det_prims(text):
    sents = [s for s in _SENT.findall(text) if s.strip()]
    fc = 1.0 if 1 <= len(sents) <= 400 and len(text) < 60000 else 0.0
    return {"fc": fc, "nr": ngram_nonredundancy(text)}


def corrupt_omission(text):
    ss = sentences(text)
    if len(ss) < 5:
        return None
    drop = drop_densest(ss)
    return " ".join(s for j, s in enumerate(ss) if j not in drop)


def corrupt_claim_flip(text):
    if len(sentences(text)) < 2:
        return None
    return text + "\n\n" + CONTRA


def corrupt_offtopic(text):
    return OFF_PREFIX + text


CORRUPTERS = {"salient_omission": corrupt_omission,
              "claim_flip": corrupt_claim_flip,
              "offtopic_graft": corrupt_offtopic}


def cluster_valid(fam, pos, neg, eps):
    """Structural check of a (pos, neg) pair plus the deterministic-criteria direction guard."""
    if fam == "salient_omission":
        ok = len(neg) < len(pos) and neg != pos and bool(set(sentences(pos)) - set(sentences(neg)))
    elif fam == "claim_flip":
        ok = neg.startswith(pos) and CONTRA in neg and CONTRA not in pos
    elif fam == "offtopic_graft":
        ok = neg.endswith(pos) and neg != pos and neg.startswith(OFF_PREFIX[:40])
    else:
        return False
    if not ok:
        return False
    dp, dn = det_prims(pos), det_prims(neg)
    return dp["fc"] == dn["fc"] == 1.0 and dp["nr"] - dn["nr"] <= eps


class RefineBenchBenchmark(Benchmark):
    name = "refinebench"
    families = FAMS
    module_ids = MODULE_IDS
    module_defs = MODULE_DEFS
    det_crit = DET_CRIT
    dims = DIMS
    crit2dim = CRIT2DIM
    scope_tags = ("general",)
    synth_system = SYNTH_S
    block_system = MSCORE_S
    TASK_CAP = 20000   # head-preserving reward view of long exam tasks, as in the paper's runs
    GEN_KW = {"max_tokens": 4096, "temperature": 0.0}
    REV_KW = {"max_tokens": 4096, "temperature": 0.3}
    SCORE_KW = {"max_tokens": 640, "temperature": 0.0}
    LOC_KW = {"max_tokens": 640, "temperature": 0.0}
    DISTILL_KW = {"max_tokens": 320, "temperature": 0.0}
    SYNTH_KW = {"max_tokens": 6000, "temperature": 0.0, "thinking": True}

    def draft_request(self, task, guidelines=""):
        return {"system": RAW_S, "user": with_guidelines(task_text(task), guidelines)}

    def revision_request(self, task, draft, witness, repair, guidelines=""):
        instr = UREV_INSTR.format(witness=witness, repair=repair, skills=guidelines or "")
        return {"system": REV_INSTR, "user": REV_U.format(task=task_text(task), draft=draft, instr=instr)}

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
        fn = CORRUPTERS.get(family)
        neg = self._apply(fn, response) if fn else None
        if neg is None or not cluster_valid(family, response, neg, self.eps):
            return None
        return neg

    def duel_request(self, task, a, b):
        return {"system": DUEL_S, "user": DUEL_U.format(prompt=task_text(task), a=a, b=b)}

    def native_evaluator(self):
        return "refinebench_checklist"
