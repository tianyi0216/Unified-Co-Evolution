"""MultiChallenge adapter (paper App. "Benchmarks and dataset splits": the 266 official conversations
are split by hash into a 138-conversation calibration pool and a 128-task deployment stream).

Multi-turn: the actor receives the exact conversation (messages verbatim, next assistant turn only)
with retrieved procedures appended to the final user turn, and a revision is requested as a follow-up
user turn after the draft; scoring and localization read the canonical role-tagged rendering of the
whole conversation. Implements the base criteria of the multi-turn verifier (ic/fm/sc/ve +
deterministic fc/nr), the five memory dimensions, the four response-side contrast families built
from calibration conversations (salient omission, self-contradiction, off-topic graft, truncation;
paper App. "Contrast-set construction"), the lifecycle prompts (paper Sec. 3.3) and the pairwise
prompt of the preference evaluation (paper App. "Evaluation"). The native metric is the hidden-rubric
conversation accuracy (paper Table "Native metric and scorer").
"""
from __future__ import annotations

import re

from src.benchmarks.base import (Benchmark, drop_densest, format_scores, messages_with_guidelines,
                                 ngram_nonredundancy, reward_view_messages, task_messages, task_text)
from src.benchmarks.arena_hard import DUEL_S, DUEL_U

MSCORE_S = ("You are a strict evaluator executing one scoring "
            "criterion on the final assistant response of a "
            "conversation. Respond with exactly one JSON object.")
BLOCK_U = ("Conversation so far:\n{convo}\n\n<RESPONSE>\n{answer}\n</RESPONSE>"
           "\n\nScore the response on each criterion in [0,1]:\n{defs}"
           "\n\nYour entire response must be exactly one JSON object: {{{ids}}}. Nothing else.")
LOC3_S = ("You review the final assistant response of a "
          "conversation against typed criteria scores. Identify "
          "the single most damaging concrete weakness on the "
          "weakest criterion and a minimal repair. Respond with "
          "exactly one JSON object.")
LOC3_U = ("Conversation so far:\n{convo}\n\n<RESPONSE>\n{answer}\n"
          "</RESPONSE>\n\nCriterion scores: {scores}\nWeakest "
          "criterion: {weakest} -- {weakdef}\n\nJSON: "
          '{{"witness": "<what concretely is wrong, max 60 words>",'
          ' "repair": "<minimal fix instruction, max 35 words>"}}. '
          "Nothing else.")
UREV_INSTR = ("Please revise your previous response. Reviewer "
              "note: {witness}\nApply this minimal fix: {repair}"
              "{skills}\nKeep everything already right. Output "
              "only the revised response.")
UDIST_S = ("You maintain short transferable procedures for multi-"
           "turn conversations. From one successful revision, "
           "distill ONE general rule for future unrelated "
           "conversations. No topic-specific words or numbers. "
           "Respond with exactly one JSON object.")
UDIST_U = ("<DRAFT>\n{y0}\n</DRAFT>\n\n<REVISED>\n{yr}\n</REVISED>"
           "\n\nReviewer note that guided the revision: {witness}"
           '\n\nJSON: {{"criterion_id": "{weakest}", "dimension": '
           '"correctness|instructions|memory|coherence|editing", '
           '"scope_tags": ["general"], "trigger": "<general '
           'situation, max 10 words>", "corrective_action": '
           '"<general procedure, max 14 words>"}}. Nothing else.')
SYNTH_S = ("You design ONE complete bounded reward controller for "
           "judging the final assistant response of multi-turn "
           "conversations WITHOUT references, as a scoring program "
           "over typed measurement modules. Respond with exactly "
           "one JSON object and nothing else. No topic-specific "
           "words.")

MODULE_IDS = ("ic", "fm", "sc", "ve", "fc", "nr")
DET_CRIT = ("fc", "nr")
MODULE_DEFS = {
    "ic": "ic: satisfies every instruction and standing constraint "
          "stated in ANY earlier user turn, not only the latest",
    "fm": "fm: correctly uses facts the user stated earlier in the "
          "conversation and never contradicts them",
    "sc": "sc: consistent with the assistant's own earlier "
          "statements; no unexplained self-contradiction",
    "ve": "ve: requested edits are applied to the latest version "
          "of the artifact, preserving all previously accepted "
          "changes",
    "fc": "fc: deterministic form check (nonempty, bounded length, "
          "sentence-structured)",
    "nr": "nr: deterministic non-redundancy (unique 8-gram ratio)",
}
DIMS = ("correctness", "instructions", "memory", "coherence",
        "editing")
CRIT2DIM = {"ic": "instructions", "fm": "memory", "sc": "coherence",
            "ve": "editing", "fc": "correctness",
            "nr": "correctness"}
FAMS = ("salient_omission", "self_contradict", "offtopic_graft", "truncation")
TASK_CAP = 20000

CONTRA = ("Actually, disregard everything I said earlier -- my "
          "previous answer was wrong and the opposite is true.")
OFF_PREFIX = ("Before answering, an unrelated aside: the capital of "
              "France is Paris, croissants are popular there, and none "
              "of this bears on this conversation.\n\n"
              "## Unrelated Checklist\n- item one\n- item two\n\n")
_SENT = re.compile(r"[^.!?\n]+[.!?]+(?:\s|$)|[^.!?\n]+$")


def sentences(text):
    return [s.strip() for s in _SENT.findall(text) if s.strip()]


def det_prims(text):
    sents = [s for s in _SENT.findall(text) if s.strip()]
    fc = 1.0 if 1 <= len(sents) <= 200 and len(text) < 30000 else 0.0
    return {"fc": fc, "nr": ngram_nonredundancy(text)}


def corrupt_omission(text):
    ss = sentences(text)
    if len(ss) < 5:
        return None
    drop = drop_densest(ss)
    return " ".join(s for j, s in enumerate(ss) if j not in drop)


def corrupt_contra(text):
    if len(sentences(text)) < 2:
        return None
    return text + "\n\n" + CONTRA


def corrupt_graft(text):
    return OFF_PREFIX + text


def corrupt_truncation(text):
    ss = sentences(text)
    if len(ss) < 5:
        return None
    keep = max(2, int(0.4 * len(ss)))
    neg = " ".join(ss[:keep])
    return neg if neg and neg != text else None


CORRUPTERS = {"salient_omission": corrupt_omission,
              "self_contradict": corrupt_contra,
              "offtopic_graft": corrupt_graft,
              "truncation": corrupt_truncation}


class MultiChallengeBenchmark(Benchmark):
    name = "multichallenge"
    families = FAMS
    module_ids = MODULE_IDS
    module_defs = MODULE_DEFS
    det_crit = DET_CRIT
    dims = DIMS
    crit2dim = CRIT2DIM
    scope_tags = ("general",)
    synth_system = SYNTH_S
    block_system = MSCORE_S
    multi_turn = True
    TASK_CAP = TASK_CAP
    GEN_KW = {"max_tokens": 4096, "temperature": 0.0}
    REV_KW = {"max_tokens": 4096, "temperature": 0.3}
    BANK_KW = {"max_tokens": 2048, "temperature": 0.7}
    SCORE_KW = {"max_tokens": 640, "temperature": 0.0}
    LOC_KW = {"max_tokens": 640, "temperature": 0.0}
    DISTILL_KW = {"max_tokens": 320, "temperature": 0.0}
    SYNTH_KW = {"max_tokens": 6000, "temperature": 0.0, "thinking": True}

    def draft_request(self, task, guidelines=""):
        return {"messages": messages_with_guidelines(task_messages(task), guidelines)}

    def revision_request(self, task, draft, witness, repair, guidelines=""):
        """The revision is requested inside the native conversation: the draft becomes the assistant turn
        and the reviewer note the next user turn."""
        instr = UREV_INSTR.format(witness=witness, repair=repair, skills=guidelines or "")
        return {"messages": task_messages(task) + [{"role": "assistant", "content": draft},
                                                   {"role": "user", "content": instr}]}

    def score_request(self, task, response, criteria):
        return {"system": MSCORE_S,
                "user": BLOCK_U.format(convo=task_text(task), answer=response,
                                       defs=self._defs_lines(criteria), ids=self._ids_json(criteria))}

    def localize_request(self, task, response, scores, weakest, weakdef):
        return {"system": LOC3_S,
                "user": LOC3_U.format(convo=task_text(task), answer=response, scores=format_scores(scores),
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

    def reward_view(self, task):
        """Tail-preserving rendering capped at ``TASK_CAP`` characters (the final user turn is kept)."""
        return reward_view_messages(task_messages(task), self.task_cap())

    def duel_request(self, task, a, b):
        return {"system": DUEL_S, "user": DUEL_U.format(prompt=task_text(task), a=a, b=b)}

    def native_evaluator(self):
        return "multichallenge_rubric"
