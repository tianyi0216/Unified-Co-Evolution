"""WritingPrompts adapter (paper App. "Benchmarks and dataset splits": a 2,048-task curriculum stream
of story prompts whose final 1,024 tasks are evaluated; calibration from 1,400 frozen prompts).

Implements the seven base criteria of the story verifier (pe/cc/od/pv/hl + deterministic nr/ln),
the six memory dimensions and the scope tags inferred from the prompt, the eight narrative contrast
families with their stage schedule (paper App. "Contrast-set construction": "story generation
replaces these edits with narrative edits of the same kind"), the story prompts of the per-task
lifecycle (paper Sec. 3.3) and the five-dimension pairwise prompt whose ``overall`` verdict drives
the preference score (paper App. "Evaluation").

The diagnosis prompt (LOC_S / LOC_U) is the one prompt of this adapter that is not taken from the
paper's WritingPrompts runs: those obtained the witness and the repair instruction from a single
scoring-and-diagnosis call in which the verifier model also named the weakest criterion itself. This
release follows the paper's procedure instead (the runner selects the weakest participating criterion
from the frozen scores), so the diagnosis call asks only for the witness and the repair, with that
call's wording for both fields.
"""
from __future__ import annotations

import difflib
import re

from src.benchmarks.base import Benchmark, format_scores, task_text, with_guidelines

WP_INSTR = ("Write a complete short story of 300-500 words responding to the "
            "writing prompt. Plain continuous prose only: no headings, lists "
            "or meta-commentary. Give the story a clear arc and ending.")
RAW_S = "You are a skilled fiction writer."
TASK_PREFIX = f"{WP_INSTR}\n\nWriting prompt: "
BLOCK_S = ("You are a strict story evaluator. Score every requested "
           "criterion in [0,1]. Respond with exactly one JSON object.")
BLOCK_U = ("Task instruction:\n{instr}\n\nWriting prompt:\n{prompt}\n\n"
           "<STORY>\n{story}\n</STORY>\n\nCriteria:\n{defs}\n\n"
           "Respond with exactly one JSON object mapping each criterion "
           "id to its score: {{{ids}}}. Nothing else.")
LOC_S = ("You localize one already-selected weakness of a short story. "
         "The weakness criterion and all scores are FROZEN inputs; do not "
         "rescore. Respond with exactly one JSON object.")
LOC_U = ("Task instruction:\n{instr}\n\nWriting prompt:\n{prompt}\n\n"
         "<STORY>\n{story}\n</STORY>\n\nFrozen criterion scores: {scores}\n"
         "Selected weakest criterion: {weakest} ({weakdef})\n\n"
         "Ground this weakness in the story text and give one actionable "
         "repair. Respond with exactly one JSON object: "
         '{{"witness": "<one localized weakness grounded in the story '
         'text, max 40 words>", '
         '"repair": "<one short repair instruction, max 25 words>"}}. '
         "Nothing else.")
REV_S = ("You carefully revise short stories. Apply the reviewer note and "
         "any general guidelines; keep everything that already works. "
         "Output only the revised story.")
REV_U = ("Task instruction:\n{instr}\n\nWriting prompt:\n{prompt}\n\n"
         "<DRAFT>\n{draft}\n</DRAFT>\n\nReviewer note: {witness} "
         "Repair: {repair}{skills}\n\nRewrite the story accordingly, "
         "300-500 words of plain prose. Output only the revised story.")
DISTILL_S = ("You maintain short transferable writing procedures. From "
             "one successful revision, distill ONE general rule for "
             "future unrelated prompts. It must not mention any "
             "character, entity, object, number, quote or plot detail "
             "from this story or prompt. Respond with exactly one JSON "
             "object.")
DISTILL_U = ("Writing prompt:\n{prompt}\n\n<DRAFT>\n{y0}\n</DRAFT>\n\n"
             "<REVISED>\n{yr}\n</REVISED>\n\nReviewer note that guided "
             "the revision: {witness}\nRetrieved rule ids: {rids}\n\n"
             "Respond with exactly one JSON object: "
             '{{"criterion_id": "{weakest}", '
             '"dimension": "prompt_use|consistency|event_order|pov_tense|'
             'nonredundancy|craft", '
             '"scope_tags": ["general" or one to two of: dialogue, '
             'opening, ending, pacing, description, structure, '
             'character], '
             '"trigger": "<general situation, max 10 words>", '
             '"corrective_action": "<general procedure, max 14 words>"}}. '
             "Nothing else.")
SYNTH_S = ("You design ONE bounded reward capsule extension for judging "
           "short fiction. Respond with exactly one JSON object and "
           "nothing else. No code, no task-specific words, no source "
           "names.")
DUEL_S = ("You are a strict, impartial judge of short fiction. Compare "
          "two stories written for the same task. Respond with exactly "
          "one JSON object.")
DUEL_U = ("Task instruction:\n{instr}\n\nWriting prompt:\n{prompt}\n\n"
          "<STORY_A>\n{a}\n</STORY_A>\n\n<STORY_B>\n{b}\n</STORY_B>\n\n"
          "Judge which story is better on each dimension. Respond with "
          "exactly one JSON object: "
          '{{"novelty": "A|B|TIE", "emotional_engagement": "A|B|TIE", '
          '"coherence": "A|B|TIE", "consistency": "A|B|TIE", '
          '"overall": "A|B|TIE"}}. Nothing else.')
DUEL_DIMS = ("novelty", "emotional_engagement", "coherence", "consistency", "overall")

DIMS = ("prompt_use", "consistency", "event_order", "pov_tense", "nonredundancy", "craft")
BASE_CRIT = ("pe", "cc", "od", "pv", "hl", "nr", "ln")
BASE_DEFS = {
    "pe": "prompt_use: the story genuinely uses the prompt's premise and "
          "its named elements",
    "cc": "consistency: no internal contradictions of fact, character or "
          "setting",
    "od": "order: events unfold in a coherent causal/temporal order",
    "pv": "pov_tense: point of view and tense are stable throughout",
    "hl": "holistic: overall quality as a short story",
    "nr": "non-redundancy: computed deterministically from the text "
          "(no LLM judgment needed)",
    "ln": "length fit: computed deterministically from the text "
          "(no LLM judgment needed)"}
CRIT2DIM = {"pe": "prompt_use", "cc": "consistency", "od": "event_order",
            "pv": "pov_tense", "nr": "nonredundancy", "hl": "craft",
            "ln": "craft"}
DET_CRIT = ("nr", "ln")
SCOPE_TAGS = ("general", "dialogue", "opening", "ending", "pacing",
              "description", "structure", "character")
FAM_SCHEDULE = {0: ("missing_element", "contradiction", "order_corruption",
                    "pov_tense_switch"),
                512: ("duplication", "truncation"),
                1024: ("scramble", "cliche_filler")}
ALL_FAMS = tuple(f for _, fs in sorted(FAM_SCHEDULE.items()) for f in fs)
N_STREAM, EVAL_RANGE, MIN_SENTS = 2048, (1024, 2048), 6

_S_RE = re.compile(r"[^.!?]+[.!?]")


def sentences(story):
    return [s.strip() for s in _S_RE.findall(story)]


def prompt_of(task):
    """The bare writing prompt of a Task/Source dict (or of the rendered task text)."""
    if isinstance(task, dict) and task.get("prompt"):
        return task["prompt"]
    text = task_text(task)
    return text[len(TASK_PREFIX):] if text.startswith(TASK_PREFIX) else text


def task_text_of(prompt):
    return TASK_PREFIX + prompt


def det_prims(story):
    """nr: unique-sentence ratio; ln: fit to the 300-500 word target; sents: sentence count."""
    sents = sentences(story)
    nr = len(set(sents)) / max(1, len(sents))
    w = len(story.split())
    if 300 <= w <= 500:
        ln = 1.0
    else:
        ln = max(0.0, 1.0 - abs(w - (300 if w < 300 else 500)) / 200)
    return {"nr": round(nr, 4), "ln": round(ln, 4), "sents": len(sents)}


def prompt_tags(prompt):
    """Deterministic scope tags inferred from the public prompt for memory retrieval."""
    low = prompt.lower()
    tags = set()
    if '"' in prompt or "says" in low or "said" in low or "dialogue" in low:
        tags.add("dialogue")
    if any(w in low for w in ("begin", "start", "opening", "first line")):
        tags.add("opening")
    if any(w in low for w in ("end", "ending", "finally", "last")):
        tags.add("ending")
    if any(w in low for w in ("you are", "character", "person", "narrator",
                              "protagonist")):
        tags.add("character")
    if any(w in low for w in ("describe", "description", "vivid",
                              "detail")):
        tags.add("description")
    if any(w in low for w in ("structure", "twist", "order", "sequence")):
        tags.add("structure")
    return tags


def corrupt(family, prompt, y, rng):
    """One narrative edit of the declared family; None when the story is ineligible."""
    sents = sentences(y)
    if len(sents) < 6:
        return None
    if family == "missing_element":
        content = [w for w in re.findall(r"[A-Za-z]{5,}", prompt)
                   if re.search(rf"\b{re.escape(w)}\b", y, re.I)]
        if not content:
            return None
        out = y
        for w in content[:3]:
            out = re.sub(rf"\b{re.escape(w)}\b", "thing", out,
                         flags=re.IGNORECASE)
        return out if out != y else None
    if family == "contradiction":
        k = len(sents) // 2
        return y.replace(sents[k], sents[k] + " In truth the opposite of "
                         "all this was the case, and everyone involved "
                         "had always known it never happened.", 1)
    if family == "order_corruption":
        a, b = sents[1], sents[-2]
        return y.replace(a, "\x00", 1).replace(b, a, 1).replace("\x00", b,
                                                                1)
    if family == "pov_tense_switch":
        k = len(sents) // 2
        s = sents[k]
        s2 = re.sub(r"\b(he|she)\b", "I", s)
        s2 = re.sub(r"\b(his|her)\b", "my", s2)
        s2 = re.sub(r"\bI\b", "you", s2) if s2 == s else s2
        return y.replace(s, s2, 1) if s2 != s else None
    if family == "duplication":
        k = len(sents) // 2
        return y.replace(sents[k], sents[k] + " " + sents[k], 1)
    if family == "truncation":
        return y[:int(len(y) * 0.6)].rstrip()
    if family == "scramble":
        order = list(range(len(sents)))
        rng.shuffle(order)
        if order == sorted(order):
            order = order[::-1]
        return " ".join(sents[i] for i in order)
    if family == "cliche_filler":
        filler = (" It was what it was. Time passed as time does. Things "
                  "happened, and then more things happened.")
        k = len(sents) // 3
        return y.replace(sents[k], sents[k] + filler * 2, 1)
    return None


def stage_families(stage):
    fams = []
    for at, fs in sorted(FAM_SCHEDULE.items()):
        if at <= stage:
            fams.extend(fs)
    return tuple(fams)


def duel_val(o):
    """Strict on ``overall`` (drives the verdict); fuzzy on the four descriptive sub-dimension keys,
    which a judge occasionally misspells or renames; a missing sub-dimension counts TIE."""
    if not isinstance(o, dict):
        return None
    canon = {}
    bare = [re.sub(r"[^a-z]", "", d) for d in DUEL_DIMS]
    for k, v in o.items():
        nk = re.sub(r"[^a-z]", "", str(k).lower())
        hit = difflib.get_close_matches(nk, bare, n=1, cutoff=0.6)
        if not hit:
            continue
        d = DUEL_DIMS[bare.index(hit[0])]
        vv = str(v).strip().upper()
        if vv in ("A", "B", "TIE"):
            canon.setdefault(d, vv)
    if "overall" not in canon:
        return None
    return {d: canon.get(d, "TIE") for d in DUEL_DIMS}


class WritingPromptsBenchmark(Benchmark):
    name = "writingprompts"
    families = ALL_FAMS
    module_ids = BASE_CRIT
    module_defs = BASE_DEFS
    det_crit = DET_CRIT
    dims = DIMS
    crit2dim = CRIT2DIM
    scope_tags = SCOPE_TAGS
    synth_system = SYNTH_S
    block_system = BLOCK_S
    GEN_KW = {"max_tokens": 1024, "temperature": 0.8}
    REV_KW = {"max_tokens": 1024, "temperature": 0.3}
    SCORE_KW = {"max_tokens": 768, "temperature": 0.0}
    LOC_KW = {"max_tokens": 768, "temperature": 0.0}
    DISTILL_KW = {"max_tokens": 384, "temperature": 0.0}
    SYNTH_KW = {"max_tokens": 2048, "temperature": 0.0, "thinking": True}

    def __init__(self, cfg):
        super().__init__(cfg)
        self.eval_range = EVAL_RANGE

    def stage_families(self, stage):
        return stage_families(stage)

    def draft_request(self, task, guidelines=""):
        return {"system": RAW_S, "user": with_guidelines(task_text_of(prompt_of(task)), guidelines)}

    def revision_request(self, task, draft, witness, repair, guidelines=""):
        return {"system": REV_S,
                "user": REV_U.format(instr=WP_INSTR, prompt=prompt_of(task), draft=draft, witness=witness,
                                     repair=repair, skills=guidelines or "")}

    def score_request(self, task, response, criteria):
        return {"system": BLOCK_S,
                "user": BLOCK_U.format(instr=WP_INSTR, prompt=prompt_of(task), story=response,
                                       defs="\n".join(f"- {c['id']}: {c['definition']}" for c in criteria),
                                       ids=self._ids_json(criteria))}

    def localize_request(self, task, response, scores, weakest, weakdef):
        return {"system": LOC_S,
                "user": LOC_U.format(instr=WP_INSTR, prompt=prompt_of(task), story=response,
                                     scores=format_scores(scores), weakest=weakest, weakdef=weakdef)}

    def distill_request(self, task, y0, yr, witness, weakest, rids=None):
        return {"system": DISTILL_S,
                "user": DISTILL_U.format(prompt=prompt_of(task), y0=y0[:2500], yr=yr[:2500], witness=witness,
                                         rids=list(rids) if rids else "(none)", weakest=weakest)}

    def det_prims(self, response):
        d = det_prims(response)
        return {"nr": d["nr"], "ln": d["ln"]}

    def valid_base(self, response):
        return bool(response) and det_prims(response)["sents"] >= MIN_SENTS

    def corrupt(self, family, source, response, rng):
        if family not in ALL_FAMS:
            return None
        return self._apply(lambda y: corrupt(family, prompt_of(source), y, rng), response)

    def prompt_tags(self, task):
        return {"general"} | prompt_tags(prompt_of(task))

    def duel_request(self, task, a, b):
        return {"system": DUEL_S, "user": DUEL_U.format(instr=WP_INSTR, prompt=prompt_of(task), a=a, b=b)}

    def duel_parse(self, obj):
        got = duel_val(obj)
        return got["overall"] if got else None
