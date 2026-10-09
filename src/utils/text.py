"""Text helpers: JSON extraction from model output, sentence/token splitting and Jaccard overlap.

``extract_json`` is the lenient parser applied to every structured model reply (scores, proposals, diagnoses,
distilled procedures; paper App. "Implementation details"). ``tokens`` and ``jaccard`` implement the token-set
overlap used to de-duplicate harness procedures (App. "Harness-state instantiation", dedup threshold 0.75) and
``sentences`` the sentence split used by the reference contrast families (App. "Contrast-set construction").
"""
from __future__ import annotations

import json
import re
from typing import Any, List, Optional, Set

_SENT = re.compile(r"[^.!?]+[.!?]")
_TOK = re.compile(r"[a-z0-9']+")


def extract_json(text: str) -> Optional[Any]:
    """Parse the JSON object in a model reply: strip code fences / a leading "json" tag, else the first
    parseable suffix starting at a "{"; None when nothing parses."""
    try:
        return json.loads(text.strip().strip("`").lstrip("json").strip())
    except (json.JSONDecodeError, ValueError):
        pass
    for m in re.finditer(r"\{", text):
        try:
            return json.loads(text[m.start():].strip())
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def sentences(text: str) -> List[str]:
    """Sentences terminated by . ! or ? (stripped)."""
    return [s.strip() for s in _SENT.findall(text or "") if s.strip()]


def tokens(text: str) -> List[str]:
    """Lower-cased alphanumeric tokens (apostrophes kept)."""
    return _TOK.findall((text or "").lower())


def jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard overlap of two token sets (0.0 when both are empty)."""
    union = len(a | b)
    return len(a & b) / union if union else 0.0
