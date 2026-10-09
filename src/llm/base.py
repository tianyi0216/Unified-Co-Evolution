"""Transport interface for every model role and the backend factory.

A transport is the only component that talks to a model (paper App. "Implementation details": every role -
actor, verifier, comparator, judge - is served by a frozen model through a sampling API; nothing is trained).
Backends: Tinker (open-weight models), any OpenAI-compatible chat-completions endpoint (vLLM, OpenAI, Azure) and
a deterministic mock for tests. All of them return the same normalized result dictionary.

Decoding keyword arguments (``kw``) are normalized here: ``max_tokens``, ``temperature``, ``seed``,
``thinking`` (Tinker / vLLM chat-template flag) and ``json_mode`` (OpenAI-compatible endpoints only). The
request-field aliases ``max_completion_tokens`` and ``enable_thinking`` are accepted and mapped.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

KW_ALIASES = {"max_completion_tokens": "max_tokens", "enable_thinking": "thinking"}
KW_DEFAULTS: Dict[str, Any] = {"max_tokens": 1024, "temperature": 0.0, "seed": None, "thinking": False,
                               "json_mode": None}
BACKENDS = ("tinker", "openai", "mock")


def normalize_kw(kw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Map the request-field aliases onto the normalized keys and fill the defaults."""
    out = dict(KW_DEFAULTS)
    for k, v in (kw or {}).items():
        out[KW_ALIASES.get(k, k)] = v
    return out


def split_messages(messages: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Default rendering of a message list for single-prompt backends: (system, last user content)."""
    msgs = list(messages or [])
    system = ""
    if msgs and msgs[0].get("role") == "system":
        system = str(msgs[0].get("content") or "")
    user = str(msgs[-1].get("content") or "") if msgs else ""
    return system, user


def make_result(text: str, finish_reason: Optional[str] = "stop", prompt_tokens: Optional[int] = None,
                completion_tokens: Optional[int] = None, reasoning_tokens: Optional[int] = None,
                model: Optional[str] = None, sdk: str = "", request_id: str = "") -> Dict[str, Any]:
    """The normalized transport result: {"text", "finish_reason", "usage", "model_meta"}."""
    return {"text": text or "",
            "finish_reason": finish_reason,
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "reasoning_tokens": reasoning_tokens},
            "model_meta": {"model": model, "id": request_id, "sdk": sdk}}


class Transport:
    """Callable ``(model, system, user, kw) -> result``; ``call_messages`` serves native multi-turn inputs."""

    name = "base"

    def __call__(self, model: str, system: str, user: str, kw: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def call_messages(self, model: str, messages: List[Dict[str, Any]], kw: Dict[str, Any]) -> Dict[str, Any]:
        """Default: render system/user from the first/last message; backends with native chat templates
        override this and pass the messages verbatim."""
        system, user = split_messages(messages)
        return self(model, system, user, kw)


def make_transport(models_cfg: Dict[str, Any]) -> Transport:
    """Build the transport named by ``models_cfg["backend"]`` ("tinker" | "openai" | "mock").

    Backend-specific settings live under ``models_cfg[backend]`` (e.g. ``models.openai.base_url``).
    """
    cfg = models_cfg or {}
    backend = str(cfg.get("backend", "mock")).lower()
    sub = cfg.get(backend) or {}
    if backend == "tinker":
        from .tinker_backend import TinkerTransport
        return TinkerTransport(sub)
    if backend == "openai":
        from .openai_backend import OpenAITransport
        return OpenAITransport(sub)
    if backend == "mock":
        from .mock_backend import MockTransport
        return MockTransport(sub)
    raise ValueError(f"unknown models.backend {backend!r}; expected one of {BACKENDS}")
