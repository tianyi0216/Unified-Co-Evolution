"""Tinker sampling backend for open-weight models.

Serves every role through Tinker's sampling API (paper App. "Implementation details": frozen Qwen3-family models;
no weights are ever updated). One SamplingClient per model; the model's native chat template is applied with
``enable_thinking`` switched per call (on only for verifier synthesis), a rank-8 LoRA training client is created
once per model purely to obtain the tokenizer (nothing is trained), every sample has a hard timeout with visible
heartbeats and bounded retries, and ``<think>...</think>`` blocks are stripped from the returned text.
Requires the ``tinker`` package and the ``TINKER_API_KEY`` environment variable.
"""
from __future__ import annotations

import concurrent.futures as cf
import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from .base import Transport, make_result, normalize_kw

TIMEOUT_S = 300            # hard cap per attempt
HEARTBEAT_S = 60           # waits print a heartbeat this often
RETRIES = 3
THINKING_BUDGET = 8192     # extra completion tokens granted when thinking is enabled
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)

_lock = threading.RLock()  # reentrant: tokenizer()/_sampler() hold it while calling _svc()
_service = None
_samplers: Dict[str, Any] = {}
_tokenizers: Dict[str, Any] = {}


def _svc():
    global _service
    with _lock:
        if _service is None:
            if not os.environ.get("TINKER_API_KEY"):
                raise RuntimeError("TINKER_API_KEY is not set")
            import tinker
            _service = tinker.ServiceClient()
        return _service


def _sampler(model: str):
    with _lock:
        if model not in _samplers:
            _samplers[model] = _svc().create_sampling_client(base_model=model)
        return _samplers[model]


def tokenizer(model: str):
    """The model's tokenizer, obtained once through a rank-8 LoRA training client (never trained)."""
    with _lock:
        if model not in _tokenizers:
            tc = _svc().create_lora_training_client(base_model=model, rank=8)
            _tokenizers[model] = tc.get_tokenizer()
        return _tokenizers[model]


def render(model: str, messages: List[Dict[str, Any]], thinking: bool = False) -> str:
    """Apply the model's chat template to a native message list (passed verbatim)."""
    tok = tokenizer(model)
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


def count_tokens(model: str, messages: List[Dict[str, Any]], thinking: bool = False) -> int:
    """Context preflight: prompt token count of a message list under the model's own chat template."""
    return len(tokenizer(model).encode(render(model, messages, thinking)))


def sample_msgs(model: str, messages: List[Dict[str, Any]], max_tokens: int = 1024, temperature: float = 0.7,
                thinking: bool = False, seed: Optional[int] = None, timeout_s: int = TIMEOUT_S,
                heartbeat_s: int = HEARTBEAT_S, retries: int = RETRIES,
                thinking_budget: int = THINKING_BUDGET) -> Tuple[str, Dict[str, Any]]:
    """Native multi-turn sampling (no system prompt injected, no flattening) -> (text, info).

    ``info`` carries prompt/completion token counts and a finish reason ("length" when the budget was used up).
    """
    from tinker import types
    tok = tokenizer(model)
    ids = tok.encode(render(model, messages, thinking))
    prompt = types.ModelInput.from_ints(ids)
    budget = int(max_tokens) + (int(thinking_budget) if thinking else 0)
    params: Dict[str, Any] = {"max_tokens": budget, "temperature": float(temperature)}
    if seed is not None:
        params["seed"] = int(seed)
    last: Optional[BaseException] = None
    for attempt in range(retries):
        try:
            fut = _sampler(model).sample(prompt=prompt, num_samples=1,
                                         sampling_params=types.SamplingParams(**params))
            waited = 0
            while True:            # visible waiting, never a silent hang
                try:
                    seq = fut.result(timeout=heartbeat_s).sequences[0]
                    break
                except (cf.TimeoutError, TimeoutError):
                    waited += heartbeat_s
                    if waited >= timeout_s:
                        raise RuntimeError(f"sample wait exceeded {timeout_s}s")
                    print(f"[ucoe-tinker] {model}: queued/generating ({waited}s)...", flush=True)
            out = tok.decode(seq.tokens, skip_special_tokens=True)
            n_out = len(seq.tokens)
            info = {"prompt_tokens": len(ids), "completion_tokens": n_out,
                    "finish_reason": "length" if n_out >= budget else "stop"}
            return _THINK.sub("", out).strip(), info
        except Exception as e:  # noqa: BLE001 - bounded retry, then surface
            last = e
            print(f"[ucoe-tinker] {model}: attempt {attempt + 1} failed: {type(e).__name__}: {str(e)[:120]}",
                  flush=True)
    raise RuntimeError(f"tinker sample failed x{retries} for {model}: {last}")


def sample(model: str, system: str, user: str, max_tokens: int = 1024, temperature: float = 0.7,
           thinking: bool = False, seed: Optional[int] = None, **kwargs: Any) -> Tuple[str, Dict[str, Any]]:
    """Single-prompt sampling: an optional system turn plus one user turn."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return sample_msgs(model, msgs, max_tokens=max_tokens, temperature=temperature, thinking=thinking,
                       seed=seed, **kwargs)


class TinkerTransport(Transport):
    """Transport over Tinker. Config keys (all optional): timeout_s, heartbeat_s, retries, thinking_budget."""

    name = "tinker"

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or {}
        self.timeout_s = int(cfg.get("timeout_s", TIMEOUT_S))
        self.heartbeat_s = int(cfg.get("heartbeat_s", HEARTBEAT_S))
        self.retries = max(1, int(cfg.get("retries", RETRIES)))
        self.thinking_budget = int(cfg.get("thinking_budget", THINKING_BUDGET))
        try:
            import tinker  # noqa: F401 - prerequisite check before any stage starts
        except ImportError:
            raise SystemExit("[ucoe] models.backend: tinker needs `pip install tinker` (Python 3.11+); use "
                             "configs/smoke_mock.yaml for an offline check") from None
        if not os.environ.get("TINKER_API_KEY"):
            raise SystemExit("[ucoe] models.backend: tinker needs the TINKER_API_KEY environment variable")

    def __call__(self, model: str, system: str, user: str, kw: Dict[str, Any]) -> Dict[str, Any]:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        return self.call_messages(model, msgs, kw)

    def call_messages(self, model: str, messages: List[Dict[str, Any]], kw: Dict[str, Any]) -> Dict[str, Any]:
        k = normalize_kw(kw)
        text, info = sample_msgs(model, list(messages), max_tokens=int(k["max_tokens"]),
                                 temperature=float(k["temperature"]), thinking=bool(k.get("thinking")),
                                 seed=k.get("seed"), timeout_s=self.timeout_s, heartbeat_s=self.heartbeat_s,
                                 retries=self.retries, thinking_budget=self.thinking_budget)
        return make_result(text, finish_reason=info["finish_reason"], prompt_tokens=info["prompt_tokens"],
                           completion_tokens=info["completion_tokens"], model=model, sdk="tinker")
