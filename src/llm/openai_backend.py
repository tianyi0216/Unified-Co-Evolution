"""OpenAI-compatible chat-completions backend.

Serves any role through an OpenAI-style endpoint (paper App. "Implementation details"): a local vLLM server
hosting the open-weight models, OpenAI, or Azure OpenAI. Config keys under ``models.openai``: ``base_url``
(optional), ``api_key_env`` (default OPENAI_API_KEY; a local server needs no key), ``api_key`` (literal,
optional), ``extra_body`` (a non-empty mapping merged into every request; when given, the per-call ``thinking``
flag is mapped onto ``chat_template_kwargs.enable_thinking`` as vLLM expects; absent or ``{}`` sends nothing
extra and ignores the flag, since the OpenAI and Azure APIs reject unknown request fields),
``azure_api_version`` (switches to the Azure client with ``base_url`` as endpoint), ``timeout_s``,
``retries``, ``strip_think``, ``max_tokens_param`` ("max_tokens" or "max_completion_tokens") and
``send_temperature``. ``json_mode`` maps to ``response_format={"type": "json_object"}``.
The ``openai`` package is imported lazily on first use.
"""
from __future__ import annotations

import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

from .base import Transport, make_result, normalize_kw

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class OpenAITransport(Transport):
    name = "openai"

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = dict(cfg or {})
        self.base_url: Optional[str] = cfg.get("base_url") or None
        self.api_key_env = str(cfg.get("api_key_env") or "OPENAI_API_KEY")
        self.api_key: Optional[str] = cfg.get("api_key") or None
        extra = cfg.get("extra_body")
        self.extra_body: Optional[Dict[str, Any]] = dict(extra) if isinstance(extra, dict) else None
        self.azure_api_version: Optional[str] = cfg.get("azure_api_version") or None
        self.timeout_s = float(cfg.get("timeout_s", 300))
        self.retries = max(1, int(cfg.get("retries", 3)))
        self.strip_think = bool(cfg.get("strip_think", True))
        self.max_tokens_param = str(cfg.get("max_tokens_param", "max_tokens"))
        self.send_temperature = bool(cfg.get("send_temperature", True))
        self._sdk = None
        self._lock = threading.Lock()
        try:
            import openai  # noqa: F401 - prerequisite check before any stage starts
        except ImportError:
            raise SystemExit("[ucoe] models.backend: openai needs `pip install openai`; use configs/smoke_mock.yaml "
                             "for an offline check") from None
        if not (self.api_key or os.environ.get(self.api_key_env) or self.base_url):
            raise SystemExit(f"[ucoe] models.backend: openai needs the {self.api_key_env} environment variable (or "
                             f"models.openai.api_key / base_url for a local server)")

    # ------------------------------------------------------------------ SDK

    def _client(self):
        with self._lock:
            if self._sdk is None:
                import openai
                key = self.api_key or os.environ.get(self.api_key_env) or ("EMPTY" if self.base_url else None)
                if not key:
                    raise RuntimeError(f"{self.api_key_env} is not set and no models.openai.api_key is configured")
                if self.azure_api_version:
                    self._sdk = openai.AzureOpenAI(api_version=self.azure_api_version,
                                                   azure_endpoint=self.base_url, api_key=key,
                                                   max_retries=0, timeout=self.timeout_s)
                else:
                    self._sdk = openai.OpenAI(base_url=self.base_url, api_key=key, max_retries=0,
                                              timeout=self.timeout_s)
            return self._sdk

    @staticmethod
    def _sdk_version() -> str:
        try:
            import openai
            return getattr(openai, "__version__", "unknown")
        except ImportError:
            return "not-installed"

    # ------------------------------------------------------------------ requests

    def _params(self, model: str, messages: List[Dict[str, Any]], k: Dict[str, Any]) -> Dict[str, Any]:
        params: Dict[str, Any] = {"model": model, "messages": list(messages),
                                  self.max_tokens_param: int(k["max_tokens"])}
        if self.send_temperature:
            params["temperature"] = float(k["temperature"])
        if k.get("seed") is not None:
            params["seed"] = int(k["seed"])
        if k.get("json_mode"):
            params["response_format"] = {"type": "json_object"}
        if self.extra_body:
            body = dict(self.extra_body)
            if "thinking" in k:
                ctk = dict(body.get("chat_template_kwargs") or {})
                ctk["enable_thinking"] = bool(k["thinking"])
                body["chat_template_kwargs"] = ctk
            params["extra_body"] = body
        return params

    def _result(self, resp: Any, model: str) -> Dict[str, Any]:
        choice = resp.choices[0]
        text = choice.message.content or ""
        if self.strip_think:
            text = _THINK.sub("", text)
        usage = getattr(resp, "usage", None)
        details = getattr(usage, "completion_tokens_details", None) if usage is not None else None
        return make_result(text.strip(), finish_reason=getattr(choice, "finish_reason", None),
                           prompt_tokens=getattr(usage, "prompt_tokens", None),
                           completion_tokens=getattr(usage, "completion_tokens", None),
                           reasoning_tokens=getattr(details, "reasoning_tokens", None),
                           model=getattr(resp, "model", None) or model, sdk=f"openai-{self._sdk_version()}",
                           request_id=getattr(resp, "id", "") or "")

    def __call__(self, model: str, system: str, user: str, kw: Dict[str, Any]) -> Dict[str, Any]:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        return self.call_messages(model, msgs, kw)

    def call_messages(self, model: str, messages: List[Dict[str, Any]], kw: Dict[str, Any]) -> Dict[str, Any]:
        """Send the messages verbatim; 3 attempts with backoff, longer sleeps on rate limits."""
        params = self._params(model, messages, normalize_kw(kw))
        last: Optional[BaseException] = None
        for attempt in range(self.retries):
            try:
                return self._result(self._client().chat.completions.create(**params), model)
            except Exception as e:  # noqa: BLE001 - bounded retry, then surface
                last = e
                name = type(e).__name__
                rate_limited = "RateLimit" in name or "429" in str(e)
                print(f"[ucoe-openai] {model}: attempt {attempt + 1} failed: {name}: {str(e)[:160]}", flush=True)
                if attempt + 1 < self.retries:
                    time.sleep(min(120.0, 15.0 * (attempt + 1)) if rate_limited else 2.0 * (attempt + 1))
        raise RuntimeError(f"openai transport failed x{self.retries} for {model}: {last}")
