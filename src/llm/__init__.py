"""LLM plumbing: transports (Tinker, OpenAI-compatible, mock), the cache-first metered client, structured JSON
calls and the order-preserving parallel map (paper App. "Implementation details").
"""
from .base import Transport, make_result, make_transport, normalize_kw
from .client import (Client, ContentInvalid, CostCeilingExceeded, LedgerIntegrityError, Meter, TransportFailure,
                     json_call, pmap, request_hash)

__all__ = [
    "Transport", "make_result", "make_transport", "normalize_kw",
    "Client", "ContentInvalid", "CostCeilingExceeded", "LedgerIntegrityError", "Meter", "TransportFailure",
    "json_call", "pmap", "request_hash",
]
