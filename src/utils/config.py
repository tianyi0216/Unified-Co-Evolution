"""YAML configuration loading with ``${a.b.c}`` interpolation.

Every run is fully specified by one config file (paper App. "Implementation details": benchmark, models,
decoding settings, UCoE hyper-parameters, evaluation protocol and output directory). String values may refer
to other keys with ``${dotted.path}`` placeholders, which are resolved recursively after loading, so a preset
can derive e.g. ``output.work_dir`` from ``benchmark.name``.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

_REF = re.compile(r"\$\{([^}]+)\}")


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config and resolve ``${dotted.path}`` interpolations inside string values."""
    try:
        import yaml
    except ImportError as e:  # pragma: no cover - dependency guard
        raise SystemExit("PyYAML is required: pip install pyyaml") from e
    if not os.path.exists(path):
        raise SystemExit(f"config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"config file must contain a mapping at top level: {path}")
    return _resolve_refs(cfg, cfg)


def cfg_get(cfg: Dict[str, Any], path: str, default: Optional[Any] = None) -> Any:
    """Read a nested key with dotted notation, e.g. ``cfg_get(cfg, "ucoe.contrasts.selection_per_family")``."""
    cur: Any = cfg
    for key in path.split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def cfg_set(cfg: Dict[str, Any], path: str, value: Any) -> Dict[str, Any]:
    """Set a nested key with dotted notation, creating intermediate mappings; returns ``cfg``."""
    keys = path.split(".")
    cur = cfg
    for key in keys[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[keys[-1]] = value
    return cfg


def _resolve_refs(value: Any, root: Dict[str, Any]) -> Any:
    """Recursively resolve ``${path.to.key}`` placeholders inside string values (unknown refs are kept)."""
    if isinstance(value, str):
        def _sub(match: "re.Match") -> str:
            ref = cfg_get(root, match.group(1))
            return str(ref) if ref is not None else match.group(0)

        prev: Optional[str] = None
        cur = value
        while cur != prev:
            prev = cur
            cur = _REF.sub(_sub, cur)
        return cur
    if isinstance(value, dict):
        return {k: _resolve_refs(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_refs(v, root) for v in value]
    return value
