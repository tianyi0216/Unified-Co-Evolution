"""Response-contrast construction (paper Sec. 3.2 "Response-contrast construction"; App. "Contrast-set construction").

A contrast pairs a base response y+ (the actor's own answer to a public calibration prompt that is disjoint from the
deployment stream) with a degraded copy y- that differs in exactly one declared property.  ``build_bank`` renders
the base responses, applies one edit per reference family through the benchmark adapter's corrupter and, when
relation modules are attached, one validated local patch per relation family (``relations.patch_contrast``).

Conventions implemented here:
  * one bank per (stage, kind) with kind in {"select", "heldout"}; the file ``banks/bank_<stage>_<kind>.json`` is
    the frozen record, and a global registry ``banks/registry.json`` (together with the ``sources_used`` of every
    bank file already on disk) keeps every bank's sources disjoint from each other, from the deployment tasks and
    from the discovery prompts (``contrasts.reuse_sources_across_stages`` relaxes this to within-stage
    disjointness);
  * sources are walked in a seeded order in chunks; a source contributes at most one contrast per family; families
    are filled in order of largest deficit; sources that carry a domain receive balanced per-domain quotas;
  * on pool exhaustion a reference family below the floor (5/6 of its quota) raises; otherwise the shortfall is
    recorded and the run continues without backfilling, as it always does for relation families.
"""

from __future__ import annotations

import glob
import os
import random
from typing import Dict, List, Optional

from src.llm.client import pmap
from src.ucoe import relations
from src.utils.config import cfg_get
from src.utils.io import ensure_dir, read_json, seed_from, sha, write_json

DEFAULT_CONTRASTS = {
    "selection_per_family": 24,
    "heldout_per_family": 96,
    "relation_selection_per_family": 12,
    "relation_heldout_per_family": 24,
    "chunk": 16,
    "floor": 5.0 / 6.0,
}
SMOKE_QUOTA = {"select": 2, "heldout": 3}
RELATION_WALK_FACTOR = 6  # extra sources walked for relation families alone, per module and per quota unit
KINDS = ("select", "heldout")


# ---------------------------------------------------------------- paths, registry, quotas
def bank_path(work_dir: str, stage: int, kind: str) -> str:
    return os.path.join(work_dir, "banks", f"bank_{stage}_{kind}.json")


def registry_path(work_dir: str) -> str:
    return os.path.join(work_dir, "banks", "registry.json")


def load_registry(work_dir: str) -> dict:
    reg = read_json(registry_path(work_dir), None) or {}
    return {"ids": list(reg.get("ids", [])), "shas": list(reg.get("shas", []))}


def register_sources(work_dir: str, ids, shas) -> dict:
    """Add source ids / task shas to the global registry so no later bank reuses them."""
    reg = load_registry(work_dir)
    reg = {"ids": sorted(set(reg["ids"]) | set(ids)), "shas": sorted(set(reg["shas"]) | set(shas))}
    ensure_dir(os.path.join(work_dir, "banks"))
    write_json(registry_path(work_dir), reg)
    return reg


def quota(cfg: dict, kind: str, relation: bool = False) -> int:
    """Per-family source quota of a bank kind (smoke mode caps it at 2 selection / 3 held-out)."""
    if kind not in KINDS:
        raise ValueError(f"unknown bank kind {kind!r}; expected one of {KINDS}")
    key = ("relation_" if relation else "") + ("selection" if kind == "select" else "heldout") + "_per_family"
    q = int(cfg_get(cfg, f"ucoe.contrasts.{key}", DEFAULT_CONTRASTS[key]))
    if cfg_get(cfg, "output.smoke", False):
        q = min(q, SMOKE_QUOTA[kind])
    return q


def _split_quota(total: int, domains: List[str]) -> Dict[str, int]:
    n = len(domains)
    return {d: total // n + (1 if k < total % n else 0) for k, d in enumerate(domains)}


def _exclusions(bench, cfg, work_dir, stage, kind):
    """Source ids and task shas that this bank may not use."""
    ids, shas = set(), set()
    if cfg_get(cfg, "ucoe.contrasts.reuse_sources_across_stages", False):
        other = read_json(bank_path(work_dir, stage, "heldout" if kind == "select" else "select"), None) or {}
        for c in other.get("clusters", []):
            ids.add(c["source_id"])
            shas.add(sha(c["task"]))
    else:
        reg = load_registry(work_dir)
        ids.update(reg["ids"])
        shas.update(reg["shas"])
        for p in sorted(glob.glob(os.path.join(work_dir, "banks", "bank_*_*.json"))):
            for c in (read_json(p, None) or {}).get("clusters", []):  # banks sealed before their registration
                ids.add(c["source_id"])
                shas.add(sha(c["task"]))
    shas.update(sha(t["task"]) for t in bench.load_tasks())
    if cfg_get(cfg, "ucoe.relation_discovery.enabled", False):
        for s in bench.discovery_sources():
            ids.add(s["source_id"])
            shas.add(sha(s["task"]))
    return ids, shas


# ---------------------------------------------------------------- the builder
class _Builder:
    """State of one bank build: counts per family and domain, clusters, registered sources, shortfalls."""

    def __init__(self, client, meter, bench, cfg, stage, kind, workers, relation):
        self.client, self.meter, self.bench, self.cfg = client, meter, bench, cfg
        self.stage, self.kind, self.workers = stage, kind, max(1, int(workers or 1))
        self.fams = tuple(bench.stage_families(stage))
        self.order = {f: k for k, f in enumerate(self.fams)}
        self.modules = list((relation or {}).get("modules") or [])
        self.applicable = (relation or {}).get("applicable")
        self.rel_ids = [m["id"] for m in self.modules]
        self.chunk = max(1, int(cfg_get(cfg, "ucoe.contrasts.chunk", DEFAULT_CONTRASTS["chunk"])))
        self.floor = float(cfg_get(cfg, "ucoe.contrasts.floor", DEFAULT_CONTRASTS["floor"]))
        self.actor = relations.model_for(bench, "actor", cfg)
        self.kw = getattr(bench, "bank_kw", None) or bench.gen_kw  # base responses: the actor's draft decoding
        self.ns = str(cfg_get(cfg, "benchmark.name", getattr(bench, "name", "ucoe")))
        self.split = f"{stage}-{kind}"
        self.clusters: List[dict] = []
        self.counts: Dict[str, Dict[str, int]] = {f: {} for f in list(self.fams) + self.rel_ids}
        self.shortfall: Dict[str, Dict[str, dict]] = {}
        self.used_ids, self.used_shas = set(), set()

    def render(self, src):
        text = relations.request_text(
            self.client,
            "bank_writer",
            self.actor,
            self.bench.draft_request(src, ""),
            self.kw,
            f"bank|{self.stage}|{self.kind}|{src['source_id']}",
        )
        return src["source_id"], text

    def _open(self, counts, need):
        return any(counts[f] < need[f] for f in need)

    def _record_shortfall(self, dom, counts, need, hard):
        bad = [f for f in need if counts[f] < need[f]]
        below = [f for f in bad if counts[f] < int(need[f] * self.floor + 1e-9)]
        if hard and below:
            raise RuntimeError(
                f"bank {self.stage}/{self.kind}/{dom}: pool exhausted with families below the floor "
                f"{ {f: (counts[f], need[f]) for f in below} }"
            )
        for f in bad:
            self.shortfall.setdefault(f, {})[dom] = {"count": counts[f], "quota": need[f]}
            print(
                f"[ucoe-banks] bank {self.stage}/{self.kind}/{dom}/{f}: SHORTFALL {counts[f]}/{need[f]} "
                f"(pool exhausted; recorded, no backfill)",
                flush=True,
            )

    def walk(self, dom: str, docs: List[dict], q_ref: int, q_rel: int) -> None:
        """Fill the reference and relation families of one domain pool from a seeded source order."""
        docs = list(docs)
        random.Random(seed_from(self.ns, "bankpick", f"{self.stage}|{self.kind}|{dom}")).shuffle(docs)
        need_ref = {f: q_ref for f in self.fams}
        need_rel = {m: q_rel for m in self.rel_ids}
        c_ref = {f: 0 for f in self.fams}
        c_rel = {m: 0 for m in self.rel_ids}
        ptr, extra = 0, None
        while self._open(c_ref, need_ref) or self._open(c_rel, need_rel):
            if ptr >= len(docs):
                self._record_shortfall(dom, c_ref, need_ref, hard=True)
                self._record_shortfall(dom, c_rel, need_rel, hard=False)
                break
            if not self._open(c_ref, need_ref):
                if extra is None:
                    extra = RELATION_WALK_FACTOR * q_rel * len(self.rel_ids)
                if extra <= 0:
                    self._record_shortfall(dom, c_rel, need_rel, hard=False)
                    break
            want = sum(need_ref[f] - c_ref[f] for f in self.fams) + sum(need_rel[m] - c_rel[m] for m in self.rel_ids)
            chunk = docs[ptr : ptr + min(self.chunk, max(2, 2 * want))]
            ptr += len(chunk)
            rendered = dict(
                pmap(self.render, chunk, self.workers, f"bank_{self.stage}_{self.kind}_{dom}", every=self.chunk)
            )
            for src in chunk:
                if not (self._open(c_ref, need_ref) or self._open(c_rel, need_rel)):
                    break
                if extra is not None:
                    extra -= 1
                y = rendered.get(src["source_id"])
                if not y or not self.bench.valid_base(y):
                    continue
                made = self._reference(dom, src, y, c_ref, need_ref)
                made = self._relation(dom, src, y, c_rel, need_rel) or made
                if made:
                    self.used_ids.add(src["source_id"])
                    self.used_shas.add(sha(src["task"]))
        for f in self.fams:
            self.counts[f][dom] = c_ref[f]
        for m in self.rel_ids:
            self.counts[m][dom] = c_rel[m]

    def _reference(self, dom, src, y, c_ref, need_ref) -> bool:
        made = False
        for fam in sorted(self.fams, key=lambda f: (c_ref[f], self.order[f])):
            if c_ref[fam] >= need_ref[fam]:
                continue
            rng = random.Random(seed_from(self.ns, "bankcorrupt", f"{self.stage}|{self.kind}|{fam}|{src['source_id']}"))
            make = getattr(self.bench, "contrast_pair", None)
            if make is not None:
                pair = make(fam, src, y, rng)
            else:  # an adapter-like object with only a corrupter: y+ is the base response
                neg = self.bench.corrupt(fam, src, y, rng)
                pair = (y, neg) if neg else None
            if not pair:
                continue
            pos, neg = pair
            if not pos or not neg or neg == pos:
                continue
            c_ref[fam] += 1
            self.clusters.append(
                {
                    "cluster_id": f"{self.split}-{dom}-{fam}-{c_ref[fam]}",
                    "family": fam,
                    "group": "reference",
                    "source_id": src["source_id"],
                    "domain": src.get("domain"),
                    "task": src["task"],
                    "messages": src.get("messages"),
                    "pos": pos,
                    "neg": neg,
                }
            )
            made = True
        return made

    def _relation(self, dom, src, y, c_rel, need_rel) -> bool:
        made = False
        for m in self.modules:
            mid = m["id"]
            if c_rel[mid] >= need_rel[mid] or not self.applicable(m, src["task"]):
                continue
            c = relations.patch_contrast(self.client, self.meter, self.bench, m, src, y, self.split)
            if c is None:
                continue
            c_rel[mid] += 1
            c["cluster_id"] = f"{self.split}-{dom}-{mid}-{c_rel[mid]}"
            self.clusters.append(c)
            made = True
        return made


def build_bank(
    client, meter, bench, cfg, stage: int, kind: str, work_dir: str, workers: int, relation: Optional[dict] = None
) -> List[dict]:
    """Build (or load) the contrast bank of one stage and kind; returns its clusters.

    ``relation`` = {"modules": [RelationModule], "applicable": callable(module, task_text) -> bool} attaches the
    relation families; each module id becomes one family with the relation quotas of the config.
    """
    path = bank_path(work_dir, stage, kind)
    cached = read_json(path, None)
    if cached is not None:
        return list(cached["clusters"])
    ensure_dir(os.path.dirname(path))
    b = _Builder(client, meter, bench, cfg, stage, kind, workers, relation)
    q_ref, q_rel = quota(cfg, kind), (quota(cfg, kind, relation=True) if b.modules else 0)
    ex_ids, ex_shas = _exclusions(bench, cfg, work_dir, stage, kind)
    pools: Dict[str, List[dict]] = {}
    for s in bench.calibration_sources():
        if s["source_id"] in ex_ids or sha(s["task"]) in ex_shas:
            continue
        pools.setdefault(s.get("domain") or "all", []).append(s)
    domains = sorted(pools)
    if not domains:
        raise RuntimeError(
            f"bank {stage}/{kind}: no calibration sources left outside the registry and the " f"deployment stream"
        )
    per_ref, per_rel = _split_quota(q_ref, domains), _split_quota(q_rel, domains)
    for dom in domains:
        b.walk(dom, pools[dom], per_ref[dom], per_rel[dom])
    totals = {f: sum(v.values()) for f, v in b.counts.items()}
    bank = {
        "stage": stage,
        "kind": kind,
        "families": list(b.fams),
        "relation_families": b.rel_ids,
        "domains": domains,
        "quota": {"reference": q_ref, "relation": q_rel, "per_domain": {"reference": per_ref, "relation": per_rel}},
        "counts": b.counts,
        "totals": totals,
        "shortfall": b.shortfall,
        "sources_used": sorted(b.used_ids),
        "clusters": b.clusters,
    }
    write_json(path, bank)
    register_sources(work_dir, b.used_ids, b.used_shas)
    print(
        f"[ucoe-banks] bank {stage}/{kind}: {len(b.clusters)} clusters; per family {totals}"
        + (f"; shortfall {b.shortfall}" if b.shortfall else ""),
        flush=True,
    )
    return b.clusters
