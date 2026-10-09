"""UCoE lifecycle orchestrator (paper Algorithm 1; Sec. 3.3 "Response revision and co-evolution").

``UCoERun`` owns one deployment: the Direct Generation arm (``work_dir/raw``), relation discovery
(``work_dir/relations``), the contrast banks (``work_dir/banks``) and the UCoE arm (``work_dir/ucoe``).

Per task (Algorithm 1 lines 5-18): the draft is the sealed Direct Generation response, byte-identical to the raw arm,
unless procedural memory retrieves guidelines at draft time (``ucoe.memory.at_draft``), in which case the actor
regenerates the draft with them; a draft-time retrieval counts as a use of the procedure without a win or a loss.
With no deployed verifier the draft is served and the harness state is untouched.  Otherwise the verifier scores the
draft, the weakest participating criterion is localized into a witness and one repair instruction, the actor makes one
local repair with the retrieved guidelines, the same verifier scores the revision, and the revision is served only when
its reward lead exceeds the margin.  Only an accepted repair may add one distilled procedure; rejected revisions
update usage statistics only.

Checkpoints (Algorithm 1 lines 20-36) run before the task whose index is listed in ``ucoe.checkpoints``: admission
while no verifier is deployed (adopt the maximin winner, reset the memory), promotion otherwise (the empirical
update rule); an adopted verifier prunes the memory to the criteria it still defines, extends the lineage and writes
``ckpt_<i>_report.json``.  Every stage resumes from ``rows.jsonl`` and ``state.json``.
"""

from __future__ import annotations

import inspect
import json
import os
import time
from typing import Dict, List, Optional, Tuple

from src.llm.base import make_transport
from src.llm.client import Client, Meter, pmap
from src.ucoe import contrasts, memory, relations, revision, scoring, selection, verifier
from src.utils.config import cfg_get
from src.utils.io import append_jsonl, atomic_write, ensure_dir, read_json, read_jsonl, sha, write_json

ROLE_CLASSES = ("actor", "verifier", "comparator", "judge")
RAW_CHUNK = 16


def _with_optional(fn, args: tuple, optional: dict):
    """Call ``fn(*args)`` adding only those keyword arguments its signature accepts."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(*args)
    var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
    kw = {k: v for k, v in optional.items() if var_kw or k in params}
    return fn(*args, **kw)


def _meter_summary(meter) -> str:
    return f"new={getattr(meter, 'new', None)} hits={getattr(meter, 'hits', None)}"


class UCoERun:
    """One UCoE deployment on one benchmark under one config (see the module docstring)."""

    def __init__(self, cfg: dict, bench, work_dir: str):
        self.cfg, self.bench, self.work_dir = cfg, bench, work_dir
        self.models = dict(cfg.get("models") or {})
        self.raw_dir = os.path.join(work_dir, "raw")
        self.ucoe_dir = os.path.join(work_dir, "ucoe")
        ensure_dir(self.raw_dir)
        ensure_dir(self.ucoe_dir)
        self.store = scoring.ScoreStore(os.path.join(self.ucoe_dir, "blockscores.json"))
        self.checkpoints = sorted(int(c) for c in (cfg_get(cfg, "ucoe.checkpoints", []) or []))
        self.eps = float(cfg_get(cfg, "ucoe.eps", 0.01))
        self.rule = str(cfg_get(cfg, "ucoe.update_rule", "relative"))
        self.rd_enabled = bool(cfg_get(cfg, "ucoe.relation_discovery.enabled", False))
        mem = cfg_get(cfg, "ucoe.memory", {}) or {}
        self.mem_cap = int(mem.get("capacity", getattr(memory, "MEM_CAP", 12)))
        self.mem_per_dim = int(mem.get("per_dimension", getattr(memory, "PER_DIM_CAP", 2)))
        self.mem_k = int(mem.get("max_retrieved", getattr(memory, "MAX_RETRIEVE", 2)))
        self.mem_tokens = int(mem.get("token_cap", getattr(memory, "TOKEN_CAP", 64)))
        self.mem_dedup = float(mem.get("dedup_jaccard", getattr(memory, "DEDUP_J", 0.75)))
        self.at_draft = bool(mem.get("at_draft", True))
        self.workers = int(cfg_get(cfg, "output.workers", 8) or 8)
        self.state_path = os.path.join(self.ucoe_dir, "state.json")
        self.rows_path = os.path.join(self.ucoe_dir, "rows.jsonl")
        self.raw_pool_path = os.path.join(self.raw_dir, "raw_pool.json")
        self.state: Optional[dict] = None
        self.rows: List[dict] = []
        self.raw_pool: Dict[str, str] = {}
        self._transport = None
        self._clients: Dict[str, Tuple[Client, Meter]] = {}

    # ------------------------------------------------------------ plumbing
    @property
    def transport(self):
        if self._transport is None:
            self._transport = make_transport(self.cfg["models"])
        return self._transport

    def client_for(self, arm: str) -> Tuple[Client, Meter]:
        """Cache-first client and cost meter of one arm directory (``raw`` or ``ucoe``)."""
        if arm not in self._clients:
            arm_dir = os.path.join(self.work_dir, arm)
            ensure_dir(arm_dir)
            ceilings = {k: v for k, v in (cfg_get(self.cfg, "output.ceilings", {}) or {}).items() if v is not None}
            meter = Meter(os.path.join(arm_dir, "cost_meter.json"), ceilings or None)
            role_models = {k: self.models[k] for k in ROLE_CLASSES if self.models.get(k)}
            client = Client(os.path.join(arm_dir, "llm_cache.jsonl"), self.transport, meter, role_models=role_models)
            self._clients[arm] = (client, meter)
        return self._clients[arm]

    def tasks(self, n: Optional[int] = None) -> List[dict]:
        tasks = list(self.bench.load_tasks())
        if n:
            tasks = tasks[: int(n)]
        if not tasks:
            raise RuntimeError("the benchmark adapter returned no deployment tasks")
        return tasks

    def _reward_view(self, task: dict) -> str:
        """Capped task text for the verifier-side prompts (the adapter's view when it defines one)."""
        if hasattr(self.bench, "reward_view"):
            return self.bench.reward_view(task)
        return task["task"][: int(self.bench.task_cap())]

    def _save_state(self) -> None:
        write_json(self.state_path, self.state)

    # ------------------------------------------------------------ Direct Generation
    def run_raw(self, n: Optional[int], workers: int) -> None:
        """One draft per task (no feedback); seals ``raw/raw_pool.json`` and writes ``raw/rows.jsonl``."""
        tasks = self.tasks(n)
        client, meter = self.client_for("raw")
        pool = read_json(self.raw_pool_path, None) or {}
        todo = [t for t in tasks if t["task_id"] not in pool]
        actor = self.models["actor"]
        print(f"[ucoe-raw] {len(pool)} drafts sealed, {len(todo)} to generate", flush=True)

        def one(task):
            text = relations.request_text(
                client, "actor", actor, self.bench.draft_request(task, ""), self.bench.gen_kw, f"raw|{task['task_id']}"
            )
            return task["task_id"], text

        for j in range(0, len(todo), RAW_CHUNK):
            pool.update(dict(pmap(one, todo[j : j + RAW_CHUNK], workers, "raw", every=RAW_CHUNK)))
            write_json(self.raw_pool_path, pool)
            print(f"  [ucoe-raw] {len(pool)}/{len(tasks)} {_meter_summary(meter)}", flush=True)
        rows_path = os.path.join(self.raw_dir, "rows.jsonl")
        rows = read_jsonl(rows_path)
        for i in range(len(rows), len(tasks)):
            text = pool[tasks[i]["task_id"]]
            append_jsonl(rows_path, {"i": i, "task_id": tasks[i]["task_id"], "final": text, "final_sha256": sha(text)})
        print(
            f"[ucoe-raw] sealed {len(tasks)} drafts ({sum(1 for t in tasks if pool[t['task_id']])} non-empty)",
            flush=True,
        )
        meter.snapshot("raw complete")

    # ------------------------------------------------------------ relation discovery
    def modules(self) -> List[dict]:
        """Frozen relation modules (empty when discovery is disabled or produced none)."""
        if not self.rd_enabled:
            return []
        got = read_json(os.path.join(relations.relations_dir(self.work_dir), "modules.json"), None) or {}
        return list(got.get("modules") or [])

    def rd_applicability(self, module: dict, task_text: str) -> bool:
        client, meter = self.client_for("ucoe")
        return relations.applicable(client, meter, self.store, self.bench, module, task_text)

    def usable_modules(self, workers: Optional[int] = None) -> List[dict]:
        """Modules whose relation family reached both stage-0 quotas (``relations/usable.json``)."""
        mods = self.modules()
        if not mods:
            return []
        up = os.path.join(relations.relations_dir(self.work_dir), "usable.json")
        got = read_json(up, None)
        if got is None:
            workers = workers or self.workers
            client, meter = self.client_for("ucoe")
            relation = {"modules": mods, "applicable": self.rd_applicability}
            banks = [
                contrasts.build_bank(
                    client, meter, self.bench, self.cfg, 0, kind, self.work_dir, workers, relation=relation
                )
                for kind in contrasts.KINDS
            ]
            quotas = {kind: contrasts.quota(self.cfg, kind, relation=True) for kind in contrasts.KINDS}
            usable = relations.constructibility_check(banks[0], banks[1], mods, quotas)
            got = {
                "modules": [m["id"] for m in mods],
                "quota": quotas,
                "usable": usable,
                "stage0_counts": {kind: relations.relation_counts(b) for kind, b in zip(contrasts.KINDS, banks)},
            }
            write_json(up, got, indent=1)
            print(
                f"[ucoe-relations] constructibility: usable {usable} "
                f"(stage-0 counts {got['stage0_counts']}, quota {quotas})",
                flush=True,
            )
            self.store.flush()
        usable = set(got.get("usable") or [])
        return [m for m in mods if m["id"] in usable]

    def run_discovery(self, workers: int) -> List[dict]:
        """Relation discovery and mining; registers the discovery prompts; records stage-0 constructibility."""
        if not self.rd_enabled:
            print("[ucoe-relations] relation discovery disabled (ucoe.relation_discovery.enabled: false)", flush=True)
            return []
        self.workers = workers
        client, meter = self.client_for("ucoe")
        n = int(cfg_get(self.cfg, "ucoe.relation_discovery.n_discovery_prompts", 32))
        srcs = list(self.bench.discovery_sources())[:n]
        contrasts.register_sources(self.work_dir, [s["source_id"] for s in srcs], [sha(s["task"]) for s in srcs])
        relations.discover(client, meter, self.bench, self.cfg, self.work_dir, workers)
        mods = relations.mine(client, meter, self.bench, self.cfg, self.work_dir)
        usable = self.usable_modules(workers)
        self.store.flush()
        meter.snapshot("discovery complete")
        print(f"[ucoe-relations] modules {[m['id'] for m in mods]}; usable {[m['id'] for m in usable]}", flush=True)
        return usable

    # ------------------------------------------------------------ banks
    def relation_for(self, stage: int, workers: Optional[int] = None) -> Optional[dict]:
        """Relation modules attached to the banks of a stage: all mined modules at stage 0, usable ones later."""
        mods = self.modules() if stage == 0 else self.usable_modules(workers)
        if not mods:
            return None
        return {"modules": mods, "applicable": self.rd_applicability}

    def banks_fn(self, stage: int, kind: str) -> List[dict]:
        """Clusters of one bank (built lazily, cached); relation clusters restricted to the usable families."""
        client, meter = self.client_for("ucoe")
        clusters = contrasts.build_bank(
            client,
            meter,
            self.bench,
            self.cfg,
            stage,
            kind,
            self.work_dir,
            self.workers,
            relation=self.relation_for(stage, self.workers),
        )
        usable = {m["id"] for m in self.usable_modules(self.workers)}
        return [c for c in clusters if c.get("group") != "relation" or c["family"] in usable]

    def reachable_checkpoints(self, n_tasks: int) -> List[int]:
        """Checkpoints the deployment loop reaches: those below the number of deployment tasks."""
        return [c for c in self.checkpoints if c < n_tasks]

    def build_banks(self, workers: int, n: Optional[int] = None) -> None:
        """Pre-build the banks of stage 0 and of every checkpoint the stream reaches, before the UCoE arm runs."""
        self.workers = workers
        n_tasks = len(self.tasks(n))
        stages = [0] + self.reachable_checkpoints(n_tasks)
        skipped = [c for c in self.checkpoints if c >= n_tasks]
        if skipped:
            print(
                f"[ucoe-banks] checkpoints {skipped} are not below the {n_tasks} deployment tasks; their banks are "
                f"not built",
                flush=True,
            )
        per_stage = sum(contrasts.quota(self.cfg, kind) for kind in contrasts.KINDS)
        fresh = not cfg_get(self.cfg, "ucoe.contrasts.reuse_sources_across_stages", False)
        need = per_stage * (len(stages) if fresh else 1)
        pool = len(self.bench.calibration_sources())
        if self.rd_enabled:
            n_disc = int(cfg_get(self.cfg, "ucoe.relation_discovery.n_discovery_prompts", 32))
            pool -= len(list(self.bench.discovery_sources())[:n_disc])
        policy = (
            f" x {len(stages)} stages = {need} (fresh sources at every stage)" if fresh
            else " (sources may recur across stages)"
        )
        print(
            f"[ucoe-banks] source budget: about {per_stage} distinct sources per stage{policy} against {pool} "
            f"calibration sources" + ("; WARNING: the budget exceeds the pool" if need > pool else ""),
            flush=True,
        )
        for stage in stages:
            for kind in contrasts.KINDS:
                clusters = self.banks_fn(stage, kind)
                counts: Dict[str, int] = {}
                for c in clusters:
                    counts[c["family"]] = counts.get(c["family"], 0) + 1
                print(
                    f"[ucoe-banks] bank {stage}/{kind}: {len(clusters)} clusters in use, per family {counts}",
                    flush=True,
                )
        self.store.flush()
        self.client_for("ucoe")[1].snapshot("banks complete")

    # ------------------------------------------------------------ verifier decisions
    def _decision_kwargs(self) -> dict:
        usable = self.usable_modules(self.workers)
        return {"rd_modules": usable or None, "rd_applicability": self.rd_applicability if usable else None}

    def _admission(self, stage: int, workers: int):
        client, meter = self.client_for("ucoe")
        self.state["extra"]["rd_modules"] = self.usable_modules(workers)
        return _with_optional(
            selection.admission,
            (client, meter, self.store, self.bench, self.cfg, self.state, stage, self.banks_fn, workers),
            {"rule": self.rule, **self._decision_kwargs()},
        )

    def _promotion(self, stage: int, workers: int):
        client, meter = self.client_for("ucoe")
        return _with_optional(
            selection.promotion,
            (client, meter, self.store, self.bench, self.cfg, self.state, stage, self.rows, self.banks_fn, workers),
            {"rule": self.rule, **self._decision_kwargs()},
        )

    def checkpoint(self, i: int, workers: int) -> None:
        """Admission while no verifier is deployed, promotion otherwise; adoption affects only later tasks."""
        st = self.state
        if st["V"] is None:
            Wn, rep = self._admission(i, workers)
            if Wn is not None:
                st["V"], st["memory"] = Wn, []
                st["lineage"].append(Wn["version"])
        else:
            newV, rep = self._promotion(i, workers)
            if newV is not None and newV["version"] != st["V"]["version"]:
                active = set(verifier.crit_ids(newV))
                st["V"] = newV
                st["lineage"].append(newV["version"])
                st["memory"] = [r for r in st["memory"] if r["criterion_id"] in active]
        st["ckpt"][str(i)] = {k: rep[k] for k in ("decision", "decision_rel", "decision_hard") if k in rep}
        write_json(os.path.join(self.ucoe_dir, f"ckpt_{i}_report.json"), rep, indent=1)
        self._save_state()
        self.store.flush()
        print(f"[ucoe] checkpoint {i}: {rep.get('decision')}", flush=True)

    # ------------------------------------------------------------ per-task step
    def step(self, task: dict, i: int) -> Tuple[dict, str]:
        """Algorithm 1 lines 5-18 for one task; returns (row fields, served response)."""
        st, V, bench = self.state, self.state["V"], self.bench
        client, meter = self.client_for("ucoe")
        pre = memory.mem_sha(st["memory"])
        crit_set = set(verifier.crit_ids(V)) if V is not None else set()
        ptags = set(bench.prompt_tags(task))
        rules_d = (
            memory.retrieve(st["memory"], None, None, crit_set, ptags, self.mem_k, self.mem_tokens)
            if self.at_draft and V is not None and st["memory"]
            else []
        )
        if rules_d:
            st["memory"] = memory.account(st["memory"], [r["rule_id"] for r in rules_d], None)
            guidelines = revision.guidelines_text(rules_d)
            y0 = relations.request_text(
                client,
                "actor",
                self.models["actor"],
                bench.draft_request(task, guidelines),
                bench.gen_kw,
                f"draft|{task['task_id']}|{sha(guidelines)[:8]}",
            )
            y0_src = "actor"
        else:
            y0, y0_src = str(self.raw_pool.get(task["task_id"]) or ""), "raw"
        f = {
            "V": V["version"] if V is not None else None,
            "y0": y0,
            "y0_src": y0_src,
            "yr": None,
            "b": 0,
            "weakest": None,
            "witness": None,
            "repair": None,
            "s0": None,
            "sR": None,
            "retrieved": [],
            "retrieved_draft": [r["rule_id"] for r in rules_d],
            "rule_status": None,
            "fallback": None,
            "proposal_noop": False,
            "pre_sha": pre,
            "mem_sha": pre,
            "mem_n": len(st["memory"]),
        }

        def done(y_final, **extra):
            f.update(extra)
            f["mem_sha"], f["mem_n"] = memory.mem_sha(st["memory"]), len(st["memory"])
            return f, y_final

        if not y0:
            return done("", fallback="source_empty")
        if V is None:
            return done(y0, incumbent="null")
        rd_app = self.rd_applicability if V.get("rd_modules") else None
        view = self._reward_view(task)
        vals = scoring.score_version(client, meter, self.store, bench, V, view, y0, rd_app)
        if vals is None:
            return done(y0, fallback="controller_invalid")
        s0 = verifier.agg_score(V, vals)
        weakest = revision.weakest_criterion(V, vals)
        loc = revision.localize(client, meter, bench, V, task, y0, vals, weakest)
        if loc is None:
            return done(y0, fallback="localize_invalid", weakest=weakest, s0=round(s0, 4))
        dim = bench.crit2dim.get(weakest, bench.dims[0] if bench.dims else None)
        rules = memory.retrieve(st["memory"], weakest, dim, crit_set, ptags, self.mem_k, self.mem_tokens)
        rids = [r["rule_id"] for r in rules]
        yr = revision.revise(client, bench, task, y0, loc["witness"], loc["repair"], revision.guidelines_text(rules))
        if not yr or yr == y0:
            st["memory"] = memory.account(st["memory"], rids, False)
            return done(
                y0,
                weakest=weakest,
                witness=loc["witness"],
                repair=loc["repair"],
                s0=round(s0, 4),
                retrieved=rids,
                proposal_noop=(yr == y0),
            )
        valsR = scoring.score_version(client, meter, self.store, bench, V, view, yr, rd_app)
        sR = verifier.agg_score(V, valsR) if valsR is not None else None
        b = int(revision.accept(s0, sR, self.eps))
        st["memory"] = memory.account(st["memory"], rids, bool(b))
        status = None
        if b:
            rule = memory.distill(client, meter, bench, task, y0, yr, loc["witness"], weakest, crit_set, rids=rids)
            if rule:
                st["memory"], status = memory.commit_rule(
                    st["memory"], rule, i, V["version"], self.mem_cap, self.mem_per_dim, self.mem_dedup
                )
            else:
                status = "invalid"
        return done(
            yr if b else y0,
            yr=yr,
            b=b,
            weakest=weakest,
            witness=loc["witness"],
            repair=loc["repair"],
            s0=round(s0, 4),
            sR=(round(sR, 4) if sR is not None else None),
            retrieved=rids,
            rule_status=status,
        )

    # ------------------------------------------------------------ the deployment loop
    def _load_rows(self) -> List[dict]:
        """Rows of the UCoE arm; a last row whose memory hash disagrees with the saved state is reprocessed."""
        rows = read_jsonl(self.rows_path)
        if rows and rows[-1].get("mem_sha") != memory.mem_sha(self.state["memory"]):
            rows = rows[:-1]
            atomic_write(self.rows_path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            print(
                "[ucoe] dropped the last row (memory hash differs from the saved state); it is replayed from the cache",
                flush=True,
            )
        return rows

    def run(self, n: Optional[int], workers: int) -> None:
        """H0 admission, then the per-task lifecycle with checkpoints; resumable from rows.jsonl and state.json."""
        self.workers = workers
        tasks = self.tasks(n)
        self.raw_pool = read_json(self.raw_pool_path, None)
        if self.raw_pool is None:
            raise RuntimeError("raw/raw_pool.json missing: run the raw stage first")
        missing = [t["task_id"] for t in tasks if t["task_id"] not in self.raw_pool]
        if missing:
            raise RuntimeError(
                f"raw pool lacks {len(missing)} deployment drafts (first {missing[:3]}): rerun the raw stage"
            )
        dead = [c for c in self.checkpoints if c >= len(tasks)]
        if dead:
            print(
                f"[ucoe] WARNING: checkpoints {dead} are not below the number of deployment tasks ({len(tasks)}) "
                f"and never run",
                flush=True,
            )
        client, meter = self.client_for("ucoe")
        self.state = read_json(self.state_path, None)
        if self.state is None:
            self.state = {"V": None, "memory": [], "lineage": ["null"], "ckpt": {}, "extra": {}}
            V0, rep = self._admission(0, workers)
            write_json(os.path.join(self.ucoe_dir, "h0_report.json"), rep, indent=1)
            if V0 is not None:
                self.state["V"] = V0
                self.state["lineage"].append(V0["version"])
            self.state["ckpt"]["0"] = {k: rep[k] for k in ("decision", "decision_rel", "decision_hard") if k in rep}
            self._save_state()
            self.store.flush()
            print(f"[ucoe] H0: {rep.get('decision')}", flush=True)
        self.rows = self._load_rows()
        t0, start = time.time(), len(self.rows)
        for i in range(len(self.rows), len(tasks)):
            if i in self.checkpoints and str(i) not in self.state["ckpt"]:
                self.checkpoint(i, workers)
            task = tasks[i]
            fields, y_final = self.step(task, i)
            row = {"i": i, "task_id": task["task_id"], **fields, "final": y_final, "final_sha256": sha(y_final)}
            append_jsonl(self.rows_path, row)
            self._save_state()
            self.store.flush()
            self.rows.append(row)
            if (i + 1) % 8 == 0 or i == len(tasks) - 1:
                rate = (time.time() - t0) / max(1, i + 1 - start)
                print(
                    f"  [ucoe] {i + 1}/{len(tasks)} ({rate:.0f}s/task) "
                    f"accepts={sum(1 for r in self.rows if r.get('b'))} "
                    f"V={(self.state['V'] or {}).get('version', 'null')} mem={len(self.state['memory'])} "
                    f"{_meter_summary(meter)}",
                    flush=True,
                )
        meter.snapshot("ucoe complete")

    # ------------------------------------------------------------ status
    def status(self) -> None:
        """Print what exists on disk for this run: raw drafts, banks, relations, rows, verifier, memory, meters."""
        pool = read_json(self.raw_pool_path, None) or {}
        raw_rows = read_jsonl(os.path.join(self.raw_dir, "rows.jsonl"))
        print(f"[ucoe-status] work_dir {self.work_dir}", flush=True)
        print(f"  raw: {len(pool)} drafts sealed, {len(raw_rows)} rows", flush=True)
        banks = []
        for stage in [0] + self.checkpoints:
            for kind in contrasts.KINDS:
                got = read_json(contrasts.bank_path(self.work_dir, stage, kind), None)
                if got is not None:
                    banks.append(f"{stage}/{kind}:{len(got.get('clusters', []))}")
        print(f"  banks: {' '.join(banks) if banks else '-'}", flush=True)
        if self.rd_enabled:
            usable = read_json(os.path.join(relations.relations_dir(self.work_dir), "usable.json"), None) or {}
            print(
                f"  relations: modules {[m['id'] for m in self.modules()]} usable {usable.get('usable', '-')}",
                flush=True,
            )
        state = read_json(self.state_path, None)
        rows = read_jsonl(self.rows_path)
        accepts = sum(1 for r in rows if r.get("b"))
        fallbacks: Dict[str, int] = {}
        for r in rows:
            if r.get("fallback"):
                fallbacks[r["fallback"]] = fallbacks.get(r["fallback"], 0) + 1
        print(f"  ucoe: {len(rows)} rows, {accepts} accepted revisions, fallbacks {fallbacks or '-'}", flush=True)
        if state is not None:
            print(
                f"  verifier: {(state.get('V') or {}).get('version', 'null')}; lineage {state.get('lineage')}; "
                f"memory {len(state.get('memory', []))} rules; checkpoints {state.get('ckpt')}",
                flush=True,
            )
        for arm in ("raw", "ucoe"):
            m = read_json(os.path.join(self.work_dir, arm, "cost_meter.json"), None)
            if m is not None:
                print(f"  meter[{arm}]: new={m.get('new')} hits={m.get('hits')}", flush=True)
