"""Offline tests of the contrast banks and relation-family construction (paper App. "Contrast-set construction").

A fake benchmark adapter (two reference families, two domains, a deterministic corrupter) and a fake cache-first
client (scripted JSON answers keyed by the prompt text) exercise: quota filling with per-domain balance, source
disjointness through the registry (deployment stream, discovery prompts, other banks), cached reload, the shortfall
floor, relation patches with the anonymized validator, mining with the static and semantic gates, and the
stage-0 constructibility check.  No network access.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import unittest

from src.ucoe import contrasts, relations
from src.utils.io import read_json, sha

SENTENCES = [
    "The answer starts with the main claim",
    "It then gives a supporting reason",
    "A worked example follows",
    "A caveat is stated for edge cases",
    "The closing sentence summarizes the result",
]
MODULE = {
    "id": "cause",
    "name": "Measure addresses stated cause",
    "relation": "each proposed measure addresses a cause the response itself stated",
    "definition": "cause: does each measure address a stated cause? "
    "(0: no measure is tied to a stated cause; 1: every measure is tied to a stated cause)",
    "applicability": "tasks that ask for recommendations",
    "applicability_keywords": ["recommend"],
    "break_recipe": "replace one measure with one unrelated to any stated cause",
    "preserve": "the stated causes and the other measures",
    "member_sources": ["s1", "s2", "s3"],
}


def base_text(user):
    return ". ".join(f"{s} for {user.split(chr(10))[0][:12]}" for s in SENTENCES) + "."


class FakeBench:
    name = "fake"
    families = ("omission", "injection")
    module_ids = ("co", "fc")
    det_crit = ("fc",)
    dims = ("correctness",)
    crit2dim = {"co": "correctness"}
    gen_kw = {"max_tokens": 64, "temperature": 0.7}

    def __init__(self, cfg, n_sources=60, domains=("code", "math")):
        self.cfg = cfg
        self.sources = [
            {
                "source_id": f"s{k}",
                "task": f"calibration prompt {k}\nrecommend a plan",
                "messages": None,
                "domain": domains[k % len(domains)] if domains else None,
            }
            for k in range(n_sources)
        ]
        self.sources[5]["task"] = "deployment task one"  # same text as a deployment task: must be excluded

    def load_tasks(self):
        return [{"task_id": "t1", "task": "deployment task one", "messages": None, "domain": None, "i": 0}]

    def calibration_sources(self):
        return list(self.sources)

    def discovery_sources(self):
        return self.sources[:4]

    def stage_families(self, stage):
        return self.families

    def draft_request(self, task, guidelines):
        return {"system": "sys", "user": task["task"] + guidelines}

    def valid_base(self, response):
        return bool(response) and "unusable" not in response

    def corrupt(self, family, source, response, rng):
        sents = [s for s in response.split(". ") if s]
        if family == "omission":
            return ". ".join(sents[:-2]) + "." if len(sents) >= 5 else None
        return response + " An external review reached the opposite conclusion."

    def task_cap(self):
        return 6000

    def prompt_tags(self, task):
        return {"general"}


class FakeMeter:
    def __init__(self):
        self.new, self.hits, self.transport_retries, self.syntax_retries = {}, {}, 0, 0

    def charge(self, cls):
        self.new[cls] = self.new.get(cls, 0) + 1

    def hit(self, cls):
        self.hits[cls] = self.hits.get(cls, 0) + 1

    def transport_retry(self):
        self.transport_retries += 1

    def syntax_retry(self):
        self.syntax_retries += 1

    def snapshot(self, label):
        return None


class FakeClient:
    """Scripted, cache-free client with the contract's call surface; ``responder(role, system, user)`` -> text."""

    ns = "test"

    def __init__(self, responder):
        self.responder, self.calls = responder, []

    def call(self, role, model, system, user, kw, nonce=0):
        self.calls.append((role, model, dict(kw), nonce, user))
        text = self.responder(role, system, user)
        return {
            "request_hash": sha(user),
            "role": role,
            "model": model,
            "text": text,
            "finish_reason": "stop",
            "usage": {},
            "model_meta": {},
        }

    def call_messages(self, role, model, messages, kw, nonce=0):
        return self.call(role, model, messages[0]["content"], messages[-1]["content"], kw, nonce)


class FakeStore:
    def __init__(self):
        self.d = {}

    def get(self, key):
        return self.d.get(key)

    def put(self, key, vals):
        self.d[key] = vals

    def flush(self):
        return None


def actor_responder(role, system, user):
    if "unusable" in user:
        return "unusable draft"
    return base_text(user)


def make_cfg(work_dir, **contrast_overrides):
    c = {
        "selection_per_family": 4,
        "heldout_per_family": 6,
        "relation_selection_per_family": 2,
        "relation_heldout_per_family": 2,
        "chunk": 4,
        "floor": 5.0 / 6.0,
    }
    c.update(contrast_overrides)
    return {
        "benchmark": {"name": "fake", "n_tasks": 1},
        "models": {
            "backend": "mock",
            "actor": "actor-model",
            "verifier": "verifier-model",
            "comparator": "cmp-model",
            "judge": "judge-model",
        },
        "ucoe": {
            "contrasts": c,
            "relation_discovery": {
                "enabled": True,
                "n_discovery_prompts": 4,
                "max_pairs": 4,
                "max_modules": 2,
                "min_recurrence": 3,
            },
        },
        "output": {"work_dir": work_dir, "smoke": False},
    }


def patch_responder(role, system, user):
    """Relation-side script: a verbatim 40+ char span replaced by an unrelated measure; validator reads the marks."""
    if role == "patch":
        m = re.search(r"<RESPONSE>\n(.*?)\n</RESPONSE>", user, re.S)
        sents = m.group(1).split(". ")
        quote = sents[1]
        return json.dumps(
            {
                "target_quote": quote,
                "positive_span": quote,
                "replacement": "It then proposes repainting the office walls in a neutral colour instead",
                "what_breaks": "the measure no longer addresses a stated cause",
            }
        )
    if role == "validate":
        x = re.search(r"<VERSION_X>\n(.*?)\n</VERSION_X>", user, re.S).group(1)
        marked = re.search(r"⟦(.*?)⟧", x, re.S).group(1)
        return json.dumps(
            {"relation_holds_in": "Y" if "repainting" in marked else "X", "other_attributes_damaged_in": False}
        )
    if role == "applicable":
        return json.dumps({"applies": True})
    return actor_responder(role, system, user)


class BankTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ucoe_contrasts_")
        self.cfg = make_cfg(self.tmp)
        self.bench = FakeBench(self.cfg)
        self.client, self.meter = FakeClient(actor_responder), FakeMeter()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _bank(self, stage, kind, relation=None, client=None, cfg=None, bench=None):
        return contrasts.build_bank(
            client or self.client,
            self.meter,
            bench or self.bench,
            cfg or self.cfg,
            stage,
            kind,
            self.tmp,
            2,
            relation=relation,
        )

    def test_quota_domains_registry_and_exclusions(self):
        clusters = self._bank(0, "select")
        per_fam = {f: sum(1 for c in clusters if c["family"] == f) for f in self.bench.families}
        self.assertEqual(per_fam, {"omission": 4, "injection": 4})
        for f in self.bench.families:
            doms = [c["domain"] for c in clusters if c["family"] == f]
            self.assertEqual(sorted(doms), ["code", "code", "math", "math"])
        for c in clusters:
            self.assertEqual(c["group"], "reference")
            self.assertNotEqual(c["pos"], c["neg"])
            self.assertNotEqual(c["task"], "deployment task one")
            self.assertNotIn(c["source_id"], {"s0", "s1", "s2", "s3"})  # discovery prompts stay out of every bank
            self.assertTrue(c["cluster_id"].startswith("0-select-"))
        used = {c["source_id"] for c in clusters}
        reg = contrasts.load_registry(self.tmp)
        self.assertEqual(set(reg["ids"]), used)
        self.assertTrue(set(reg["shas"]) >= {sha(c["task"]) for c in clusters})
        bank = read_json(contrasts.bank_path(self.tmp, 0, "select"))
        self.assertEqual(bank["totals"], {"omission": 4, "injection": 4})
        self.assertEqual(bank["shortfall"], {})
        n_calls = len(self.client.calls)
        self.assertEqual(self._bank(0, "select"), clusters)  # cached reload, no new calls
        self.assertEqual(len(self.client.calls), n_calls)

    def test_disjoint_across_banks_and_seeded_order(self):
        a = self._bank(0, "select")
        b = self._bank(0, "heldout")
        self.assertFalse({c["source_id"] for c in a} & {c["source_id"] for c in b})
        self.assertEqual(
            {f: sum(1 for c in b if c["family"] == f) for f in self.bench.families}, {"omission": 6, "injection": 6}
        )
        c = self._bank(1, "select")
        self.assertFalse({x["source_id"] for x in c} & ({x["source_id"] for x in a} | {x["source_id"] for x in b}))
        # the same config on an empty work dir reproduces the same source picks
        other = tempfile.mkdtemp(prefix="ucoe_contrasts2_")
        try:
            again = contrasts.build_bank(
                FakeClient(actor_responder), FakeMeter(), FakeBench(self.cfg), make_cfg(other), 0, "select", other, 2
            )
            self.assertEqual([x["source_id"] for x in again], [x["source_id"] for x in a])
        finally:
            shutil.rmtree(other, ignore_errors=True)

    def test_invalid_bases_are_skipped(self):
        bench = FakeBench(self.cfg)
        for s in bench.sources[4:12]:
            s["task"] += " unusable"
        clusters = self._bank(0, "select", bench=bench)
        self.assertEqual(len(clusters), 8)
        self.assertFalse(any("unusable" in c["pos"] for c in clusters))

    def test_shortfall_recorded_above_floor_and_error_below(self):
        bench = FakeBench(self.cfg, n_sources=10 + 4, domains=())  # 4 discovery prompts + 1 deployment duplicate
        cfg = make_cfg(self.tmp, selection_per_family=10)
        clusters = self._bank(0, "select", bench=bench, cfg=cfg)
        bank = read_json(contrasts.bank_path(self.tmp, 0, "select"))
        self.assertEqual(sum(1 for c in clusters if c["family"] == "omission"), 9)
        self.assertEqual(bank["shortfall"]["omission"]["all"], {"count": 9, "quota": 10})
        tiny = FakeBench(self.cfg, n_sources=7, domains=())
        with self.assertRaises(RuntimeError):
            contrasts.build_bank(
                self.client, self.meter, tiny, make_cfg(self.tmp, selection_per_family=10), 1, "select", self.tmp, 2
            )

    def test_smoke_quota_cap(self):
        cfg = make_cfg(self.tmp)
        cfg["output"]["smoke"] = True
        self.assertEqual(contrasts.quota(cfg, "select"), 2)
        self.assertEqual(contrasts.quota(cfg, "heldout"), 3)
        self.assertEqual(contrasts.quota(self.cfg, "heldout", relation=True), 2)
        with self.assertRaises(ValueError):
            contrasts.quota(cfg, "cert")

    def test_relation_families_in_bank(self):
        client = FakeClient(patch_responder)
        relation = {"modules": [MODULE], "applicable": lambda m, t: "recommend" in t}
        clusters = self._bank(0, "select", relation=relation, client=client)
        rel = [c for c in clusters if c["group"] == "relation"]
        self.assertEqual(
            {f: sum(1 for c in clusters if c["family"] == f) for f in ("omission", "injection", "cause")},
            {"omission": 4, "injection": 4, "cause": 2},
        )
        self.assertEqual(sorted(c["domain"] for c in rel), ["code", "math"])
        for c in rel:
            self.assertEqual(c["edit"]["kind"], "patch")
            self.assertIn("repainting", c["neg"])
            self.assertNotIn("repainting", c["pos"])
            self.assertEqual(c["edit"]["validator"]["relation_holds_in"], "Y" if c["edit"]["validator_flip"] else "X")
        self.assertEqual(relations.relation_counts(clusters), {"cause": 2})
        self.assertEqual(
            relations.constructibility_check(clusters, clusters, [MODULE], {"select": 2, "heldout": 2}), ["cause"]
        )
        self.assertEqual(
            relations.constructibility_check(clusters, clusters, [MODULE], {"select": 2, "heldout": 3}), []
        )


class RelationHelperTests(unittest.TestCase):
    def test_locate_patch_ok_mark(self):
        base = "alpha beta gamma. " + "x" * 50 + ". alpha beta gamma."
        self.assertIsNone(relations._locate(base, "alpha beta gamma"))  # ambiguous
        self.assertEqual(relations._locate(base, "x" * 50), "x" * 50)
        self.assertEqual(relations._locate("one   two three", "one two three"), "one   two three")
        quote = "x" * 50
        self.assertEqual(relations._patch_ok(base, "short", "a", "b"), "span length/uniqueness")
        self.assertEqual(relations._patch_ok(base, quote, "same words here", "same  words here"), "pos == neg")
        self.assertEqual(relations._patch_ok(base, quote, "the cat sat", "the cat ran"), "trivial pos/neg delta")
        self.assertTrue(
            relations._patch_ok(base, quote, "the cat sat", "the cat sat and then walked away quickly").startswith(
                "negative merely"
            )
        )
        self.assertIsNone(relations._patch_ok(base, quote, "the cat sat on the mat", "the dog ran in the park"))
        self.assertEqual(relations._mark("a b c", "b"), "a ⟦b⟧ c")
        self.assertTrue(
            relations._single_factor_valid({"relation_holds_in": "X", "other_attributes_damaged_in": False}, "X")
        )
        self.assertFalse(
            relations._single_factor_valid({"relation_holds_in": "X", "other_attributes_damaged_in": True}, "X")
        )
        self.assertFalse(
            relations._single_factor_valid({"relation_holds_in": "both", "other_attributes_damaged_in": False}, "X")
        )

    def test_module_static_issue(self):
        m = {k: v for k, v in MODULE.items() if k in relations.MODULE_TEXT_LIMITS or k == "applicability_keywords"}
        m.update(
            {
                "question": "does each measure address a stated cause?",
                "anchor0": "no measure is tied to a stated cause",
                "anchor1": "every measure is tied to a stated cause",
            }
        )
        self.assertIsNone(relations.module_static_issue(m, "cause"))
        bad = dict(m, break_recipe="Check whether the measure is tied to a cause")
        self.assertEqual(relations.module_static_issue(bad, "cause"), "break recipe is a check, not an edit")
        bad = dict(m, anchor1=m["anchor0"])
        self.assertEqual(relations.module_static_issue(bad, "cause"), "identical 0/1 anchors")
        bad = dict(m, relation="in 2023 the plan ...")
        self.assertIn("truncation", relations.module_static_issue(bad, "cause"))
        bad = dict(m, relation="the plan from 2023 matches")
        self.assertIn("task-specific", relations.module_static_issue(bad, "cause"))
        bad = dict(m, applicability_keywords=["ok", ""])
        self.assertEqual(relations.module_static_issue(bad, "cause"), "invalid applicability keywords")
        self.assertIsNone(relations.module_audit_issue({k: True for k in relations.MODULE_AUDIT_KEYS}))
        self.assertIn("same_axis", relations.module_audit_issue({"same_axis": False, "reason": "no"}))
        self.assertTrue(relations.REJECT_RE.search("rewards longer answers"))
        self.assertTrue(relations.PRESENCE_RE.search("includes an explanation of the code"))

    def test_module_id_normalization(self):
        used, taken = set(), {"co", "fc"}
        self.assertEqual(relations._module_id({"id": "cause"}, used, taken), "cause")
        self.assertEqual(
            relations._module_id({"id": "Bad Id", "name": "Measure addresses the stated cause"}, used, taken),
            "meas_add",
        )
        used.add("meas_add")
        self.assertEqual(
            relations._module_id({"id": "co", "name": "Measure addresses the stated cause"}, used, taken), "meas_a2"
        )


class DiscoveryMiningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ucoe_relations_")
        self.cfg = make_cfg(self.tmp)
        self.bench = FakeBench(self.cfg)
        self.meter, self.store = FakeMeter(), FakeStore()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _responder(self, role, system, user):
        if role == "discovery_comparator":
            return "Comparator answer. It is longer and names causes. " + base_text(user)
        if role == "screen":
            return json.dumps({"a_complete": True, "b_complete": True, "a_truncated": False, "b_truncated": False})
        if role == "mine":
            return json.dumps(
                {
                    "modules": [
                        {
                            "name": "Measure addresses stated cause",
                            "relation": MODULE["relation"],
                            "question": "does each measure address a stated cause?",
                            "anchor0": "no measure is tied to a stated cause",
                            "anchor1": "every measure is tied to a stated cause",
                            "applicability": "recommendation tasks",
                            "evidence_a": "ev a",
                            "evidence_b": "ev b",
                            "break_recipe": MODULE["break_recipe"],
                            "preserve": MODULE["preserve"],
                        }
                    ]
                }
            )
        if role == "cluster":
            return json.dumps(
                {
                    "modules": [
                        {
                            "id": "cause",
                            "name": "Measure addresses stated cause",
                            "relation": MODULE["relation"],
                            "question": "does each measure address a stated cause?",
                            "anchor0": "no measure is tied to a stated cause",
                            "anchor1": "every measure is tied to a stated cause",
                            "applicability_keywords": ["Recommend"],
                            "applicability": "recommendation tasks",
                            "break_recipe": MODULE["break_recipe"],
                            "preserve": MODULE["preserve"],
                            "member_sources": ["s0", "s1", "s2", "s3"],
                        },
                        {
                            "id": "verbose",
                            "name": "Longer explanation",
                            "relation": "the response is longer and more verbose",
                            "question": "is it longer?",
                            "anchor0": "short",
                            "anchor1": "long",
                            "applicability_keywords": [],
                            "applicability": "all",
                            "break_recipe": "shorten",
                            "preserve": "facts",
                            "member_sources": ["s0", "s1", "s2"],
                        },
                        {
                            "id": "thin",
                            "name": "Thin support",
                            "relation": "a claim is supported by a reason",
                            "question": "is each claim supported?",
                            "anchor0": "no",
                            "anchor1": "yes",
                            "applicability_keywords": [],
                            "applicability": "all",
                            "break_recipe": "delete the reason",
                            "preserve": "the claim",
                            "member_sources": ["s0"],
                        },
                    ],
                    "rejected": [],
                }
            )
        if role == "assign":
            return json.dumps({"member_sources": ["s0", "s1"]})
        if role == "module_audit":
            return json.dumps({k: True for k in relations.MODULE_AUDIT_KEYS})
        return actor_responder(role, system, user)

    def test_discover_and_mine(self):
        client = FakeClient(self._responder)
        got = relations.discover(client, self.meter, self.bench, self.cfg, self.tmp, 2)
        self.assertEqual(sorted(got["pairs"]), ["s0", "s1", "s2", "s3"])
        self.assertEqual(got["qualifying"], ["s0", "s1", "s2", "s3"])
        for sid, pr in got["pairs"].items():
            self.assertTrue(pr["qualifies"])
            actor_text = base_text(self.bench.sources[int(sid[1:])]["task"])
            self.assertEqual(pr["B" if pr["flip"] else "A"], actor_text)  # the seeded flip hides the system identity
        self.assertEqual({sid: len(v) for sid, v in got["proposals"].items()}, {"s0": 1, "s1": 1, "s2": 1, "s3": 1})
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "relations", "pairs.json")))
        mods = relations.mine(client, self.meter, self.bench, self.cfg, self.tmp)
        self.assertEqual([m["id"] for m in mods], ["cause"])
        self.assertEqual(
            mods[0]["definition"],
            "cause: does each measure address a stated cause? (0: no measure is tied to a stated cause; "
            "1: every measure is tied to a stated cause)",
        )
        self.assertEqual(mods[0]["applicability_keywords"], ["recommend"])
        man = read_json(os.path.join(self.tmp, "relations", "modules.json"))
        reasons = {r["name"]: r["reason"] for r in man["rejected"]}
        self.assertEqual(reasons["Longer explanation"], "length/wording/style-type module")
        self.assertEqual(reasons["Thin support"], "recurs on 2 < 3 sources")
        self.assertEqual(man["assignment_fallback"], {"Thin support": ["s0", "s1"]})
        self.assertEqual(relations.mine(client, self.meter, self.bench, self.cfg, self.tmp), mods)  # frozen
        self.assertTrue(relations.applicable(client, self.meter, self.store, self.bench, mods[0], "please recommend"))
        self.assertEqual(self.store.d, {})
        self.assertFalse(relations.applicable(client, FakeMeter(), self.store, self.bench, mods[0], "no keyword here"))
        self.assertEqual(list(self.store.d.values()), [{"applies": False}])

    def test_mine_rejects_everything_then_freezes_empty(self):
        def responder(role, system, user):
            if role == "cluster":
                return "not json at all"
            return self._responder(role, system, user)

        client = FakeClient(responder)
        relations.discover(client, self.meter, self.bench, self.cfg, self.tmp, 2)
        self.assertEqual(relations.mine(client, self.meter, self.bench, self.cfg, self.tmp), [])
        man = read_json(os.path.join(self.tmp, "relations", "modules.json"))
        self.assertEqual(len(man["mining_attempts"]), 2)
        cluster_prompts = [c[4] for c in client.calls if c[0] == "cluster"]  # two attempts, one nonce retry each
        self.assertEqual(len(set(cluster_prompts)), 2)
        self.assertNotIn("Repair the first cluster attempt", cluster_prompts[0])
        self.assertIn("Repair the first cluster attempt", cluster_prompts[-1])

    def test_patch_and_null_contrasts(self):
        client = FakeClient(patch_responder)
        src = self.bench.sources[7]
        y = base_text(src["task"])
        c = relations.patch_contrast(client, self.meter, self.bench, MODULE, src, y, "0-select")
        self.assertIsNotNone(c)
        self.assertEqual((c["group"], c["family"], c["source_id"]), ("relation", "cause", "s7"))
        self.assertEqual(c["pos"], y)
        self.assertEqual(c["edit"]["positive_repaired"], 0)
        self.assertEqual(c["neg"].count("repainting"), 1)
        rejecting = FakeClient(
            lambda role, s, u: (
                json.dumps({"relation_holds_in": "both", "other_attributes_damaged_in": False})
                if role == "validate"
                else patch_responder(role, s, u)
            )
        )
        self.assertIsNone(
            relations.patch_contrast(rejecting, self.meter, self.bench, MODULE, src, y, "0-select", nonce_budget=2)
        )
        self.assertEqual(sum(1 for x in rejecting.calls if x[0] == "patch"), 2)

        def null_responder(role, s, u):
            q = base_text(src["task"]).split(". ")[2]
            return json.dumps(
                {"target_quote": q, "paraphrase": "An illustrative example comes next for the same prompt"}
            )

        n = relations.null_contrast(FakeClient(null_responder), self.meter, self.bench, src, y, "0-select")
        self.assertEqual((n["group"], n["family"]), ("null", "null"))
        self.assertIn("illustrative", n["neg"])
        self.assertEqual(n["pos"], y)


if __name__ == "__main__":
    unittest.main()
