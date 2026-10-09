"""Unit tests of scoring, driving-capacity selection and the empirical update rule (paper Sec. 3.2-3.3, App. B.3
Eq. empirical-admission) with a scripted verifier model: block scores depend on marker words in the response, so
candidates with different criteria separate different contrast families."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from src.ucoe import scoring, selection, synthesis, verifier


class FakeMeter:
    def __init__(self):
        self.syntax_retries = 0

    def syntax_retry(self):
        self.syntax_retries += 1

    def snapshot(self, label):
        return None


class FakeClient:
    """Scores every criterion 0.9 for a 'good' response and 0.4 otherwise; the criterion 'nw' scores 0.2 when the
    response is 'messy'; responses containing 'invalid' get no parseable JSON."""

    ns = "test"

    def __init__(self):
        self.calls = 0

    def call(self, role, model, system, user, kw, nonce=0):
        self.calls += 1
        req = json.loads(user)
        text = req["response"]
        if "invalid" in text:
            return {"text": "no json here", "finish_reason": "stop", "usage": {}, "model_meta": {}}
        q = 0.9 if "good" in text else 0.4
        nw = 0.2 if "messy" in text else 0.9
        return {"text": json.dumps({cid: (nw if cid == "nw" else q) for cid in req["ids"]}),
                "finish_reason": "stop", "usage": {}, "model_meta": {}}


class FakeBench:
    name = "fake"
    families = ("f1", "f2")
    module_ids = ("co", "ia")
    module_defs = {"co": "co: correct? (0: no; 1: yes)", "ia": "ia: instructed? (0: no; 1: yes)"}
    det_crit = ()
    dims = ("correctness",)
    crit2dim = {"co": "correctness", "ia": "correctness"}
    score_kw = {"max_tokens": 64, "temperature": 0.0}
    synth_kw = {"max_tokens": 64, "temperature": 0.0}
    synth_system = "synth"
    cfg = {"models": {"actor": "actor-model", "verifier": "verifier-model"}}

    def stage_families(self, stage):
        return self.families

    def score_request(self, task_text, response, criteria):
        return {"system": "score", "user": json.dumps({"response": response, "ids": [c["id"] for c in criteria]})}

    def det_prims(self, response):
        return {}

    def reward_view(self, task):
        return task["task"]


def cl(cid, fam, pos, neg):
    return {"cluster_id": cid, "family": fam, "group": "reference", "source_id": cid, "domain": None,
            "task": "task " + cid, "messages": None, "pos": pos, "neg": neg}


SEL = [cl("s1", "f1", "good answer one", "bad answer one"), cl("s2", "f1", "good answer two", "bad answer two"),
       cl("s3", "f2", "good tidy one", "good messy one"), cl("s4", "f2", "good tidy two", "good messy two")]
HO = [cl("h1", "f1", "good answer h", "bad answer h"), cl("h2", "f2", "good tidy h", "good messy h"),
      cl("h3", "f2", "good tidy k", "good messy k")]
HO_REGRESS = [cl("h1", "f1", "good answer messy", "bad answer"), cl("h2", "f2", "good tidy h", "good messy h"),
              cl("h3", "f2", "good tidy k", "good messy k")]
B0 = verifier.base_block(FakeBench())
XB = {"block_id": "X_b", "criteria": [{"id": "nw", "definition": "nw: tidy? (0: messy; 1: tidy)"}]}
A = verifier.make_verifier("S_h0a", [B0], "(co + ia) / 2")
B = verifier.make_verifier("S_h0b", [B0, XB], "(co + ia + nw) / 3")
CFG = {"ucoe": {"eps": 0.01, "initial_candidates": 2, "regression_tolerance": 0.10, "max_new_criteria": 2}}
EPS = 0.01


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ucoe_sel_")
        self.store = scoring.ScoreStore(os.path.join(self.tmp, "scores.json"))
        self.client, self.meter, self.bench = FakeClient(), FakeMeter(), FakeBench()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def banks(self, heldout=HO):
        return lambda stage, kind: list(SEL) if kind == "select" else list(heldout)

    def test_score_version_cache_and_failures(self):
        vals = scoring.score_version(self.client, self.meter, self.store, self.bench, B, "t", "good tidy")
        self.assertEqual(vals, {"co": 0.9, "ia": 0.9, "nw": 0.9})
        n = self.client.calls
        scoring.score_version(self.client, self.meter, self.store, self.bench, A, "t", "good tidy")
        self.assertEqual(self.client.calls, n, "the shared base block is reused from the store")
        self.assertIsNone(scoring.score_version(self.client, self.meter, self.store, self.bench, A, "t", "invalid"))
        self.assertGreaterEqual(self.meter.syntax_retries, 1)
        self.store.flush()
        self.assertEqual(len(scoring.ScoreStore(self.store.path)), len(self.store))
        rd = verifier.make_verifier("S_rd", [B0, {"block_id": "X_rd", "criteria": XB["criteria"]}], "nw",
                                    rd_modules=[{"id": "nw", "definition": "nw"}])
        vals = scoring.score_version(self.client, self.meter, self.store, self.bench, rd, "t", "good messy",
                                     rd_applicability=lambda m, t: False)
        self.assertEqual(vals["nw"], 1.0, "inapplicable relation modules score neutral")

    def test_head_J_and_capacity(self):
        per, diag, wins = selection.head_J(self.client, self.meter, self.store, self.bench, A, SEL, EPS, 1)
        self.assertEqual(per, {"f1": (2, 2), "f2": (0, 2)})
        self.assertEqual(diag["f2"]["exact_ties"], 2)
        self.assertEqual(wins, {"s1": 1, "s2": 1, "s3": 0, "s4": 0})
        self.assertEqual(selection.family_J(per), {"f1": 1.0, "f2": -1.0})
        self.assertEqual(selection.driving_capacity({"f1": 1.0, "f2": -1.0}), -0.25)
        per, diag, wins = selection.head_J(self.client, self.meter, self.store, self.bench, A,
                                           SEL + [cl("s5", "f1", "invalid", "bad")], EPS, 1)
        self.assertEqual(per["f1"], (2, 3), "an unscorable cluster counts as a failure")
        self.assertEqual(diag["f1"]["unscored"], 1)
        self.assertEqual(wins["s5"], 0)

    def test_select_rows_and_pick(self):
        rows = selection.select_rows(self.client, self.meter, self.store, self.bench, [A, B], SEL, EPS, 1,
                                     ["f1", "f2"])
        self.assertEqual([r["worst_j"] for r in rows], [-1.0, 1.0])
        self.assertEqual(selection.pick(rows)["candidate_id"], "S_h0b")
        tie = [dict(rows[1], candidate_id="S_z", V=dict(B, aggregation="(co + ia + nw) / 3 ")),
               dict(rows[1], candidate_id="S_y")]
        self.assertEqual(selection.pick(tie)["candidate_id"], "S_y", "ties: shorter aggregation, then candidate order")
        self.assertEqual(selection.select_rows(self.client, self.meter, self.store, self.bench, [A], SEL, EPS, 1,
                                               ["f1", "f2", "f9"]), [], "a candidate missing a family is dropped")

    def test_admission_relative_and_hard(self):
        state = {"V": None, "memory": [], "lineage": ["null"], "ckpt": {}, "extra": {}}
        cands = {"h0a": A, "h0b": B}
        with mock.patch.object(synthesis, "synthesize", lambda c, m, b, cfg, tag, **kw: cands.get(tag)):
            V, rep = selection.admission(self.client, self.meter, self.store, self.bench, CFG, state, 0,
                                         self.banks(), 1)
        self.assertEqual(V["version"], "S_h0b")
        self.assertEqual(rep["decision"], "DEPLOY S_h0b (selection winner)")
        self.assertEqual(rep["winner"], "S_h0b")
        self.assertEqual(sorted(rep["candidates"]), ["S_h0a", "S_h0b"])
        self.assertEqual(rep["heldout_J"], {"f1": 1.0, "f2": 1.0})
        self.assertEqual(len(state["extra"]["archive"]), 2)
        for key in ("selection", "selection_point", "heldout_per_family", "cp_bounds", "decision_rel",
                    "decision_hard"):
            self.assertIn(key, rep)
        self.assertEqual(rep["decision_hard"], "STOP_H0_CERT", "tiny held-out banks cannot certify")
        self.assertEqual(len(state["extra"]["fail_reports"]), 1, "the failed certification is recorded")
        with mock.patch.object(synthesis, "synthesize", lambda *a, **kw: None):
            V2, rep2 = selection.admission(self.client, self.meter, self.store, self.bench, CFG, state, 0,
                                           self.banks(), 1, rule="hard")
        self.assertIsNone(V2)
        self.assertEqual(rep2["decision"], "STOP_H0_CERT")
        self.assertEqual(len(state["extra"]["fail_reports"]), 2)
        empty = {"V": None, "memory": [], "lineage": ["null"], "ckpt": {}, "extra": {}}
        with mock.patch.object(synthesis, "synthesize", lambda *a, **kw: None):
            V3, rep3 = selection.admission(self.client, self.meter, self.store, self.bench, CFG, empty, 0,
                                           self.banks(), 1)
        self.assertIsNone(V3)
        self.assertTrue(rep3["decision"].startswith("RETAIN null"))

    def promote(self, challenger, heldout=HO):
        state = {"V": A, "memory": [], "lineage": ["null", "S_h0a"], "ckpt": {}, "extra": {}}
        rows = [{"b": 1, "weakest": "co"}, {"b": 0, "weakest": "ia"}]
        with mock.patch.object(synthesis, "synthesize", lambda c, m, b, cfg, tag, **kw: challenger):
            return selection.promotion(self.client, self.meter, self.store, self.bench, CFG, state, 2, rows,
                                       self.banks(heldout), 1)

    def test_promotion_promotes_on_gain_and_paired_wins(self):
        D = verifier.make_child("V_ck2", A, XB, "(co + ia + nw) / 3")
        V, rep = self.promote(D)
        self.assertEqual(V["version"], "V_ck2")
        self.assertTrue(rep["decision"].startswith("PROMOTE V_ck2"), rep["decision"])
        self.assertEqual(rep["paired"], {"n10": 2, "n01": 0, "lb": rep["paired"]["lb"]})
        self.assertEqual(rep["regressions"], [])
        self.assertTrue(all(rep["gate_rel"].values()))
        self.assertEqual(rep["selection_point"]["incumbent"]["worst_j"], -1.0)

    def test_promotion_abstains_on_regression(self):
        D = verifier.make_child("V_ck2", A, XB, "0.2 * (co + ia) / 2 + 0.8 * nw")
        V, rep = self.promote(D, heldout=HO_REGRESS)
        self.assertEqual(V["version"], "S_h0a")
        self.assertTrue(rep["decision"].startswith("ABSTAIN"), rep["decision"])
        self.assertEqual(rep["regressions"], ["f1"])
        self.assertEqual((rep["paired"]["n10"], rep["paired"]["n01"]), (2, 1))

    def test_promotion_abstains_without_strict_gain_or_challenger(self):
        E = verifier.make_child("V_ck2", A, None, "(ia + co) / 2")
        V, rep = self.promote(E)
        self.assertEqual(V["version"], "S_h0a")
        self.assertEqual(rep["decision"], "ABSTAIN (no strict selection gain)")
        V, rep = self.promote(None)
        self.assertEqual(rep["decision"], "ABSTAIN (no valid challenger)")
        self.assertEqual(V["version"], "S_h0a")

    def test_residuals(self):
        rows = [{"b": 1, "weakest": "co"}, {"b": 0, "weakest": "co"}, {"b": 1, "weakest": "ia"}]
        txt = synthesis.checkpoint_residual(rows, [{"rule_id": "r1"}])
        self.assertEqual(txt, 'Aggregate prefix diagnostics: repair win rate 2/3; weakest-criterion counts '
                              '{"co": 2, "ia": 1}; memory rules 1.\n\n')
        self.assertEqual(synthesis.failure_residual([]), "")
        rep = {"stage": 0, "winner": "S_h0a", "per_family": {"f1": [1, 2], "f2": [2, 2]},
               "bounds": {"f1": 0.1, "f2": 0.3}, "diag": {"f1": {"exact_ties": 1}}}
        self.assertIn("THE WEAKEST FAILED FAMILY IS 'f1'", synthesis.failure_residual([rep]))


if __name__ == "__main__":
    unittest.main()
