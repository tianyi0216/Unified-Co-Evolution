"""Unit tests of verifier programs (paper Sec. 3.2, Eq. reward-interface): blocks and content hashes, cumulative
children, the aggregation whitelist and probe validator, and proposal parsing."""

from __future__ import annotations

import unittest

from src.ucoe import revision, verifier as VF


class Bench:
    module_ids = ("co", "ia", "fc")
    module_defs = {"co": "co: Is the content correct? (0: wrong; 1: correct)",
                   "ia": "ia: Are the instructions followed? (0: ignored; 1: followed)",
                   "fc": "fc: deterministic -- sane length/structure"}


B0 = VF.base_block(Bench())
XB = {"block_id": "X_b", "criteria": [{"id": "nw", "definition": "nw: new? (0: no; 1: yes)"}]}


class VerifierTests(unittest.TestCase):
    def test_base_block_and_sha(self):
        self.assertEqual(B0["block_id"], VF.BASE_BLOCK_ID)
        self.assertEqual([c["id"] for c in B0["criteria"]], list(Bench.module_ids))
        self.assertEqual(VF.block_sha(B0), VF.block_sha(VF.base_block(Bench())))
        self.assertNotEqual(VF.block_sha(B0), VF.block_sha(XB))

    def test_make_verifier_and_child(self):
        V = VF.make_verifier("S_a", [B0], "(co + ia + fc) / 3")
        self.assertEqual(V["sha256"], VF.verifier_sha(V))
        self.assertEqual(V["parent_sha256"], "root")
        self.assertEqual(VF.crit_ids(V), ["co", "ia", "fc"])
        self.assertEqual(VF.ext_ids(V), [])
        self.assertEqual(VF.active_ids(V), ["co", "ia", "fc"])
        child = VF.make_child("V_b", V, XB, "min(co, nw)")
        self.assertEqual(child["parent_sha256"], V["sha256"])
        self.assertEqual(VF.crit_ids(child), ["co", "ia", "fc", "nw"])
        self.assertEqual(VF.ext_ids(child), ["nw"])
        self.assertEqual(VF.active_ids(child), ["co", "nw"])
        self.assertNotEqual(child["sha256"], V["sha256"])
        self.assertEqual(len(V["blocks"]), 1, "the parent is immutable")
        self.assertEqual(VF.defs(child)["nw"], XB["criteria"][0]["definition"])
        rd = VF.make_verifier("S_r", [B0], "co", rd_modules=[{"id": "m1", "definition": "m1: x"}])
        self.assertEqual(VF.make_child("V_r", rd, None, "ia")["rd_modules"], rd["rd_modules"])

    def test_check_expr_whitelist(self):
        names = ["co", "ia", "fc"]
        for ok in ("(co + ia) / 2", "min(co, ia)", "sorted([co, ia, fc])[1]", "co if co > 0.5 else ia",
                   "max(0, co - 0.1)", "abs(co - ia)", "0.5 * min(co, ia) + 0.5 * fc", "(co + ia) / 2 if fc else 0"):
            self.assertTrue(VF.check_expr(ok, names), ok)
        for bad in ("__import__('os')", "co ** 2", "co.real", "(lambda: 1)()", "unknown + co", "co; ia",
                    "[x for x in [co]][0]", "open('f')", "co % 2", "f'{co}'", "", "'a' * 999999999999",
                    "min(1, 1e400 * co)", "co * 10000000", "co + ''", "co + None"):
            self.assertFalse(VF.check_expr(bad, names), bad)
        self.assertFalse(VF.check_expr("co" + " + 0" * 2000, names), "oversized expression")

    def test_agg_ok_bounds_and_usage(self):
        ids = ["co", "ia", "nw"]
        self.assertTrue(VF.agg_ok("(co + ia + nw) / 3", ids, ["nw"]))
        self.assertFalse(VF.agg_ok("co + ia", ids, []), "exceeds 1")
        self.assertFalse(VF.agg_ok("co - ia", ids, []), "negative")
        self.assertFalse(VF.agg_ok("co / ia", ids, []), "division by zero on the all-0 probe")
        self.assertFalse(VF.agg_ok("(co + ia) / 2", ids, ["nw"]), "appended id unused")
        self.assertFalse(VF.agg_ok("sorted([co, ia])[5]", ids, []), "index error")
        self.assertTrue(VF.agg_ok("co > 0.5", ids, []), "a comparison is a 0/1 score")
        self.assertFalse(VF.agg_ok("min(1, " + " * ".join(["1000000"] * 60) + " * co) / 2", ids, []),
                         "a product that overflows float is rejected, not raised")

    def test_eval_and_agg_score(self):
        V = VF.make_verifier("S_a", [B0], "(co + ia + fc) / 3")
        self.assertAlmostEqual(VF.agg_score(V, {"co": 1.0, "ia": 0.5, "fc": 1.0}), 2.5 / 3)
        self.assertAlmostEqual(VF.agg_score(V, {"co": 1.0}), 1 / 3, msg="missing criteria score 0")
        self.assertAlmostEqual(VF.eval_expr("sorted([co, ia])[0]", {"co": 0.7, "ia": 0.2}), 0.2)

    def test_parse_proposal(self):
        retained = ["co", "ia", "fc"]
        new = [{"id": "nw", "question": "Is it new?", "anchor0": "no", "anchor1": "yes"}]
        got = VF.parse_proposal({"criteria": retained + ["nw"], "new_criteria": new,
                                 "aggregation": "(co + ia + fc + nw) / 4"}, retained)
        self.assertIsNotNone(got)
        ext, agg = got
        self.assertEqual(ext["criteria"], [{"id": "nw", "definition": "nw: Is it new? (0: no; 1: yes)"}])
        self.assertEqual(agg, "(co + ia + fc + nw) / 4")
        self.assertEqual(VF.parse_proposal({"criteria": retained, "new_criteria": [], "aggregation": "min(co, ia)"},
                                           retained), (None, "min(co, ia)"))
        bad = [
            {"criteria": ["co", "ia"], "new_criteria": [], "aggregation": "co"},                        # retained set
            {"criteria": retained + ["co"], "new_criteria": [dict(new[0], id="co")], "aggregation": "co"},  # collision
            {"criteria": retained + ["Bad-Id"], "new_criteria": [dict(new[0], id="Bad-Id")], "aggregation": "co"},
            {"criteria": retained + ["nw", "n2", "n3"], "aggregation": "(nw + n2 + n3) / 3",
             "new_criteria": [dict(new[0], id=i) for i in ("nw", "n2", "n3")]},                        # > max_new
            {"criteria": retained + ["nw"], "new_criteria": [dict(new[0], question="x" * 221)], "aggregation": "nw"},
            {"criteria": retained + ["nw"], "new_criteria": [dict(new[0], anchor0="case 12345")], "aggregation": "nw"},
            {"criteria": retained + ["nw"], "new_criteria": new, "aggregation": "(co + ia) / 2"},       # nw unused
            {"criteria": retained + ["nw"], "new_criteria": new, "aggregation": "co + nw"},              # out of [0,1]
            {"criteria": retained, "new_criteria": [], "aggregation": ""},
            "not a dict",
        ]
        for obj in bad:
            self.assertIsNone(VF.parse_proposal(obj, retained), obj)
        leaky = {"criteria": retained + ["nw"], "aggregation": "nw",
                 "new_criteria": [dict(new[0], question="Does the judge agree?")]}
        self.assertIsNone(VF.parse_proposal(leaky, retained, leak_terms=("judge",)))
        self.assertIsNotNone(VF.parse_proposal(leaky, retained))
        self.assertIsNone(VF.parse_proposal({"criteria": retained, "new_criteria": [], "aggregation": "co"},
                                            retained, min_new=1))


class WeakestCriterionTests(unittest.TestCase):
    def test_weakest_includes_deterministic_criteria(self):
        V = VF.make_verifier("S_a", [B0], "(co + ia + fc) / 3")
        vals = {"co": 1.0, "ia": 0.8, "fc": 0.0}
        self.assertEqual(revision.weakest_criterion(V, vals), "fc", "the paper's rule: the weakest participating one")
        self.assertEqual(revision.weakest_criterion(V, vals, exclude=("fc",)), "ia", "opt-in exclusion")
        self.assertEqual(revision.weakest_criterion(V, {"co": 0.5, "ia": 0.5, "fc": 0.5}), "co", "catalogue order")
        only = VF.make_verifier("S_b", [B0], "fc")
        self.assertEqual(revision.weakest_criterion(only, vals, exclude=("fc",)), "fc", "never an empty pool")


if __name__ == "__main__":
    unittest.main()
