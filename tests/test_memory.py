"""Unit tests of the bounded procedural memory (paper App. "Harness-state instantiation"): commit statuses and
eviction order, retrieval eligibility and ranking, usage accounting, and the generality checks of a distilled
procedure."""

from __future__ import annotations

import unittest

from src.ucoe import memory as M


class Bench:
    dims = ("correctness", "clarity")
    scope_tags = ("general", "coding")


def rule(cid, dim, trigger, action, tags=("general",)):
    return {"criterion_id": cid, "dimension": dim, "scope_tags": list(tags), "trigger": trigger,
            "corrective_action": action}


TASK = {"task_id": "t1", "task": "Write a short note about the quarterly budget review meeting."}


class MemoryTests(unittest.TestCase):
    def test_commit_dedup_and_caps(self):
        first = rule("co", "correctness", "a claim lacks support", "add the reasoning")
        rules, st = M.commit_rule([], first, 0, "S_a")
        self.assertEqual(st, "appended")
        self.assertEqual((rules[0]["wins"], rules[0]["losses"], rules[0]["retrievals"]), (1, 0, 0))
        self.assertEqual(rules[0]["source_reward_version"], "S_a")
        dup = rule("co", "correctness", "a claim lacks support", "add the reasoning")
        self.assertEqual(M.commit_rule(rules, dup, 1, "S_a")[1], "duplicate_skipped")
        other_dim = rule("co", "clarity", "a claim lacks support", "add the reasoning")
        rules, st = M.commit_rule(rules, other_dim, 1, "S_a")
        self.assertEqual(st, "appended", "dedup applies within a dimension only")
        ending = rule("co", "clarity", "the ending is abrupt", "close with the result")
        rules, st = M.commit_rule(rules, ending, 2, "S_a", per_dim_cap=2)
        self.assertEqual(st, "appended")
        rules, st = M.commit_rule(rules, rule("co", "clarity", "terms are undefined", "define each term"), 3, "S_a",
                                  per_dim_cap=2)
        self.assertEqual(st, "evicted_dim")
        self.assertEqual(sum(1 for r in rules if r["dimension"] == "clarity"), 2)
        self.assertNotIn(1, [r["created_at"] for r in rules if r["dimension"] == "clarity"],
                         "the oldest record of the dimension is evicted on equal records")
        rules, st = M.commit_rule(rules, rule("ia", "correctness", "steps are skipped", "write out each step"), 4,
                                  "S_a", cap=3, per_dim_cap=5)
        self.assertEqual(st, "evicted_cap")
        self.assertEqual(len(rules), 3)
        self.assertEqual(M.commit_rule(rules, rule("ia", "clarity", "x", "y"), 5, "S_a", cap=0), (rules, "disabled"))

    def test_retrieve_ranking_and_eligibility(self):
        rules = []
        for i, (cid, dim, tr) in enumerate((("co", "correctness", "a claim lacks support"),
                                            ("ia", "clarity", "the ending is abrupt"),
                                            ("ia", "correctness", "steps are skipped"),
                                            ("cp", "clarity", "terms are undefined"))):
            rules, st = M.commit_rule(rules, rule(cid, dim, tr, "fix it plainly"), i, "S_a", per_dim_cap=5)
            self.assertEqual(st, "appended")
        active = {"co", "ia", "cp"}
        got = M.retrieve(rules, "ia", "clarity", active, {"general"}, k=3, token_cap=64)
        self.assertEqual([r["criterion_id"] for r in got][:2], ["ia", "ia"], "criterion match ranks first")
        self.assertEqual(got[0]["dimension"], "clarity", "then dimension match")
        self.assertEqual(len(M.retrieve(rules, "ia", "clarity", active, {"general"}, k=1, token_cap=64)), 1)
        self.assertEqual(M.retrieve(rules, "ia", "clarity", active, {"general"}, k=3, token_cap=5), [],
                         "the word budget is a hard cap")
        self.assertEqual(len(M.retrieve(rules, None, None, {"co"}, {"general"}, k=3, token_cap=64)), 1,
                         "criteria no longer defined are ineligible")
        rules[0]["losses"] = rules[0]["wins"] + 3
        self.assertNotIn("co", [r["criterion_id"] for r in M.retrieve(rules, None, None, active, {"general"}, 4, 64)])
        rules[0]["losses"] = 0
        rules[0]["scope_tags"] = ["coding"]
        self.assertNotIn("co", [r["criterion_id"] for r in M.retrieve(rules, None, None, active, {"math"}, 4, 64)])
        self.assertIn("co", [r["criterion_id"] for r in M.retrieve(rules, None, None, active, {"coding"}, 4, 64)])
        rules[2]["wins"] = 5
        top = M.retrieve(rules, None, None, active, {"general"}, k=1, token_cap=64)[0]
        self.assertEqual(top["created_at"], 2, "draft-time retrieval ranks by record")

    def test_account_and_sha(self):
        rules, _ = M.commit_rule([], rule("co", "correctness", "a claim lacks support", "add the reasoning"), 0, "S_a")
        before = M.mem_sha(rules)
        M.account(rules, [rules[0]["rule_id"]], True)
        M.account(rules, [rules[0]["rule_id"]], False)
        M.account(rules, ["missing"], True)
        self.assertEqual((rules[0]["wins"], rules[0]["losses"], rules[0]["retrievals"]), (2, 1, 2))
        self.assertNotEqual(before, M.mem_sha(rules))
        self.assertEqual(M.mem_sha([]), M.mem_sha([]))

    def test_validate_rule(self):
        bench, active = Bench(), {"co", "ia"}
        ok = {"criterion_id": "co", "dimension": "clarity", "scope_tags": ["coding", "bogus", "general", "math"],
              "trigger": "the conclusion drifts from the question", "corrective_action": "answer the question directly"}
        got = M.validate_rule(ok, bench, TASK, "draft text", "revised text", "co", active)
        self.assertEqual(got["scope_tags"], ["coding", "general"], "unknown tags dropped, at most two kept")
        self.assertEqual(got["criterion_id"], "co")
        bogus = M.validate_rule(dict(ok, scope_tags=["bogus"]), bench, TASK, "", "", "co", active)
        self.assertEqual(bogus["scope_tags"], ["general"])
        no_id = M.validate_rule(dict(ok, criterion_id=None), bench, TASK, "", "", "ia", active)
        self.assertEqual(no_id["criterion_id"], "ia", "a missing criterion id defaults to the weakest criterion")
        for bad in (dict(ok, dimension="speed"), dict(ok, criterion_id="zz"),
                    dict(ok, trigger="one two three four five six seven eight nine ten eleven"),
                    dict(ok, corrective_action=" ".join(["w"] * 15)), dict(ok, trigger="use 3 bullets"),
                    dict(ok, corrective_action='quote "this"'), dict(ok, corrective_action="see http://x.y"),
                    dict(ok, trigger=""), "not a dict",
                    dict(ok, trigger="the quarterly budget review meeting")):
            self.assertIsNone(M.validate_rule(bad, bench, TASK, "draft", "revised", "co", active), bad)


if __name__ == "__main__":
    unittest.main()
