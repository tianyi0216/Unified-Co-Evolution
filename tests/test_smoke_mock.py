"""End-to-end offline smoke of the UCoE pipeline on the deterministic mock backend (configs/smoke_mock.yaml with a
smaller stream): Direct Generation, contrast banks, H0 admission, the per-task lifecycle with one checkpoint, the
procedural memory and the preference evaluation, followed by a replay that must add no model call."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from src.ucoe import memory
from src.utils.io import read_json, read_jsonl, sha

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_TASKS = 4
CHECKPOINT = 2


def _config(tmp):
    try:
        from src.utils.config import load_config
        cfg = load_config(os.path.join(REPO, "configs", "smoke_mock.yaml"))
    except SystemExit as e:  # pyyaml missing: load_config exits with an install hint
        raise unittest.SkipTest(f"configs/smoke_mock.yaml cannot be loaded: {e}")
    data_dir = os.path.join(REPO, cfg["benchmark"]["data_dir"])
    if not os.path.exists(os.path.join(data_dir, "tasks.jsonl")):
        from data import prepare_arena_hard
        data_dir = os.path.join(tmp, "processed")
        prepare_arena_hard.main(cfg, mode="frozen", out_dir=data_dir)
    cfg["benchmark"]["data_dir"] = data_dir
    cfg["benchmark"]["n_tasks"] = N_TASKS
    cfg["ucoe"]["checkpoints"] = [CHECKPOINT]
    cfg["output"]["workers"] = 2
    return cfg


def _run(cfg, work_dir):
    from src.benchmarks.registry import get_benchmark
    from src.eval.preference import evaluate_preference
    from src.ucoe.lifecycle import UCoERun
    cfg = json.loads(json.dumps(cfg))
    cfg["output"]["work_dir"] = work_dir
    bench = get_benchmark(cfg)
    run = UCoERun(cfg, bench, work_dir)
    run.run_raw(N_TASKS, 2)
    run.run_discovery(2)
    run.build_banks(2)
    run.run(N_TASKS, 2)
    results = evaluate_preference(cfg, bench, work_dir, arms=("ucoe",), baseline="raw", workers=2)
    return run, results


class SmokeMockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="ucoe_smoke_")
        cls.cfg = _config(cls.tmp)
        cls.work = os.path.join(cls.tmp, "run")
        cls.pipeline, cls.results = _run(cls.cfg, cls.work)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def path(self, *parts):
        return os.path.join(self.work, *parts)

    def test_artifacts_exist(self):
        for rel in (("raw", "rows.jsonl"), ("raw", "raw_pool.json"), ("banks", "bank_0_select.json"),
                    ("banks", "bank_0_heldout.json"), ("banks", f"bank_{CHECKPOINT}_select.json"),
                    ("banks", "registry.json"), ("ucoe", "rows.jsonl"), ("ucoe", "state.json"),
                    ("ucoe", "h0_report.json"), ("ucoe", f"ckpt_{CHECKPOINT}_report.json"),
                    ("ucoe", "blockscores.json"), ("ucoe", "llm_cache.jsonl"), ("eval", "preference.json"),
                    ("eval", "preference_verdicts.json")):
            self.assertTrue(os.path.exists(self.path(*rel)), rel)

    def test_lifecycle_rows_and_state(self):
        rows = read_jsonl(self.path("ucoe", "rows.jsonl"))
        self.assertEqual([r["i"] for r in rows], list(range(N_TASKS)))
        for r in rows:
            self.assertEqual(sha(r["final"]), r["final_sha256"])
            self.assertIn(r["y0_src"], ("raw", "actor"))
            self.assertIsNone(r["fallback"])
        self.assertGreaterEqual(sum(r["b"] for r in rows), 1, "at least one accepted revision")
        accepted = [r for r in rows if r["b"]]
        self.assertTrue(all(r["sR"] > r["s0"] + 0.01 for r in accepted))
        self.assertTrue(all(r["final"] == r["yr"] for r in accepted))
        self.assertTrue(all(r["witness"] and r["repair"] and r["weakest"] for r in rows))
        state = read_json(self.path("ucoe", "state.json"))
        self.assertIsNotNone(state["V"])
        self.assertTrue(state["lineage"][1].startswith("S_h0"))
        self.assertGreaterEqual(len(state["memory"]), 1, "an accepted repair committed a procedure")
        for rule in state["memory"]:
            for key in ("rule_id", "criterion_id", "dimension", "scope_tags", "trigger", "corrective_action",
                        "created_at", "source_reward_version", "wins", "losses", "retrievals"):
                self.assertIn(key, rule)
            self.assertLessEqual(len(rule["trigger"].split()), 10)
            self.assertLessEqual(len(rule["corrective_action"].split()), 14)
        self.assertTrue(any(r["rule_status"] == "appended" for r in rows))
        self.assertEqual(rows[-1]["mem_sha"], memory.mem_sha(state["memory"]))

    def test_decisions(self):
        h0 = read_json(self.path("ucoe", "h0_report.json"))
        self.assertTrue(h0["decision"].startswith("DEPLOY S_h0"), h0["decision"])
        self.assertEqual(sorted(h0["candidates"]), ["S_h0a", "S_h0b"])
        self.assertEqual(h0["families"], ["constraint_omission", "claim_injection", "off_topic_graft"])
        self.assertTrue(all(j > 0 for j in h0["heldout_J"].values()), "every reference family is separable")
        ck = read_json(self.path("ucoe", f"ckpt_{CHECKPOINT}_report.json"))
        self.assertTrue(ck["decision"].startswith(("PROMOTE", "ABSTAIN")), ck["decision"])
        self.assertEqual(ck["incumbent"], h0["winner"])
        self.assertTrue(ck["challenger"].startswith("V_ck"))
        state = read_json(self.path("ucoe", "state.json"))
        self.assertEqual(set(state["ckpt"]), {"0", str(CHECKPOINT)})

    def test_preference(self):
        res = self.results
        row = res["ucoe_vs_raw"]
        self.assertEqual(row["n"], N_TASKS)
        self.assertEqual(sum(row["W_L_T"]), N_TASKS)
        self.assertGreater(row["pref"], 50.0, "accepted revisions beat the drafts under the mock judge")
        self.assertIn("holm_p", row)
        self.assertEqual(read_json(self.path("eval", "preference.json"))["n"], N_TASKS)

    def test_replay_is_free_and_deterministic(self):
        before = read_jsonl(self.path("ucoe", "rows.jsonl"))
        meter = read_json(self.path("ucoe", "cost_meter.json"))
        run2, _ = _run(self.cfg, self.work)
        self.assertEqual(read_jsonl(self.path("ucoe", "rows.jsonl")), before)
        self.assertEqual(read_json(self.path("ucoe", "cost_meter.json"))["new"], meter["new"],
                         "a finished run replays without new model calls")
        other = os.path.join(self.tmp, "run2")
        _run(self.cfg, other)
        self.assertEqual(read_jsonl(os.path.join(other, "ucoe", "rows.jsonl")), before,
                         "the mock pipeline is deterministic across work directories")


if __name__ == "__main__":
    unittest.main()
