# Data

Everything the method sees is a public prompt (or conversation) plus its own model-generated artifacts.
Reference answers, checklists, rubrics, tests and golden workbooks are **evaluation-only**: they live in
separate sidecar files that only `src/eval/` (and, for SpreadsheetBench-2, `src/agent_harness/evaluate.py`)
ever open.

```
data/
├── prepare.py                 dispatcher: python -m data.prepare --config C [--benchmark NAME|all] [--from-frozen|--from-source]
├── prepare_<benchmark>.py     one script per benchmark (same flags; --out_dir overrides the processed directory)
├── prepare_ssb2.py            SpreadsheetBench-2 pin + agent-visible task tree (agent-harness study)
├── frozen/<benchmark>/        frozen public bundles shipped with the repository (see below)
└── processed/<benchmark>/     written by the prepare scripts (git-ignored)
```

`bash process_data.sh [config.yaml]` runs the dispatcher for `benchmark.name` of the config and writes to
`benchmark.data_dir` (default `data/processed/<benchmark>`). `--from-frozen` is the default whenever the frozen
bundle exists; `--from-source` re-pins the official release (Hugging Face hub or GitHub, pinned revisions and
hashes) and must produce the same files.

## Processed layout (`data/processed/<benchmark>/`)

| File | Content | Read by |
|------|---------|---------|
| `tasks.jsonl` | Frozen deployment stream, one Task per line: `{"i", "task_id", "task", "messages", "domain", ...}`. `task` is the model-visible text (a single-turn prompt, or the canonical `[role] content` rendering of a conversation); `messages` holds the native chat turns for multi-turn benchmarks (WildBench, MultiChallenge), else `null` | method and evaluation |
| `calibration.jsonl` | Prompt-only calibration pool, one Source per line: `{"source_id", "task", "messages", "domain", ...}`; sha-checked disjoint from the deployment stream. Response contrasts are built from the actor's own answers to these prompts | method |
| `discovery.jsonl` | Optional development prompts for relation discovery; when absent, the first `ucoe.relation_discovery.n_discovery_prompts` calibration sources in hash order are used (and excluded from the banks) | method |
| `eval_sidecar.json.gz` | `{task_id: evaluation-only fields}` (see the table below) | `src/eval` only |
| `manifest.json` | Source pins, split rule, counts and file or id-list hashes; identical on every rerun (the gzip files are written without timestamps) | humans |

Extra per-row fields: WritingPrompts tasks carry `prompt` (the bare prompt without the story instruction) and `cx`
(curriculum complexity); Hard-MATH rows carry `subject`, `level` and `source_domain`; IFBench also writes
`tasks_full.jsonl` (all 300 prompts) and `calibration_arena.jsonl` for `benchmark.protocol: full`.

## Benchmarks

`matchedbank_v2` and `rewardbank_v3` below are fixed salt strings inherited from the original split code; they are hard-coded in the prepare scripts and must not be changed, or the calibration/deployment split (and the manifest hashes) will differ from the paper's.

| Benchmark | Official source (pinned) | Deployment / calibration | Split | Evaluation-only sidecar fields |
|-----------|--------------------------|--------------------------|-------|--------------------------------|
| `arena_hard` | HF `lmarena-ai/arena-hard-auto`, `data/arena-hard-v2.0/question.jsonl`, `category == hard_prompt` | 500 / 2,940 (540 `code_*` HumanEval+/MBPP+ instructions, 2,400 `math_*` competition problems) | official order; calibration = frozen prompt-only pool, deduplicated and deployment-disjoint | `category`, `subcategory`, `language` (no native metric in this release) |
| `writingprompts` | HF `euclaise/writingprompts` (train, `prompt` column only) | 2,048 / 1,400 | first 2,048 prompts of the frozen 6,400-prompt stream, ordered as a curriculum by complexity then sha256; the next 1,400 are calibration; only tasks 1,024-2,047 are evaluated (`eval_range`) | empty (no native metric) |
| `wildbench` | HF `allenai/WildBench`, config `v2`, split `test` | 1,024 / 6,940 (4,000 `hb_*` health-assistant conversations, 2,940 `ar_*` single-turn requests) | official order; calibration = frozen conversation pool | `checklist`, `references`, `intent`, `primary_tag`, `secondary_tags`, `toxic`, `redacted`, `turn_meta` |
| `writingbench` | GitHub `X-PLUG/WritingBench`, `benchmark_query/benchmark_all.jsonl` | 500 / 500 | ids ordered by `sha256("<jsonl_sha16>|matchedbank_v2|<index>")`: first 500 calibration, rest deployment | `checklist` (five scored criteria), `domain1`, `domain2`, `lang`, `subsets` |
| `multichallenge` | HF `ScaleAI/MultiChallenge`, test split | 128 / 138 | ids ordered by `sha256("<parquet_sha>|matchedbank_v2|<question_id>")`: first 138 calibration, rest deployment | `axis`, `target_question`, `pass_criteria`, `num_turns` |
| `refinebench` | HF `RefineBench/RefineBench`, `exam_type == "Free Form"` (478) | 278 / 200 | field-stratified largest-remainder quotas (200 calibration), within a field by `sha256("<revision>|rewardbank_v3|<task_id>")`; deployment keeps the official order | `reference_answer`, `checklist`, `comment`, `field`, `subject`, `institution` |
| `ifbench` | GitHub `allenai/IFBench`, `data/IFBench_test.jsonl` (300) | 150 / 150 (`benchmark.protocol: matched`, the paper) | keys ordered by `sha256("<jsonl_sha16>|matchedbank_v2|<key>")`: first 150 calibration, rest deployment; `protocol: full` deploys all 300 and calibrates on the Arena pool | `instruction_id_list`, `kwargs` (drive the vendored official verifiers) |
| `hard_math` | HF `HuggingFaceH4/MATH-500` (levels 4-5), `AI-MO/aimo-validation-aime` (2022-2024), `MathArena/aime_2025`, `EleutherAI/hendrycks_math` (training, levels 4-5) | 382 / 2,400 | 262 MATH-500 problems (official order) then 120 AIME problems (30 per year); calibration = 2,400 MATH training problems disjoint by id and text | gold answer string per task |
| `bigcodebench_hard` | HF `bigcode/bigcodebench-hard`, `data/v0.1.4-00000-of-00001.parquet` (Instruct prompts) | 148 / 540 | official order; calibration = HumanEval+ / MBPP+ prompts (`evalplus/humanevalplus`, `evalplus/mbppplus`), content-deduplicated | `test`, `entry_point`, `canonical_solution`, `code_prompt`, `complete_prompt` |

Deployment and calibration are disjoint by prompt sha256 in every benchmark (`write_outputs` refuses overlaps),
and the processed id lists are pinned by hash in the manifest so a rebuilt split cannot drift.

## Frozen bundles (`data/frozen/<benchmark>/`)

The bundles hold the public prompts in their frozen order plus the calibration pools used in the paper; every
file is gzip-compressed JSON. Roles: `public` (deployment prompts), `sidecar` (evaluation-only fields),
`pool` (calibration prompts; for `hard_math` a reference-only MATH training file that no script opens, its calibration pool being the `battery_pool` inside `public`), `pool_ucal` (an unused auxiliary pool kept for completeness).

| Benchmark | `public.json.gz` | `sidecar.json.gz` | `pool.json.gz` |
|-----------|------------------|-------------------|----------------|
| `arena_hard` | `{"revision", "file_sha256", "items": [{"i", "uid", "prompt"}]}` (500) | `{uid: {"category", "subcategory", "language"}}` | `[{"id", "prompt"}]` (`code_*`, `math_*`) |
| `writingprompts` | `[{"id": "wps000001", "prompt"}]` (6,400, seeded shuffle order) | - | `pool_ucal.json.gz`: `[{"id": "wpu000001", "prompt"}]` (500, unused) |
| `wildbench` | `{"revision", "file_sha256", "items": [{"i", "task_id", "conv": [{"role", "content"}]}]}` (1,024) | `{task_id: {"checklist", "references", "intent", "primary_tag", "secondary_tags", "toxic", "redacted", "turn_meta"}}` | `[{"id", "conv"}]` (6,940; `hb_*`, `ar_*`) |
| `writingbench` | `{"commit", "jsonl_sha16", "items": [{"i", "task_id", "query"}]}` (1,000) | `{task_id: {"checklist": [{"name", "criteria_description", "1-2", ..., "9-10"}], "domain1", "domain2", "lang", "subsets"}}` | - |
| `multichallenge` | `{"revision", "parquet_sha", "items": [{"i", "question_id", "messages"}]}` (266) | `{question_id: {"axis", "target_question", "pass_criteria", "num_turns"}}` | - |
| `refinebench` | `{"revision", "items": [{"i", "task_id", "question", "passages", "materials"}]}` (478) | `{task_id: {"reference_answer", "checklist", "comment", "field", "subject", "institution"}}` | `[{"id", "task"}]` (522 Exact-Match tasks, unused by default) |
| `ifbench` | `{"commit", "jsonl_sha16", "items": [{"i", "task_id", "prompt"}]}` (300) | `{task_id: {"instruction_id_list", "kwargs"}}` | `[{"id", "prompt"}]` (2,940 Arena battery prompts, `protocol: full`) |
| `hard_math` | `{"domain_order", "test_stream": [{"id", "problem", "domain", "subject", "level"}] (382), "battery_pool": [...] (2,400), "u_cal": [...] (600, unused)}` | `{id: gold}` (stream + pool problems) | `[{"subject", "level", "problem", "solution", "gold"}]` (5,582 MATH training problems, reference only) |
| `bigcodebench` | `{"revision", "items": [{"i", "task_id", "instruction"}]}` (148) | `{task_id: {"test", "entry_point", "canonical_solution", "code_prompt", "complete_prompt"}}` | `[{"id", "instruction"}]` (542 HumanEval+ / MBPP+ prompts) |

Frozen files are verified by sha256 (or by their recorded revision and item count) before use.

## Regenerating from the official sources

```bash
python -m data.prepare --config configs/wildbench_qwen8b.yaml --from-source      # one benchmark
python -m data.prepare --config config.yaml --benchmark all                       # every frozen benchmark
```

`--benchmark all` covers the nine benchmarks with frozen bundles. SpreadsheetBench-2 is prepared separately with `bash process_data.sh configs/ssb2_codex.yaml` (or `python -m data.prepare --config configs/ssb2_codex.yaml`); it has no frozen bundle and needs network access (git clone of the checker, `huggingface_hub` for the data).

`--from-source` needs `huggingface_hub` (all hub sources), `datasets` (Hard-MATH) and `pyarrow` or `pandas`
(parquet files). Two pools have no other public origin than the frozen bundles and are kept from them in both
modes: the WildBench calibration conversations and (when the sibling BigCodeBench / Hard-MATH bundles are absent)
the Arena-Hard calibration pool. WritingPrompts `--from-source` re-derives the stream from the dataset with the
paper's recipe (clean, filter to 40-300 characters, deduplicate, seeded shuffle), which yields the same
distribution but not necessarily the identical prompt set of the frozen stream.

## SpreadsheetBench-2 (agent-harness study)

`python -m data.prepare_ssb2 --config configs/ssb2_codex.yaml [--steps pin,integrity,toycheck,pilot,export]`
(also reached through `bash process_data.sh configs/ssb2_codex.yaml`) makes no model calls and writes
everything under `benchmark.data_dir` (default `data/raw/ssb2`, git-ignored):

```
data/raw/ssb2/
├── SpreadsheetBench-2/          official checker repository (benchmark.code_repo at benchmark.code_commit)
├── hf/                          official data (benchmark.hf_repo at benchmark.hf_revision), unpacked:
│   └── spreadsheetbench-v2/<Category>/dataset.json + task directories (input, reference workbooks, answer positions)
├── pin.json                     recorded repository commit, dataset revision and data root
├── integrity.json               per category: task count, workbook families, inputs openpyxl cannot read
├── toycheck.json                self-test of the official checker on synthetic workbooks
├── pilot.json                   the 12 development tasks excluded from the stream and from calibration
└── tasks/                       the agent-visible tree, the only part the method path reads
    ├── index.json               {category: [{"id", "family", "readable"}]}
    └── <Category>/<id>/{task.json, input.xlsx}      task.json = {id, category, family, instruction, input_file}
```

The runner (`src/agent_harness/runner.py`) builds the 100-task stream (`benchmark.category_targets`, excluded
families, pilot ids removed) and the calibration chunks from `tasks/index.json`; the reference workbooks and
`answer_position` fields stay in the official data directory and are opened only by `src/agent_harness/evaluate.py`
(`official`).
