<div align="center">

  <h3>Verifier&ndash;Harness Co-Evolution for Training-Free Self-Improvement on Open-Ended Tasks</h3>

  <p><b>UCoE (Unified Co-Evolution)</b> &nbsp;&middot;&nbsp; ICLR 2027 (under review)</p>

  <p>
    <a href="https://tianyi0216.github.io/">Tianyi Xu</a><sup>1,*</sup>&nbsp;&middot;&nbsp;
    <a href="https://mercury7353.github.io/Yaolun-Zhang.github.io/">Yaolun Zhang</a><sup>1,4,*</sup>&nbsp;&middot;&nbsp;
    Yiran Wu<sup>2,*</sup>&nbsp;&middot;&nbsp;
    Joe Nguyen<sup>1</sup>&nbsp;&middot;&nbsp;
    Shengyu Dai<sup>3</sup>&nbsp;&middot;&nbsp;
    Saba Moeinizade<sup>3</sup>&nbsp;&middot;&nbsp;
    Zhenwen Shao<sup>3</sup>&nbsp;&middot;&nbsp;
    Qingyun Wu<sup>2,4</sup>&nbsp;&middot;&nbsp;
    <a href="https://huazhengwang.github.io/">Huazheng Wang</a><sup>1&#9993;</sup>
  </p>

  <p>
    <sup>1</sup> Oregon State University &nbsp;&nbsp;
    <sup>2</sup> Pennsylvania State University &nbsp;&nbsp;
    <sup>3</sup> Johnson &amp; Johnson &nbsp;&nbsp;
    <sup>4</sup> AG2 AI
  </p>

  <p>
    <sup>*</sup> Equal contribution &nbsp;&nbsp;
    <sup>&#9993;</sup> Corresponding author
  </p>

  <p>
    <img src="https://img.shields.io/badge/ICLR%202027-under%20review-4B0082?style=flat-square" alt="ICLR 2027 (under review)">
    <!-- arXiv badge: uncomment once the preprint is posted.
    <a href="https://arxiv.org/abs/XXXX.XXXXX">
      <img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-B31B1B?style=flat-square&logo=arxiv" alt="arXiv">
    </a>
    -->
    <a href="https://github.com/tianyi0216/Unified-Co-Evolution">
      <img src="https://img.shields.io/badge/Code-GitHub-181717?style=flat-square&logo=github" alt="Code">
    </a>
    <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
    <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python" alt="Python">
  </p>

</div>

---

## Overview

<p align="center">
  <img src="assets/pipeline.png" alt="UCoE pipeline" width="100%">
</p>

UCoE is a training-free self-improvement method for open-ended tasks. It evolves an **executable verifier program** together with the **harness state** that shapes how a fixed actor handles later tasks, while every model parameter stays frozen. The verifier model synthesizes candidate verifier programs (criterion scorers plus an aggregation expression) and tests them on **response contrasts**: pairs of responses that differ in exactly one declared property with a known preferred side. UCoE deploys the candidate whose discrimination on its *weakest* contrast family is highest, a quantity the paper calls **driving capacity**. On every task the deployed verifier diagnoses one weakness in the draft, guides a single local repair, decides whether the draft or the revision is served, and gates whether the accepted repair may enter the harness state as a generalized procedure. At scheduled checkpoints, diagnostics from completed tasks seed new candidate verifiers, which must beat the current verifier on the same contrasts before adoption.

Across six open-ended benchmarks UCoE averages **59.00%** tie-adjusted preference against matched direct generation with Qwen3-8B and **57.19%** with Qwen3.8-27B (50 is parity). On three verifiable benchmarks it raises the mean from **36.06%** (direct generation) to **38.48%**. Inside the Codex coding-agent harness on SpreadsheetBench-2 it reaches **22.7%** exact-match accuracy against **17.3%** for both direct generation and a fixed LLM judge.

This repository contains the UCoE pipeline (`src/`), the benchmark adapters and data preparation scripts (`data/`), the evaluation protocol (`src/eval/`), and the agent-harness port used for the SpreadsheetBench-2 study (`src/agent_harness/`).

## Setup

```bash
conda create -n ucoe python=3.11 -y
conda activate ucoe
pip install -r requirements.txt
```

The pipeline itself runs on Python 3.9 or newer with PyYAML as its only hard dependency; `requirements.txt` adds the optional packages with Python-version markers, so the Tinker backend (`tinker`, Python 3.11+) and the Hard-MATH answer matcher (`math-verify`, Python 3.10+) are installed only where their floor is met and silently skipped elsewhere.

Model access is selected by `models.backend` in the config:

```bash
# Tinker backend (paper main experiments; Qwen3-8B / Qwen3.8-27B / Qwen3.6-35B-A3B / Qwen3.5-397B-A17B);
# needs Python 3.11+ and `pip install tinker`
export TINKER_API_KEY=...

# Any OpenAI-compatible chat-completions server (vLLM, OpenAI, Azure): set models.backend: openai
export OPENAI_API_KEY=...            # or the variable named by models.openai.api_key_env
# and point models.openai.base_url at the server in the config
```

The agent-harness study additionally needs **LibreOffice** (`soffice` on `PATH`, or `SOFFICE=/path/to/soffice`) for workbook recalculation, `openpyxl`, and the **Codex CLI** app server with access to the GPT-5.6 models named in `configs/ssb2_codex.yaml`.

## Quick start

```bash
# 1. Build the frozen task stream, calibration pool and evaluation sidecar for the configured benchmark.
bash process_data.sh

# 2. Run the UCoE lifecycle: direct-generation drafts -> relation discovery -> contrast banks -> deployment stream.
bash run_ucoe.sh

# 3. Evaluate: tie-adjusted preference against direct generation, plus the benchmark's native metric.
bash eval.sh

# 4. (Optional) The SpreadsheetBench-2 study inside the Codex agent harness
#    (manifest -> pin-probe -> smoke -> run for the native, judge and ucoe arms -> official -> report).
bash run_agent_harness.sh configs/ssb2_codex.yaml all
```

Each launcher accepts an optional config path, and `run_ucoe.sh` / `eval.sh` an optional stage or mode:

```bash
bash process_data.sh configs/wildbench_qwen27b.yaml
bash run_ucoe.sh configs/wildbench_qwen27b.yaml            # stage defaults to "all"
bash run_ucoe.sh configs/wildbench_qwen27b.yaml banks      # all | raw | discover | banks | run | status
bash eval.sh configs/wildbench_qwen27b.yaml native         # preference | native | all
```

The launchers wrap the module CLIs, which can also be called directly from the repo root:

```bash
python -m src.run --config config.yaml --stage all [--n N] [--workers W]
python -m src.eval.run --config config.yaml --mode preference [--arms ucoe,raw]
python -m src.agent_harness.run --config configs/ssb2_codex.yaml <manifest|pin-probe|smoke|calib-drafts|banks|h0|run --arm A|status|official|report|audit>
```

`configs/smoke_mock.yaml` runs the whole lifecycle offline on a deterministic mock backend (six Arena-Hard tasks, one checkpoint, no API keys, any Python 3.9+) and is the fastest way to check an installation; `configs/smoke_mock_rd.yaml` does the same with relation discovery enabled:

```bash
bash process_data.sh configs/smoke_mock.yaml                      # Arena-Hard stream from the frozen bundle
bash run_ucoe.sh configs/smoke_mock.yaml && bash eval.sh configs/smoke_mock.yaml preference
python -m unittest discover -s tests -v                           # verifier, selection, memory, contrasts, agent core, mock smoke
```

## Repository layout

```
Unified-Co-Evolution/
├── config.yaml                 default run: Arena-Hard, Qwen3-8B actor, Qwen3.6-35B-A3B verifier model, Tinker backend
├── configs/                    one preset per paper setting (see "Reproducing paper results")
├── process_data.sh             bash process_data.sh [config.yaml]
├── run_ucoe.sh                 bash run_ucoe.sh [config.yaml] [stage]
├── eval.sh                     bash eval.sh [config.yaml] [preference|native|all]
├── run_agent_harness.sh        bash run_agent_harness.sh [configs/ssb2_codex.yaml] <cmd>
├── data/                       prepare_<benchmark>.py scripts, frozen prompt bundles (data/frozen/), processed streams (data/processed/)
├── src/
│   ├── run.py                  stage driver (raw -> discover -> banks -> run)
│   ├── ucoe/                   verifier programs, synthesis, contrasts, relation discovery, selection, revision, memory, lifecycle
│   ├── benchmarks/             one adapter per benchmark (prompts, criteria, contrast families, splits)
│   ├── llm/                    Tinker / OpenAI-compatible / mock transports, cached client, call metering
│   ├── eval/                   preference protocol and native metrics
│   └── agent_harness/          UCoE inside the Codex harness for SpreadsheetBench-2
└── tests/
```

Every stage is resumable: rows are appended atomically, state is written with atomic replace, and every LLM call is cached in a JSONL ledger keyed by request hash, so re-running a finished stage costs nothing. A run leaves `raw/` (direct-generation drafts), `relations/` (discovered relation modules), `banks/` (contrast banks per checkpoint and split), `ucoe/` (per-task rows, harness state, checkpoint reports) and `eval/` under `output.work_dir`.

## Configuration

All runtime behavior is driven by a single YAML file (`config.yaml` by default, which reproduces the Arena-Hard / Qwen3-8B cell). Values may reference other keys with `${a.b.c}`. Top-level groups:

| Key | Purpose |
|-----|---------|
| `benchmark.{name,data_dir,n_tasks,seed}` | Which benchmark adapter to run and where its processed stream lives |
| `models.{backend,actor,verifier,comparator,judge}` | `tinker`, `openai` or `mock` transport; the fixed actor, the verifier model, the larger comparison model for relation discovery, and the final judge. `models.openai.{base_url,api_key_env,extra_body}` configures an OpenAI-compatible server |
| `decoding.{generation,revision,scoring,synthesis,localize,distill,judge}` | Per-role `max_tokens` / `temperature` (and `thinking` for synthesis); adapters supply the paper defaults, config overrides them |
| `ucoe.{initial_candidates,candidates_per_checkpoint,max_new_criteria,eps,checkpoints,regression_tolerance,update_rule}` | Lifecycle hyperparameters: candidate budgets, acceptance margin, checkpoint schedule, regression tolerance, and the update rule (`relative` = paper; `hard` = conservative confidence-bound ablation) |
| `ucoe.contrasts.{selection_per_family,heldout_per_family,relation_selection_per_family,relation_heldout_per_family}` | Contrast quotas per family and split |
| `ucoe.relation_discovery.{enabled,n_discovery_prompts,max_pairs,max_modules,min_recurrence}` | Relation discovery on development prompts (benchmark-specific contrast families) |
| `ucoe.memory.{capacity,per_dimension,max_retrieved,token_cap,dedup_jaccard,at_draft}` | Bounded procedural memory (the harness state of the main experiments) |
| `evaluation.{judge_both_orders,bootstrap,native}` | Preference protocol (both presentation orders, block bootstrap) and native-metric knobs |
| `output.{work_dir,workers,smoke,ceilings}` | Run directory, thread pool size, smoke mode, and per-model-class call ceilings |

To switch benchmarks or actors, start from a preset in `configs/`, or copy `config.yaml` to `configs/my_run.yaml`, change `benchmark.name` / `benchmark.data_dir` / `models.*`, and pass it to the launchers.

## Datasets

The method only ever sees public prompts and its own model-generated artifacts. Evaluation-only fields (reference answers, checklists, rubrics, tests, golden workbooks) are stored in separate `*_sidecar.json.gz` files that only `src/eval` opens.

| Benchmark | Official source | Evaluated tasks | Notes |
|-----------|-----------------|-----------------|-------|
| **Arena-Hard-v2.0** | Hugging Face `lmarena-ai/arena-hard-auto`, `hard_prompt` set (500) | 500 | Official order; calibration prompts from the frozen battery pool |
| **WritingPrompts** | Hugging Face `euclaise/writingprompts` | 1,024 | 2,048-task curriculum stream; the last 1,024 are evaluated; 1,400 frozen calibration prompts |
| **WildBench** | Hugging Face `allenai/WildBench`, config `v2`, split `test` (1,024) | 1,024 | Official order; calibration prompts from the frozen mixed battery pool |
| **WritingBench** | GitHub `X-PLUG/WritingBench` (1,000) | 500 | Hash-ordered split: 500 calibration / 500 deployment |
| **MultiChallenge** | Hugging Face `ScaleAI/MultiChallenge` (266) | 128 | Split 138 calibration / 128 deployment; multi-turn conversations kept as native messages |
| **RefineBench** | Hugging Face `RefineBench/RefineBench`, Free-Form split (478) | 278 | Field-stratified split 200 calibration / 278 deployment |
| **IFBench** | GitHub `allenai/IFBench`, `IFBench_test.jsonl` (300) | 150 | Matched split 150 calibration / 150 deployment |
| **Hard-MATH** | MATH-500 (levels 4 and 5) followed by AIME 2022&ndash;2025 | 382 | 262 MATH-500 problems then 120 AIME problems; calibration prompts from the frozen math pool |
| **BigCodeBench-Hard** | Hugging Face `bigcode/bigcodebench-hard`, Instruct split, v0.1.4 (148) | 148 | Official order; calibration prompts from the frozen HumanEval+ / MBPP+ pool (`evalplus/humanevalplus`, `evalplus/mbppplus`) |
| **SpreadsheetBench-2** | GitHub `RUCKBReasoning/SpreadsheetBench-2` (checker) + Hugging Face `KAKA22/SpreadsheetBench-v2` (data) | 75 | Frozen 100-task manifest (Template 30 / Financial Model 40 / Debugging 30); the paper and `configs/ssb2_codex.yaml` deploy its first 75 tasks (`benchmark.n_tasks`) with checkpoints after tasks 34 and 67 |

Prompt-only frozen bundles of every benchmark (the deployment prompts in their frozen order plus the calibration pools used in the paper) ship under `data/frozen/<benchmark>/`, and `bash process_data.sh` builds `data/processed/<benchmark>/{tasks.jsonl,calibration.jsonl,eval_sidecar.json.gz}` from them by default. Each `data/prepare_<benchmark>.py` can instead re-pin the official source from the Hugging Face hub or GitHub; see `data/README.md` for the schema of every file. SpreadsheetBench-2 is prepared by `data/prepare_ssb2.py`, which pins the dataset and exports the agent-visible task tree (instruction and input workbook) under `data/raw/ssb2/`; the runner adds the task skill file (`AGENTS.md`) and the recalculation helper to each task's working directory at run time. It is the one benchmark that `python -m data.prepare --benchmark all` leaves out, because it has no frozen bundle and needs the network.

## Pipeline

The lifecycle follows Algorithm 1 of the paper. The harness state is a bounded procedural memory in the main experiments and the experience block of a coding agent in the agent-harness study; everything else is shared.

1. **Relation discovery** (`src/ucoe/relations.py`, optional). The actor and a larger frozen comparison model answer the same development prompts. The verifier model reads each pair in random order with system identities hidden and names atomic, task-general differences. A relation is kept only if it recurs in at least three prompts and can be written as one binary criterion with one edit that breaks that property alone. Retained relations become benchmark-specific contrast families and extra criteria offered to synthesis.
2. **Candidate synthesis** (`src/ucoe/verifier.py`, `src/ucoe/synthesis.py`). From the public task requirements, the shared criteria and the discovered relations, the verifier model proposes candidate verifier programs. A program is a set of criteria, each with explicit 0/1 anchors (semantic criteria are scored by the verifier model, mechanical ones are computed directly), plus one aggregation expression over the scores that must stay within [0, 1]. A candidate may append at most two criteria of its own. A static validator discards programs with undefined criteria, new criteria that do not affect the output, or probe score vectors that produce non-finite or out-of-range values.
3. **Response contrasts** (`src/ucoe/contrasts.py`). Each contrast starts from the actor's own answer to a public calibration prompt disjoint from deployment and applies one targeted edit: omission of the two most information-dense sentences (or lines of a code-dominated answer), injection of an unsupported claim, or insertion of irrelevant material; story generation uses narrative edits of the same kind, and each relation family uses its own declared edit. Sources are assigned to the selection or held-out split before any verifier is scored, a source contributes at most one contrast per family, and every checkpoint draws fresh sources. A family that cannot fill its quota records the shortfall and the run continues.
4. **Driving-capacity selection** (`src/ucoe/selection.py`). Every candidate scores the same contrasts with transformation names and preference directions hidden. A candidate succeeds on a contrast only if its reward for the base response exceeds the degraded one by more than the margin ε; a tie, a sub-margin lead or an invalid score is a failure. Each family gets a centered accuracy J = 2p − 1, and the candidate with the largest worst-family J (rounded to four decimals) wins, with ties broken by mean accuracy, then aggregation simplicity, then fixed candidate order. The initial winner becomes the deployed verifier.
5. **Verifier-guided repair, acceptance and procedural memory** (`src/ucoe/revision.py`, `src/ucoe/memory.py`, `src/ucoe/lifecycle.py`). For each task the actor drafts with the retrieved procedures as guidelines (Algorithm 1; `ucoe.memory.at_draft: false` instead serves the sealed Direct Generation draft byte for byte and retrieves only for the revision, which is how the runs behind the reported tables were configured). The deployed verifier scores the draft, takes the weakest participating criterion (deterministic criteria included), and the diagnosis call returns a localized witness and one repair instruction. The actor makes one local repair, the same verifier rescores it, and the revision is served only if its reward exceeds the draft's by more than ε; unusable feedback, a missing revision or an invalid score serves the draft. Only after the response is returned may an accepted repair be distilled into at most one generalized trigger/action procedure (trigger ≤ 10 words, action ≤ 14 words, no task-specific entities or numbers); a rejected revision updates usage statistics only. Retrieval ranks procedures by criterion match, dimension match and acceptance record and returns at most two within 64 words; near duplicates are dropped at token-set Jaccard 0.75, and the capacity is 12 procedures overall and 2 per dimension, evicting the weakest record with ties broken by age.
6. **Checkpoint evolution** (`src/ucoe/selection.py`, `promotion`). At each scheduled checkpoint UCoE summarizes up to the last 128 completed tasks (revision-acceptance rate, most-often-weakest criteria, harness-state size) and synthesizes one challenger from the current verifier's definitions, which may append criteria or change the aggregation. The current verifier joins the pool, maximin selection is repeated on that checkpoint's fresh contrasts, and the winner replaces the current verifier only if it (i) strictly improves worst-family accuracy on the selection split, (ii) wins more held-out contrasts than it loses against the current verifier on the same contrasts, and (iii) regresses on no existing family by more than κ = 0.10. Any failed or undefined comparison retains the current verifier, and an adoption first affects the next task.

Nothing from benchmark answers or final-evaluator feedback enters deployment at any step.

### Evaluation

`bash eval.sh` scores the served responses of an arm against the matched direct-generation response of the same actor. The judge model compares the two full responses in **both presentation orders**, blinded and with deterministic decoding; the two verdicts are merged by sign into a task-level win, tie or loss (identical texts are mechanical ties), and

    Pref = 100 · (W + T/2) / (W + T + L),

so 50 is parity. The report includes a block bootstrap interval (blocks of 64 tasks, 50,000 resamples), an exact sign test and Holm correction over the arms compared. The `native` mode runs each benchmark's own metric where one exists: WildBench score (1&ndash;10, official template), WritingBench Critic-7B score, MultiChallenge hidden-rubric accuracy, RefineBench checklist accuracy, IFBench prompt-level accuracy from the vendored official evaluator (loose by default, the 34.67 / 30.00 of Table 2; `evaluation.native.ifbench_primary: strict_prompt_acc` selects the strict variant, and all four official figures, strict and loose at the prompt and instruction level, are written), Hard-MATH answer accuracy, and BigCodeBench-Hard Pass@1 by execution. The Arena-Hard style-controlled win rate comes from the official `arena-hard-auto` toolkit, which is not part of this release: `bash eval.sh configs/arena_hard_qwen8b.yaml native` exports each arm's served answers in the toolkit's `model_answer` layout under `<work_dir>/eval/arena_hard_official/` and prints the `gen_judgment` / `show_result` commands with the paper's judge (gpt-4.1-2025-04-14) and anchor (o3-mini-2025-01-31). SpreadsheetBench-2 uses the benchmark checker with LibreOffice recalculation (`run_agent_harness.sh ... official`).

## Reproducing paper results

Every preset in `configs/` reproduces one paper setting; its header comment names the table and row. The actors are `Qwen/Qwen3-8B` and `Qwen/Qwen3.8-27B`, the verifier model (program synthesis, semantic criterion scoring, diagnosis and distillation) is `Qwen/Qwen3.6-35B-A3B`, the comparison model for relation discovery is Qwen3.8-27B for the 8B actor and Qwen3.5-397B-A17B for the 27B actor, and the final judge is `Qwen/Qwen3.5-397B-A17B`. The main experiments serve all of them through the Tinker API.

**Table 1, open-ended preference against direct generation (50 = parity).**

| Preset | Benchmark | Relation discovery | Checkpoints | UCoE (paper) |
|--------|-----------|--------------------|-------------|--------------|
| `arena_hard_qwen8b.yaml` | Arena-Hard, Qwen3-8B | off | 128, 256 | 58.30 |
| `writingprompts_qwen8b.yaml` | WritingPrompts, Qwen3-8B | off | 512, 1024 | 62.79 |
| `wildbench_qwen8b.yaml` | WildBench, Qwen3-8B | off | 256, 512 | 57.42 |
| `writingbench_qwen8b.yaml` | WritingBench, Qwen3-8B | on | 128, 256 | 56.90 |
| `multichallenge_qwen8b.yaml` | MultiChallenge, Qwen3-8B | off | 48, 96 | 57.42 |
| `refinebench_qwen8b.yaml` | RefineBench, Qwen3-8B | on | 96, 192 | 61.15 |
| `arena_hard_qwen27b.yaml` | Arena-Hard, Qwen3.8-27B | on | 128, 256 | 59.70 |
| `writingprompts_qwen27b.yaml` | WritingPrompts, Qwen3.8-27B | on | 512, 1024 | 63.92 |
| `wildbench_qwen27b.yaml` | WildBench, Qwen3.8-27B | on | 256, 512 | 55.42 |
| `writingbench_qwen27b.yaml` | WritingBench, Qwen3.8-27B | on | 128, 256 | 54.40 |
| `multichallenge_qwen27b.yaml` | MultiChallenge, Qwen3.8-27B | off | 48, 96 | 51.95 |
| `refinebench_qwen27b.yaml` | RefineBench, Qwen3.8-27B | off | 96, 192 | 57.73 |

The six-benchmark means are 59.00 (Qwen3-8B) and 57.19 (Qwen3.8-27B).

**Table 2, verifiable tasks with Qwen3-8B** (accuracy for IFBench and Hard-MATH, Pass@1 for BigCodeBench-Hard; direct generation in parentheses).

| Preset | Benchmark | Checkpoints | UCoE (paper) |
|--------|-----------|-------------|--------------|
| `ifbench_qwen8b.yaml` | IFBench | 48, 96 | 34.67 (30.00) |
| `hard_math_qwen8b.yaml` | Hard-MATH | 128, 256 | 59.16 (56.54) |
| `bigcodebench_qwen8b.yaml` | BigCodeBench-Hard | 48, 96 | 21.62 (21.62) |

The three-benchmark mean rises from 36.06 to 38.48. Run `bash eval.sh <preset> all` to get both the preference score and the native metric.

The relation-discovery column and the checkpoint schedules are those of the research runs behind each cell; the paper states them only for Arena-Hard. Where a run departed from the Arena-Hard defaults, the preset header says so and the preset carries the run's values: IFBench, MultiChallenge and RefineBench draw 8 selection / 40 (36 for MultiChallenge) held-out sources per family from their small calibration pools and may reuse a source across checkpoints; WritingPrompts, Hard-MATH and BigCodeBench-Hard use 48 held-out sources; every benchmark other than Arena-Hard revises at temperature 0.3, and WritingBench, MultiChallenge, RefineBench and IFBench draft greedily.

**Arena-Hard ablations with Qwen3-8B** (preference against direct generation). The paper removes one component at a time from `arena_hard_qwen8b.yaml` (58.30): *fixed verifier* (56.60) freezes the initial verifier, which corresponds to `ucoe.checkpoints: []`; *no memory updates* (56.70) disables harness-state updates from accepted repairs, which corresponds to `ucoe.memory.capacity: 0`; *random initial verifier* (56.70) replaces driving-capacity selection with a seeded random draw from the candidate pool, and *harness-state updates only* (50.80) removes the current-task repair so that the verifier only updates the harness state. The last two change the algorithm rather than a configuration value and are not shipped as presets.

**Agent-harness study on SpreadsheetBench-2** (`configs/ssb2_codex.yaml`). UCoE runs inside the Codex coding-agent harness with GPT-5.6-luna (medium reasoning effort) as the actor and GPT-5.6-sol (extra-high reasoning effort) as the verifier model, both through the Codex app server. Every actor and verifier call is a fresh Codex session in an isolated task directory with a shell running Python and `openpyxl`, file editing confined to that directory, a workbook recalculation helper, no network and no approvals. The harness state is an experience block of at most 12 trigger/procedure entries over five dimensions (requirements, formulas, references, preservation, numbers), at most 3 per dimension and 2,400 characters in total. Candidate verifiers keep a fixed catalogue of base criteria (requirements implemented, formulas correct, references intended, unrelated content preserved, values plausible, no recalculated error values, workbook changed), may append at most two criteria, and are selected on the contrast families `formula_error`, `broken_reference` and `regression_damage` with 2 initial candidates, 4 selection / 2 held-out sources per family, ε = 0.01 and checkpoints after tasks 34 and 67. The commands run in order:

```bash
bash process_data.sh configs/ssb2_codex.yaml                      # pin the checker and the dataset, build the task tree
bash run_agent_harness.sh configs/ssb2_codex.yaml manifest        # freeze the task stream and the calibration chunks
bash run_agent_harness.sh configs/ssb2_codex.yaml pin-probe       # verify the served model and effort of every role
bash run_agent_harness.sh configs/ssb2_codex.yaml smoke           # live smoke on two reserved calibration tasks
bash run_agent_harness.sh configs/ssb2_codex.yaml run --arm ucoe  # also: --arm native (direct generation), --arm judge (LLM as a Judge)
bash run_agent_harness.sh configs/ssb2_codex.yaml official        # benchmark checker with LibreOffice recalculation
bash run_agent_harness.sh configs/ssb2_codex.yaml report
```

`run` is fail-closed: it refuses to start without a passing pin probe and a smoke report bound to the current protocol and roles. The UCoE arm builds its calibration drafts, contrast banks and initial verifier on demand; `calib-drafts`, `banks` and `h0` run those preparatory steps separately, `status` and `audit` inspect a run, and `bash run_agent_harness.sh configs/ssb2_codex.yaml all` executes the whole sequence for the three arms. Everything lands under `output.work_dir` (`manifest.json`, `state/`, `tasks/<arm>/`, `ucoe/`, `smoke/` for the live smoke's own work directory, `official.json`, `report.json`). `--initial-candidates K` must accompany `manifest` (or be set as `ucoe.initial_candidates` before it runs), because K is frozen into the manifest and every later command refuses a different value.

The paper reports 22.7% exact-match accuracy for UCoE against 17.3% for both direct generation and LLM as a Judge on the 75-task stream.

**Default hyperparameters** (shared by the Qwen3-8B and Qwen3.8-27B runs; `config.yaml` already matches them):

- Verifier synthesis: 2 initial candidates, 1 new candidate per checkpoint, at most 2 new criteria per candidate
- Response contrasts: 3 reference families with 24 selection / 96 held-out sources per family; 12 / 24 sources for each relation family
- Response revision: 1 revision attempt per task, acceptance margin ε = 0.01
- Procedural memory: capacity 12, at most 2 entries per dimension, at most 2 retrieved procedures within 64 words, at most 1 new procedure per accepted revision, token-Jaccard deduplication at 0.75
- Verifier updates: checkpoints after tasks 128 and 256 on Arena-Hard (per-benchmark schedules above), regression tolerance κ = 0.10, the relative update rule
- Decoding: generation / revision temperature 0.7 on Arena-Hard (the other benchmarks' suite values are listed in their presets), scoring / synthesis / distillation temperature 0

`output.ceilings` caps the number of new calls per model class (actor, verifier, judge, comparator) and every run records actor, verifier-model and judge calls separately.

## Citation

```bibtex
@misc{xu2026ucoe,
      title={Verifier--Harness Co-Evolution for Training-Free Self-Improvement on Open-Ended Tasks},
      author={Tianyi Xu and Yaolun Zhang and Yiran Wu and Joe Nguyen and Shengyu Dai and Saba Moeinizade and Zhenwen Shao and Qingyun Wu and Huazheng Wang},
      year={2026},
      note={Under review},
}
```

## License

MIT. See [LICENSE](LICENSE).
