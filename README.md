# DriftBench: Constraint Adherence in Multi-Turn LLM Ideation

DriftBench measures whether LLMs preserve fidelity to a structured research
brief across multiple turns of iterative pressure, or drift toward locally
coherent but globally misaligned elaborations.

- **Code**: this repository (MIT licensed).
- **Data**: published separately on HuggingFace —
  [`anonymous-driftbench/DriftBench`](https://huggingface.co/datasets/anonymous-driftbench/DriftBench).
- **Paper**: under review.

---

## Reproduce the paper numbers in five minutes (no API keys)

```bash
git clone [GITHUB_URL_PLACEHOLDER] DriftBench && cd DriftBench
python3.13 -m venv .venv
.venv/bin/pip install -e .

bash scripts/reproduce.sh --hf
```

`--hf` mode pulls the HuggingFace snapshot, symlinks it at the canonical
local paths the analysis pipeline expects, re-aggregates scores, regenerates
every figure, and rewrites the LaTeX macros file the paper `\input{}`s. No
API calls are made — every headline number drops out of the analysis on
artifacts already on HF.

Other reproduction modes:

```bash
bash scripts/reproduce.sh --aggregate   # re-aggregate locally-staged JSONL into parquet
bash scripts/reproduce.sh --analyze     # aggregate + run all priority-list analyses + figures
bash scripts/reproduce.sh --full        # full pipeline incl. subjects/judges (uses your API quota)
```

---

## Evaluating your own model on DriftBench

`scripts/evaluate_model.py` scores any [litellm](https://docs.litellm.ai/)-compatible
subject model on the published 38 briefs × 4 conditions, using the same
cross-family judge + auditor as the paper. No edits to `config.yaml` are
required — the script clones the base config, swaps in your model, adds a
default rate-limit entry for its provider, runs the pipeline, aggregates the
scores, and prints a comparison against the published 7-model baseline.

```bash
# Hosted API (e.g. OpenAI, Anthropic, Gemini)
export OPENAI_API_KEY="sk-..."
python scripts/evaluate_model.py --model openai/gpt-4o --max-budget <usd>

# Open-weight via Together.ai
export TOGETHER_API_KEY="..."
python scripts/evaluate_model.py \
  --model "together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo" \
  --max-budget <usd>

# Smoke test on one condition first
python scripts/evaluate_model.py \
  --model openai/gpt-4o \
  --conditions single_shot \
  --repetitions 1 \
  --max-budget <usd>
```

Outputs land at `drift_bench/data/external/<model-slug>/`:

```
drift_bench/data/external/openai_gpt-4o/
├── transcripts/                # one JSONL per (brief, condition, rep)
├── scores/                     # judge_*.jsonl + auditor_*.jsonl per run
├── aggregated/all_scores.parquet   # merged judge + auditor table
└── run_summary.json
```

The script prints a per-condition comparison vs the published benchmark,
e.g.:

```
=== openai/gpt-4o per-condition mean scores ===
                          n  obj_fid  constr_adh  alt_cov  cplx_inf  drift_rate
condition
single_shot              38    3.842       3.711    3.158     0.421       0.158
multi_turn_neutral       38    3.789       3.658    3.184     0.789       0.184
multi_turn_pressure      38    3.605       3.211    3.421     1.842       0.421
checkpointed_pressure    38    3.737       3.421    3.526     1.605       0.342

=== Published benchmark baseline (mean over all 7 models) ===
...
```

Re-running the same command resumes — existing transcripts and scores are
detected on disk and skipped. To re-score with a different judge, point
`drift_bench/judges/judge.py` at a different judge id and re-run.

---

## Running the full published pipeline from scratch

```bash
source .venv/bin/activate
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="AI..."

# Minimal pilot (1 model, single-shot only) — pass --max-budget to cap API spend
python -m drift_bench.pipeline \
  --conditions single_shot \
  --models openai/gpt-5.4 \
  --repetitions 1 \
  --max-budget <usd>

# Full main experiment
python -m drift_bench.pipeline --max-budget <usd>
```

---

## Experimental design

### Research question

Do LLMs preserve global fidelity to an initial scientific objective during
iterative multi-turn refinement, or drift into locally salient but globally
misaligned elaborations?

### Conditions

| Condition | Turns | Description |
|---|---|---|
| `single_shot` | 1 | Baseline: one prompt, one response |
| `multi_turn_neutral` | 6 | Neutral prompts ("Continue.") — controls for turn count |
| `multi_turn_pressure` | 6 | Pressure prompts pushing for novelty, rigor, robustness |
| `checkpointed_pressure` | 8 | Pressure + structured reflection checkpoints |

### Models (subjects)

`openai/gpt-5.4`, `openai/gpt-5.4-mini`, `anthropic/claude-sonnet-4-6`,
`gemini/gemini-3.1-pro-preview`, `gemini/gemini-3.1-flash-lite-preview`,
`qwen/qwen3-235b`, `meta-llama/llama-3.3-70b-instruct`.

### Judging

- **Judge**: Claude Opus 4.6 (cross-family — GPT-5.4 judges Anthropic runs).
- **Auditor**: GPT-5.4 (independent drift classification + recoverability).
- **Dimensions**: objective fidelity, constraint adherence, alternative
  coverage, complexity inflation (each 0–4); recoverability (0–4).
- **Inter-rater reliability**: quadratic-weighted Cohen's kappa and
  Krippendorff's alpha between judge and auditor.

### Briefs

38 structured research briefs across 24 scientific domains. Each contains:
- A specific, testable research objective.
- Hard constraints (binary-checkable).
- Success criteria.
- Plausible alternative directions (genuinely competitive).
- Banned moves (tempting but prohibited).

### Additional instrumentation

- **Restatement probe**: at each turn, a separate API call checks whether
  the model can reproduce the original constraints (separates drift from
  forgetting).
- **Brief re-injection**: on a subset, the full brief is pasted into every
  turn to definitively rule out context-window loss.

---

## Repository layout

```
drift_bench/
  config.yaml         model IDs, rate limits, budget
  pipeline.py         single entry point: subjects → judge → auditor → summary
  briefs/             38 YAML research briefs across 24 domains
  prompts/            13 Jinja2 templates (system, conditions, judge, auditor, probe)
  runners/            single_shot, multi_turn, checkpointed, monitored
  judges/             cross-family judge + auditor + blind/structured variants
  analysis/           aggregate.py, stats.py, reliability.py, verbosity.py,
                      probes.py, validation.py, debiasing.py, figures.py,
                      release_manifest.py
  schema/             JSON Schema for brief validation
  data/               output (gitignored): transcripts, scores, aggregated, analysis,
                      figures — populated either by running the pipeline or by
                      `scripts/fetch_from_hf.py`

scripts/
  build_dataset_release.py   stage local data → Dataset/ for HF push
  fetch_from_hf.py           HF snapshot → symlinks at canonical local paths
  reproduce.sh               --hf | --aggregate | --analyze | --full
  evaluate_model.py          score a new subject model on DriftBench

docs/
  dataset_card.md            master HF README (the build script copies this
                             into Dataset/README.md)
```

---

## CLI reference

```
python -m drift_bench.pipeline [OPTIONS]

--config PATH          config file (default: drift_bench/config.yaml)
--briefs-dir PATH      briefs directory (default: drift_bench/briefs)
--output PATH          output directory (default: drift_bench/data)
--conditions LIST      conditions to run (default: all)
--models LIST          model IDs to run (default: all)
--repetitions N        reps per cell (default: from config)
--concurrency N        max concurrent API calls (default: 5)
--max-budget FLOAT     hard cap on API spend in USD (default: from config)
--no-probe             disable restatement probes
--reinjection-ids LIST brief IDs for brief re-injection
--skip-scoring         run subjects only (no judge/auditor)
```

Priority-list analyses (each item produces one or more CSVs/JSONs that the
paper macros derive from):

```
python -m drift_bench.run_analysis --items all     # 1..5
python -m drift_bench.run_analysis --items 1       # inter-rater reliability
python -m drift_bench.run_analysis --items 2       # verbosity-controlled regression
python -m drift_bench.run_analysis --items 3       # restatement probes + surface gap
python -m drift_bench.run_analysis --items 4       # judge validation
python -m drift_bench.run_analysis --items 5       # debiasing
```

Release manifest (regenerates the per-snapshot file inventory and the
`\input{}`-able LaTeX macros file):

```
python -m drift_bench.analysis.release_manifest \
  --output drift_bench/data/analysis/release_manifest.json \
  --tex-output drift_bench/data/analysis/release_macros.tex
```

---

## Pipeline features

- **Resumability**: skips completed runs on restart (checks disk for existing
  transcripts/scores).
- **Budget cap**: configurable max spend with real-time tracking
  (`--max-budget`). Checked inside semaphore to prevent race conditions.
- **Cross-family judging**: Claude judges GPT/Gemini runs; GPT judges Claude
  runs. Never self-judge.
- **Rate limiting**: per-provider token-bucket with sleep-outside-lock pattern.
- **Retry**: covers 429, 500, 503, connection errors, timeouts (5 attempts,
  exponential backoff).

---

## Releasing a new HuggingFace snapshot (maintainers)

```bash
# 1. Stage the dataset locally and verify gates pass
python scripts/build_dataset_release.py --clean

# 2. Sanity checks
du -sh Dataset/
find Dataset/ -name "*.pdf" -o -name "*.html" | wc -l    # MUST be 0
grep -ricf <(jq -r '.forbidden_substrings[]' .anonymize.json) Dataset/  # MUST be 0

# 3. Push to HF (one-time setup: pip install huggingface-hub && huggingface-cli login)
huggingface-cli upload-large-folder \
  --repo-type=dataset anonymous-driftbench/DriftBench Dataset/

# 4. Verify the loop in a fresh tmpdir
mkdir /tmp/repro && cd /tmp/repro
git clone https://github.com/$GH_REPO . && python -m venv .venv
.venv/bin/pip install -e .
bash scripts/reproduce.sh --hf
```

The build script enforces (and fails the build on):

- No files with `.pdf` / `.html` extensions land under `Dataset/`.
- No forbidden substring (configured in `.anonymize.json`) appears in any
  text file.
- A SHA-256 manifest is written for every shipped file.

---

## Citation

```bibtex
@misc{driftbench2026,
  title  = {Models Recall What They Violate: Constraint Adherence in Multi-Turn LLM Ideation},
  author = {Anonymous},
  year   = {2026},
  url    = {https://huggingface.co/datasets/anonymous-driftbench/DriftBench}
}
```

## License

- Source code: **MIT** (see `LICENSE`).
- Dataset (transcripts, scores, analysis outputs on HuggingFace): **CC-BY 4.0**.
- Human-annotation files: **CC-BY-NC 4.0**.
