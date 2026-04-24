# DriftBench: Trajectory Drift in Multi-Turn LLM-Assisted Scientific Ideation

A benchmark and evaluation protocol for measuring **trajectory drift** — the process-level failure mode where LLMs become increasingly detailed and locally coherent while progressively losing fidelity to the original research objective, constraints, or alternative lines of inquiry during multi-turn interaction.

## Quick Start

```bash
# Setup
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Set API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="AI..."

# Run a minimal pilot (1 brief, 1 model, single-shot)
python -m drift_bench.pipeline \
  --conditions single_shot \
  --models openai/gpt-5.4 \
  --repetitions 1 \
  --max-budget 5

# Run full experiment
python -m drift_bench.pipeline --max-budget 400

# Aggregate and analyze
python -c "
from drift_bench.analysis.aggregate import aggregate_scores
from drift_bench.analysis.stats import *
from drift_bench.analysis.figures import generate_all_figures
from pathlib import Path

df = aggregate_scores(Path('drift_bench/data/scores'), Path('drift_bench/data/aggregated/all_scores.parquet'))
print(compute_condition_effects(df))
print(pairwise_condition_tests(df))
print(compute_inter_rater_reliability(df))
generate_all_figures(Path('drift_bench/data/aggregated/all_scores.parquet'), Path('drift_bench/data/figures'))
"
```

## Experimental Design

### Research Question

Do LLMs preserve global fidelity to an initial scientific objective during iterative multi-turn refinement, or do they drift into locally salient but globally misaligned elaborations?

### Conditions

| Condition | Turns | Description |
|---|---|---|
| `single_shot` | 1 | Baseline: one prompt, one response |
| `multi_turn_neutral` | 6 | Neutral prompts ("Continue.") — controls for turn count without pressure |
| `multi_turn_pressure` | 6 | Pressure prompts pushing for novelty, rigor, robustness |
| `checkpointed_pressure` | 8 | Pressure + structured reflection checkpoints after turns 2 and 4 |

### Models (Subjects)

- `openai/gpt-5.4` (OpenAI)
- `anthropic/claude-sonnet-4-6` (Anthropic)
- `gemini/gemini-3.1-pro-preview` (Google)

### Scoring

- **Judge**: Claude Opus 4.6 (cross-family: GPT-5.4 judges Claude runs)
- **Auditor**: GPT-5.4 (independent drift classification + recoverability)
- **Dimensions**: objective fidelity (0-4), constraint adherence (0-4), alternative coverage (0-4), complexity inflation (0-2), recoverability (0-4)
- **Inter-rater reliability**: quadratic-weighted Cohen's kappa between judge and auditor

### Briefs

38 structured research briefs across 24 scientific domains. Each brief contains:
- A specific, testable research objective
- Hard constraints (binary-checkable)
- Success criteria
- Plausible alternative directions (genuinely competitive)
- Banned moves (tempting but prohibited)

### Additional Instrumentation

- **Restatement probe**: At each turn, a separate API call checks whether the model can reproduce the original constraints (separates drift from forgetting)
- **Brief re-injection**: On a subset, the full brief is pasted into every turn to definitively rule out context-window loss

## Repository Structure

```
drift_bench/
  briefs/           38 YAML research briefs
  prompts/          Jinja2 templates (system, conditions, judge, auditor)
  runners/          Conversation execution (single-shot, multi-turn, checkpointed)
  judges/           LLM-as-judge scoring + fresh-auditor evaluation
  analysis/         Statistics (Wilcoxon, effect sizes, kappa) + figures
  schema/           JSON Schema for brief validation
  config.yaml       Model IDs, rate limits, budget
  pipeline.py       Single entry point: runs subjects, judges, auditors
  data/             Output (gitignored): transcripts, scores, aggregated
```

## Release Audit

The repository contains multiple analysis branches: the main seven-model benchmark release, the five-model core subset, the open-weight extension, monitored-warning runs, and ancillary temperature / rigor-pressure follow-ups. To keep manuscript counts tied to the files that are actually present on disk, generate a machine-readable manifest before freezing any paper version:

```bash
python -m drift_bench.analysis.release_manifest \
  --output drift_bench/data/analysis/release_manifest.json
```

The same command also writes `drift_bench/data/analysis/release_macros.tex`, which the paper can `\input{}` directly so benchmark and validation counts refresh from the audited snapshot rather than from manual edits.

The manifest reports:

- benchmark-release counts and model/condition breakdowns
- core/open-weight/monitored/follow-up scope separation
- blind-judge and structured-judge coverage for the core subset
- duplicate / irregular cell counts in the current snapshot
- human-validation completion status without exposing raw rater identities
- hashes for key config, manuscript, and analysis files

Human-validation protocol and data are available in the [HuggingFace dataset](https://huggingface.co/datasets/kruthof/DriftBench).

## Pipeline Features

- **Resumability**: Skips completed runs on restart (checks disk for existing transcripts/scores)
- **Budget cap**: Configurable max spend with real-time tracking (`--max-budget`)
- **Cross-family judging**: Claude judges GPT/Gemini runs; GPT judges Claude runs
- **Rate limiting**: Per-provider token-bucket with sleep-outside-lock pattern
- **Retry**: Covers 429, 500, 503, connection errors, timeouts (5 attempts, exponential backoff)

## CLI Reference

```
python -m drift_bench.pipeline [OPTIONS]

--config PATH          Config file (default: drift_bench/config.yaml)
--briefs-dir PATH      Briefs directory (default: drift_bench/briefs)
--output PATH          Output directory (default: drift_bench/data)
--conditions LIST      Conditions to run (default: all)
--models LIST          Model IDs to run (default: all)
--repetitions N        Reps per cell (default: from config)
--concurrency N        Max concurrent API calls (default: 5)
--max-budget FLOAT     Budget cap in USD (default: from config)
--no-probe             Disable restatement probes
--reinjection-ids LIST Brief IDs for brief re-injection
--skip-scoring         Run subjects only (no judge/auditor)
```

