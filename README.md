# DriftBench: Constraint Adherence Under Iterative Pressure in Multi-Turn LLM Ideation

DriftBench measures whether LLMs keep the hard constraints of a structured
research brief across multiple turns of iterative pressure, or drift toward
locally coherent but globally misaligned elaborations.

- **Code**: this repository, <https://github.com/kruthof/driftbench>.
- **Data**: published separately on Hugging Face,
  [`driftbench/DriftBench`](https://huggingface.co/datasets/driftbench/DriftBench).
- **Paper**: Garvin Kruthof. *Constraint Adherence Under Iterative Pressure in
  Multi-Turn LLM Ideation.* NeurIPS 2026 Evaluations and Datasets Track. Read
  the PDF on [OpenReview](https://openreview.net/forum?id=9kbTg0tKKr) or arXiv.
  This repository holds the code only, not the LaTeX sources.
- **Changes since the submission**: [`CHANGELOG.md`](CHANGELOG.md) records every
  correction to the released artifact and every camera-ready addition.

---

## Reproduce the paper's numbers

No API keys are needed. Use Python 3.13 and the pinned versions in
`requirements-lock.txt` (the versions that produced the paper's numbers):

```bash
git clone https://github.com/kruthof/driftbench DriftBench && cd DriftBench
python3.13 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/pip install -e . --no-deps

export HF_TOKEN=hf_...   # optional: a free Hugging Face read token avoids rate limits
bash scripts/reproduce.sh --hf
```

The run downloads about 19,000 files and then takes a few minutes (about 5 on
a laptop). Without `HF_TOKEN` the download can stop with HTTP 429 (rate limit);
set the token and re-run, and the download resumes where it stopped. The
fetch step checks the download against the snapshot's `MANIFEST.txt` and stops
with an error if listed files are missing. The
download is the Hugging Face tag `camera-ready`, the exact snapshot the paper
used (`bash scripts/reproduce.sh --hf --revision main` follows the latest
commit instead).

The run ends by comparing every regenerated file that carries the paper's
numbers with the snapshot's `MANIFEST.txt` (`scripts/verify_reproduction.py`).
Expected final output:

```
verify_reproduction: regenerated files under drift_bench/data vs drift_bench/data/hf_cache/MANIFEST.txt (… entries)
  cr_macros.json             1/1 identical
  camera-ready CSVs          75/75 identical
  release_macros.tex         1/1 identical
  figures (numbers drawn)    6/6 identical
  figures (PDF bytes)        6/6 identical
  note: camera_ready/analysis/swe_verify_results.csv is compared as shipped (swe_verify executes model-written code and is run by hand)
verify_reproduction: identical. Every checked file matches the release byte for byte.
```

`cr_macros.json` holds every `\cr...` macro value of the paper. The run is
byte-identical on macOS arm64; on other platforms the last floating-point
digits of intermediate CSV values may differ, and every printed number is
identical (the verify script checks this). There a file whose hash differs is
compared with the released copy, which `reproduce.sh` saves before the run: it
passes when the shape, the header and every text cell are equal, every number
is within a relative 1e-9, and every printed macro value is equal. The run
then ends with "identical at printed precision (last float digits differ)"
and exits 0. Anything else is listed cell by cell and exits 1. The figure PDFs
embed the font STIX Two Text, which ships with macOS; elsewhere their bytes
differ and the script only warns, because the numbers each figure draws are
checked separately.

`--hf` mode pulls the Hugging Face snapshot (`driftbench/DriftBench`), symlinks
it at the canonical local paths the analysis pipeline expects, re-aggregates
the scores, runs analysis items 1 to 6 (item 6 is every camera-ready module),
regenerates the submission-era figures and the six camera-ready figures
(`figures/`), rewrites the LaTeX macros file the paper `\input{}`s, and runs
the check above. No API calls are made. Because of the symlinks, the
regenerated files overwrite the downloaded copies; `MANIFEST.txt` keeps the
released hashes, and the run first saves the released copies of the checked
files to `drift_bench/data/released_reference/`.

The camera-ready numbers are `\cr...` macros (values kept from the submitted
version or the rebuttal are labelled so in the paper). The macros are
generated from CSVs under `drift_bench/data/camera_ready/analysis/`, each with
a `.provenance.json` sidecar that lists its inputs, filters and row counts. To
regenerate only the camera-ready layer:

```bash
# Every camera-ready CSV, then cr_macros.tex / cr_macros.json (macros run last)
python -m drift_bench.run_analysis --items 6

# Check the shipped macros against the CSVs without rewriting anything
python -m drift_bench.analysis.camera_ready.macros --verify

# Look up a macro: name, value and source CSV (regenerates cr_macros.tex/.json;
# pass --out-dir elsewhere to leave them untouched)
python -m drift_bench.analysis.camera_ready.macros --list | grep -i kbv
```

Item 6 imports every module in `drift_bench/analysis/camera_ready/` that has a
`run()` function and runs them in alphabetical order, with `e7_summary` and
`macros` last. A failing module is reported and does not stop the others. It
makes no API calls. `reproduce.sh` sets `LITELLM_LOCAL_MODEL_COST_MAP=True`, so litellm
uses the model map bundled with the pinned version instead of downloading the
current one (the token analysis reads context windows from it); set it yourself when
running item 6 by hand. Two modules are skipped
on purpose. `reproduce.sh` runs `figures_cr` after item 6; the SWE harness must
be run by hand:

```bash
# The six camera-ready figures (vector PDFs plus provenance sidecars in figures/)
python -m drift_bench.analysis.camera_ready.figures_cr
python -m drift_bench.analysis.camera_ready.figures_cr --only kbv_by_judge_arm trajectories

# The SWE executable-verification harness. It EXECUTES model-written code
# (temp directory, timeout, process-group kill; no sandbox). Read the code first.
python -m drift_bench.analysis.camera_ready.swe_verify --extract-only
python -m drift_bench.analysis.camera_ready.swe_verify --config drift_bench/config_swe_code.yaml
```

`figures_cr` reads only the CSVs and data on disk. Two runs on the same inputs
give byte-identical PDFs.

Other reproduction modes:

```bash
bash scripts/reproduce.sh --aggregate   # re-aggregate locally-staged JSONL into parquet
bash scripts/reproduce.sh --analyze     # aggregate + run analysis items 1-6 + figures + macros + check
bash scripts/reproduce.sh --full        # full pipeline incl. subjects/judges (uses your API quota)
```

### Camera-ready data

The camera-ready experiments (crossed-judge calibration, Sonnet without the
output cap, Gemini Pro probe replay, structured Opus scores, per-turn
re-scoring, the extended SWE pilot, strict probe re-grading, and the LLM
annotator labels for the typology and the mechanism hand-check) ship on
Hugging Face under `camera_ready/`. `scripts/fetch_from_hf.py` (run by
`reproduce.sh --hf`) links them at `drift_bench/data/camera_ready/`. Each
experiment directory has a `README.md` with the exact command, date, model ids
and cost. The one exception is `e6_swe_verify/` (the executable SWE arm): its
commands are in `e6_swe/README.md`. See [`CHANGELOG.md`](CHANGELOG.md) for
what each experiment changed and why.

---

## Evaluating your own model on DriftBench

`scripts/evaluate_model.py` scores any [litellm](https://docs.litellm.ai/)-compatible
subject model on the published 38 briefs × 4 conditions, using the same
published-arm judge and auditor as the paper (Claude Opus 4.6 judges
non-Anthropic subjects, GPT-5.4 judges Anthropic subjects; GPT-5.4 audits).
No edits to `config.yaml` are required. The script clones the base config,
swaps in your model, adds a default rate-limit entry for its provider, runs the
pipeline, aggregates the scores, and prints a comparison against the published
7-model baseline. The comparison needs the published scores at
`drift_bench/data/aggregated/all_scores.parquet`; without that file the script
prints your model's table only. Fetch the Hugging Face data first
(`bash scripts/reproduce.sh --hf`), or pass `--baseline-parquet <path>`.

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

When the baseline parquet is present, the script prints a per-condition
comparison vs the published benchmark, e.g.:

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

Re-running the same command resumes: existing transcripts and scores are
detected on disk and skipped.

**Report a new model the way the paper does.** The camera-ready paper scores
the single-shot and pressure cells with both judges (Claude Opus 4.6 and
GPT-5.4), because binary KBV levels depend on the judge. Re-score your
transcripts with the judge the pipeline did not use, into a directory of its
own:

```bash
python -m drift_bench.rescore \
  --transcripts-dir drift_bench/data/external/openai_gpt-4o/transcripts \
  --scores-dir drift_bench/data/external/openai_gpt-4o/scores_gpt \
  --judge-override openai/gpt-5.4 --output-prefix judge_gpt \
  --mode judge --conditions single_shot multi_turn_pressure \
  --max-budget <usd> --dry-run      # drop --dry-run to score
```

`drift_bench/rescore.py` re-judges stored transcripts with any judge, mode
(`judge`, `structured` or `blind`) and turn index. It never calls the subject
model, is resumable, and refuses to write into the published score
directories. To swap the judge for a full pipeline run instead, change
`models.judges` in `drift_bench/config.yaml`.

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

When researchers refine ideas with an LLM over several turns, does the model
keep the constraints of the original brief?

### Conditions

| Condition | Turns | Description |
|---|---|---|
| `single_shot` | 1 | Baseline: one prompt, one response |
| `multi_turn_neutral` | 6 | Neutral prompts ("Continue.") — controls for turn count |
| `multi_turn_pressure` | 6 | Pressure prompts pushing for novelty, rigor, robustness |
| `checkpointed_pressure` | 8 | Pressure + checkpoints that re-elicit the brief after turns 2 and 4 |

### Models (subjects)

`openai/gpt-5.4`, `openai/gpt-5.4-mini`, `anthropic/claude-sonnet-4-6`,
`gemini/gemini-3.1-pro-preview`, `gemini/gemini-3.1-flash-lite-preview`,
`together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput`,
`together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo`.

### Judging

- **Published arm (all four conditions)**: cross-family. Claude Opus 4.6 judges
  non-Anthropic runs; GPT-5.4 judges Anthropic runs. No model judges its own
  outputs.
- **Crossed-judge design (camera-ready)**: both Claude Opus 4.6 and GPT-5.4
  score every run of all seven models in the single-shot and pressure cells.
  The common Opus 4.6 judge is the primary arm and the common GPT-5.4 judge the
  robustness check. Neutral and checkpointed cells have published-arm scores
  only.
- **Auditor**: GPT-5.4 (holistic drift classification + recoverability;
  supplementary to the judge).
- **Dimensions**: objective fidelity, constraint adherence, alternative
  coverage, complexity inflation (each 0–4); recoverability (0–4).
- **Derived metrics**: non-compliance (judge adherence < 4); knows-but-violates
  (KBV: non-compliant on a pressure run whose final restatement probe scores
  `overall_accuracy >= 0.8`); gKBV (mean share of hard constraints a structured
  judge marks stretched or violated).
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

- **Restatement probe**: after each of turns 1 to 5, a separate API call asks
  the model to restate the objective, hard constraints and banned moves. The
  answer is stored with `is_probe: true` and never shown back to the model. It
  tests whether the constraints stay legible and reproducible on demand.
- **Brief re-injection**: an optional pipeline flag (`--reinjection-ids`)
  pastes the full brief into every turn of the listed briefs.

---

## Repository layout

```
drift_bench/
  config.yaml         model IDs, rate limits, budget
  pipeline.py         single entry point: subjects → judge → auditor → summary
  rescore.py          re-judge stored transcripts with a chosen judge (camera-ready)
  run_analysis.py     analysis items 1-6
  briefs/             38 YAML research briefs across 24 domains
  briefs_swe/         SWE pilot briefs (outside the 38) and contract tests
  prompts/            Jinja2 templates (system, conditions, judge, auditor, probe)
  prompts_swe/        SWE pilot templates; prompts_swe_code/ for the executable arm
  runners/            single_shot, multi_turn, checkpointed, monitored, reprobe
  judges/             cross-family judge + auditor + blind/structured variants
  analysis/           aggregate.py, stats.py, reliability.py, verbosity.py,
                      probes.py, validation.py, debiasing.py, figures.py,
                      release_manifest.py
  analysis/camera_ready/
                      one module per camera-ready analysis; each has run()
                      and a CLI (python -m drift_bench.analysis.camera_ready.<name>)
  schema/             JSON Schema for brief validation
  data/               output (not in git): transcripts, scores, aggregated,
                      analysis, camera_ready, figures — populated either by
                      running the pipeline or by `scripts/fetch_from_hf.py`

figures/              camera-ready figure PDFs + provenance, written by
                      figures_cr.py (created on first run)

requirements-lock.txt pinned dependency versions that produced the paper's numbers

scripts/
  build_dataset_release.py   stage local data → Dataset/ for HF push
  fetch_from_hf.py           HF snapshot (tag camera-ready) → symlinks at canonical local paths
  reproduce.sh               --hf | --aggregate | --analyze | --full
  verify_reproduction.py     regenerated numbers vs the snapshot's MANIFEST.txt
  figures_reference.json     reference hashes of the six camera-ready figures
  evaluate_model.py          score a new subject model on DriftBench

docs/
  dataset_card.md            master HF README (the build script copies this
                             into Dataset/README.md)
  swe_pilot_recovered/       recovered SWE pilot briefs and prompts (read by
                             camera_ready/checkers.py and the SWE tests)
  swe_pilot_reconstruction.md  how the SWE pilot was reconstructed
```

---

## CLI reference

```
python -m drift_bench.pipeline [OPTIONS]

--config PATH          config file (default: drift_bench/config.yaml)
--briefs-dir PATH      briefs directory (default: drift_bench/briefs)
--prompts-dir PATH     prompt templates directory (default: drift_bench/prompts)
--rubrics PATH         judge rubrics file (default: drift_bench/judges/rubrics.yaml)
--calibration PATH     judge calibration examples (default: drift_bench/judges/calibration.yaml)
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

Analysis items (each item produces one or more CSVs/JSONs that the paper
macros derive from):

```
python -m drift_bench.run_analysis --items all     # 1..6
python -m drift_bench.run_analysis --items 1       # inter-rater reliability
python -m drift_bench.run_analysis --items 2       # verbosity-controlled regression
python -m drift_bench.run_analysis --items 3       # restatement probes + surface gap
python -m drift_bench.run_analysis --items 4       # judge validation
python -m drift_bench.run_analysis --items 5       # debiasing
python -m drift_bench.run_analysis --items 6       # every camera-ready module, macros last
```

Release manifest (regenerates the per-snapshot file inventory and the
`\input{}`-able LaTeX macros file; it also regenerates `cr_macros.tex` and
appends the `\input` line that loads it):

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
- **Cross-family judging**: Claude judges GPT/Gemini/open-weight runs; GPT
  judges Claude runs. Never self-judge.
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
grep -rlF "$(pwd)/" Dataset/ | wc -l                     # MUST be 0 (no local root path)

# 3. Push to HF (one-time setup: pip install huggingface-hub && huggingface-cli login)
huggingface-cli upload-large-folder \
  --repo-type=dataset driftbench/DriftBench Dataset/
# Then point the HF tag `camera-ready` at the commit the paper's numbers come
# from. scripts/fetch_from_hf.py downloads that tag by default and does not
# fall back to main when it is missing. Add --revision <commit> to tag an
# earlier commit than the latest:
hf repos tag create driftbench/DriftBench camera-ready --repo-type dataset

# 4. Verify the loop in a fresh tmpdir
mkdir /tmp/repro && cd /tmp/repro
git clone https://github.com/kruthof/driftbench . && python3.13 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt && .venv/bin/pip install -e . --no-deps
bash scripts/reproduce.sh --hf    # must exit 0 (ends with "identical" on macOS arm64)
```

After a dependency change, regenerate `requirements-lock.txt` with the command
in its header. After a figure change, rewrite the reference hashes with
`python scripts/verify_reproduction.py --write-figure-reference`.

The build script enforces (and fails the build on):

- No files with `.pdf` / `.html` extensions land under `Dataset/`.
- No forbidden substring (configured in `.anonymize.json`) appears in any
  text file. The list is empty since the camera-ready de-anonymization: its
  entries were forms of the author's name, which is now public.
- No local path remains. The build strips the absolute repository root (and
  any Claude Code scratchpad prefix) from the staged copies of text files
  (provenance sidecars, run summaries, logs), so recorded paths ship
  repository-relative. Source files are not modified. The build fails if the
  root path or the user's home directory is still present in a staged file.
- A SHA-256 manifest is written for every shipped file.

`--out <dir>` stages elsewhere (relative to the repository root, or absolute);
the script refuses a directory that overlaps a source path (including
`docs/dataset_card.md` and `.anonymize.json`), and a non-empty directory without a
MANIFEST.txt from an earlier build.

---

## Citation

```bibtex
@inproceedings{kruthof2026constraint,
  title     = {Constraint Adherence Under Iterative Pressure in Multi-Turn {LLM} Ideation},
  author    = {Kruthof, Garvin},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS) Evaluations and Datasets Track},
  year      = {2026},
  url       = {https://github.com/kruthof/driftbench}
}

@misc{kruthof2026driftbenchdata,
  title        = {{DriftBench}: Transcripts, Scores and Analysis Outputs},
  author       = {Kruthof, Garvin},
  year         = {2026},
  howpublished = {Hugging Face dataset},
  url          = {https://huggingface.co/datasets/driftbench/DriftBench}
}
```

## License

- Source code (this repository): **MIT** (see `LICENSE`).
- Dataset on Hugging Face (briefs, prompts, transcripts, scores, analysis
  outputs): **CC-BY 4.0**.
- Anonymized human-rater files (`human_validation/raters/` in the dataset):
  **CC-BY-NC 4.0** (non-commercial).
