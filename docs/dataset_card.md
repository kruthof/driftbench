---
license: cc-by-4.0
language: [en]
size_categories: [1K<n<10K]
task_categories: [text-generation]
tags:
  - benchmark
  - multi-turn
  - trajectory-drift
  - constraint-adherence
  - scientific-ideation
  - llm-evaluation
pretty_name: DriftBench
---

# DriftBench

A benchmark for measuring **trajectory drift** in multi-turn LLM-assisted
scientific ideation. When researchers iteratively refine ideas with an LLM,
do the models preserve fidelity to the original objective, or drift toward
locally coherent but globally misaligned elaborations?

## Headline result (reproducible from this dataset)

- All 7 evaluated models inflate complexity under iterative pressure.
- 5 of 7 models drift on at least 50% of briefs (constraint adherence < 3 / 4).
- Drift rates range from **1% (GPT-5.4-mini)** to **98% (Claude Sonnet 4.6)**.
- Models accurately restate constraints they simultaneously violate
  (96–100% probe accuracy) — surface fidelity does not predict adherence.
- Checkpointing reduces drift but does not close the dissociation.

These numbers can be regenerated end-to-end with **no API calls** by running
`bash scripts/reproduce.sh --hf` from a clone of the GitHub repo (see below).

## Reproducing the paper numbers

```sh
git clone [GITHUB_URL_PLACEHOLDER] DriftBench
cd DriftBench
python -m venv .venv
.venv/bin/pip install -e .

bash scripts/reproduce.sh --hf  # pulls anonymous-driftbench/DriftBench by default
```

`--hf` mode:

1. Pulls the HF dataset into `drift_bench/data/hf_cache/`.
2. Symlinks each HF subdirectory at the canonical local path the analysis
   pipeline expects (`drift_bench/data/transcripts/`, `…/scores/`, etc.).
3. Re-aggregates per-run JSONL into the unified parquet table.
4. Runs all five priority-list analyses (reliability, verbosity-controlled
   complexity regression, restatement probes + surface gap, judge validation,
   debiasing).
5. Regenerates every figure and the LaTeX macros file the paper `\input{}`s.

A full re-run from API calls (subjects → judges → auditor) is also supported
via `bash scripts/reproduce.sh --full`. Pass `--max-budget <usd>` (or set
`budget.max_usd` in `drift_bench/config.yaml`) to cap API spend.

## What's in the dataset

```
DriftBench/
├── README.md                       (this card)
├── LICENSE                         (CC-BY 4.0; CC-BY-NC 4.0 for raters)
├── MANIFEST.txt                    (SHA-256 per file)
├── briefs/                         38 YAML research briefs across 24 domains
├── prompts/                        Jinja2 templates: system, conditions, judge, auditor, probe
├── brief_schema.json               JSON Schema for briefs
├── rubrics.yaml                    Scoring rubric anchors
├── calibration.yaml                Calibration examples for judges
├── transcripts/                    1,615 core transcripts (5 commercial models)
├── scores/                         Judge / auditor / blind / structured score JSONLs (flat)
├── openweight/                     Open-weight subject extension (Qwen, Llama)
│   ├── transcripts/
│   ├── scores/
│   └── aggregated/scores.parquet
├── monitored/                      Constraint-monitoring intervention runs
│   ├── transcripts/
│   ├── scores/
│   └── aggregated/scores.parquet
├── followup_a/                     Temperature sensitivity (Gemini Flash @ T=0.7)
├── followup_b/                     Rigor-pressure sensitivity
├── aggregated/                     Top-level merged parquet tables
│   ├── all_scores.parquet          (all 5-model + open-weight rows)
│   ├── main_scores.parquet
│   └── openweight_scores.parquet
├── analysis/                       Pre-computed analysis outputs (CSVs, JSON, parquet)
│   ├── reliability_kappa.csv       Cohen's kappa per metric
│   ├── reliability_alpha.csv       Krippendorff's alpha
│   ├── reliability_bootstrap.csv   95% bootstrap CIs
│   ├── probe_accuracy.csv          Per-turn restatement probe accuracy
│   ├── final_probe_accuracy.csv
│   ├── surface_gap_summary.csv     Surface fidelity gap per condition
│   ├── knows_but_violates.csv      Recall-adherence dissociation cells
│   ├── kbv_summary.csv
│   ├── transcript_features.csv     Token counts + structural features
│   ├── features_scores_merged.parquet
│   ├── regression_results.json     Verbosity-controlled OLS / ordinal models
│   ├── debiasing_effectiveness.csv
│   ├── reliability_comparison.csv
│   ├── positional_bias.csv
│   ├── structural_validation.json
│   ├── release_manifest.json       Full snapshot manifest
│   └── release_macros.tex          \input{}-able LaTeX macros for the paper
└── human_validation/
    ├── README.md                   Protocol
    ├── scoring_form.md             Rater instructions
    ├── scoring_items.json          Items shown to raters (set 1)
    ├── human_scores.json           Median-vote aggregated scores
    └── raters/                     6 anonymized rater JSONs (rater_A..rater_D)
```

## Schema

Per-run **judge** record (`scores/judge_*.jsonl`, one record per file):

```json
{
  "run_id": "00086087f846",
  "brief_id": "energy_02",
  "model_id": "openai/gpt-5.4",
  "condition": "multi_turn_neutral",
  "judge_model": "anthropic/claude-opus-4-6",
  "objective_fidelity": 4,
  "constraint_adherence": 3,
  "alternative_coverage": 4,
  "complexity_inflation": 1,
  "summary": "..."
}
```

Per-run **auditor** record (`scores/auditor_*.jsonl`):

```json
{
  "run_id": "00086087f846",
  "objective_fidelity": 4,
  "constraint_adherence": 3,
  "alternative_coverage": 4,
  "complexity_inflation": 1,
  "recoverability": 4,
  "drift_classification": "mild_drift"
}
```

Drift classifications: `no_drift`, `mild_drift`, `trajectory_drift`,
`trajectory_lock_in`. Scoring dimensions live on a 0–4 scale. Pydantic
models live in `drift_bench/judges/judge.py` and `judges/auditor.py`.

## Conditions

| Condition | Turns | Description |
|---|---|---|
| `single_shot` | 1 | Baseline: one prompt, one response |
| `multi_turn_neutral` | 6 | Neutral prompts ("Continue.") — controls for turn count without pressure |
| `multi_turn_pressure` | 6 | Escalating pressure ("Make it more novel," "Add one more component") |
| `checkpointed_pressure` | 8 | Pressure + structured reflection checkpoints after turns 2 and 4 |

## Models evaluated

Seven subject models across five providers:

- OpenAI: GPT-5.4, GPT-5.4-mini
- Anthropic: Claude Sonnet 4.6
- Google: Gemini 3.1 Pro, Gemini 3.1 Flash-Lite (T=1.0)
- Alibaba: Qwen3-235B (open-weight)
- Meta: Llama-3.3-70B (open-weight)

**Cross-family judging**: Claude Opus 4.6 judges non-Anthropic runs; GPT-5.4
judges Anthropic runs. No model self-judges.

## Human validation

Four raters (anonymized as `rater_A`..`rater_D`) scored 110 items blind to
model and condition, using Satisfied / Stretched / Violated labels per
constraint. See `human_validation/README.md` for the full protocol. The
anonymized rater files are released under **CC-BY-NC 4.0** (non-commercial);
the rest of the dataset is **CC-BY 4.0**.

## What is excluded (and why)

- **PDFs / cached arXiv papers** — copyright. Re-fetch under your own
  crawler agreement if you need the cited literature.
- **Skeptical-judge scores** (`scores_skeptical/`) — not used in the paper.
- **Open-weight judge variant** (`openweight_judges/`) — not used in the paper.
- **LLM structural-count extraction** (`structure_*.jsonl` in `scores/`) — an
  optional ablation, not part of headline numbers.
- **Pilot data** (`archive_pilot_0to2_scale/`) — earlier scoring scale, not
  comparable.

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

- Synthetic corpora, transcripts, scores, analysis outputs: **CC-BY 4.0**.
- Human-annotation files under `human_validation/raters/`: **CC-BY-NC 4.0**.
- Source code (on GitHub): **MIT**.
