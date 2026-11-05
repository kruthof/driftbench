---
license:
  - cc-by-4.0
  - cc-by-nc-4.0
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
configs:
  - config_name: all_scores
    data_files: aggregated/all_scores.parquet
    default: true
  - config_name: main_scores
    data_files: aggregated/main_scores.parquet
  - config_name: openweight_scores
    data_files: aggregated/openweight_scores.parquet
---

# DriftBench

A benchmark for **constraint adherence** in multi-turn LLM-assisted scientific
ideation. When researchers refine ideas with an LLM over several turns, do the
models keep the constraints of the original brief, or drift toward locally
coherent but globally misaligned elaborations?

Paper: Garvin Kruthof. *Constraint Adherence Under Iterative Pressure in
Multi-Turn LLM Ideation.* NeurIPS 2026 Evaluations and Datasets Track.
Code: <https://github.com/kruthof/driftbench>.

## Headline results (reproducible from this dataset)

All numbers come from the camera-ready paper. They use the clean analysis
sample of 2,116 scored runs: the 2,146 published rows minus 30 stray Claude
Sonnet 4.6 pilot rows. "Non-compliance" is a judge score: constraint adherence
below 4 on the 0–4 rubric (a hard constraint stretched or violated). The
knows-but-violates (KBV) rate is the share of pressure runs that are
non-compliant although the model's final restatement probe reproduces the
constraints (accuracy ≥ 0.8).

- **Pressure raises complexity in every model and lowers adherence.** Under a
  common Claude Opus 4.6 judge, mean complexity inflation rises from 0.16
  (single-shot) to 1.65 (multi-turn pressure) and mean constraint adherence
  falls from 3.77 to 3.45.
- **The constraints stay legible.** Pooled over multi-turn runs, final
  restatement-probe accuracy is 97.3%, and every run with an untruncated final
  probe clears the 80% cut (1,423 of 1,423). The constraints are reproducible
  on demand while the proposal departs from them.
- **KBV is judge-conditional.** Across the seven models, KBV under pressure
  ranges from 7.9% to 93.4% under a common Claude Opus 4.6 judge and from 42.1%
  to 98.7% under a common GPT-5.4 judge. 4/7 models (Opus judge) and 6/7
  models (GPT-5.4 judge) are above 50%.
- **The top of the ranking is stable; the bottom is not identified.** GPT-5.4
  has the lowest KBV under both judges. The most non-compliant model differs
  between judges (Llama-3.3-70B under Opus 4.6, Claude Sonnet 4.6 under
  GPT-5.4).
- **The pressure effect is judge-stable; KBV levels are not.** On the same
  runs, GPT-5.4 scores adherence 0.535 points lower than Opus 4.6, and the
  offset does not interact with condition (p = 0.565).
- **Checkpoint re-elicitation lowers KBV for some models but not others.**
  Re-eliciting the brief after turns 2 and 4 lowers KBV for Qwen3-235B (55.4%
  to 36.0%) and Claude Sonnet 4.6 (98.7% to 86.8%); it stays near the pressure
  level for Llama-3.3-70B, Gemini 3.1 Flash-Lite and Gemini 3.1 Pro (published
  judge arm). Complexity inflation persists (1.83 under pressure, 1.64 with
  checkpoints).
- **Reported adherence is conservative.** Blind human raters find more
  violations than the LLM judge: on 486 constraint items labelled by both,
  humans mark 5.6% violated and the structured judge 0 of 486. Raters saw the proposal without the model's closing constraint checklist (cut by design, so that raters judged the proposal itself); the judge saw it, and part of the gap may reflect this.

Every number above is a `\cr...` macro of the paper, generated from a CSV under
`camera_ready/analysis/`. `python -m drift_bench.analysis.camera_ready.macros --list`
prints each macro with its value and source CSV.

## Reproducing the paper numbers

No API calls are needed.

```sh
git clone https://github.com/kruthof/driftbench DriftBench
cd DriftBench
python3.13 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt   # the versions behind the paper's numbers
.venv/bin/pip install -e . --no-deps

export HF_TOKEN=hf_...           # optional: a free read token avoids rate limits
bash scripts/reproduce.sh --hf  # pulls driftbench/DriftBench, tag camera-ready
```

`--hf` mode:

1. Pulls the HF dataset at the tag `camera-ready` into `drift_bench/data/hf_cache/`
   and checks that every file `MANIFEST.txt` lists arrived.
2. Symlinks each HF subdirectory at the canonical local path the analysis
   pipeline expects (`drift_bench/data/transcripts/`, `…/scores/`, etc.).
3. Re-aggregates per-run JSONL into the unified parquet table.
4. Runs analysis items 1 to 6: the five submission-era analyses (reliability,
   verbosity-controlled complexity regression, restatement probes + surface
   gap, judge validation, debiasing) and every camera-ready module (item 6;
   the macros run last).
5. Regenerates the submission-era figures, the six camera-ready figures and the
   LaTeX macros file the paper `\input{}`s.
6. Compares the regenerated files that carry the paper's numbers
   (`camera_ready/analysis/cr_macros.json`, the 75 CSVs in
   `camera_ready/analysis/`, `analysis/release_macros.tex`) with
   `MANIFEST.txt`, and the six figures with reference hashes in the code
   repository, and prints "N/N identical" per group
   (`scripts/verify_reproduction.py`). Before step 3 the run saves the released
   copies of these files, since the regenerated files overwrite the snapshot
   through the symlinks.

The tag `camera-ready` marks the exact snapshot the paper used; `--revision main`
(`bash scripts/reproduce.sh --hf --revision main`) follows the latest commit.
Anonymous downloads of the about 19,000 files can hit the Hugging Face rate limit
(HTTP 429); set `HF_TOKEN` to a free read token and re-run, and the download
resumes. With the library versions of `requirements-lock.txt` the run is
byte-identical on macOS arm64; on other platforms the last floating-point digits
of intermediate CSV values may differ, and every printed number is identical
(the verify script checks this). A file whose hash differs passes "identical at
printed precision" when its shape, header and text cells are equal, its numbers
are within a relative 1e-9 of the released copy and every printed macro value is
equal; otherwise the script lists the differing cells and exits 1. The repository
README lists every command.

A full re-run from API calls (subjects → judges → auditor) is also supported
via `bash scripts/reproduce.sh --full`, which caps spend at USD 400. For
another cap, run `python -m drift_bench.pipeline --max-budget <usd>` directly
(or set `budget.max_usd` in `drift_bench/config.yaml`).

## What's in the dataset

File counts are the files on disk that `scripts/build_dataset_release.py`
stages (its `.pdf`/`.html` exclusions applied), counted on 2026-10-27.
`MANIFEST.txt` lists every shipped file with its SHA-256. The build strips the
local repository root from paths recorded in text files (provenance sidecars,
run summaries, logs), so those paths are repository-relative.

```
DriftBench/
├── README.md                       (this card)
├── LICENSE                         (licence text; see License below)
├── MANIFEST.txt                    (SHA-256 per file)
├── run_summary.json                Run counts and cost written by the pipeline.
│                                   It held the main run's summary until a later
│                                   rigor-pressure pass overwrote it; it now
│                                   describes that pass (201 runs, 3 models,
│                                   single_shot + multi_turn_pressure_rigor)
├── briefs/                         38 YAML research briefs across 24 domains
├── prompts/                        13 Jinja2 templates (system, conditions,
│                                   judge, blind/structured judge, auditor,
│                                   probe, strict probe grader, structural-count
│                                   extraction) + loader
├── brief_schema.json               JSON Schema for briefs
├── rubrics.yaml                    Scoring rubric anchors
├── calibration.yaml                Calibration examples for judges
├── transcripts/                    1,615 files, 5 commercial models: 1,528 in
│                                   the four main conditions + 87
│                                   multi_turn_pressure_rigor runs
├── scores/                         8,265 per-run score JSONLs (flat): judge
│                                   1,547, auditor 1,544, blind 2,123,
│                                   structured 1,524, structure 1,527 (LLM
│                                   structural counts)
├── openweight/                     Open-weight subject extension (Qwen, Llama)
│   ├── transcripts/                603 files
│   ├── scores/                     1,205 files (judge 602, auditor 603)
│   ├── aggregated/scores.parquet
│   └── run_summary.json            Run counts and cost of this run
├── monitored/                      Constraint-monitoring intervention runs
│   ├── transcripts/                152 files
│   ├── scores/                     303 files (judge 151, auditor 152)
│   ├── aggregated/scores.parquet
│   └── run_summary.json            Run counts and cost of this run
├── followup_a/                     Temperature sensitivity (Gemini Flash-Lite
│   │                               at T=0.7): 304 transcripts, 608 scores
│   └── run_summary.json            Run counts and cost of this run
├── followup_b/                     Rigor-pressure sensitivity: 176
│   │                               transcripts, 352 scores
│   └── run_summary.json            Run counts and cost of this run
├── scores_skeptical/               1,286 files: GPT-5.4 judge with a skeptical
│                                   prompt, 5 subjects (no GPT-5.4, no
│                                   GPT-5.4-mini); judge-prompt sensitivity
├── openweight_judges/              799 files: Llama-3.3-70B (399) and
│                                   Qwen3-235B (400) as judges of a subset
│                                   of the 5 commercial models' runs; not
│                                   analysed in the paper
├── aggregated/                     Top-level merged parquet tables
│   ├── all_scores.parquet          (2,146 rows: 5-model + open-weight)
│   ├── main_scores.parquet
│   └── openweight_scores.parquet
├── analysis/                       29 submission-era analysis outputs (CSVs,
│                                   JSON, parquet), e.g. reliability_*.csv,
│                                   probe_accuracy.csv, final_probe_accuracy.csv,
│                                   knows_but_violates.csv, kbv_summary.csv,
│                                   transcript_features.csv, regression_results.json,
│                                   release_manifest.json, release_macros.tex
│                                   (see the note below on reproduce.sh --hf)
├── camera_ready/                   3,335 files: camera-ready experiments and
│   │                               the CSVs the camera-ready numbers derive from
│   ├── analysis/                   157 files: 75 CSVs, each with a
│   │                               <name>.csv.provenance.json sidecar; cr_macros.tex,
│   │                               cr_macros.json (the paper's macros),
│   │                               cr_macros.tex.provenance.json,
│   │                               per_model_condition_means.tex;
│   │                               corrections_full.md, the full list of
│   │                               corrections the paper's corrections appendix
│   │                               refers to (source corrections_full.tex,
│   │                               with a provenance sidecar)
│   ├── e1_crossed_judge/           Crossed judge: Opus 4.6 on Sonnet (152),
│   │                               GPT-5.4 on the other six (908), 50-run
│   │                               original-judge control
│   ├── e2_sonnet_uncapped/         Sonnet 4.6 rerun without the 4,096-token
│   │                               output cap, scored by both judges
│   ├── e3_reprobe_gemini_pro/      Gemini Pro final probes replayed at a
│   │                               4,096-token cap
│   ├── e4_structured_opus/         Structured Opus 4.6 scores for Sonnet,
│   │                               Llama and Qwen (single-shot, pressure)
│   ├── e5_per_turn/                Per-turn Opus 4.6 re-scores (turns 1-5)
│   ├── e6_swe/, e6_swe_verify/     Extended SWE pilot and its executable arm
│   ├── e7_strict_probes/           Strict re-grading of 304 final probes
│   ├── e8_typology_llm/            Constraint-typology labels, LLM annotator
│   ├── e9_mechanism_handcheck/     Mechanism hand-check labels, LLM annotator
│   ├── human_validation_set1_items.csv  Set-1 items as raters saw them
│   └── human_validation_set2_items.csv  Set-2 items as raters saw them
└── human_validation/
    ├── README.md                   Protocol
    ├── rater_instructions.md       Rater instructions: the text the rater app
    │                               showed (verbatim)
    ├── scoring_form.md             Printable scoring form (its text differs
    │                               from what the rater app showed)
    ├── scoring_items.json          The 55 set-1 items (brief fields, full
    │                               final message, judge and auditor scores);
    │                               raters saw camera_ready/
    │                               human_validation_set1_items.csv
    ├── scoring_items_set2.json     The 55 set-2 items as the rater app
    │                               showed them (same text as camera_ready/
    │                               human_validation_set2_items.csv)
    ├── human_scores.json           Median-vote aggregated scores (110 runs)
    └── raters/                     6 anonymized rater JSONs (rater_A..rater_D)
```

Each `camera_ready/<experiment>/` directory has a `README.md` with the exact
command, date, model ids, litellm version and cost. The one exception is
`e6_swe_verify/` (the executable SWE arm): its commands are in
`e6_swe/README.md`. The labels in
`e8_typology_llm/` and `e9_mechanism_handcheck/` come from an LLM annotator
(Claude Opus 4.6), not a human. The SWE pilot's briefs and prompt templates
(`briefs_swe/`, `prompts_swe/`, `prompts_swe_code/`, `briefs_swe_exec/`) and
`rubrics_swe.yaml` live in the code repository.

Some `analysis/` files ship as written on 2026-05-03.
`reliability_{kappa,alpha,bootstrap}.csv` and
`reliability_drift_agreement.json` cover the 1,544 commercial-model runs;
`spotcheck_sample.csv` has 24 rows. `reproduce.sh --hf` reruns analysis items
1–5 on all 2,146 runs and overwrites these files in place (through the symlink
into `hf_cache/`); they then cover 7 models and the spot check has 28 rows.
`debiased_scores.parquet` and `features_scores_merged.parquet` are rewritten
with equal values, and `regression_results.json` with values equal within
2e-8 (its model summaries also carry the new run date). Every other item 1–5
output is rewritten byte-identical. No camera-ready macro reads these files, so
`cr_macros.tex` is unchanged. The rewritten snapshot no longer matches
`MANIFEST.txt` for these files and for the provenance sidecars (they carry the
run date); `scripts/verify_reproduction.py` checks only the files that carry the
paper's numbers.

`analysis/` holds two generations of the probe tables. `probe_accuracy.csv`
and `final_probe_accuracy.csv` (regenerated 2026-07-26, all seven models) are
canonical; every analysis reads them. `final_probe_accuracy_combined.csv`,
`openweight_probe_accuracy.csv` and `openweight_final_probe_accuracy.csv` are
earlier outputs (2026-05-03). Their rows equal the matching rows of the
canonical files, and no analysis reads them.

In provenance files, "April" means the main benchmark run (April 2026);
"follow-up A" is `followup_a/` (Flash-Lite temperature 0.7) and "follow-up B"
is `followup_b/` (rigor pressure).

The analysis sample drops 30 stray Sonnet pilot rows of
`aggregated/all_scores.parquet` (judge file dated 2026-04-16 UTC); the rule is
`drift_bench/analysis/camera_ready/sample.py`. Eleven of the 30 have a
transcript in `transcripts/`. Four main-condition transcripts (three in
`transcripts/`, one in `openweight/transcripts/`) have no row in
`aggregated/all_scores.parquet`: aggregation keeps runs with both a judge and
an auditor record. Three GPT-5.4-mini runs lack an auditor file and one
Llama-3.3-70B run lacks a judge file; `camera_ready/analysis/failed_runs.csv`
lists them.

## Brief schema

Each brief in `briefs/` is a YAML file validated against `brief_schema.json`.
All seven fields are required:

| Field | Type | Content |
|---|---|---|
| `id` | string, `^[a-z]+_[0-9]{2}$` | brief id, e.g. `energy_02` |
| `domain` | string | scientific domain (24 domains) |
| `objective` | string | a specific, testable research objective |
| `hard_constraints` | list, 3–8 items | binary-checkable constraints the proposal must respect |
| `success_criteria` | list, ≥ 2 items | what a good proposal achieves |
| `plausible_directions` | list, 2–5 items | genuinely competitive alternative designs |
| `banned_moves` | list, 2–5 items | tempting moves the proposal must not make |

The 38 released briefs carry 5–8 hard constraints each (234 in total) and 3–5
banned moves each.

## Score schema

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

Crossed-judge records (`camera_ready/e1_crossed_judge/{opus,gpt}/judge_{opus,gpt}_<run_id>.jsonl`)
have the judge record's fields plus `judge_response_model` (the model id the
API reported), `turn_index`, `cost_usd` and `cost_estimated`.

Drift classifications: `no_drift`, `mild_drift`, `trajectory_drift`,
`trajectory_lock_in`. Scoring dimensions live on a 0–4 scale. Pydantic
models live in `drift_bench/judges/judge.py` and `judges/auditor.py`.

## Conditions

| Condition | Turns | Description |
|---|---|---|
| `single_shot` | 1 | Baseline: one prompt, one response |
| `multi_turn_neutral` | 6 | Neutral prompts ("Continue.") — controls for turn count without pressure |
| `multi_turn_pressure` | 6 | Escalating pressure ("Make it more novel," "Add one more component") |
| `checkpointed_pressure` | 8 | Pressure + checkpoints that re-elicit the brief after turns 2 and 4 |

## Models evaluated

Seven subject models across five providers:

- OpenAI: GPT-5.4, GPT-5.4-mini
- Anthropic: Claude Sonnet 4.6
- Google: Gemini 3.1 Pro, Gemini 3.1 Flash-Lite
- Alibaba: Qwen3-235B (open-weight)
- Meta: Llama-3.3-70B (open-weight)

Subject temperature is 0.7, except 1.0 for the two Gemini models (provider
recommendation).

**Judging.** The published arm is cross-family: Claude Opus 4.6 judges
non-Anthropic runs and GPT-5.4 judges Anthropic runs, so no model judges its
own outputs. The camera-ready release adds a crossed-judge design: both Claude
Opus 4.6 and GPT-5.4 score every run of all seven models in the single-shot and
pressure cells (`camera_ready/e1_crossed_judge/`). The common Opus 4.6 judge is
the paper's primary arm, the common GPT-5.4 judge the robustness check. The
neutral and checkpointed cells have published-arm scores only.

## Human validation

Four unpaid volunteer raters (anonymized as `rater_A`..`rater_D`; three
external helpers and the author) scored 110 runs in two sets of 55, three
raters per set. Raters saw only the brief's domain, the research objective,
the hard constraints, the banned moves, and the final proposal without the
model's closing self-check (cut by design in 107 of 110 proposals, so that
raters judged the proposal itself rather than the model's claims about it).
The LLM judge saw the full transcript, including that section. Raters were
blind to model, condition and automated scores. Each rater labelled every hard
constraint as Satisfied,
Stretched or Violated, and a deterministic rule turns the labels into a 0–4
score (4: none stretched or violated; 3: at least one stretched, none
violated; 2, 1, 0: one, two, three or more violated). `human_scores.json`
stores the median of the three raters' scores per run (the aggregation the
paper reports), the per-constraint median labels, and each rater's score. The
full protocol is in `human_validation/README.md` and the rater instructions in
`human_validation/rater_instructions.md`; `human_validation/scoring_form.md` is
a printable scoring form whose text differs from what the rater app showed.
The text the raters saw is in
`camera_ready/human_validation_set1_items.csv` (set 1) and
`camera_ready/human_validation_set2_items.csv` (set 2; also
`human_validation/scoring_items_set2.json`, the rater app's item list);
`human_validation/scoring_items.json` holds the full set-1 final messages.
Before rating, the raters were told separately of the minimal risk, the
voluntary nature of participation, anonymization and the public release of
the anonymized individual ratings; the hosted rating page gave the task
instructions. The page asked for a name or ID, which the released rater files
replace with an anonymous rater ID; no other personal information was
collected.

## Intended use

- Measuring whether a model keeps the hard constraints of a research brief
  under iterative multi-turn pressure, and separating that from recall with the
  restatement probe.
- Re-scoring the released transcripts with another judge, prompt or rubric;
  the release holds everything a re-score needs.
- Evaluating a new subject model with the released pipeline. Report it the way
  the paper does: score the single-shot and pressure cells with both judges
  (Claude Opus 4.6 and GPT-5.4), because KBV levels depend on the judge.

The results can inform model selection for research workflows. They are not
general safety or capability assessments. All briefs are benign research
topics.

## Limitations

- **Judge dependence.** Every adherence score comes from an LLM judge. Condition
  effects are judge-stable, but KBV levels and the model ranking are
  judge-conditional; the two judge arms bracket them rather than identify
  them. No third judge family is used to resolve the bracket; the Llama/Qwen
  judge scores in `openweight_judges/` cover a subset of the five commercial
  models' runs and are not analysed in the paper.
- **Output truncation.** Claude Sonnet 4.6 ran with a 4,096-token output cap,
  and 221 of its 304 clean runs on the four main conditions hit it at least
  once. A one-repetition re-run without the cap
  (`camera_ready/e2_sonnet_uncapped/`) does not raise Sonnet's KBV.
- **Short horizon by design.** Conversations have six turns, and the largest
  final context fills 6.3% of the model's window. Longer horizons are
  untested.
- **Instruction arbitration.** Part of the effect may be adaptation to the
  latest user turn rather than drift from the brief.
- **Neutral condition.** "Continue." is not inert: it licenses elaboration and
  may act as weak pressure.
- **Non-compliance is mostly stretching.** Under the Opus 4.6 judge, 91.1% of
  KBV-positive pressure runs sit at adherence 3 (a constraint stretched, not
  violated). Human labels license "violation" for a minority of pressure runs.
- **SWE pilot.** The coding-domain pilot under `camera_ready/e6_swe*` is small
  and its brief difficulty is not calibrated against the research briefs.

## What is excluded (and why)

- **PDFs / cached arXiv papers** — copyright. Re-fetch under your own
  crawler agreement if you need the cited literature.
- **Pilot data** (`archive_pilot_0to2_scale/`) — earlier scoring scale, not
  comparable.

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

- Dataset (briefs, prompts, transcripts, scores, analysis outputs, everything
  outside `human_validation/raters/`): **CC-BY 4.0**.
- Anonymized human-rater files under `human_validation/raters/`:
  **CC-BY-NC 4.0** (non-commercial).
- Source code (<https://github.com/kruthof/driftbench>): **MIT**.
