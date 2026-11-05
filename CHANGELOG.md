# Changelog

All corrections to the released DriftBench artifact are recorded here.

## 2026-11-05 — full corrections list added to the release; pinned and checked reproduction

- `camera_ready/analysis/corrections_full.tex` is the full corrections list the paper's
  appendix was shortened from; `drift_bench/analysis/camera_ready/corrections_full.py`
  renders it with the macro values to `corrections_full.md`, and
  `tests/test_cr_corrections_full.py` checks the render.
- Reproduction from a fresh clone. `requirements-lock.txt` pins the dependency
  versions that produced the paper's numbers (Python 3.13). `scripts/fetch_from_hf.py`
  downloads the Hugging Face tag `camera-ready` by default (`--revision main` follows
  the latest commit) and answers an HTTP 429 or network error with one line: set
  `HF_TOKEN` and re-run; the download resumes. It also checks the download against
  `MANIFEST.txt` and stops if listed files are missing. `scripts/reproduce.sh --hf` and
  `--analyze` now also draw the six camera-ready figures (`figures_cr`), re-run
  `sensitivity` so its provenance records the check against the new figure sidecar,
  and end with the new `scripts/verify_reproduction.py`. That script compares
  `cr_macros.json`, every CSV in `camera_ready/analysis/` and `release_macros.tex`
  with the snapshot's `MANIFEST.txt`, and the figures with
  `scripts/figures_reference.json`. The run is byte-identical on macOS arm64; on
  other platforms the last floating-point digits of intermediate CSV values may
  differ, and every printed number is identical. So a file whose hash differs is
  compared with the released copy that `reproduce.sh` saves before the run
  (`--save-reference`): it passes "identical at printed precision" when shape,
  header and text cells are equal, numbers are within rtol 1e-9 and every printed
  macro value is equal; otherwise the script lists the differing cells and exits 1.
  `tests/test_verify_reproduction.py` covers it. `reproduce.sh` sets
  `LITELLM_LOCAL_MODEL_COST_MAP=True`, so litellm uses the model map bundled with the
  pinned version instead of downloading the current one (`tokens_context` reads context
  windows from it; the Llama 3.3 70B window then comes from the fallback table, with the
  same value). No number changes.
- `camera_ready/analysis/sonnet_truncation.csv` also holds Gemini 3.1 Pro rows. Gemini Pro is the
  only other subject that reaches the 4,096-token output cap: 48 of its 303 clean runs have a
  message at 4,092 tokens (Gemini's logged count at the cap, as 1,020 at the 1,024-token probe
  cap), and its judged final response is capped in 2 single-shot runs. An earlier camera-ready
  draft said that no other model reached the cap; the paper now gives these counts (macros `\crGeminiProCap...`). The
  Sonnet rows and every earlier macro value are unchanged.

## 2026-10-27 — camera-ready release: SWE pilot provenance, LLM-labelled hand-check, remaining analysis modules, public names

These items complete the 2026-10-26 entry. Every table named below sits under
`drift_bench/data/camera_ready/analysis/` with a provenance sibling.

### 1. The released SWE pilot is a re-execution

`drift_bench/data/camera_ready/e6_swe/README.md`, `docs/swe_pilot_reconstruction.md`

The first coding-domain pilot (2026-07-26; 32 runs, 8 briefs, 2 Gemini models) lost its
raw transcripts, its score files, and its adapted judge prompt and coding rubric with
the session scratchpad. No transcript or score file of that pilot is in this release;
its per-run pressure scores survive only as a table
(`docs/swe_pilot_recovered/swe_pilot_run_level_results.md`). The SWE data under
`camera_ready/e6_swe/` and `camera_ready/e6_swe_verify/` is a re-execution of
2026-10-26 (E6, entry of 2026-10-26, section 7): the 8 recovered briefs plus 2 new ones,
three subjects, new judge calls. The judge prompt (`drift_bench/prompts_swe/judge.j2`)
and the coding rubric (`drift_bench/judges/rubrics_swe.yaml`) were reconstructed from
the pilot report's list of edits; two rubric lines are guesses
(`docs/swe_pilot_reconstruction.md`, section 6). The paper says so in Section 6 and
Appendix G.

### 2. Mechanism hand-check and constraint typology labelled by an LLM annotator

`drift_bench/analysis/camera_ready/mechanism_handcheck.py`,
`drift_bench/analysis/camera_ready/mechanism_handcheck_codebook.md`,
`camera_ready/e9_mechanism_handcheck/`

The rebuttal promised a hand-check of 50 mechanism detections. No human has labelled
them: the `human_*` columns of `analysis/mechanism_handcheck_sample.csv` are empty. An
LLM annotator labelled the 50 rows on 2026-10-27 (E9): Claude Opus 4.6, temperature 0,
one call per row, the codebook as system prompt (sha256 `ce1d7855...`). It is an LLM,
not a human, and the paper says so. The annotator saw the brief's objective, the target
and the other hard constraints, and the final assistant message. It did not see the
detector's output or the judge's status. Detector vs LLM: `appears` 50 of 50, `verdict`
46 of 50 (kappa 0.78). The LLM confirms the judge's departure in 6 of 50 rows (19 no,
25 unclear). Tables `analysis/mechanism_handcheck_{llm,agreement}.csv`; USD 1.09.
`run()` rebuilds both tables from `responses.jsonl` without network; only
`--annotate` calls the API. Human labels can be added later (see the E9 README).

The constraint typology is labelled the same way (E8, `camera_ready/e8_typology_llm/`,
module `typology_annotate.py`): `label_author` in `analysis/typology_sheet.csv` is
empty, so every type rate uses the LLM labels (234 of 234 constraints, Claude Opus 4.6,
codebook pass 3).

### 3. Monitoring cells recomputed from the monitored runs

`drift_bench/analysis/camera_ready/monitoring.py`

No camera-ready CSV held KBV under automated monitoring, so the mechanism section and
Appendix E could not cite it through macros. `monitoring.py` recomputes it from
`monitored/`: 152 transcripts (condition `multi_turn_pressure_monitored`, 4 subjects x
38 briefs, one repetition) and 151 judge files (one Llama run has none). Recall uses
the main probe scorer on the final probe; the judge is the published arm (GPT-5.4 for
Sonnet, Opus 4.6 for the other three). Monitored KBV: Sonnet 4.6 37/38, Gemini
Flash-Lite 19/38, Qwen3-235B 22/38, Llama-3.3-70B 33/37, against 98.7%, 76.3%, 55.4%
and 93.4% in the clean unmonitored pressure cells. Each row carries the unmonitored
cell and the difference. Table `analysis/monitoring_cells.csv` (4 rows). Disk only.

### 4. Probe accuracy, Sonnet output cap and surface gap as tables of record

`drift_bench/analysis/camera_ready/probe_accuracy.py`,
`drift_bench/analysis/camera_ready/sonnet_truncation.py`,
`drift_bench/analysis/camera_ready/surface_gap.py`

- `probe_accuracy.py`: final-probe accuracy on four samples (the probe file, as
  published, clean, clean without Gemini Pro's capped probes), per scope and model,
  with brief-clustered bootstrap CIs. Pooled over multi-turn runs it is 97.3% as
  published and on the clean sample. Every untruncated final probe of the clean sample
  clears the 0.8 cut (1,423 of 1,423). Table `analysis/probe_accuracy_summary.csv`
  (180 rows).
- `sonnet_truncation.py`: Sonnet 4.6's 4,096-token output cap per sample, condition and
  rule. Any assistant message at the cap (the prevalence rule): 221 of 304 clean runs,
  226 of 315 as published. Final message at the cap (the rule of the adherence test):
  125 of 304 clean, 127 of 315 as published. Table `analysis/sonnet_truncation.csv`
  (20 rows).
- `surface_gap.py`: surface fidelity gap (fidelity minus the mean of adherence and
  coverage) per judge arm, condition and model, rebuilt from the crossed-judge run
  table, with brief-clustered bootstrap CIs. Under the Opus arm Sonnet's pressure gap
  is -0.30 [-0.41, -0.17] (published arm +0.14 on the clean sample; the submission
  printed +0.18). The submitted reading does not hold under the primary arm, so the
  main text drops it; the corrections appendix gives the regenerated values. Table
  `analysis/surface_gap.csv` (48 rows).

### 5. Sensitivity cells

`drift_bench/analysis/camera_ready/sensitivity.py`

KBV in the two follow-up experiments beside the April cells: Gemini Flash-Lite at
temperature 0.7 (`followup_a`; neutral, pressure, checkpointed) and rigor pressure
(`followup_b`; GPT-5.4, Sonnet 4.6, Flash-Lite). Published judge arm; recall from the
main probe scorer. Before this module the cells existed only inside `figures_cr.py`,
which recomputes them on the fly, so no macro could cite them. Tables
`analysis/sensitivity_cells.csv` (12 rows) and `analysis/sensitivity_runs.csv` (one row
per follow-up: planned, completed and failed runs; follow-up B completed 176 of 228).
Disk only.

### 6. Public names and camera-ready title

`README.md`, `docs/dataset_card.md`, `scripts/fetch_from_hf.py`, `LICENSE`

- Code: <https://github.com/kruthof/driftbench>. Data: Hugging Face
  [`driftbench/DriftBench`](https://huggingface.co/datasets/driftbench/DriftBench).
  `scripts/fetch_from_hf.py` (and so `reproduce.sh --hf`) now defaults to
  `driftbench/DriftBench`; `$DRIFTBENCH_HF_REPO` overrides it (`--repo-id` when
  fetch_from_hf.py is run directly). The anonymous review mirror
  `anonymous-driftbench/DriftBench` is no longer referenced by the code, the README or
  the dataset card.
- Title: the submission's "Models Recall What They Violate: Constraint Adherence in
  Multi-Turn LLM Ideation" is now "Constraint Adherence Under Iterative Pressure in
  Multi-Turn LLM Ideation". KBV stays a named metric. The README and the dataset card
  cite the camera-ready paper (author Garvin Kruthof).
- Dataset card: the headline is rewritten from the camera-ready numbers. Withdrawn:
  "5 of 7 models drift on at least 50% of briefs", "1% to 98%", "96–100% probe
  accuracy", "Checkpointing reduces drift". File counts are recounted on disk.
  `scores_skeptical/` and `openweight_judges/` are shipped by
  `scripts/build_dataset_release.py` since 2026-10-26, when they were added to its copy
  plan and to the path map of `scripts/fetch_from_hf.py` (`tests/test_release_maps.py`
  checks that the two maps stay inverse); the submission snapshot did not include them,
  and the card now lists them as shipped. The card gains
  sections on the brief schema, the human-validation protocol, intended use and
  limitations.
- Licences, unchanged and now stated once in each of the README and the card: code MIT;
  data CC-BY 4.0; the anonymized rater files (`human_validation/raters/`) CC-BY-NC 4.0.
  The card's YAML header now lists both data licences (`cc-by-4.0`, `cc-by-nc-4.0`;
  it listed only `cc-by-4.0`). The MIT `LICENSE` names Garvin Kruthof as copyright
  holder instead of "Anonymous Authors".

### 7. Release build: structural-count files shipped, local paths stripped

`scripts/build_dataset_release.py`, `tests/test_release_build.py`

- The build no longer excludes `structure_*` files. `prompts/structure_extraction.j2`
  and the 1,527 `scores/structure_<run_id>.jsonl` files (LLM structural counts) now
  ship: Appendix A reports the structural counts, and the paper promises all prompts
  and scores. The old exclusion comment called them "not used by the paper", which was
  wrong.
- The build strips the absolute repository root from the staged copies of text files
  (provenance sidecars, run summaries, logs), so recorded paths ship
  repository-relative. Claude Code scratchpad prefixes become `<scratchpad>/`. The
  2026-10-27 build rewrote 68 staged files. The source files are not modified. The
  build fails if the root path or the user's home directory remains in a staged file.
- The build refuses a staging dir that is the repository root, overlaps a source path
  (now including `docs/dataset_card.md` and `.anonymize.json`, which it reads), or is a
  non-empty directory without a `MANIFEST.txt` from an earlier build. `--clean` deletes
  the staging dir and the path transform rewrites files in it, so a wrong `--out` could
  otherwise delete or rewrite repository files.
- `.anonymize.json`'s forbidden-substring list is now empty: its entries were forms of
  the author's name, which is public since the camera-ready.

### 8. Checker-human agreement uses the set-1 text the raters saw

`drift_bench/analysis/camera_ready/checkers.py`,
`camera_ready/human_validation_set1_items.csv`, `camera_ready/analysis/checker_agreement.csv`

Both rater apps showed each final proposal cut before the model's closing self-check
("Final Constraint Checklist" or "Objective Restatement"; 54 of 55 set-1 and 53 of 55
set-2 proposals had one, and each holds a constraint checklist). The LLM judge saw the
full transcript. The checker-human comparison on "the text the raters saw" read set 2
from the app's items but set 1 from `human_validation/scoring_items.json`, which holds
the full final message. So for set 1 the checker read the self-check that the raters
did not see. `checkers.py --export-set1 scoring_app_standalone/scoring_items.json` now
writes the set-1 items as shown to `camera_ready/human_validation_set1_items.csv`
(same columns as the set-2 file, no judge or auditor scores). The export checks that
each shown text is a prefix of the full final message and that the constraint order
matches the brief. `load_human_labels` reads both sets from these CSVs and fails if one
is missing. Changed values (as-rated value pairs, checker vs human median labels):
pairs 191 to 172, agreement 91.1% to 94.2%, kappa 0.10 to 0.16, checker violated rate
0.5% to 0.6%, human not-satisfied rate 9.4% to 6.4%. On set 1 alone kappa moves from
0.19 to 0.66. The restatement-only rows and the human-human ceiling on value pairs
change too; the ceiling on all and on checkable constraints does not. The
transcript-text rows and the judge rows do not change.

### 9. Release layout: one sidecar name, set-2 items, stale probe tables

`drift_bench/analysis/camera_ready/{graded_severity,reprobe_accuracy,typology,typology_annotate}.py`,
`scripts/build_dataset_release.py`, `drift_bench/analysis/release_manifest.py`,
`docs/dataset_card.md`, `human_validation/README.md`

- Every CSV provenance sidecar is now `<name>.csv.provenance.json`. Four
  writers used `<name>.provenance.json`: `graded_severity.write_csv_with_provenance`
  (also used by `stretch_violate.py` and `human_severity.py`),
  `reprobe_accuracy._write_csv`, `typology._provenance` and
  `typology_annotate.write_provenance`. The 20 sidecars they wrote under
  `camera_ready/analysis/` (`graded_severity_*`, `human_severity_*`, `human_vs_judge`,
  `reprobe_*`, `stretch_violate_*`, `typology_*`) were renamed, not regenerated; their
  content is unchanged. Figure sidecars (`figures/fig_*.provenance.json`) keep their
  name.
- The set-2 rater items ship as `human_validation/scoring_items_set2.json`, a
  byte copy of the gitignored rater app's list. Its text equals
  `camera_ready/human_validation_set2_items.csv`. The release manifest reads the set-2
  item count from it.
- The release manifest's `key_files` no longer lists the four gitignored configs that
  the run scripts derive from `config.yaml`.
- `analysis/` keeps three probe tables of 2026-05-03 beside their 2026-07-26
  regenerations. The dataset card names `probe_accuracy.csv` and
  `final_probe_accuracy.csv` as canonical. The older files' rows equal the matching
  rows of these, and no analysis reads them.

## 2026-10-26 — camera-ready sample rule, analysis modules and regenerated rebuttal experiments

The rebuttal-period experiments of 2026-07 lost their raw files. Every number the camera-ready
version prints is regenerated from the released files (Hugging Face `driftbench/DriftBench`)
into `drift_bench/data/camera_ready/`, each CSV with a provenance sibling, and reaches the
manuscript through macros (`drift_bench/analysis/camera_ready/macros.py`, input by
`release_macros.tex`). The decisions and the spend ledger are in the camera-ready plan,
which is not part of the public repository.

### 1. Stray Sonnet runs are dropped from every analysis

`drift_bench/analysis/camera_ready/sample.py`

`all_scores.parquet` holds 2,146 rows, 30 of which are Claude Sonnet 4.6 pilot runs whose
judge file is dated 2026-04-16 UTC (the main pass is 2026-04-17 onward). They inflate the
Sonnet cells beyond 76 runs. The paper's sample is now the 2,116 clean rows; every
camera-ready module drops the 30 runs through `sample.drop_stray` and records the rule in
its provenance. The published parquet is not rewritten.

### 2. Judge JSON fallback kept text after the object

`drift_bench/judges/judge.py`, `drift_bench/judges/judge_structured.py`

The fallback parser used a greedy `\{.*\}` span, which fails when a judge emits two JSON
objects (one structured-judge response in the E4 re-score did: a draft, a self-correction
paragraph, then the final object). `first_json_object` now decodes the first complete
object. The one affected record (`camera_ready/e4_structured_opus/structured_opus_e301452993b4.jsonl`)
stores the judge's final object (constraint_adherence 4, not the retracted 3); field
`manual_override` records the change and `raw_response` is unchanged.

### 3. Structured-judge checks matched to the brief's hard constraints

`drift_bench/analysis/camera_ready/graded_severity.py`

The structured judge sometimes appends banned moves or success criteria to its
`constraint_checks`. The loader now matches checks to the brief's hard constraints by
exact text and drops the extras (2 runs in the published structured files, 2 in E4).
gKBV changed for `0bdfba9a53be` (0.143 to 0.0) and `n_checks` for `252515d6bef5`
(13 to 7), both Gemini checkpointed-pressure runs.

### 4. Release manifest double-counted the open-weight runs

`drift_bench/analysis/release_manifest.py`

Since 2026-07-26 `all_scores.parquet` already contains the 602 open-weight rows, but
`build_manifest` concatenated `openweight_subjects/aggregated/scores.parquet` on top,
so `\BenchmarkRuns` regenerated as 2,748. The frame is now deduplicated on `run_id`
(2,146), the five-model subset is read from `config.yaml`, and `None`/`nan` values are
refused when writing macros. The set-2 human-validation item list now also resolves from
`camera_ready/human_validation_set2_items.csv` on a clean checkout.

### 5. Gemini 3.1 Pro probe truncation measured, not imputed

`drift_bench/runners/reprobe.py`, `drift_bench/analysis/camera_ready/reprobe_accuracy.py`

The final restatement probes of Gemini 3.1 Pro were cut at the probe's 1,024-token cap.
Replaying all 228 final probes at a 4,096-token cap on the stored contexts gives full
recall, so its pressure KBV is 25.0% (19/76) with recall measured
(`camera_ready/e3_reprobe_gemini_pro/`).

### 6. SWE pilot briefs and prompts adopted into the repository

The coding-domain pilot of 2026-07-26 (32 runs) lost its raw files. Its 8 briefs and 4
subject templates were recovered verbatim from the pilot agent's session log and now live
in `drift_bench/briefs_swe/` and `drift_bench/prompts_swe/`, outside the 38 research
briefs. The adapted judge template (`prompts_swe/judge.j2`) and rubrics
(`judges/rubrics_swe.yaml`) were reconstructed from the pilot report's list of edits
(`docs/swe_pilot_recovered/pilot_coding_domain.md`, section 4); two rubric lines are
guesses (`docs/swe_pilot_reconstruction.md`, section 6). Two new agentic briefs (swe_09,
swe_10), code-output templates (`prompts_swe_code/`), contract briefs
(`briefs_swe_exec/`), a Phase C config (`config_swe_code.yaml`) and a local test harness
(`analysis/camera_ready/swe_verify.py`) were added for the extended pilot. The pilot's
per-run pressure scores survive only as a table
(`docs/swe_pilot_recovered/swe_pilot_run_level_results.md`); its single-shot per-run
scores are lost.

### 7. Rebuttal-period experiments regenerated (E2, E5, E6)

**E2, Sonnet 4.6 without the output cap** (`camera_ready/e2_sonnet_uncapped/`,
`config_e2_sonnet.yaml`, module `analysis/camera_ready/e2_uncapped.py`). 38 briefs x
single-shot and pressure, one repetition, `max_tokens` 16,384 (never bound; longest turn
7,650 tokens), judged by both Opus 4.6 and GPT-5.4. 47% of pressure turns exceed 4,096
tokens, matching the 47% of clean April turns at the cap (221 of 304 clean runs, 226 of
315 as published). Pressure KBV: Opus 25/38 = 65.8% [49.9, 78.8] vs 75.0% capped;
GPT-5.4 37/38 = 97.4% vs 98.7%. Removing the cap does not raise KBV; mean adherence rises
(+0.26 [+0.06, +0.47] under GPT-5.4) and alternative coverage is 4/4 on every uncapped run.
Tables `analysis/e2_uncapped_{cells,deltas,offset}.csv`. USD 41.64 recorded plus about
USD 1.0-1.7 charged for three runs cut off by a credit outage.

**E5, per-turn re-scoring** (`camera_ready/e5_per_turn/`, module
`analysis/camera_ready/per_turn.py`). 152 runs (repetition 0 of GPT-5.4 and Gemini
Flash-Lite in the neutral and pressure cells) re-judged by Opus 4.6 on the transcript
truncated after turns 1 to 5 (760 files) plus a 20-run turn-6 check that reproduces the
April scores (fidelity and adherence 20/20 exact). Flash-Lite's adherence under pressure
falls by 0.45 [0.27, 0.63] from turn 1 to 5, 88% of it at turn 2; GPT-5.4 does not move;
complexity inflation rises in every cell. The rebuttal's "+1.41" for alternative coverage
matches the pressure-pooled turn-6-minus-turn-5 difference (+1.42), which conflates the
sectioned final-proposal format with one more turn. Tables
`analysis/e5_per_turn_{cells,deltas}.csv`. USD 33.96.

**E6, extended SWE pilot.** The extended pilot (E6) was run on 2026-10-26 with litellm 1.83.8 and is complete:
120 main-arm runs (10 briefs x single_shot/multi_turn_pressure x Gemini 3.1 Flash-Lite,
Gemini 3.1 Pro, GPT-5.4-mini x 2 repetitions) judged by Claude Opus 4.6 against
`rubrics_swe.yaml` with the research-domain calibration examples; all 24 Pro pressure
final probes replayed at a 4,096-token cap (`<arm>/reprobe_gemini_pro/`: 14 of 20
main-arm and 3 of 4 executable-arm originals were capped at 1,024 (flagged at >= 1,020
tokens); all 24 score 1.000 replayed, the 7 untruncated ones identical; the replay flips
one Pro classification, so Pro KBV is 4/20 with capped probes graded as-is and 5/20
measured, the other 4 KBV runs "knows" under both, none in the executable arm; both
columns are in `e6_swe_cells.csv`); and 16
executable-arm runs (swe_02/06/08/10 with interface contracts, Pro and GPT-5.4-mini,
8,192-token cap) judged against the contract briefs and, as a sensitivity check, the
plain briefs, and tested locally (16 of 16 pass 116 of 116 contract tests). Pressure-run
KBV rates: Flash-Lite 16/20, Pro 5/20, GPT-5.4-mini 2/20 (executable arm: Pro 1/4,
mini 1/4); constraint adherence never scored below 3 ("stretched", not "violated").
Objective fidelity is 4 on all 16 executable-arm runs under both brief versions; the two
judge passes differ on one of 64 scores. 9 of 20 Pro single-shot finals hit the
4,096-token cap and were not re-run. Data under `drift_bench/data/camera_ready/e6_swe/`
and `e6_swe_verify/`, tables `analysis/e6_swe_{runs,cells,costs,exec_fidelity}.csv`;
USD 20.26 in total (subjects 12.40, judge 7.19, replays 0.67), all in
`e6_swe_costs.csv`. The judge's system message stays research-flavoured ("expert
research evaluation judge") and the restatement probe (`prompts_swe/restatement_probe.j2`,
a byte-identical copy of the research probe) asks for "The original research objective",
both as in the July pilot.

### 8. New tooling

`drift_bench/rescore.py` re-judges stored transcripts with a chosen judge, mode
(`judge`, `structured` or `blind`) and turn index, resumably and under a budget cap, into a
directory of its own (it refuses the published score directories).
`python -m drift_bench.run_analysis --items 6` runs every camera-ready module
(macros last; the SWE harness, which executes model-written code, must be run on
purpose).

## 2026-07-26 — analysis-code and human-validation aggregation fixes

Three defects were found in the released artifact while preparing analyses in response
to peer review. None of them changes a number reported in the paper: all seven per-model
KBV rates, the human-validation statistics, and the aggregate condition means were
recomputed from the raw data and reproduce the published values. The defects were in the
*shipped analysis path* and in one *aggregated convenience file*, both of which a third
party re-running the benchmark would have hit.

### 1. `knows_but_violates_table` did not reproduce the paper's KBV definition

`drift_bench/analysis/probes.py`

- `adherence_threshold` defaulted to `3`. The paper defines constraint non-compliance as
  adherence `< 4` (any hard constraint stretched or violated). `run_probe_analysis` called
  the function with defaults, so `python -m drift_bench.run_analysis --items 3` produced
  KBV rates computed at the wrong threshold — e.g. Claude Sonnet 4.6 under multi-turn
  pressure appeared as 56.1% instead of 98.8%. Default corrected to `4` and documented.
- The "knows" criterion was hard-coded to `constraint_recall` (hard constraints only),
  whereas the reported KBV rates condition on `overall_accuracy` (hard constraints *and*
  banned moves). This affected only Gemini 3.1 Pro, the one model whose probe recall is
  off the ceiling, which appeared as 21.1% instead of 18.4%. The criterion is now an
  explicit `recall_column` parameter defaulting to `overall_accuracy`.

After both fixes, the shipped entry point reproduces all seven published per-model KBV
rates: 7.9 / 15.8 / 18.4 / 55.4 / 76.3 / 93.4 / 98.8%.

### 2. Probe analysis covered only five of seven subject models

`drift_bench/analysis/probes.py`, `drift_bench/run_analysis.py`

`run_probe_analysis` read only the main transcripts directory, silently excluding the
open-weight subject runs (Llama-3.3-70B, Qwen3-235B) stored under
`data/openweight_subjects/transcripts/`. It now accepts `extra_transcripts_dirs`, and
`run_analysis.py` passes the open-weight tree. Probe coverage rises from 1,080 to 1,683
runs and the regenerated `probe_accuracy.csv`, `final_probe_accuracy.csv`,
`knows_but_violates.csv` and `kbv_summary.csv` now span all seven models.

### 3. `human_scores.json` was not a median-vote aggregation

`drift_bench/data/human_validation/human_scores.json` (and its copy in `hf_staging/`)

The file was documented in `README.md` and `docs/dataset_card.md` as "aggregated
median-vote scores across raters". It was in fact byte-identical to
`raters/human_rater_set1_rater_A.json`: a single rater, on set 1 only (55 of 110 runs).
Rater A is the most lenient of the three set-1 raters (15 flagged violations vs 33 and 30),
so anyone re-deriving the human-validation results from this file would have obtained
systematically lenient numbers and concluded the paper's were inflated.

Replaced with a true aggregation over all six rater exports (110 runs, 684 constraint
items, 2,052 labels, 3 raters per set). Because score-level and constraint-level median
votes are not identical (they differ on 11 of 110 runs), both are now stored explicitly:

- `computed_score` — median of the three raters' deterministic 0–4 scores. This is the
  aggregation the paper reports, and it reproduces the published figures exactly:
  constraint adherence 3.704 (single-shot) → 2.585 (multi-turn pressure), Δ = −1.118
  against the judge's −0.398 on the same 110 runs.
- `constraint_statuses` / `computed_score_from_constraint_median` — per-constraint median
  label across raters, and the score derived from it.
- `per_rater_scores` — the individual scores behind each aggregate, so the rater spread
  is inspectable rather than hidden.

The raw rater exports in `raters/` were correct and are unchanged; only the derived
aggregate was wrong.

### Verification

`pytest tests` — 65 passed, 3 xfailed. All seven per-model KBV rates and the
human-validation statistics re-derive from the shipped code and data.
