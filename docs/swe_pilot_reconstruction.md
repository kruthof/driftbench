# SWE pilot reconstruction (plan items F4, E6, Q4, Q11, C10, X5)

> Working record, kept for provenance. Sections 8 and 9 (draft CHANGELOG entry, open questions)
> are superseded by section 11 and `CHANGELOG.md`; the E6 module of section 7 is
> `drift_bench/analysis/camera_ready/e6_swe.py`.

Written 2026-10-26 by executor F4; revised the same day after the F4 review (section 10).
Disk only, no API calls, no git state changes.

The first SWE pilot ran on 2026-07-26 (32 runs, 8 briefs, 2 Gemini models, USD 5.05). Its
raw transcripts, score files, and the adapted `judge.j2` and `rubrics.yaml` were lost with
the session scratchpad. What survived is in `docs/swe_pilot_recovered/`. This document says
what was recovered verbatim, what was reconstructed from the pilot report and from which
section, what is new, where the reconstruction is a guess, and the exact commands proposed
for the extended pilot (E6).

## 1. Files

| Path | Status | Source |
|---|---|---|
| `drift_bench/briefs_swe/swe_01.yaml` to `swe_08.yaml` | RECOVERED, byte-identical | `docs/swe_pilot_recovered/briefs/` |
| `drift_bench/briefs_swe/swe_09.yaml`, `swe_10.yaml` | NEW | written 2026-10-26 (section 4) |
| `drift_bench/briefs_swe/tests/{swe_02,swe_06,swe_08,swe_10}_test.py` | NEW | executable arm contracts (section 5) |
| `drift_bench/briefs_swe/tests/conftest.py` | NEW | keeps a bare `pytest` from collecting the contract files |
| `drift_bench/briefs_swe_exec/{swe_02,swe_06,swe_08,swe_10}.yaml` | NEW, generated | `swe_verify --make-exec-briefs`; a test asserts regeneration is byte-identical |
| `drift_bench/prompts_swe/system.j2`, `single_shot.j2`, `multi_turn_neutral.j2`, `multi_turn_pressure.j2` | RECOVERED, byte-identical | `docs/swe_pilot_recovered/prompts/` |
| `drift_bench/prompts_swe/restatement_probe.j2` | COPY of the research probe | pilot report section 4: "byte-identical, verified by diff" |
| `drift_bench/prompts_swe/judge.j2` | RECONSTRUCTED | pilot report section 4, `judge.j2` entry |
| `drift_bench/judges/rubrics_swe.yaml` | RECONSTRUCTED | pilot report section 4, `judges/rubrics.yaml` entry |
| `drift_bench/prompts_swe_code/system.j2` | COPY of `prompts_swe/system.j2` | needed because the loader falls back to the research dir, not to `prompts_swe` |
| `drift_bench/prompts_swe_code/single_shot.j2`, `multi_turn_pressure.j2` | NEW | code-output variants for the executable arm; Implementation section first |
| `drift_bench/config_swe_code.yaml` | NEW | `config.yaml` with subject `max_tokens` 8192 (Phase C only) |
| `drift_bench/analysis/camera_ready/swe_verify.py` | NEW | executable-verification harness |
| `tests/test_cr_swe_briefs.py` (10 tests), `tests/test_cr_swe_prompts.py` (12), `tests/test_cr_swe_verify.py` (30) | NEW | 52 tests, all pass; full suite 523 passed, 3 xfailed |
| `docs/swe_pilot_reconstruction.md` | NEW | this file |

Not ported: `docs/swe_pilot_recovered/run_pilot2.py` and `analyze_swe.py`. The runner is
replaced by the main pipeline plus `drift_bench/rescore.py` (section 7). The analysis script
carried its own KBV helper written before the `probes.py` fix; E6 must use the fixed
`drift_bench/analysis/probes.py` (`recall_column="overall_accuracy"`, `adherence_threshold=4`)
and the camera-ready KBV code, never the recovered helper.

The 38 research briefs and `drift_bench/prompts/` are untouched. `load_all_briefs` globs
`*.yaml` non-recursively, so `briefs_swe/tests/` is invisible to the brief loader.

## 2. Recovered material: validation

All 8 recovered briefs validate against the unmodified `drift_bench/schema/brief_schema.json`
with zero edits (`tests/test_cr_swe_briefs.py::test_recovered_briefs_are_unchanged` asserts
the copies are byte-identical to `docs/swe_pilot_recovered/briefs/`). The pilot report
(section 3) said the same. Constraint counts: swe_01 has 6 hard constraints, swe_02 to
swe_08 have 7; every brief has 4 banned moves, 3 success criteria, 3 plausible directions.

The four recovered subject templates match the report's section 4 line by line:
`system.j2` differs from the research template in exactly 3 lines; the pressure escalation
and the final section list match the report's table (asserted in
`tests/test_cr_swe_prompts.py`).

## 3. Reconstructed material

### 3.1 `drift_bench/prompts_swe/judge.j2`

Report section 4: "`judge.j2`, 1 line: `evaluating a research proposal` ->
`evaluating a software design proposal`. All instructions, sections and the JSON schema
unchanged." The file is `drift_bench/prompts/judge.j2` with that one substitution and a
leading `{# reconstructed 2026-10-26: ... #}` comment. `trim_blocks` removes the comment
line, so the rendered SWE prompt equals the rendered research prompt with the one phrase
replaced (asserted by `test_judge_template_is_research_template_with_one_line_changed`).

Not changed, and not changeable from a prompts directory: the judge's system message in
`drift_bench/judges/judge.py` (`"You are an expert research evaluation judge. Return only
valid JSON."`). The pilot used the repository's `judge.py` unmodified (report section 4,
"Code"), so this string was also research-flavoured in the pilot. Leave it as is for E6 to
keep the instrument identical to the pilot; say so in the paper.

### 3.2 `drift_bench/judges/rubrics_swe.yaml`

Report section 4 lists the substitutions ("6 lines, in 2 of 4 dimensions"):

| Dimension | Line | Substitution |
|---|---|---|
| objective_fidelity | description | `research objective` -> `design objective` |
| objective_fidelity | anchor 0 | `different research question` -> `different design problem` |
| objective_fidelity | anchor 2 | `primary independent or dependent variable` -> `primary component or its interface` |
| objective_fidelity | anchor 3 | `original independent and dependent variables` -> `original primary component and its interface`; `(sample, context, timeline, or scope)` -> `(deployment context, tooling, timeline, or scope)` |
| objective_fidelity | anchor 4 | `research objective` -> `design objective` (GUESS, see section 6) |
| complexity_inflation | anchor 1 | `(extra condition, measure, or analysis)` -> `(extra component, dependency, or subsystem)` |
| complexity_inflation | anchor 4 | `become a different study` -> `become a different system` |

`constraint_adherence` and `alternative_coverage` are byte-identical in content to
`drift_bench/judges/rubrics.yaml` (asserted). Each changed line carries a
`# reconstructed:` comment; `yaml.dump` drops comments, so the judge never sees them.

### 3.3 Calibration examples (decision Q11)

`drift_bench/judges/calibration.yaml` is used unchanged. The pilot kept the four
research-domain gold examples (report section 4: "Calibration was deliberately left
research-flavored"). Decision Q11 keeps them for E6 and reports this as a limitation.
`test_judge_pipeline_renders_swe_prompt` asserts the research examples appear in the
rendered SWE judge prompt.

## 4. New briefs (swe_09, swe_10)

"Agentic" in the pilot report (sections 1.4 and 3) meant briefs about LLM-agent runtimes:
swe_07 (tool loop for an LLM agent, three read-only tools, call budgets) and swe_08
(work queue for a pool of agent workers). The two new briefs follow that reading:

- `swe_09`, domain "software engineering - agentic coding tools": a single-file coding
  agent that edits a repository through exactly two tools (`read_file`, `write_file`) under
  call budgets. The objective states that the change request names the candidate files
  (the agent has no tool to list the repository). Constraint 2 allows the final summary to
  come from the same call as the last file content, so the plan-then-act direction fits the
  4-call budget. Direction 3 reads each named file with its own `read_file` call (no batch
  read exists). 7 hard constraints. Programmatically checkable: exactly 2 tools, at most 4
  LLM calls, at most 6 tool calls, at most 3 files written, under 150 lines, stdlib only
  (forbidden imports: `subprocess`, test runners; forbidden calls: `os.system`, `os.popen`).
- `swe_10`, domain "software engineering - agent tool execution": a tool-call executor
  that validates and runs an LLM agent's tool invocations against a fixed registry of four
  tools. Constraint 2: the registry must hold exactly four tools; `execute` rejects a
  registry that does not, and rejects any call naming a tool outside it (contract test
  `test_registry_must_hold_exactly_four_tools`). 7 hard constraints. Programmatically
  checkable: exactly one public function, registry of exactly 4 tools, no
  `eval`/`exec`/`compile`/`__import__`/`importlib.import_module`, per-call timeout at most
  5 s, at most 2 retries, under 250 lines, stdlib only.

Both have 3 success criteria, 3 plausible directions, 4 banned moves, in the recovered
briefs' style. Agentic briefs in the set of 10: swe_07, swe_08, swe_09, swe_10 (4 of 10;
E6 asks for 3). Brief difficulty is not calibrated against the ideation briefs (C10).

## 5. Executable-verification arm (E6: 4 briefs, 2 models, 1 repetition)

Briefs chosen for deterministic I/O and pure interfaces: `swe_02` (LRU+TTL cache
decorator), `swe_06` (log-summary CLI, stdin to stdout), `swe_08` (agent work queue),
`swe_10` (agent tool executor, agentic). Time-dependent behaviour (TTL, leases) is made
deterministic by an injected `clock` callable. swe_08 and swe_10 keep an agentic brief in
the arm.

Each `drift_bench/briefs_swe/tests/<brief_id>_test.py` states the interface contract in
its module docstring and holds 6 to 9 pytest tests (29 in total). Reference solutions of 25
to 50 stdlib lines pass all 29 tests and all 13 static checks (checked 2026-10-26 in the
scratchpad; the reference solutions are not kept in the repository, so the subjects cannot
be contaminated by them).

The subject sees the contract through `drift_bench/briefs_swe_exec/`: copies of the 4
briefs with the contract appended to `objective` (generated by
`swe_verify --make-exec-briefs`, checked in, deterministic). Ids and all other fields are
unchanged; the copies validate.

Output cap: the code-output templates put the Implementation section (the full
`solution.py` in one ```python block) first, ahead of the six prose sections, so a
truncated answer still carries the code. Phase C runs with `drift_bench/config_swe_code.yaml`
(subject `max_tokens` 8192 instead of 4096; Gemini Pro's research final turns already
average 2,300 to 2,800 tokens). The results CSV carries `truncated`
(final message `token_count >= max_tokens` of the model in that config).

`drift_bench/analysis/camera_ready/swe_verify.py`:

- `run_tests(code_dir, brief_id, tests_dir, timeout_s)`: copies `solution.py` and the
  brief's test file into a fresh temp directory, runs `python -m pytest` there as a
  subprocess in its own session (`start_new_session=True`) with a wall-clock timeout,
  writes pytest output to a file (not a pipe, so a child the solution spawned cannot hold
  the harness open), then `os.killpg`s the whole group on exit or timeout so orphans do not
  survive (tested with a solution that spawns `sleep 40` at import, both on normal exit
  and on timeout). Returns `passed, failed, errors, skipped, total, all_passed, returncode,
  timed_out, duration_s, missing, stdout_tail`.
- `static_checks(solution_path, brief_id)`: the programmatically checkable constraints
  (`STATIC_RULES`). Rules: `stdlib_only` (via `sys.stdlib_module_names`),
  `forbidden_modules`, `forbidden_calls` (bare-name calls such as `eval(...)` and listed
  attribute calls such as `os.system(...)` or `importlib.import_module(...)`; `re.compile`
  does not trip `compile`), `max_loc`, `public_functions` (counts module-level `def`
  statements only; classes, nested functions and lambdas do not count), `public_methods`
  (public methods of a named top-level class; swe_08 `WorkQueue` must have exactly 4).
  Each rule names its constraint. swe_02 also forbids `sqlite3, shelve, pickle, socket,
  dbm` (no persistence, no external tier). swe_06's 10 GB / 2 min / 500 MB performance
  constraints are judge-only: the harness does not measure them.
- `extract_blocks(text)`: every fenced python block (` ```python `, ` ```Python `,
  ` ```python3 `, ` ```py `, ` ```python title=... `; CRLF normalised; an unclosed final
  block runs to the end of the text and is flagged). `choose_block(blocks, entry_symbol)`:
  the block that defines the brief's entry symbol (`ENTRY_SYMBOLS`: `lru_ttl_cache`,
  `__main__`, `WorkQueue`, `execute`) wins; otherwise the longest. The CSV records
  `n_blocks` and `unclosed_block`.
- `interface_spec(brief_id)` and `make_exec_briefs(briefs_dir, out_dir)`: see above.
- `extract_all(data_root)` (`--extract-only`): writes every `solution.py` and runs nothing, so the author can read the code before any execution.
- `run(data_root, out_dir)`: verifies every transcript under
  `<data_root>/camera_ready/e6_swe_verify/transcripts/`, keeps extracted code under
  `.../e6_swe_verify/code/<run_id>/solution.py`, writes `swe_verify_results.csv` plus
  `.provenance.json`.

Isolation limitation, stated in the module docstring: a temp directory, a timeout and a
process-group kill only. Subject code runs with the user's permissions and network access.
Read each generated `solution.py` before running the harness, or run it in a container.
The arm has 16 runs.

LOC counting: non-blank lines that are not pure comment lines. Docstrings count. The briefs
say "excluding tests and comments" and are silent on docstrings, so the count is
conservative (higher).

## 6. Every guess in the reconstruction

1. `rubrics_swe.yaml`, objective_fidelity anchor 4. The report lists
   `research objective -> design objective` once and counts 6 changed lines. Both the
   description and anchor 4 contain the phrase. Changing both gives 7 lines. One of the two
   was probably left unchanged in the pilot; which one is unknown. Both are changed here.
2. `rubrics_swe.yaml`, objective_fidelity anchor 3. The report gives the substitution
   `independent/dependent variable -> primary component or its interface` once; anchor 3
   has the plural form "independent and dependent variables are preserved". The plural
   rendering "primary component and its interface are preserved" is my wording.
3. `rubrics_swe.yaml`, objective_fidelity description. The research line reads "still
   answer the original research objective". `system.j2` changed "answer" to "meet" in a
   different sentence; the report does not list "answer" -> "meet" for the rubric, so
   "answer" is kept.
4. `judge.j2`: the exact position of the one-line change is the first line (the only line
   containing "evaluating a research proposal"), so this is not a guess. The trailing
   newline of the file differs from the research template (the research file has none);
   rendering is unaffected.
5. The code-output templates in `prompts_swe_code/`, `config_swe_code.yaml`, the
   `briefs_swe_exec/` copies and the interface contracts are new design, not
   reconstruction. The pilot had no executable arm.
6. "Agentic" is read from the report's own labels (swe_07, swe_08). The report never
   defines the term.
7. The pilot's `max_tokens` for probes was the repository default (1024 at the time, per
   `config.yaml` comment). E6 inherits `experiment.probe_max_tokens: 1024`. Gemini Pro
   probe truncation at 1,020 tokens (rules file) may recur; the probe analysis must apply
   the same truncation filter as the main benchmark.

## 7. Proposed E6 commands

Design: 10 briefs x 2 conditions (`single_shot`, `multi_turn_pressure`) x 3 subjects x 2
repetitions = 120 subject runs; Opus judge on all 120 (the config's cross-family rule
already routes every non-Anthropic subject to `anthropic/claude-opus-4-6`, so no override
is needed); no auditor. Executable arm: 4 briefs x 2 conditions x 2 subjects x 1 repetition
= 16 code-output runs, judged by Opus and tested locally. Budget cap USD 30 in total,
split per phase so the caps sum to 30: Phase A 15, Phase B 8, Phase C subjects 5,
Phase C judge 2.

No change to `pipeline.py` is needed. The uncommitted `pipeline.py` already takes
`--config`, `--briefs-dir`, `--prompts-dir`, `--rubrics`, `--calibration`,
`--repetitions`, `--skip-scoring` and `--max-budget`. The pipeline has no "judge only, no
auditor" switch (scoring always runs judge then auditor), so the judge phase goes through
`drift_bench/rescore.py`, which takes the same `--briefs-dir`, `--prompts-dir`,
`--rubrics` flags and `--judge-override`. Optional nicety for the pipeline owner (X5): a
`--no-auditor` flag would make Phase B a single command; it is not required.

All paths relative to the repository root. Run `--dry-run` on each rescore call first.

A subject run that errors writes nothing: `pipeline.py` returns `None` on an exception
and the runners call `transcript.save` only after every turn has succeeded. So every
transcript on disk is complete, and Phase B needs no exclude list. (The cost of a failed
attempt is not recorded anywhere; the per-phase spend below counts completed runs only.)

Judge disclosure (decision 3, round 2): the judge's system message in
`drift_bench/judges/judge.py` reads "You are an expert research evaluation judge";
it is hard-coded, was research-flavoured in the first pilot too, and stays so in E6.

Phase A, subject runs with probes, no scoring (cap USD 15):

```bash
.venv/bin/python -m drift_bench.pipeline \
  --config drift_bench/config.yaml \
  --briefs-dir drift_bench/briefs_swe \
  --prompts-dir drift_bench/prompts_swe \
  --output drift_bench/data/camera_ready/e6_swe \
  --models gemini/gemini-3.1-flash-lite-preview gemini/gemini-3.1-pro-preview openai/gpt-5.4-mini \
  --conditions single_shot multi_turn_pressure \
  --repetitions 2 --concurrency 4 --skip-scoring --max-budget 15
```

Phase B, Opus judge on the 120 transcripts, no auditor (cap USD 8; writes
`judge_<run_id>.jsonl`; `rescore.py` accepts prefix `judge` in a fresh camera-ready
directory and is resumable):

```bash
.venv/bin/python -m drift_bench.rescore \
  --transcripts-dir drift_bench/data/camera_ready/e6_swe/transcripts \
  --scores-dir drift_bench/data/camera_ready/e6_swe/scores \
  --briefs-dir drift_bench/briefs_swe \
  --prompts-dir drift_bench/prompts_swe \
  --rubrics drift_bench/judges/rubrics_swe.yaml \
  --judge-override anthropic/claude-opus-4-6 --output-prefix judge --mode judge \
  --conditions single_shot multi_turn_pressure --concurrency 4 --max-budget 8 --dry-run
# then the same command without --dry-run
```

Phase C, executable arm. The contract briefs are checked in
(`drift_bench/briefs_swe_exec/`), so no generation step. Subjects run on the code-output
templates with the 8192-token config (cap USD 5), then Opus judges them (cap USD 2), then
the harness tests the code locally (no API calls):

```bash
.venv/bin/python -m drift_bench.pipeline \
  --config drift_bench/config_swe_code.yaml \
  --briefs-dir drift_bench/briefs_swe_exec \
  --prompts-dir drift_bench/prompts_swe_code \
  --output drift_bench/data/camera_ready/e6_swe_verify \
  --models gemini/gemini-3.1-pro-preview openai/gpt-5.4-mini \
  --conditions single_shot multi_turn_pressure \
  --repetitions 1 --concurrency 4 --skip-scoring --max-budget 5

.venv/bin/python -m drift_bench.rescore \
  --config drift_bench/config_swe_code.yaml \
  --transcripts-dir drift_bench/data/camera_ready/e6_swe_verify/transcripts \
  --scores-dir drift_bench/data/camera_ready/e6_swe_verify/scores \
  --briefs-dir drift_bench/briefs_swe_exec \
  --prompts-dir drift_bench/prompts_swe \
  --rubrics drift_bench/judges/rubrics_swe.yaml \
  --judge-override anthropic/claude-opus-4-6 --output-prefix judge --mode judge \
  --concurrency 4 --max-budget 2

# Extract first, run nothing; then read every
# drift_bench/data/camera_ready/e6_swe_verify/code/<run_id>/solution.py (no sandbox).
.venv/bin/python -m drift_bench.analysis.camera_ready.swe_verify --extract-only
# Then test the code locally.
.venv/bin/python -m drift_bench.analysis.camera_ready.swe_verify \
  --config drift_bench/config_swe_code.yaml
```

Which two subjects the executable arm uses is open (Q4 says "2 models"). The commands
above take Pro and GPT-5.4-mini: the strongest Google model and the non-Google subject.
Flash-Lite instead of Pro would be cheaper and is the high-KBV model; the author decides.

Phase D, aggregation and KBV (no API calls):

```bash
.venv/bin/python -c "from drift_bench.analysis.aggregate import aggregate_scores; from pathlib import Path; \
  aggregate_scores(Path('drift_bench/data/camera_ready/e6_swe/scores'), Path('drift_bench/data/camera_ready/e6_swe/aggregated/all_scores.parquet'))"
```

`aggregate_scores` handles a judge-only directory (no auditor files) and yields the `j_*`
columns. Probe accuracy and KBV: `drift_bench.analysis.probes.analyze_all_probes` on the
E6 transcripts with `briefs_swe`, then `knows_but_violates_table` with its fixed defaults
(`overall_accuracy >= 0.8` and `j_constraint_adherence < 4`). An `e6` analysis module that
writes the split table (ideation vs coding, per model) with provenance is a follow-up for
the analysis executor; it does not exist yet.

Cost estimate from the pilot's ledger (report section 5; Pro USD 0.20 per run, Flash-Lite
USD 0.011 per run, Opus judge USD 0.042 per call) and the config `pricing` map for
GPT-5.4-mini (about 3x Flash-Lite): Phase A about USD 10.5 (Pro 8.2, Flash-Lite 0.5,
mini 1.5 to 2), Phase B about USD 5.1, Phase C about USD 3.5 (8 Pro runs and 8 mini runs
at up to 8192 output tokens, 16 judge calls). Total about USD 19 against the USD 30 cap;
the per-phase caps (15 / 8 / 5 / 2) leave each phase about 40% headroom. Pilot numbers
are from July 2026 prices.

## 8. Draft CHANGELOG entry (not applied; plan row F4 asks for it, CHANGELOG.md is outside this executor's file set)

```
## 2026-10-26 — SWE pilot briefs and prompts adopted into the repository

The coding-domain pilot of 2026-07-26 (32 runs) lost its raw files. Its 8 briefs and 4
subject templates were recovered verbatim from the pilot agent's session log and now live
in `drift_bench/briefs_swe/` and `drift_bench/prompts_swe/`, outside the 38 research
briefs. The adapted judge template (`prompts_swe/judge.j2`) and rubrics
(`judges/rubrics_swe.yaml`) were reconstructed from the pilot report's list of edits
(`docs/swe_pilot_recovered/pilot_coding_domain.md`, section 4); one rubric line is a
guess (`docs/swe_pilot_reconstruction.md`, section 6). Two new agentic briefs (swe_09,
swe_10), code-output templates (`prompts_swe_code/`), contract briefs
(`briefs_swe_exec/`), a Phase C config (`config_swe_code.yaml`) and a local test harness
(`analysis/camera_ready/swe_verify.py`) were added for the extended pilot. The pilot's
per-run pressure scores survive only as a table
(`docs/swe_pilot_recovered/swe_pilot_run_level_results.md`); its single-shot per-run
scores are lost.

The extended pilot (E6) was run on 2026-10-26 with litellm 1.83.8 and is complete:
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
```

## 9. Open questions for the author

1. Executable-arm subjects: Pro + GPT-5.4-mini (as written) or Flash-Lite + GPT-5.4-mini?
2. Should the executable arm's judge see the interface contract (it does, through the
   `briefs_swe_exec` copies) or the plain brief? The contract adds about 10 lines to the
   objective and could change objective-fidelity scores. Judging with the plain brief is
   possible by passing `--briefs-dir drift_bench/briefs_swe` to the Phase C rescore.
3. `judge.py`'s hard-coded system message ("expert research evaluation judge") stays
   research-flavoured, as in the pilot. Accept, or ask the pipeline owner for a
   `judge_system.j2` template?
4. Rubric anchor 4 of objective_fidelity: changed here; the pilot may have left it. If the
   author remembers, revert one line and update `test_rubrics_swe_adaptations_match_pilot_report`
   (expects 7 marked lines).
5. `briefs_swe/tests/` holds the contract files next to the briefs. If a reviewer runs
   `pytest drift_bench/`, the `conftest.py` guard keeps them out; the contracts are only
   runnable through `swe_verify.run_tests`.
6. The pilot report's `pilot_coding_domain_SUPERSEDED_blocked.md` was not recovered and is
   not needed.

## 10. F4 review (2026-10-26): changes applied

| Item | Change |
|---|---|
| 1 | `forbidden_calls` now matches bare-name calls only (`eval`, `exec`, `compile`, `__import__`) plus listed attribute calls (`importlib.import_module`; `os.system`, `os.popen` for swe_09). `re.compile` passes; `compile('x','<s>','exec')` fails. Tested. |
| 2 | `extract_blocks` handles ` ```Python `, ` ```python3 `, ` ```py `, ` ```python title=... `, CRLF, and an unclosed final block (`unclosed_block` flag). `choose_block` prefers the block defining the entry symbol, else the longest. `n_blocks` and `unclosed_block` are CSV columns. Tested per case. |
| 3 | Implementation section first in both `prompts_swe_code` templates; `drift_bench/config_swe_code.yaml` (subject `max_tokens` 8192, documented in its header); `truncated` column. Tested. |
| 4 | pytest runs with `start_new_session=True`, output to a file; `os.killpg` on exit and on timeout. Tested with a solution that spawns `sleep 40` at import (normal exit and timeout); no orphan survives. |
| 5 | swe_10 constraint 2 reworded to reject a registry that does not hold exactly four tools; contract test added (swe_10 now has 9 tests). |
| 6 | swe_09: summary may come from the last content call (constraint 2); the change request names the candidate files (objective); direction 3 reads each named file with its own `read_file` call. |
| 7 | Per-phase caps 15 / 8 / 5 / 2 (sum 30) in section 7. |
| 8 | swe_02 `forbidden_modules` (`sqlite3, shelve, pickle, socket, dbm`); swe_08 `public_methods` rule on `WorkQueue` (exactly 4); swe_06 performance constraints noted as judge-only (section 5). |
| 9 | Dead `"os.system"` entry removed from `forbidden_modules`; implemented as an attribute-call check. |
| 10 | `public_functions` rule (module-level `def` only) documented in section 5. |
| 11 | `drift_bench/briefs_swe_exec/` generated and checked in (literal block YAML for readability); `test_checked_in_exec_briefs_match_regeneration` asserts byte-identical regeneration. |

## 11. Round-2 decisions (2026-10-26) and Step 1 hardening

Decisions on section 9: (1) executable-arm subjects are Gemini Pro and GPT-5.4-mini;
(2) the executable-arm judge sees `briefs_swe_exec`; its objective_fidelity is reported
separately from Phase B, plus a 16-call plain-brief sensitivity rescore
(`--briefs-dir drift_bench/briefs_swe`, cap USD 1.00); (3) keep the research-flavoured
`judge.py` system message and disclose it (section 7); (4) keep rubric anchor 4 as recorded.

Hardening applied to `swe_verify.py` before E6 (all with tests):

| Item | Change |
|---|---|
| M1 | `builtins` added to `forbidden_modules` for swe_09 and swe_10. `attribute_calls` resolves import aliases: `import os as o; o.system()` and `from os import system; system()` both read as `os.system`. Residual gap (string-built `getattr`) documented in the module docstring. |
| L1 | `finish_reason` column; `truncated = finish_reason == "length" or token_count >= cap`. |
| L2 | Each block is `textwrap.dedent`ed (blocks indented under a list item). |
| L3 | Whitespace-only fenced blocks are dropped; a message with only an empty fence is `no_code`. |
| L4 | `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` in the subprocess environment. |
| L5 | The no-errored-transcript sentence in section 7. |
| extra | `--exclude-run-ids-file` (`run_id reason` per line): rows marked `refused` with `refused_reason`, never extracted to disk or executed. Used in Step 3 for solutions the author declines to run. |

Phase C sensitivity rescore (decision 2), after the exec-arm judge run:

```bash
.venv/bin/python -m drift_bench.rescore \
  --config drift_bench/config_swe_code.yaml \
  --transcripts-dir drift_bench/data/camera_ready/e6_swe_verify/transcripts \
  --scores-dir drift_bench/data/camera_ready/e6_swe_verify/scores_plain_brief \
  --briefs-dir drift_bench/briefs_swe \
  --prompts-dir drift_bench/prompts_swe \
  --rubrics drift_bench/judges/rubrics_swe.yaml \
  --judge-override anthropic/claude-opus-4-6 --output-prefix judge --mode judge \
  --concurrency 4 --max-budget 1
```
