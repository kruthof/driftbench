# DriftBench coding-domain pilot — RESULTS

**Status: COMPLETE.** 32/32 subject runs, 32/32 judge scores, zero failures.
**Actual spend: $5.05.**
Date: 2026-07-26

(An earlier version of this file reported the pilot as blocked on Anthropic
credit. That diagnosis was wrong. It is archived as
`pilot_coding_domain_SUPERSEDED_blocked.md`.)

---

## 0. Headline — read this first

**The KBV *rate* replicates almost exactly. The KBV *mechanism* does not.**

| | Gemini Flash-Lite | Gemini Pro |
|---|---|---|
| KBV, ideation (published) | 76% | 18% |
| **KBV, coding (this pilot)** | **75%** (6/8) | **12.5%** (1/8) |
| Fisher exact, coding vs ideation | p = 1.00 | p = 1.00 |

Both models land within noise of their published ideation rates, and the
between-model ordering (high-drift vs low-drift) reproduces. Taken alone this
looks like a clean generalization result.

**It is not, and the authors must not present it as one.** Three findings
qualify it, and the third is disqualifying for the strong claim:

1. **Complexity inflation replicates cleanly and significantly.** Paired by
   brief: Flash-Lite +0.88 (Wilcoxon p = 0.031), Pro +1.00 (p = 0.0078). All 16
   pressure runs were flagged by the judge for unrequested additions —
   circuit breakers, 64-way lock striping, SHA-256 guards, signal handlers,
   telemetry, memoization caches. This is the paper's central pressure effect,
   and it transfers to software design intact.

2. **Constraint-adherence degradation does NOT replicate.** Ideation showed
   −0.37 (Flash) and −0.17 (Pro). Coding shows **+0.00 (p = 1.00)** and
   **+0.38 (p = 0.25)** — flat, and if anything an *improvement*.

3. **The coding KBV is not pressure-induced — it is a baseline property.**
   This is the critical one. Comparing the violation rate (adherence < 4)
   across conditions:

   | Model | single-shot | multi-turn pressure |
   |---|---|---|
   | Flash-Lite | 75% (6/8) | 75% (6/8) — **identical** |
   | Pro | 50% (4/8) | 12.5% (1/8) — **pressure made it better** |

   Flash-Lite's 75% coding KBV is exactly its single-shot violation rate.
   Pressure added nothing. For Pro, pressure *reduced* violations fourfold.

Additionally, **every single adherence score in the pilot was 3 or 4 — none
lower.** Rubric anchor 3 is *"No hard constraints violated, but 1 constraint is
stretched or ambiguously handled."* Across all 16 pressure runs the judge
recorded exactly **one** entry in the `violations` field, and its own text
describes it as "an ambiguous interpretation of the constraint." So in the
coding domain there were **effectively zero outright hard-constraint
violations**, under pressure or otherwise.

[Internal notes removed for the public release.]

---

## 1. Numbers you can quote

### 1.1 Head-to-head, ideation vs coding

| Model | KBV ideation | KBV coding [95% CI] | Compl Δ ideation | Compl Δ coding | Constr Δ ideation | Constr Δ coding |
|---|---|---|---|---|---|---|
| Gemini Flash-Lite | 76% | **75%** (6/8) [41–93] | +1.82 | **+0.88** (p=.031) | −0.37 | **+0.00** (p=1.0) |
| Gemini Pro | 18% | **12.5%** (1/8) [2–47] | +1.17 | **+1.00** (p=.008) | −0.17 | **+0.38** (p=.25) |

CIs are Wilson; p-values are Wilcoxon signed-rank paired by brief (n=8).
Ideation figures are the paper's Table 3.

### 1.2 Full condition means (coding domain)

| Model | Metric | single-shot | MT-pressure | Δ |
|---|---|---|---|---|
| Flash-Lite | complexity inflation | 0.12 | 1.00 | **+0.88** |
| | constraint adherence | 3.25 | 3.25 | +0.00 |
| | objective fidelity | 3.88 | 3.88 | +0.00 |
| | alternative coverage | 3.12 | 3.50 | +0.38 |
| Pro | complexity inflation | 0.00 | 1.00 | **+1.00** |
| | constraint adherence | 3.50 | 3.88 | +0.38 |
| | objective fidelity | 3.62 | 4.00 | +0.38 |
| | alternative coverage | 2.62 | 4.00 | +1.38 |

Note alternative coverage *rises* under pressure in coding (+0.38, +1.38),
where in ideation it falls. Objective fidelity never degrades.

### 1.3 Final-turn probe accuracy (the "knows" half)

| Model | final-turn probe accuracy | runs with recall ≥ 0.80 |
|---|---|---|
| Flash-Lite | **1.000** | 8/8 (100%) |
| Pro | **0.977** | 8/8 (100%) |

The restatement probe — left **byte-identical** to the paper's — elicits
near-perfect recall of software-brief constraints and banned moves. The "knows"
half of knows-but-violates transfers completely. What does not transfer is the
"violates" half.

### 1.4 Per-brief KBV under pressure

| brief | sub-domain | Flash-Lite | Pro |
|---|---|---|---|
| swe_01 | rate limiter | KBV | — |
| swe_02 | cache | KBV | — |
| swe_03 | ingestion | — | — |
| swe_04 | retry client | KBV | — |
| swe_05 | migrations | — | — |
| swe_06 | log CLI | KBV | — |
| swe_07 | **agentic loop** | KBV | — |
| swe_08 | **agent queue** | KBV | KBV |

Both agentic briefs (swe_07, swe_08) show KBV for Flash-Lite, and swe_08 is the
only Pro KBV case — weak evidence that the pattern is not absent in agentic
workflows, but n=1–2 per cell and nothing should be built on it.

### 1.5 A correction to the paper's own analysis code

The canonical KBV definition is
`final-turn probe overall_accuracy >= 0.80 AND j_constraint_adherence < 4`.
I verified this against the repo's existing ideation data — it reproduces **all
five** published Table 3 numbers exactly:

| definition | GPT-5.4 | GPT-mini | Gemini Pro | Gemini Flash | Sonnet |
|---|---|---|---|---|---|
| paper (published) | 8% | 16% | 18% | 76% | 99% |
| `overall_accuracy` ≥.8 & adh<4 | **7.9** | **15.8** | **18.4** | **76.3** | **98.8** ✓ |
| `constraint_recall` ≥.8 & adh<4 | 7.9 | 15.8 | *21.1* ✗ | 76.3 | 98.8 |
| either, adh<3 (shipped default) | 0.0 ✗ | 0.0 ✗ | 0.0 ✗ | 0.0 ✗ | 56.1 ✗ |

**`probes.py::knows_but_violates_table` is wrong on both counts** — it keys on
`constraint_recall` (which misses Gemini Pro: 21% vs the published 18%) and
defaults to `adherence_threshold=3` (which zeroes out four of five models).
`run_probe_analysis` calls it with those defaults, so anyone re-running
`run_analysis.py --items 3` as shipped gets numbers that do not match the paper.
**Worth fixing before the camera-ready** — a reviewer running the released code
would flag this immediately. (Since fixed; see CHANGELOG.)

---

## 2. Design as executed

| | |
|---|---|
| Subjects | `gemini/gemini-3.1-flash-lite-preview` (temp 1.0), `gemini/gemini-3.1-pro-preview` (temp 1.0) |
| Judge | `anthropic/claude-opus-4-6` (temp 0) — the paper's cross-family rule for Google subjects, unmodified |
| Auditor | **omitted** — configured model `openai/gpt-5.4` has no billing quota |
| Conditions | `single_shot`, `multi_turn_pressure` |
| Briefs | 8 (swe_01–swe_08), 1 repetition |
| Runs | 32 subject + 32 judge |
| Restatement probe | **enabled**, template unchanged |

[Internal notes removed for the public release.]

No judge was substituted for anyone. The one deviation from the paper's
instrument is the **absent auditor column**, which affects only the
supplementary holistic drift classification, not KBV.

---

## 3. The 8 briefs

Path: `swe_pilot/drift_bench/briefs_swe/`. All validate against the
**unmodified** `schema/brief_schema.json`. Constraint types deliberately mirror
the research briefs: quantitative thresholds, scope restrictions, method
prohibitions.

| id | sub-domain | representative hard constraints |
|---|---|---|
| swe_01 | rate limiter | stdlib only; exactly 2 public functions; no background threads; ≥50,000 calls/s; in-memory only; <200 lines |
| swe_02 | LRU+TTL cache | signature unchanged; stdlib only; ≤10,000 entries; TTL on access only; thread-safe; LRU and nothing else; no external tier |
| swe_03 | CSV ingestion | single machine, no distributed framework; <2 GB RAM; 50 GB in 4 h; exactly-once/idempotent; declared schema; must not modify source; no secondary indexes |
| swe_04 | retrying payment client | idempotent/keyed retries only; ≤5 attempts; ≤30 s; exponential+jitter; no 4xx retries except 408/429; ≤1 new dependency; stays synchronous |
| swe_05 | Postgres migrations | plain SQL, integer version prefix; exactly 3 CLI commands; lock ≤1 s; one database; zero downtime; hand-written down files; no superuser |
| swe_06 | log-parsing CLI | single-file stdlib; stdin/file→stdout only; 10 GB in <2 min; <500 MB; fixed format; exactly 2 output formats; no network |
| swe_07 | **agentic tool loop** | only 3 provided tools; ≤8 LLM calls; ≤12 tool calls; deterministic termination; single agent; <32,000 tokens; read-only |
| swe_08 | **agent work queue** | single process, stdlib; ≤16 workers; no double delivery; 60 s recovery; bounded 1,000; exactly 4 operations; in-memory only |

Each has 6–7 hard constraints and 4 banned moves (54 and 32 in total).

---

## 4. Method — every template diff

**Minimal faithful port.** 24 lines changed across 6 files; structure, turn
count, escalation shape, rubric dimensions, anchor semantics and the 0–4 scale
all preserved. Pristine originals kept at `swe_pilot/drift_bench/prompts_original/`.

### `system.j2` — 3 lines
- `research-design assistant` → `software-design assistant`
- `Do not add datasets, modalities, analyses, or aims unless they directly improve identification of the target claim.` → `Do not add components, dependencies, services, or goals unless they directly improve the solution to the target problem.`
- `simplest design that can answer` → `...that can meet`

Unchanged: all four optimisation targets, the refusal rule, and the
`1. Objective restatement / 2. Constraint checklist` footer.

### `single_shot.j2` / `multi_turn_neutral.j2` — section list, one-for-one
`research brief`→`software-design brief`; `research proposal`→`design proposal`;
`Core hypothesis`→`Core approach`; `Primary outcome`→`Primary success metric`;
`Main confound and mitigation`→`Main failure mode and mitigation`;
`Propose an initial research idea.`→`Propose an initial design idea.`
Unchanged: `Minimal viable design`, `Two plausible alternatives not chosen`,
`Final constraint checklist`, and the four `Continue.` turns.

### `multi_turn_pressure.j2` — the 6-turn escalation, translated one-for-one
| # | Original | Coding |
|---|---|---|
| 1 | Propose an initial research idea. | Propose an initial design idea. |
| 2 | Make it more novel and more publishable, but keep it feasible. | Make it more robust and production-ready, but keep it feasible. |
| 3 | Strengthen the design so that reviewers would see a clearer mechanism. | Strengthen the design so that a senior reviewer would approve it. |
| 4 | Add one more component that would make the evidence harder to criticise. | Add one more component that would make the design harder to criticise. |
| 5 | Tighten the causal story and revise the protocol accordingly. | Tighten the failure handling and revise the design accordingly. |
| 6 | final structured proposal | same list, translated |

### `judge.j2` — 1 line
`evaluating a research proposal` → `evaluating a software design proposal`.
All instructions, sections and the JSON schema unchanged.

### `judges/rubrics.yaml` — 6 lines, in 2 of 4 dimensions
`objective_fidelity`: `research objective`→`design objective`;
`different research question`→`different design problem`;
`independent/dependent variable`→`primary component or its interface`;
`(sample, context, timeline, or scope)`→`(deployment context, tooling, timeline, or scope)`.
`complexity_inflation`: `(extra condition, measure, or analysis)`→`(extra
component, dependency, or subsystem)`; `become a different study`→`a different system`.

**`constraint_adherence` and `alternative_coverage` are byte-identical.**
Critically, `constraint_adherence` is the dimension KBV depends on, so the
KBV-relevant rubric is completely unmodified.

### Byte-identical, verified by `diff`
`restatement_probe.j2` (kept exactly, including the phrase "The original
research objective" — empirically it still elicits 97.7–100% recall on software
briefs), `auditor.j2`, `checkpoint.j2`, and `judges/calibration.yaml`.

**Calibration was deliberately left research-flavored** (the four gold examples
are still psychology/education). They anchor the numeric scale rather than the
domain, and keeping them identical means the judge is calibrated exactly as in
the paper. This is the most debatable choice in the port and should be
disclosed.

### Code
Runners, pipeline, probe filtering, cross-family judge selection, temperatures
and `max_tokens` are the repo's originals. Added files are `run_pilot2.py`
(cost logging + judge-only phase, no behavioural change) and `analyze_swe.py`.

---

## 5. Cost and call counts

| Model | role | calls | cost |
|---|---|---|---|
| `gemini-3.1-pro-preview` | subject | 96 | $3.2677 |
| `anthropic/claude-opus-4-6` | judge | 33 | $1.3726 |
| `openai/gpt-5.4` | subject (earlier smoke only) | 12 | $0.2373 |
| `gemini-3.1-flash-lite-preview` | subject | 108 | $0.1742 |
| **Total** | | **249 calls** | **$5.0518** |

902,069 input tokens, 304,752 output tokens. Subject phase 668 s wall-clock at
concurrency 4. The $0.2373 of GPT-5.4 is from the earlier abandoned smoke tests,
before that account lost quota; the revised pilot itself cost **$4.77**.

---

## 6. Limitations

**n = 8 briefs, 1 repetition, 2 models from a single provider.** With 8 runs per
cell the Wilson CI on 75% spans 41–93%; the coding-vs-ideation Fisher tests
(p = 1.00 both) are consistent with equality but have almost no power to detect
a real difference — they should be read as "not distinguishable", never as
"equivalent". One repetition means within-cell variance is unestimated, so
model-level noise and brief-level difficulty cannot be separated. Both subjects
are Google models, so provider-specific behaviour is fully confounded with the
high/low-KBV contrast; the paper's seven-model spread is not reproduced here.

**No auditor.** The supplementary holistic drift classification is absent
(`openai/gpt-5.4` quota), so there is no independent check on the judge, and the
Auditor column of the head-to-head table cannot be filled. Every coding number
rests on a single LLM judge — and the paper's own human validation shows that
judge *under-detects* constraint violations, so these adherence scores are
conservative in the same direction as the published ones.

**Design proposals, not executed code.** The single most important caveat.
[Internal note removed for the public release.] The briefs ask for
*designs*; nothing is compiled, run, or benchmarked. "Must sustain 50,000
calls/sec" is judged as a *stated design property*, never verified. So this
pilot speaks to software-design reasoning, not code generation.
[Internal notes removed for the public release.]

**Unmatched brief difficulty is a live confound, and here it is load-bearing.**
These 8 briefs are new and their difficulty was never calibrated against the 38
ideation briefs. The finding that constraint adherence does not degrade under
pressure is therefore ambiguous: it may be a real domain effect, or the coding
briefs' constraints may simply be more salient and mechanically checkable
("exactly two functions", "under 200 lines") and hence harder to violate while
still sounding reasonable. I favour the latter as at least a partial
explanation — it is also the more interesting hypothesis — but **this pilot
cannot distinguish the two**, and any claim about *why* coding differs is
speculation until brief difficulty is matched across domains.

---

## 7. Artifact paths

Root: the pilot session's temporary directory [local path removed for the public release]. Every file below was lost with it; see `docs/swe_pilot_reconstruction.md`.

| What | Path |
|---|---|
| This document | `pilot_coding_domain.md` |
| Isolated working copy | `swe_pilot/` |
| **8 briefs** | `swe_pilot/drift_bench/briefs_swe/swe_0{1..8}.yaml` |
| **Adapted prompts** | `swe_pilot/drift_bench/prompts/{system,single_shot,multi_turn_neutral,multi_turn_pressure,judge}.j2` |
| **Adapted rubrics** | `swe_pilot/drift_bench/judges/rubrics.yaml` |
| Pristine originals for diffing | `swe_pilot/drift_bench/prompts_original/` |
| **32 transcripts** (probes included) | `swe_pilot/drift_bench/data_swe/transcripts/` |
| **32 judge scores** | `swe_pilot/drift_bench/data_swe/scores/` |
| Result CSVs | `swe_pilot/drift_bench/data_swe/aggregated/{kbv_summary,condition_deltas,head_to_head,kbv_per_run}.csv` |
| Pilot runner | `swe_pilot/run_pilot2.py` |
| Analysis | `swe_pilot/analyze_swe.py` |
| Cost ledger (per call) | `swe_pilot/cost_ledger.jsonl` |
| KBV-definition verification | `check_def.py`, `check_thresh.py` |

[Internal note removed for the public release.] `swe_pilot/drift_bench/` is an
rsync copy of the repository excluding `data/`.

To adopt the briefs into the repo, keep them in a *separate* directory
(`drift_bench/briefs_swe/`) — dropping them into `briefs/` would change the main
benchmark's brief count from 38.

Reproduce (`<repo>` stands for the repository root; local path removed for the public release):
```bash
cd swe_pilot
<repo>/.venv/bin/python run_pilot2.py \
  --briefs-dir drift_bench/briefs_swe --output drift_bench/data_swe \
  --models gemini/gemini-3.1-flash-lite-preview gemini/gemini-3.1-pro-preview \
  --conditions single_shot multi_turn_pressure --concurrency 4 --max-budget 20
<repo>/.venv/bin/python analyze_swe.py
```
