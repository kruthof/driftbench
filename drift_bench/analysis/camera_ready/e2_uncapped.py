"""E2: Sonnet 4.6 at a non-binding output cap vs the April capped runs (plan items E2, B3, B16).

Disk only. No API calls.

E2 re-ran Sonnet 4.6 on the single-shot and pressure cells, one repetition
per brief (76 runs), with ``max_tokens`` 16384 instead of the April 4096
(``camera_ready/e2_sonnet_uncapped/``). Both judge arms scored every run
on 2026-10-26: Opus (``scores_opus/judge_opus_*``) and GPT-5.4
(``scores_gpt/judge_gpt_*``).

April comparator: the clean Sonnet transcripts in the same two cells
(``sample.drop_stray``; 76 + 76 runs, two repetitions per brief). Its Opus
scores are the E1 crossed re-score (``camera_ready/e1_crossed_judge/opus/``,
2026-10-26); its GPT-5.4 scores are the published ``j_*`` columns of
``aggregated/all_scores.parquet`` (judged 2026-04-17). April probes come
from ``analysis/final_probe_accuracy.csv``; E2 probes are recomputed with
``drift_bench.analysis.probes`` and checked against the CSV in the E2 tree.

Cap rules (non-probe assistant turns only): April, ``token_count == 4096``
(the cap bound); E2, ``token_count > 4096`` (would have been truncated in
April; the 16384 cap never bound, every finish_reason is ``stop``).

KBV: final-probe ``overall_accuracy >= 0.8`` and ``constraint_adherence < 4``.

Designs. ``unpaired_runs``: 38 E2 runs vs 76 April runs (Newcombe hybrid
score CI + Fisher exact for proportions; Welch CI + Mann-Whitney for
means). ``brief_paired``: per brief, the E2 value minus the April mean over
that brief's repetitions (paired t CI, brief bootstrap CI, Wilcoxon
signed-rank; nan when every brief ties).

Wording limit: removing the cap does not raise KBV; the data cannot
exclude a drop up to the CI bound. Never "does not change" or "does not
lower".

Outputs (in ``out_dir``), each with a ``.provenance.json`` sibling:

1. ``e2_uncapped_cells.csv``   one row per (source, condition, judge arm)
2. ``e2_uncapped_deltas.csv``  E2 minus April per (condition, arm, metric, design)
3. ``e2_uncapped_offset.csv``  Opus minus GPT-5.4 on the 76 paired E2 runs,
   beside the E1 Sonnet value from ``judge_offset.csv``

Run: ``python -m drift_bench.analysis.camera_ready.e2_uncapped``
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from drift_bench.analysis.camera_ready.kbv_table import (
    ADHERENCE_THRESHOLD,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUT_DIR,
    PROBE_ACCURACY_THRESHOLD,
    SONNET,
    read_jsonl_first,
    wilson_ci,
    write_csv_with_provenance,
)
from drift_bench.analysis.camera_ready.sample import STRAY_RULE, drop_stray, stray_run_ids
from drift_bench.analysis.probes import analyze_all_probes, compute_final_probe_accuracy

OPUS = "anthropic/claude-opus-4-6"
GPT = "openai/gpt-5.4"

E2_DIR = Path("camera_ready") / "e2_sonnet_uncapped"
E1_OPUS_DIR = Path("camera_ready") / "e1_crossed_judge" / "opus"
E2_PROBE_CSV = "final_probe_accuracy.csv"
JUDGE_OFFSET_CSV = "judge_offset.csv"
JUDGE_DRIFT_CSV = "judge_drift_control.csv"

APRIL_CAP = 4096
E2_CAP = 16384
SS = "single_shot"
MTP = "multi_turn_pressure"
CELLS = [SS, MTP]
N_BRIEFS = 38

SOURCE_APRIL = "april_clean"
SOURCE_E2 = "e2_uncapped"
SOURCES = [SOURCE_APRIL, SOURCE_E2]

ARMS: dict[str, dict[str, str]] = {
    "opus": {"judge_model": OPUS, "e2_subdir": "scores_opus", "e2_prefix": "judge_opus"},
    "gpt": {"judge_model": GPT, "e2_subdir": "scores_gpt", "e2_prefix": "judge_gpt"},
}
ARM_ORDER = ["opus", "gpt"]

DIMS = ["objective_fidelity", "constraint_adherence", "alternative_coverage", "complexity_inflation"]
TOKEN_METRICS = ["output_tokens_per_run", "final_tokens", "cap_turn_share", "final_cap_share"]
PROPORTION_METRICS = {"kbv_rate", "final_cap_share"}
METRICS = ["kbv_rate", *DIMS, *TOKEN_METRICS]
DESIGN_UNPAIRED = "unpaired_runs"
DESIGN_PAIRED = "brief_paired"
SCOPE_OVERALL = "overall"

BOOT_SEED = 20261026
B = 5000
Z95 = 1.959964

CAP_RULE_APRIL = f"non-probe assistant turn with token_count == {APRIL_CAP} (the April cap bound)"
CAP_RULE_E2 = (
    f"non-probe assistant turn with token_count > {APRIL_CAP} (would have been truncated in April; "
    f"the E2 cap of {E2_CAP} never bound, every finish_reason is stop)"
)
KBV_RULE = (
    f"final-probe overall_accuracy >= {PROBE_ACCURACY_THRESHOLD} (keyword scorer, drift_bench.analysis.probes) "
    f"and constraint_adherence < {ADHERENCE_THRESHOLD}; denominator = runs with a final probe"
)
CI_BRIEF_BOOT = f"brief-cluster bootstrap, percentile 2.5/97.5, B = {B}, numpy default_rng seed {BOOT_SEED}, briefs resampled with replacement"
CI_WILSON = f"Wilson score interval, z = {Z95}"
CI_NEWCOMBE = "Newcombe hybrid score interval (method 10) for the difference of two independent proportions, Wilson components"
CI_WELCH = "Welch t interval for the difference of two independent means (Welch-Satterthwaite df), 95%"
CI_PAIRED_T = "paired t interval on the per-brief deltas, df = n_briefs - 1, 95%"
TEST_FISHER = "scipy.stats.fisher_exact, two-sided"
TEST_MWU = "scipy.stats.mannwhitneyu, two-sided; nan when every value is identical"
TEST_WILCOXON = "scipy.stats.wilcoxon on the per-brief deltas, zero_method = wilcox, method = approx, two-sided; nan when every brief ties"
DESIGN_STATEMENT = (
    f"{DESIGN_UNPAIRED}: E2 runs (one repetition) vs April clean runs (two repetitions) as independent samples; "
    f"{DESIGN_PAIRED}: per brief, E2 value minus the mean of the April repetitions of that brief"
)
WORDING_LIMIT = "removing the cap does not raise KBV; the data cannot exclude a drop up to the CI bound"
TOKEN_ROWS_NOTE = "token columns and token-metric rows do not depend on the judge; they repeat across the two judge-arm rows"


def _dim_cols() -> dict[str, str]:
    out: dict[str, str] = {}
    for d in DIMS:
        out[f"{d}_mean"] = f"mean {d}"
        out[f"{d}_sd"] = f"sample sd (ddof 1) of {d}"
        out[f"{d}_ci_low"] = f"lower 95% bound of the mean of {d} ({CI_BRIEF_BOOT})"
        out[f"{d}_ci_high"] = f"upper 95% bound of the mean of {d} ({CI_BRIEF_BOOT})"
    return out


COLUMNS: dict[str, dict[str, str]] = {
    "e2_uncapped_cells": {
        "source": f"{SOURCE_APRIL} (April clean transcripts) or {SOURCE_E2} (E2 transcripts)",
        "condition": f"{SS} or {MTP}",
        "judge_arm": "opus or gpt",
        "judge_model": "judge id requested",
        "judge_response_model": "judge id reported by the API (semicolon list); 'not recorded' for the April GPT-5.4 arm",
        "judge_date_utc": "UTC dates of the judge call timestamps (semicolon list)",
        "n_runs": "runs in the cell",
        "n_briefs": "distinct briefs in the cell",
        "repetitions": "runs per brief",
        "max_tokens": "subject max_tokens (April 4096, E2 16384)",
        "cap_rule": "cap rule applied to this source",
        "n_turns": "non-probe assistant turns in the cell",
        "output_tokens_per_run_mean": "mean metadata.total_output_tokens per run",
        "input_tokens_per_run_mean": "mean metadata.total_input_tokens per run",
        "cost_usd_per_run_mean": "mean metadata.total_cost_usd per run (litellm cost field)",
        "turn_tokens_mean": "mean token_count over non-probe assistant turns",
        "turn_tokens_median": "median token_count over non-probe assistant turns",
        "turn_tokens_max": "max token_count over non-probe assistant turns",
        "turn_tokens_by_turn": "mean token_count by turn position (semicolon list)",
        "cap_turns": "turns meeting the cap rule",
        "cap_turn_share": "cap_turns / n_turns",
        "cap_turn_share_by_turn": "share of turns meeting the cap rule, by turn position (semicolon list)",
        "runs_any_cap": "runs with at least one turn meeting the cap rule",
        "runs_any_cap_share": "runs_any_cap / n_runs",
        "final_cap": "runs whose final turn meets the cap rule",
        "final_cap_share": "final_cap / n_runs",
        "final_tokens_mean": "mean token_count of the final non-probe assistant turn",
        "per_run_max_median": "median over runs of the run's largest turn token_count",
        "per_run_max_min": "min over runs of the run's largest turn token_count",
        "per_run_max_max": "max over runs of the run's largest turn token_count",
        "probe_tokens_max": "max token_count over probe answers (NaN in single_shot)",
        **_dim_cols(),
        "adherence_lt4": f"runs with constraint_adherence < {ADHERENCE_THRESHOLD}",
        "adherence_lt3": "runs with constraint_adherence < 3",
        "adherence_eq4": "runs with constraint_adherence == 4",
        "n_with_final_probe": f"{MTP} rows: runs with a final probe (KBV denominator); NaN in {SS}",
        "knows": f"{MTP} rows: runs with final-probe overall_accuracy >= {PROBE_ACCURACY_THRESHOLD}",
        "violates": f"{MTP} rows: runs with constraint_adherence < {ADHERENCE_THRESHOLD} among runs with a final probe",
        "kbv_count": f"{MTP} rows: knows and violates",
        "kbv_rate": "kbv_count / n_with_final_probe",
        "kbv_ci_low": f"lower bound of kbv_rate ({CI_WILSON})",
        "kbv_ci_high": f"upper bound of kbv_rate ({CI_WILSON})",
    },
    "e2_uncapped_deltas": {
        "condition": f"{SS} or {MTP}",
        "judge_arm": "opus or gpt (token-metric rows repeat across arms)",
        "metric": ", ".join(METRICS) + f" (kbv_rate on {MTP} only)",
        "design": f"{DESIGN_UNPAIRED} or {DESIGN_PAIRED}",
        "n_e2": "E2 runs with a value",
        "n_april": "April runs with a value",
        "n_briefs": "briefs (paired units on brief_paired rows)",
        "e2_value": "E2 cell mean (or rate)",
        "april_value": "April cell mean (or rate)",
        "delta": "E2 minus April (brief_paired: mean of the per-brief deltas)",
        "ci_low": "lower 95% bound of delta (ci_method)",
        "ci_high": "upper 95% bound of delta (ci_method)",
        "ci_method": "Newcombe (unpaired proportions), Welch (unpaired means) or paired t (brief_paired)",
        "boot_ci_low": f"brief_paired rows: lower 95% bound of delta ({CI_BRIEF_BOOT}); NaN on unpaired rows",
        "boot_ci_high": f"brief_paired rows: upper 95% bound of delta ({CI_BRIEF_BOOT}); NaN on unpaired rows",
        "p_value": "two-sided p of the test in `test`",
        "test": "Fisher exact (unpaired proportions), Mann-Whitney U (unpaired means) or Wilcoxon signed-rank (brief_paired)",
        "briefs_up": "brief_paired rows: briefs with delta > 0",
        "briefs_down": "brief_paired rows: briefs with delta < 0",
        "briefs_tie": "brief_paired rows: briefs with delta == 0",
        "ci_excludes_zero": "True when [ci_low, ci_high] excludes 0; False when a bound is NaN",
        "judge_comparability": "judge and date comparability of the two sides (per arm); n/a on token metrics",
    },
    "e2_uncapped_offset": {
        "scope": f"{SCOPE_OVERALL}, {SS} or {MTP}",
        "dimension": "scored dimension",
        "n_runs": "paired E2 runs (same transcript scored by both judges)",
        "n_briefs": "distinct briefs",
        "e2_offset": "mean per-run (Opus minus GPT-5.4) on the E2 transcripts",
        "e2_ci_low": f"lower 95% bound of e2_offset ({CI_BRIEF_BOOT})",
        "e2_ci_high": f"upper 95% bound of e2_offset ({CI_BRIEF_BOOT})",
        "e1_offset": "E1 Sonnet offset from judge_offset.csv (block offset_by_model, ss+mtp); NaN where no row exists",
        "e1_ci_low": "E1 lower bound from judge_offset.csv",
        "e1_ci_high": "E1 upper bound from judge_offset.csv",
        "e1_n_runs": "E1 runs behind e1_offset",
        "e1_source": "judge_offset.csv row used, or why none",
    },
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def cap_rule_april(token_count: int) -> bool:
    return int(token_count) == APRIL_CAP


def cap_rule_e2(token_count: int) -> bool:
    return int(token_count) > APRIL_CAP


def summarize_run(transcript: dict, cap_rule: Callable[[int], bool]) -> dict:
    """Per-run token facts of one transcript under ``cap_rule``.

    Only non-probe assistant turns enter the turn statistics. Probe answers
    give ``probe_tokens_max``.
    """
    md = transcript.get("metadata", {})
    messages = transcript.get("messages", [])
    assistant = [m for m in messages if m.get("role") == "assistant" and not m.get("is_probe")]
    probes = [m for m in messages if m.get("role") == "assistant" and m.get("is_probe")]
    tokens = [int(m.get("token_count") or 0) for m in assistant]
    caps = [bool(cap_rule(t)) for t in tokens]
    probe_tokens = [int(m.get("token_count") or 0) for m in probes]
    return {
        "run_id": md.get("run_id"),
        "brief_id": md.get("brief_id"),
        "model_id": md.get("model_id"),
        "condition": md.get("condition"),
        "repetition": md.get("repetition"),
        "error": md.get("error"),
        "max_tokens": md.get("max_tokens"),
        "finish_reasons": [m.get("finish_reason") for m in assistant + probes],
        "output_tokens": md.get("total_output_tokens"),
        "input_tokens": md.get("total_input_tokens"),
        "cost_usd": md.get("total_cost_usd"),
        "n_turns": len(tokens),
        "turn_tokens": tokens,
        "cap_flags": caps,
        "cap_turns": int(sum(caps)),
        "cap_turn_share": (sum(caps) / len(caps)) if caps else math.nan,
        "any_cap": bool(any(caps)),
        "final_tokens": tokens[-1] if tokens else math.nan,
        "final_cap": bool(caps[-1]) if caps else False,
        "run_max": max(tokens) if tokens else math.nan,
        "probe_tokens_max": max(probe_tokens) if probe_tokens else math.nan,
    }


def _check_balanced(runs: pd.DataFrame, n_briefs: int, what: str) -> int:
    """Every brief appears the same number of times in every cell. Returns that count."""
    if sorted(runs["condition"].unique()) != sorted(CELLS):
        raise ValueError(f"{what}: conditions {sorted(runs['condition'].unique())} != {sorted(CELLS)}")
    reps: set[int] = set()
    for cond, g in runs.groupby("condition"):
        counts = g.groupby("brief_id").size()
        if len(counts) != n_briefs:
            raise ValueError(f"{what}: {cond} has {len(counts)} briefs, expected {n_briefs}")
        if counts.nunique() != 1:
            raise ValueError(f"{what}: {cond} is unbalanced over briefs: {counts[counts != counts.iloc[0]].to_dict()}")
        reps.add(int(counts.iloc[0]))
    if len(reps) != 1:
        raise ValueError(f"{what}: cells differ in runs per brief: {sorted(reps)}")
    return reps.pop()


def load_e2_transcripts(e2_dir: Path, n_briefs: int = N_BRIEFS) -> pd.DataFrame:
    """One row per E2 transcript, with the E2 cap rule.

    Guards (ValueError): a transcript with ``metadata.error`` set,
    ``max_tokens`` != 16384, a subject other than Sonnet, a condition outside
    the two cells, an assistant message whose ``finish_reason`` is not
    ``stop``; not exactly one run per brief per cell.
    """
    tdir = e2_dir / "transcripts"
    if not tdir.is_dir():
        raise FileNotFoundError(f"E2 transcripts not found: {tdir}")
    rows = []
    for path in sorted(tdir.glob("*.jsonl")):
        row = summarize_run(read_jsonl_first(path), cap_rule_e2)
        row["source_file"] = path.name
        if row["error"] is not None:
            raise ValueError(f"{path.name}: metadata.error is set: {row['error']!r}")
        if row["max_tokens"] != E2_CAP:
            raise ValueError(f"{path.name}: max_tokens {row['max_tokens']!r} != {E2_CAP}")
        if row["model_id"] != SONNET:
            raise ValueError(f"{path.name}: model_id {row['model_id']!r} != {SONNET}")
        if row["condition"] not in CELLS:
            raise ValueError(f"{path.name}: condition {row['condition']!r} not in {CELLS}")
        bad = [fr for fr in row["finish_reasons"] if fr != "stop"]
        if bad:
            raise ValueError(f"{path.name}: {len(bad)} assistant messages with finish_reason != stop, e.g. {bad[:3]}")
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"no E2 transcripts under {tdir}")
    if df["run_id"].duplicated().any():
        raise ValueError("duplicate run ids among the E2 transcripts")
    reps = _check_balanced(df, n_briefs, "E2 transcripts")
    if reps != 1:
        raise ValueError(f"E2 transcripts: {reps} runs per brief per cell, expected 1")
    return df.sort_values(["condition", "brief_id"]).reset_index(drop=True)


def load_april_transcripts(data_root: Path, n_briefs: int = N_BRIEFS) -> tuple[pd.DataFrame, dict]:
    """Clean April Sonnet transcripts of the two cells, with the April cap rule.

    Returns the frame and the ``drop_stray`` provenance record. Guards
    (ValueError): a transcript with ``metadata.error`` set; the clean sample
    not balanced over briefs within each cell.
    """
    tdir = data_root / "transcripts"
    rows = []
    for path in sorted(tdir.glob("*.jsonl")):
        t = read_jsonl_first(path)
        md = t.get("metadata", {})
        if md.get("model_id") != SONNET or md.get("condition") not in CELLS:
            continue
        row = summarize_run(t, cap_rule_april)
        row["source_file"] = path.name
        if row["error"] is not None:
            raise ValueError(f"{path.name}: metadata.error is set: {row['error']!r}")
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"no April Sonnet transcripts in the two cells under {tdir}")
    df, info = drop_stray(df, data_root)
    _check_balanced(df, n_briefs, "April clean transcripts")
    return df.sort_values(["condition", "brief_id", "repetition"]).reset_index(drop=True), info


def _score_frame(records: list[dict], what: str, expected_judge: str) -> pd.DataFrame:
    cols = ["run_id", "judge_model", "judge_response_model", "timestamp", "judge_cost_usd", *DIMS]
    rows = []
    for rec in records:
        if rec.get("judge_model") != expected_judge:
            raise ValueError(f"{what}: judge_model {rec.get('judge_model')!r} != {expected_judge!r} on {rec.get('run_id')}")
        if bool(rec.get("cost_estimated", False)):
            raise ValueError(f"{what}: cost_estimated is true on {rec.get('run_id')}")
        for d in DIMS:
            if rec.get(d) is None:
                raise ValueError(f"{what}: missing score {d} on {rec.get('run_id')}")
        row = {c: rec.get(c) for c in cols if c != "judge_cost_usd"}
        row["judge_cost_usd"] = rec.get("cost_usd")  # the subject cost keeps the name cost_usd in the runs frame
        rows.append(row)
    df = pd.DataFrame(rows, columns=cols)
    for d in DIMS:
        df[d] = pd.to_numeric(df[d])
    return df


def load_e2_scores(e2_dir: Path, arm: str, run_ids: list[str]) -> pd.DataFrame:
    """E2 score files of one arm. Ids must equal ``run_ids`` exactly."""
    spec = ARMS[arm]
    sdir = e2_dir / spec["e2_subdir"]
    if not sdir.is_dir():
        raise FileNotFoundError(f"E2 {arm} scores not found: {sdir}")
    recs = [read_jsonl_first(p) for p in sorted(sdir.glob(f"{spec['e2_prefix']}_*.jsonl"))]
    df = _score_frame(recs, f"E2 {arm} scores", spec["judge_model"])
    if df["run_id"].duplicated().any():
        raise ValueError(f"E2 {arm} scores: duplicate run ids")
    if set(df["run_id"]) != set(run_ids):
        missing = sorted(set(run_ids) - set(df["run_id"]))
        extra = sorted(set(df["run_id"]) - set(run_ids))
        raise ValueError(f"E2 {arm} scores: ids differ from the transcripts (missing {missing[:5]}, extra {extra[:5]})")
    return df


def load_april_scores(data_root: Path, arm: str, run_ids: list[str]) -> pd.DataFrame:
    """April scores of one arm for ``run_ids``.

    ``gpt``: the published ``j_*`` columns of ``all_scores.parquet``
    (judge must be GPT-5.4); the judge timestamp comes from
    ``scores/judge_<run_id>.jsonl``. ``opus``: the E1 crossed files
    ``camera_ready/e1_crossed_judge/opus/judge_opus_<run_id>.jsonl``.
    """
    spec = ARMS[arm]
    if arm == "gpt":
        pq = pd.read_parquet(data_root / "aggregated" / "all_scores.parquet")
        sub = pq[pq["run_id"].isin(run_ids)].copy()
        missing = sorted(set(run_ids) - set(sub["run_id"]))
        if missing:
            raise ValueError(f"{len(missing)} April runs have no parquet row, e.g. {missing[:5]}")
        if sub["run_id"].duplicated().any():
            raise ValueError("duplicate run ids in all_scores.parquet for the April runs")
        bad = sub[sub["judge_model"] != spec["judge_model"]]
        if len(bad):
            raise ValueError(f"{len(bad)} April parquet rows judged by {sorted(bad['judge_model'].unique())}, not {spec['judge_model']}")
        recs = []
        for _, r in sub.iterrows():
            ts = read_jsonl_first(data_root / "scores" / f"judge_{r['run_id']}.jsonl").get("timestamp")
            recs.append(
                {
                    "run_id": r["run_id"],
                    "judge_model": r["judge_model"],
                    "judge_response_model": None,
                    "timestamp": ts,
                    "cost_usd": None,
                    **{d: r[f"j_{d}"] for d in DIMS},
                }
            )
        return _score_frame(recs, "April gpt scores (parquet)", spec["judge_model"])
    e1 = data_root / E1_OPUS_DIR
    recs = []
    for rid in run_ids:
        path = e1 / f"judge_opus_{rid}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"E1 Opus file missing for April run {rid}: {path}")
        recs.append(read_jsonl_first(path))
    return _score_frame(recs, "April opus scores (E1)", spec["judge_model"])


def load_april_probes(data_root: Path, run_ids: list[str]) -> pd.DataFrame:
    """``run_id, overall_accuracy`` of the April final probes for ``run_ids``."""
    fp = pd.read_csv(data_root / "analysis" / "final_probe_accuracy.csv")
    sub = fp[fp["run_id"].isin(run_ids)][["run_id", "overall_accuracy"]].drop_duplicates("run_id")
    return sub.reset_index(drop=True)


def compute_e2_probes(e2_dir: Path, briefs_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Recompute the E2 final probes with the keyword scorer; check the CSV in the E2 tree.

    Returns ``run_id, overall_accuracy`` and a note. ValueError when the E2
    CSV exists and disagrees on any run.
    """
    probe_df = analyze_all_probes(e2_dir / "transcripts", briefs_dir)
    final = compute_final_probe_accuracy(probe_df) if not probe_df.empty else pd.DataFrame(columns=["run_id", "overall_accuracy"])
    final = final[["run_id", "overall_accuracy"]].reset_index(drop=True)
    csv_path = e2_dir / E2_PROBE_CSV
    if csv_path.exists():
        on_disk = pd.read_csv(csv_path)[["run_id", "overall_accuracy"]]
        m = final.merge(on_disk, on="run_id", how="outer", suffixes=("", "_disk"), indicator=True)
        off = m[(m["_merge"] != "both") | ~np.isclose(m["overall_accuracy"], m["overall_accuracy_disk"])]
        if len(off):
            raise ValueError(f"E2 probes recomputed differ from {csv_path.name} on {len(off)} runs, e.g. {off['run_id'].head(3).tolist()}")
        note = {"checked_against": str(E2_DIR / E2_PROBE_CSV), "n_runs": int(len(final)), "equal": True}
    else:
        note = {"checked_against": None, "n_runs": int(len(final)), "equal": None}
    return final, note


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def brief_bootstrap_mean_ci(values, briefs, b: int = B, seed: int = BOOT_SEED) -> tuple[float, float]:
    """Percentile CI of the mean of ``values`` with briefs resampled with replacement.

    NaN values are dropped with their brief label. NaN bounds when nothing is left.
    """
    v = np.asarray(values, dtype=float)
    br = np.asarray(briefs)
    keep = ~np.isnan(v)
    v, br = v[keep], br[keep]
    if len(v) == 0:
        return (math.nan, math.nan)
    ub, inv = np.unique(br, return_inverse=True)
    sums = np.bincount(inv, weights=v, minlength=len(ub))
    counts = np.bincount(inv, minlength=len(ub)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ub), size=(b, len(ub)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def newcombe_ci(k1: int, n1: int, k2: int, n2: int, z: float = Z95) -> tuple[float, float]:
    """Newcombe hybrid score interval for ``k1/n1 - k2/n2`` (independent samples)."""
    if n1 == 0 or n2 == 0:
        return (math.nan, math.nan)
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson_ci(k1, n1, z)
    l2, u2 = wilson_ci(k2, n2, z)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (lo, hi)


def fisher_p(k1: int, n1: int, k2: int, n2: int) -> float:
    if n1 == 0 or n2 == 0:
        return math.nan
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def welch_ci(a, b, level: float = 0.95) -> tuple[float, float, float]:
    """(mean_a - mean_b, ci_low, ci_high) with Welch df. NaN bounds when a side has n < 2 or both variances are 0."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return (math.nan, math.nan, math.nan)
    d = float(x.mean() - y.mean())
    if len(x) < 2 or len(y) < 2:
        return (d, math.nan, math.nan)
    vx, vy = float(x.var(ddof=1)), float(y.var(ddof=1))
    se2 = vx / len(x) + vy / len(y)
    if se2 == 0:
        return (d, math.nan, math.nan)
    df = se2**2 / ((vx / len(x)) ** 2 / (len(x) - 1) + (vy / len(y)) ** 2 / (len(y) - 1))
    q = float(stats.t.ppf(0.5 + level / 2, df))
    se = math.sqrt(se2)
    return (d, d - q * se, d + q * se)


def mannwhitney_p(a, b) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0 or np.ptp(np.concatenate([x, y])) == 0:
        return math.nan
    return float(stats.mannwhitneyu(x, y, alternative="two-sided").pvalue)


def paired_t_ci(d, level: float = 0.95) -> tuple[float, float, float]:
    """(mean, ci_low, ci_high) of the deltas with a t interval, df = n - 1."""
    x = np.asarray(d, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return (math.nan, math.nan, math.nan)
    mean = float(x.mean())
    if n < 2:
        return (mean, math.nan, math.nan)
    se = float(x.std(ddof=1) / math.sqrt(n))
    q = float(stats.t.ppf(0.5 + level / 2, n - 1))
    return (mean, mean - q * se, mean + q * se)


def wilcoxon_p(d) -> float:
    """Two-sided Wilcoxon signed-rank p of the deltas; nan when every delta is 0 or n == 0."""
    x = np.asarray(d, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0 or np.all(x == 0):
        return math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(stats.wilcoxon(x, zero_method="wilcox", method="approx").pvalue)


def ci_excludes_zero(lo: float, hi: float) -> bool:
    if lo is None or hi is None or math.isnan(lo) or math.isnan(hi):
        return False
    return bool(lo > 0 or hi < 0)


def brief_paired_deltas(e2: pd.DataFrame, april: pd.DataFrame, value: str) -> pd.DataFrame:
    """Per brief: the E2 value minus the April mean over that brief's repetitions.

    Both frames need ``brief_id`` and ``value``. Briefs lacking a value on
    either side are dropped. Returns ``brief_id, e2, april, delta``.
    """
    a = april[["brief_id", value]].dropna().groupby("brief_id")[value].mean().rename("april")
    e = e2[["brief_id", value]].dropna()
    if e["brief_id"].duplicated().any():
        raise ValueError("brief_paired_deltas: more than one E2 run per brief")
    e = e.set_index("brief_id")[value].rename("e2")
    out = pd.concat([e, a], axis=1, join="inner").reset_index()
    out["delta"] = out["e2"] - out["april"]
    return out.sort_values("brief_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _utc_dates(ts: pd.Series) -> str:
    vals = pd.to_numeric(ts, errors="coerce").dropna()
    dates = sorted({dt.datetime.fromtimestamp(float(v), dt.timezone.utc).date().isoformat() for v in vals})
    return ";".join(dates)


def _per_run_values(runs: pd.DataFrame, scores: pd.DataFrame, probes: pd.DataFrame) -> pd.DataFrame:
    """One row per run: token facts, the arm's scores, the final probe, and the KBV flags."""
    df = runs.merge(scores, on="run_id", how="left", validate="one_to_one")
    if df[DIMS].isna().any().any():
        raise ValueError("a run has no score in this arm")
    df = df.merge(probes.rename(columns={"overall_accuracy": "probe_accuracy"}), on="run_id", how="left")
    df["has_probe"] = df["probe_accuracy"].notna()
    df["violates"] = df["constraint_adherence"] < ADHERENCE_THRESHOLD
    df["knows"] = (df["probe_accuracy"] >= PROBE_ACCURACY_THRESHOLD).where(df["has_probe"])
    df["kbv"] = (df["violates"] & (df["probe_accuracy"] >= PROBE_ACCURACY_THRESHOLD)).where(df["has_probe"])
    df["kbv_rate"] = df["kbv"].astype(float)
    df["output_tokens_per_run"] = pd.to_numeric(df["output_tokens"], errors="coerce")
    df["final_cap_share"] = df["final_cap"].astype(float)
    return df


def _by_turn(list_col: pd.Series, fmt: str) -> str:
    lists = [list(x) for x in list_col]
    if not lists:
        return ""
    width = max(len(x) for x in lists)
    out = []
    for i in range(width):
        vals = [float(x[i]) for x in lists if len(x) > i]
        out.append(fmt.format(np.mean(vals)))
    return ";".join(out)


def cell_row(df: pd.DataFrame, source: str, condition: str, arm: str, cap_rule: str, max_tokens: int) -> dict:
    """One cells-table row from the per-run frame of one (source, condition, arm)."""
    turn_tokens = np.array([t for x in df["turn_tokens"] for t in x], dtype=float)
    n_runs = int(len(df))
    row: dict = {
        "source": source,
        "condition": condition,
        "judge_arm": arm,
        "judge_model": ARMS[arm]["judge_model"],
        "judge_response_model": ";".join(sorted(df["judge_response_model"].dropna().astype(str).unique())) or "not recorded",
        "judge_date_utc": _utc_dates(df["timestamp"]),
        "n_runs": n_runs,
        "n_briefs": int(df["brief_id"].nunique()),
        "repetitions": n_runs / df["brief_id"].nunique() if n_runs else math.nan,
        "max_tokens": max_tokens,
        "cap_rule": cap_rule,
        "n_turns": int(len(turn_tokens)),
        "output_tokens_per_run_mean": float(pd.to_numeric(df["output_tokens"]).mean()),
        "input_tokens_per_run_mean": float(pd.to_numeric(df["input_tokens"]).mean()),
        "cost_usd_per_run_mean": float(pd.to_numeric(df["cost_usd"]).mean()),
        "turn_tokens_mean": float(turn_tokens.mean()) if len(turn_tokens) else math.nan,
        "turn_tokens_median": float(np.median(turn_tokens)) if len(turn_tokens) else math.nan,
        "turn_tokens_max": float(turn_tokens.max()) if len(turn_tokens) else math.nan,
        "turn_tokens_by_turn": _by_turn(df["turn_tokens"], "{:.1f}"),
        "cap_turns": int(df["cap_turns"].sum()),
        "cap_turn_share": float(df["cap_turns"].sum() / len(turn_tokens)) if len(turn_tokens) else math.nan,
        "cap_turn_share_by_turn": _by_turn(df["cap_flags"].apply(lambda x: [float(v) for v in x]), "{:.3f}"),
        "runs_any_cap": int(df["any_cap"].sum()),
        "runs_any_cap_share": float(df["any_cap"].mean()),
        "final_cap": int(df["final_cap"].sum()),
        "final_cap_share": float(df["final_cap"].mean()),
        "final_tokens_mean": float(pd.to_numeric(df["final_tokens"]).mean()),
        "per_run_max_median": float(pd.to_numeric(df["run_max"]).median()),
        "per_run_max_min": float(pd.to_numeric(df["run_max"]).min()),
        "per_run_max_max": float(pd.to_numeric(df["run_max"]).max()),
        "probe_tokens_max": float(pd.to_numeric(df["probe_tokens_max"]).max()) if df["probe_tokens_max"].notna().any() else math.nan,
    }
    for d in DIMS:
        lo, hi = brief_bootstrap_mean_ci(df[d], df["brief_id"])
        row[f"{d}_mean"] = float(df[d].mean())
        row[f"{d}_sd"] = float(df[d].std(ddof=1)) if n_runs > 1 else math.nan
        row[f"{d}_ci_low"] = lo
        row[f"{d}_ci_high"] = hi
    row["adherence_lt4"] = int((df["constraint_adherence"] < 4).sum())
    row["adherence_lt3"] = int((df["constraint_adherence"] < 3).sum())
    row["adherence_eq4"] = int((df["constraint_adherence"] == 4).sum())
    if condition == MTP:
        wp = df[df["has_probe"]]
        n = int(len(wp))
        k = int(wp["kbv"].astype(bool).sum()) if n else 0
        lo, hi = wilson_ci(k, n)
        row.update(
            {
                "n_with_final_probe": n,
                "knows": int(wp["knows"].astype(bool).sum()) if n else 0,
                "violates": int(wp["violates"].sum()) if n else 0,
                "kbv_count": k,
                "kbv_rate": k / n if n else math.nan,
                "kbv_ci_low": lo,
                "kbv_ci_high": hi,
            }
        )
    else:
        row.update({c: math.nan for c in ("n_with_final_probe", "knows", "violates", "kbv_count", "kbv_rate", "kbv_ci_low", "kbv_ci_high")})
    return row


def delta_rows(e2: pd.DataFrame, april: pd.DataFrame, condition: str, arm: str, comparability: dict[str, str]) -> list[dict]:
    """Unpaired and brief-paired rows for every metric of one (condition, arm)."""
    rows = []
    metrics = [m for m in METRICS if not (m == "kbv_rate" and condition != MTP)]
    n_briefs = int(e2["brief_id"].nunique())
    for metric in metrics:
        comp = comparability["none"] if metric in TOKEN_METRICS else comparability[arm]
        x = e2[metric].astype(float)
        y = april[metric].astype(float)
        xv, yv = x.dropna(), y.dropna()
        base = {
            "condition": condition,
            "judge_arm": arm,
            "metric": metric,
            "n_e2": int(len(xv)),
            "n_april": int(len(yv)),
            "n_briefs": n_briefs,
            "e2_value": float(xv.mean()) if len(xv) else math.nan,
            "april_value": float(yv.mean()) if len(yv) else math.nan,
            "judge_comparability": comp,
        }
        # unpaired runs
        if metric in PROPORTION_METRICS:
            k1, n1 = int(xv.sum()), int(len(xv))
            k2, n2 = int(yv.sum()), int(len(yv))
            lo, hi = newcombe_ci(k1, n1, k2, n2)
            d = (k1 / n1 - k2 / n2) if n1 and n2 else math.nan
            p, ci_method, test = fisher_p(k1, n1, k2, n2), CI_NEWCOMBE, TEST_FISHER
        else:
            d, lo, hi = welch_ci(xv, yv)
            p, ci_method, test = mannwhitney_p(xv, yv), CI_WELCH, TEST_MWU
        rows.append(
            {
                **base,
                "design": DESIGN_UNPAIRED,
                "delta": d,
                "ci_low": lo,
                "ci_high": hi,
                "ci_method": ci_method,
                "boot_ci_low": math.nan,
                "boot_ci_high": math.nan,
                "p_value": p,
                "test": test,
                "briefs_up": math.nan,
                "briefs_down": math.nan,
                "briefs_tie": math.nan,
                "ci_excludes_zero": ci_excludes_zero(lo, hi),
            }
        )
        # brief paired
        pr = brief_paired_deltas(e2, april, metric)
        mean, lo, hi = paired_t_ci(pr["delta"])
        blo, bhi = brief_bootstrap_mean_ci(pr["delta"], pr["brief_id"])
        rows.append(
            {
                **base,
                "n_briefs": int(len(pr)),
                "design": DESIGN_PAIRED,
                "delta": mean,
                "ci_low": lo,
                "ci_high": hi,
                "ci_method": CI_PAIRED_T,
                "boot_ci_low": blo,
                "boot_ci_high": bhi,
                "p_value": wilcoxon_p(pr["delta"]),
                "test": TEST_WILCOXON,
                "briefs_up": int((pr["delta"] > 0).sum()),
                "briefs_down": int((pr["delta"] < 0).sum()),
                "briefs_tie": int((pr["delta"] == 0).sum()),
                "ci_excludes_zero": ci_excludes_zero(lo, hi),
            }
        )
    return rows


def offset_rows(e2_opus: pd.DataFrame, e2_gpt: pd.DataFrame, runs: pd.DataFrame, e1_offsets: pd.DataFrame | None) -> list[dict]:
    """Opus minus GPT-5.4 per dimension on the paired E2 runs, overall and per cell."""
    pair = e2_opus[["run_id", *DIMS]].merge(e2_gpt[["run_id", *DIMS]], on="run_id", suffixes=("_opus", "_gpt"), validate="one_to_one")
    pair = pair.merge(runs[["run_id", "brief_id", "condition"]], on="run_id", validate="one_to_one")
    rows = []
    for scope in [SCOPE_OVERALL, *CELLS]:
        g = pair if scope == SCOPE_OVERALL else pair[pair["condition"] == scope]
        for d in DIMS:
            diff = g[f"{d}_opus"] - g[f"{d}_gpt"]
            lo, hi = brief_bootstrap_mean_ci(diff, g["brief_id"])
            row = {
                "scope": scope,
                "dimension": d,
                "n_runs": int(len(g)),
                "n_briefs": int(g["brief_id"].nunique()),
                "e2_offset": float(diff.mean()) if len(g) else math.nan,
                "e2_ci_low": lo,
                "e2_ci_high": hi,
                "e1_offset": math.nan,
                "e1_ci_low": math.nan,
                "e1_ci_high": math.nan,
                "e1_n_runs": math.nan,
                "e1_source": "no judge_offset.csv",
            }
            if e1_offsets is not None:
                e1 = e1_offsets[
                    (e1_offsets["block"] == "offset_by_model")
                    & (e1_offsets["model_id"] == SONNET)
                    & (e1_offsets["dimension"] == d)
                    & (e1_offsets["judge"] == "opus_minus_gpt")
                ]
                if scope == SCOPE_OVERALL and len(e1) == 1:
                    r = e1.iloc[0]
                    row.update(
                        {
                            "e1_offset": float(r["estimate"]),
                            "e1_ci_low": float(r["ci_low"]),
                            "e1_ci_high": float(r["ci_high"]),
                            "e1_n_runs": int(r["n_runs"]),
                            "e1_source": f"judge_offset.csv block offset_by_model, model_id {SONNET}, condition {r['condition']}",
                        }
                    )
                elif scope == SCOPE_OVERALL:
                    row["e1_source"] = f"judge_offset.csv has {len(e1)} offset_by_model rows for {SONNET}/{d}, expected 1"
                else:
                    row["e1_source"] = "judge_offset.csv has no per-cell Sonnet row (offset_by_model pools ss+mtp)"
            rows.append(row)
    return rows


def judge_comparability(cells: pd.DataFrame, drift: pd.DataFrame | None) -> dict[str, str]:
    """Per-arm text on judge and date comparability, from the cells table and the drift control CSV."""
    def _dates(source: str, arm: str) -> str:
        sub = cells[(cells["source"] == source) & (cells["judge_arm"] == arm)]
        return ";".join(sorted({d for s in sub["judge_date_utc"] for d in str(s).split(";") if d}))

    def _resp(source: str, arm: str) -> str:
        sub = cells[(cells["source"] == source) & (cells["judge_arm"] == arm)]
        return ";".join(sorted({d for s in sub["judge_response_model"] for d in str(s).split(";") if d}))

    out = {
        "opus": (
            f"same judge pass on both sides: April Opus scores are the E1 crossed re-score judged {_dates(SOURCE_APRIL, 'opus')} "
            f"({_resp(SOURCE_APRIL, 'opus')}); E2 judged {_dates(SOURCE_E2, 'opus')} ({_resp(SOURCE_E2, 'opus')}); no judge-date gap"
        ),
        "none": "n/a: token metric, no judge involved",
    }
    control = "judge_drift_control.csv not found"
    if drift is not None:
        r = drift[(drift["control_judge"] == "gpt") & (drift["dimension"] == "constraint_adherence")]
        if len(r) == 1:
            r = r.iloc[0]
            control = (
                f"control (judge_drift_control.csv): GPT-5.4 re-scored n = {int(r['n'])} Sonnet runs on 2026-10-26, "
                f"constraint_adherence today minus April {float(r['mean_diff_today_minus_april']):+.2f}, exact agreement {float(r['exact_agreement']):.2f}"
            )
        else:
            control = f"judge_drift_control.csv has {len(r)} gpt/constraint_adherence rows, expected 1"
    out["gpt"] = (
        f"judge-date gap: April GPT-5.4 scores judged {_dates(SOURCE_APRIL, 'gpt')} (judge_response_model not recorded); "
        f"E2 judged {_dates(SOURCE_E2, 'gpt')} ({_resp(SOURCE_E2, 'gpt')}); {control}"
    )
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _display_path(path: Path, data_root: Path) -> str:
    """``path`` relative to ``data_root`` when it lies inside it (machine-independent provenance)."""
    try:
        return str(Path(path).resolve().relative_to(Path(data_root).resolve()))
    except ValueError:
        return str(path)


def run(
    data_root: Path = DEFAULT_DATA_ROOT,
    out_dir: Path = DEFAULT_OUT_DIR,
    briefs_dir: Path | None = None,
    n_briefs: int = N_BRIEFS,
    comparators_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, pd.DataFrame]:
    """``comparators_dir`` holds ``judge_offset.csv`` (E1 Sonnet offset) and
    ``judge_drift_control.csv`` (GPT-5.4 judge drift control). It defaults to the
    analysis directory so a non-default ``out_dir`` still finds them; a missing
    file warns and leaves the E1 columns NaN and the control text explicit."""
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    comparators_dir = Path(comparators_dir)
    briefs_dir = Path(briefs_dir) if briefs_dir is not None else data_root.parent / "briefs"
    e2_dir = data_root / E2_DIR

    e2_runs = load_e2_transcripts(e2_dir, n_briefs)
    e2_ids = e2_runs["run_id"].tolist()
    april_runs, stray_info = load_april_transcripts(data_root, n_briefs)
    april_ids = april_runs["run_id"].tolist()

    e2_scores = {arm: load_e2_scores(e2_dir, arm, e2_ids) for arm in ARM_ORDER}
    april_scores = {arm: load_april_scores(data_root, arm, april_ids) for arm in ARM_ORDER}
    e2_probes, e2_probe_note = compute_e2_probes(e2_dir, briefs_dir)
    april_probes = load_april_probes(data_root, april_runs.loc[april_runs["condition"] == MTP, "run_id"].tolist())

    per_run: dict[tuple[str, str], pd.DataFrame] = {}
    for arm in ARM_ORDER:
        per_run[(SOURCE_APRIL, arm)] = _per_run_values(april_runs, april_scores[arm], april_probes)
        per_run[(SOURCE_E2, arm)] = _per_run_values(e2_runs, e2_scores[arm], e2_probes)

    cell_rows = []
    for source in SOURCES:
        cap_rule, max_tokens = (CAP_RULE_APRIL, APRIL_CAP) if source == SOURCE_APRIL else (CAP_RULE_E2, E2_CAP)
        for condition in CELLS:
            for arm in ARM_ORDER:
                df = per_run[(source, arm)]
                cell_rows.append(cell_row(df[df["condition"] == condition], source, condition, arm, cap_rule, max_tokens))
    cells = pd.DataFrame(cell_rows, columns=list(COLUMNS["e2_uncapped_cells"]))

    drift_path = comparators_dir / JUDGE_DRIFT_CSV
    if drift_path.exists():
        drift = pd.read_csv(drift_path)
    else:
        drift = None
        warnings.warn(f"{drift_path} not found; judge_comparability will say so", stacklevel=2)
    comparability = judge_comparability(cells, drift)
    rows = []
    for condition in CELLS:
        for arm in ARM_ORDER:
            e2 = per_run[(SOURCE_E2, arm)]
            ap = per_run[(SOURCE_APRIL, arm)]
            rows.extend(delta_rows(e2[e2["condition"] == condition], ap[ap["condition"] == condition], condition, arm, comparability))
    deltas = pd.DataFrame(rows, columns=list(COLUMNS["e2_uncapped_deltas"]))

    offset_path = comparators_dir / JUDGE_OFFSET_CSV
    if offset_path.exists():
        e1_offsets = pd.read_csv(offset_path)
    else:
        e1_offsets = None
        warnings.warn(f"{offset_path} not found; e1_* columns will be NaN", stacklevel=2)
    offset = pd.DataFrame(offset_rows(e2_scores["opus"], e2_scores["gpt"], e2_runs, e1_offsets), columns=list(COLUMNS["e2_uncapped_offset"]))

    judge_dates = {
        f"{source}/{arm}": cells.loc[(cells["source"] == source) & (cells["judge_arm"] == arm), "judge_date_utc"].iloc[0]
        for source in SOURCES
        for arm in ARM_ORDER
    }
    inputs = [
        f"{E2_DIR}/transcripts/*.jsonl (E2 subject runs)",
        f"{E2_DIR}/scores_opus/judge_opus_*.jsonl and {E2_DIR}/scores_gpt/judge_gpt_*.jsonl (E2 judge arms)",
        f"{E2_DIR}/{E2_PROBE_CSV} (checked against the recomputed E2 probes)",
        f"transcripts/*.jsonl with model_id {SONNET} and condition in {CELLS} (April subject runs)",
        "scores/judge_*.jsonl and openweight_subjects/scores/judge_*.jsonl (stray rule and April GPT-5.4 judge dates)",
        "aggregated/all_scores.parquet j_* (April GPT-5.4 arm)",
        f"{E1_OPUS_DIR}/judge_opus_<run_id>.jsonl (April Opus arm, E1 crossed re-score)",
        "analysis/final_probe_accuracy.csv (April final probes)",
        f"{briefs_dir.name}/*.yaml (keyword probe scorer for the E2 probes)",
        f"{_display_path(comparators_dir, data_root)}/{JUDGE_OFFSET_CSV} (E1 Sonnet offset) and "
        f"{_display_path(comparators_dir, data_root)}/{JUDGE_DRIFT_CSV} (GPT-5.4 judge drift control)",
    ]
    base = {
        "inputs": inputs,
        "filters": {
            "april_sample": stray_info,
            "stray_rule": STRAY_RULE,
            "april_cells": {c: int((april_runs["condition"] == c).sum()) for c in CELLS},
            "e2_cells": {c: int((e2_runs["condition"] == c).sum()) for c in CELLS},
            "e2_guards": [
                "metadata.error is None on every E2 transcript",
                f"metadata.max_tokens == {E2_CAP} on every E2 transcript",
                "finish_reason == stop on every E2 assistant message (turns and probes)",
                "exactly one E2 run per brief per cell",
                "E2 score ids equal the E2 transcript ids in both arms; cost_estimated false",
                "April parquet rows judged by GPT-5.4; E1 Opus file present for every April run",
            ],
            "e2_probes": e2_probe_note,
            "april_probes": {"n_pressure_runs": int((april_runs["condition"] == MTP).sum()), "n_with_final_probe": int(len(april_probes))},
        },
        "cap_rules": {SOURCE_APRIL: CAP_RULE_APRIL, SOURCE_E2: CAP_RULE_E2, "scope": "non-probe assistant turns only"},
        "kbv_rule": KBV_RULE,
        "ci_methods": {
            "dimension_means": CI_BRIEF_BOOT,
            "kbv_rate": CI_WILSON,
            "unpaired_proportions": f"{CI_NEWCOMBE}; {TEST_FISHER}",
            "unpaired_means": f"{CI_WELCH}; {TEST_MWU}",
            "brief_paired": f"{CI_PAIRED_T}; {CI_BRIEF_BOOT}; {TEST_WILCOXON}",
            "seed": BOOT_SEED,
            "B": B,
        },
        "judge_dates_utc": judge_dates,
        "judges": {arm: {"judge_model": ARMS[arm]["judge_model"], "temperature": 0} for arm in ARM_ORDER},
        "design": DESIGN_STATEMENT,
        "wording_limit": WORDING_LIMIT,
        "token_rows_note": TOKEN_ROWS_NOTE,
    }
    write_csv_with_provenance(
        cells,
        out_dir / "e2_uncapped_cells.csv",
        {
            **base,
            "description": "E2/B16: token, cap, score and KBV facts per (source, condition, judge arm); April clean vs E2 uncapped",
            "row_counts": {"april_runs": int(len(april_runs)), "e2_runs": int(len(e2_runs)), "cells_out": int(len(cells))},
            "columns": COLUMNS["e2_uncapped_cells"],
        },
    )
    write_csv_with_provenance(
        deltas,
        out_dir / "e2_uncapped_deltas.csv",
        {
            **base,
            "description": "E2/B3/B16: E2 minus April per (condition, judge arm, metric, design)",
            "judge_comparability": comparability,
            "row_counts": {"deltas_out": int(len(deltas))},
            "columns": COLUMNS["e2_uncapped_deltas"],
        },
    )
    write_csv_with_provenance(
        offset,
        out_dir / "e2_uncapped_offset.csv",
        {
            **base,
            "description": "Opus minus GPT-5.4 per dimension on the 76 paired E2 runs, beside the E1 Sonnet offset",
            "row_counts": {"paired_runs": int(len(e2_runs)), "offset_out": int(len(offset))},
            "columns": COLUMNS["e2_uncapped_offset"],
        },
    )
    return {"e2_uncapped_cells": cells, "e2_uncapped_deltas": deltas, "e2_uncapped_offset": offset}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--briefs-dir", type=Path, default=None)
    ap.add_argument("--comparators-dir", type=Path, default=DEFAULT_OUT_DIR, help=f"directory holding {JUDGE_OFFSET_CSV} and {JUDGE_DRIFT_CSV}")
    args = ap.parse_args()
    tables = run(args.data_root, args.out_dir, args.briefs_dir, comparators_dir=args.comparators_dir)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 300):
        c = tables["e2_uncapped_cells"]
        print(c[["source", "condition", "judge_arm", "n_runs", "output_tokens_per_run_mean", "cap_turn_share", "runs_any_cap_share", "constraint_adherence_mean", "alternative_coverage_mean", "kbv_count", "kbv_rate", "kbv_ci_low", "kbv_ci_high"]])
        d = tables["e2_uncapped_deltas"]
        print(d[d["metric"].isin(["kbv_rate", "constraint_adherence", "output_tokens_per_run"])][["condition", "judge_arm", "metric", "design", "delta", "ci_low", "ci_high", "boot_ci_low", "boot_ci_high", "p_value"]])
        print(tables["e2_uncapped_offset"][["scope", "dimension", "n_runs", "e2_offset", "e2_ci_low", "e2_ci_high", "e1_offset"]])


if __name__ == "__main__":
    main()
