"""Per-turn trajectories from experiment E5 (plan items C8, C14; decision P14).

Disk only. No API calls.

E5 re-scored 152 runs (GPT-5.4 and Flash-Lite, neutral and pressure,
repetition 0 of every brief) with Opus 4.6 on the transcript truncated
after non-probe assistant turn K, K in 1..5:
``camera_ready/e5_per_turn/opus/judge_opus_t<K>_<run_id>.jsonl``.
Turn 6 is the published April score of the same run (``j_*`` in
``aggregated/all_scores.parquet``, Opus 4.6). Turn 6 is the sectioned
final proposal, a different elicitation format from turns 1 to 5, so the
turn-6 rows carry ``format = final_sectioned`` and every turn-6 contrast
carries ``format_matched = False``. A separate check re-judged turn 6 on
20 runs today: ``camera_ready/e5_per_turn/opus_t6_check/``.

Probe recall at turn t is the keyword score of the restatement probe that
follows assistant turn t (``drift_bench.analysis.probes``). KBV at turn t
is ``constraint_adherence_t < 4 and probe_t overall_accuracy >= 0.8``.
Turn 6 uses probe 5, as ``kbv_table.py`` does for the published scores.

Sample: the clean sample (``sample.drop_stray``). The 152 runs are
GPT-5.4 and Flash-Lite, so the Sonnet stray rule drops 0 rows; the filter
is applied and recorded anyway.

Outputs (in ``out_dir``), each with a ``.provenance.json`` sibling:

1. ``e5_per_turn_cells.csv``   one row per (model, condition, turn 1..6)
2. ``e5_per_turn_deltas.csv``  paired contrasts per (model, condition,
   dimension, contrast), plus pooled rows

Run: ``python -m drift_bench.analysis.camera_ready.per_turn``
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from drift_bench.analysis.camera_ready.kbv_table import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUT_DIR,
    PROBE_ACCURACY_THRESHOLD,
    ADHERENCE_THRESHOLD,
    read_jsonl_first,
    wilson_ci,
    write_csv_with_provenance,
)
from drift_bench.analysis.camera_ready.sample import STRAY_RULE, drop_stray, stray_run_ids
from drift_bench.analysis.probes import analyze_probes_for_run

OPUS = "anthropic/claude-opus-4-6"
GPT = "openai/gpt-5.4"
FLASH_LITE = "gemini/gemini-3.1-flash-lite-preview"
MODEL_ORDER = [GPT, FLASH_LITE]

MTN = "multi_turn_neutral"
MTP = "multi_turn_pressure"
CONDITION_ORDER = [MTN, MTP]

DIMS = ["objective_fidelity", "constraint_adherence", "alternative_coverage", "complexity_inflation"]
PER_TURN_TURNS = [1, 2, 3, 4, 5]
FINAL_TURN = 6
PROBE_TURNS = [1, 2, 3, 4, 5]

E5_DIR = Path("camera_ready") / "e5_per_turn"
PER_TURN_SUBDIR = "opus"
T6_CHECK_SUBDIR = "opus_t6_check"
FILE_PREFIX = "judge_opus"
RUN_IDS_FILE = "run_ids.txt"
T6_CHECK_IDS_FILE = "t6_check_run_ids.txt"

SOURCE_PER_TURN = "per_turn_opus_20261026"
SOURCE_T6_APRIL = "published_april_opus"
SOURCE_T6_CHECK = "t6_check_opus_20261026"
FORMAT_REFINEMENT = "refinement_turn"
FORMAT_FINAL = "final_sectioned"

POOLED = "pooled"
Z95 = 1.959964

# contrast name -> (later turn key, earlier turn key, same_judge_date, format_matched)
CONTRASTS: dict[str, tuple[str, str, bool, bool]] = {
    "t2-t1": ("t2", "t1", True, True),
    "t5-t1": ("t5", "t1", True, True),
    "t5-t2": ("t5", "t2", True, True),
    "t6april-t5": ("t6april", "t5", False, False),
    "t6check-t5": ("t6check", "t5", True, False),
}
STEP_SHARE_CONTRAST = "t2-t1"
STEP_SHARE_DENOMINATOR = "t5-t1"

REPETITION_RULE = (
    "one run per (brief, model, condition): the transcript with metadata.repetition == 0 "
    "(select_ids.py in camera_ready/e5_per_turn; 152 cells, 0 substitutions)"
)
CI_T = "t interval on the mean, df = n - 1, 95%; one observation per run, so this equals the run-clustered CI"
CI_PAIRED = "paired t interval on the within-run difference, df = n - 1, 95%"
CI_WILSON = f"Wilson score interval, z = {Z95}"
WILCOXON_METHOD = "scipy.stats.wilcoxon, zero_method = wilcox, method = approx (asymptotic, continuity-corrected); p = 1 when every delta is 0"
PAIRED_T_METHOD = "scipy.stats.ttest_1samp on the deltas against 0, two-sided; p = 1 when every delta is 0"

_FILE_RE = re.compile(r"^(?P<prefix>.+)_t(?P<turn>\d+)_(?P<run_id>[^_]+)\.jsonl$")


def _dim_columns(prefix: str, what: str) -> dict[str, str]:
    return {f"{prefix}{d}": f"{what} {d}" for d in DIMS}


COLUMNS: dict[str, dict[str, str]] = {
    "e5_per_turn_cells": {
        "model_id": "subject model",
        "condition": f"{MTN} or {MTP}",
        "turn": "assistant turn scored (1..6)",
        "source": f"{SOURCE_PER_TURN} for turns 1..5 (E5 files), {SOURCE_T6_APRIL} for turn 6 (parquet j_*)",
        "format": f"{FORMAT_REFINEMENT} for turns 1..5, {FORMAT_FINAL} for turn 6 (plan item C14)",
        "n": "runs in the cell",
        **_dim_columns("mean_", "mean of"),
        **_dim_columns("se_", "standard error of the mean of"),
        **_dim_columns("ci_lo_", "lower 95% t bound (df n-1) of the mean of"),
        **_dim_columns("ci_hi_", "upper 95% t bound (df n-1) of the mean of"),
        "share_ca_lt4": f"share of runs with constraint_adherence < {ADHERENCE_THRESHOLD}",
        "share_ca_lt4_ci_lo": "Wilson 95% lower bound of share_ca_lt4",
        "share_ca_lt4_ci_hi": "Wilson 95% upper bound of share_ca_lt4",
        "kbv_rate": f"share of runs with constraint_adherence < {ADHERENCE_THRESHOLD} and probe overall_accuracy >= {PROBE_ACCURACY_THRESHOLD} (probe after this turn; turn 6 uses probe 5)",
        "kbv_n": "runs with a usable probe at this turn (KBV denominator)",
        "kbv_ci_lo": "Wilson 95% lower bound of kbv_rate",
        "kbv_ci_hi": "Wilson 95% upper bound of kbv_rate",
        "probe_recall_mean": "mean probe overall_accuracy (keyword scorer) of the probe after this turn; turn 6 uses probe 5",
    },
    "e5_per_turn_deltas": {
        "model_id": f"subject model, or {POOLED} for both models together",
        "condition": f"{MTN}, {MTP}, or {POOLED} for both conditions together",
        "dimension": "scored dimension",
        "contrast": "later minus earlier: " + ", ".join(CONTRASTS),
        "n": "paired runs (76 on pooled per-condition rows for the t1..t5 contrasts; 20 check ids on t6check-t5 pooled over everything)",
        "mean_delta": "mean within-run difference",
        "se": "standard error of mean_delta",
        "ci_lo": "lower 95% paired t bound, df n-1",
        "ci_hi": "upper 95% paired t bound, df n-1",
        "p_paired_t": "two-sided paired t p-value (1 when every delta is 0)",
        "p_wilcoxon": "two-sided Wilcoxon signed-rank p-value, asymptotic (1 when every delta is 0)",
        "n_down": "runs whose score fell",
        "n_same": "runs whose score did not change",
        "n_up": "runs whose score rose",
        "step_share": f"on {STEP_SHARE_CONTRAST} rows only: mean({STEP_SHARE_CONTRAST}) / mean({STEP_SHARE_DENOMINATOR}); NaN when the denominator is 0; NaN on other rows",
        "step_share_denominator_ci_excludes_zero": f"on {STEP_SHARE_CONTRAST} rows only: True when the paired 95% CI of {STEP_SHARE_DENOMINATOR} excludes 0 (quote step_share only then); empty on other rows",
        "same_judge_date": "True when both sides were judged on 2026-10-26; False for t6april-t5 (April score)",
        "format_matched": "False for both t6 contrasts (turn 6 is the sectioned final proposal); True otherwise",
    },
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def read_id_file(path: Path) -> list[str]:
    """Run ids, one per line; blank lines and ``#`` comments are skipped."""
    ids: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s and not s.startswith("#"):
                ids.append(s)
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate run ids in {path}")
    return ids


def parse_turn_filename(name: str, prefix: str = FILE_PREFIX) -> tuple[int, str]:
    """``judge_opus_t3_<run_id>.jsonl`` -> (3, run_id). Raises ValueError otherwise."""
    m = _FILE_RE.match(name)
    if m is None or m.group("prefix") != prefix:
        raise ValueError(f"file name {name!r} does not match {prefix}_t<K>_<run_id>.jsonl")
    return int(m.group("turn")), m.group("run_id")


def load_turn_dir(path: Path, prefix: str = FILE_PREFIX, expected_judge: str = OPUS) -> pd.DataFrame:
    """One row per ``<prefix>_t<K>_<run_id>.jsonl`` file in ``path``.

    Guards (ValueError): ``judge_model`` differs from ``expected_judge``;
    ``cost_estimated`` is true; ``turn_index`` missing or not equal to the
    file name's K; ``run_id`` in the record differs from the file name.
    FileNotFoundError when ``path`` does not exist.
    """
    if not path.is_dir():
        raise FileNotFoundError(f"judge directory not found: {path}")
    cols = ["run_id", "brief_id", "model_id", "condition", "turn", "judge_model", "judge_response_model",
            "timestamp", "cost_usd", "cost_estimated", *DIMS, "source_file"]
    rows = []
    for file in sorted(path.glob(f"{prefix}_t*_*.jsonl")):
        turn, run_id = parse_turn_filename(file.name, prefix)
        rec = read_jsonl_first(file)
        if rec.get("judge_model") != expected_judge:
            raise ValueError(f"{file.name}: judge_model {rec.get('judge_model')!r} != {expected_judge!r}")
        if bool(rec.get("cost_estimated", False)):
            raise ValueError(f"{file.name}: cost_estimated is true")
        if rec.get("turn_index") is None:
            raise ValueError(f"{file.name}: record has no turn_index")
        if int(rec["turn_index"]) != turn:
            raise ValueError(f"{file.name}: turn_index {rec['turn_index']} != file name turn {turn}")
        if rec.get("run_id") not in (None, run_id):
            raise ValueError(f"{file.name}: run_id {rec.get('run_id')!r} != file name run id {run_id!r}")
        for d in DIMS:
            if rec.get(d) is None:
                raise ValueError(f"{file.name}: missing score {d}")
        row = {c: rec.get(c) for c in cols if c not in ("turn", "source_file", "run_id")}
        row["run_id"] = run_id
        row["turn"] = turn
        row["source_file"] = file.name
        rows.append(row)
    df = pd.DataFrame(rows, columns=cols)
    for d in DIMS:
        df[d] = pd.to_numeric(df[d])
    return df


def check_complete(df: pd.DataFrame, run_ids: list[str], turns: list[int], what: str) -> None:
    """Raise ValueError unless every (run, turn) pair is present exactly once."""
    have = df.groupby(["run_id", "turn"]).size()
    missing = [(r, t) for r in run_ids for t in turns if (r, t) not in have.index]
    if missing:
        raise ValueError(f"{what}: {len(missing)} missing (run, turn) pairs, e.g. {missing[:5]}")
    dup = have[have > 1]
    if len(dup):
        raise ValueError(f"{what}: duplicate (run, turn) pairs, e.g. {dup.index[:5].tolist()}")


def load_april(data_root: Path, run_ids: list[str], expected_judge: str = OPUS) -> pd.DataFrame:
    """The April (turn 6) scores of ``run_ids`` from the published parquet.

    Columns: ``run_id, brief_id, model_id, condition, judge_model`` and the
    four dimensions (the ``j_`` prefix stripped). ValueError when a run has
    no parquet row or its judge is not ``expected_judge``.
    """
    df = pd.read_parquet(data_root / "aggregated" / "all_scores.parquet")
    sub = df[df["run_id"].isin(run_ids)].copy()
    missing = sorted(set(run_ids) - set(sub["run_id"]))
    if missing:
        raise ValueError(f"{len(missing)} runs have no April row in all_scores.parquet, e.g. {missing[:5]}")
    if sub["run_id"].duplicated().any():
        raise ValueError("duplicate run ids in all_scores.parquet for the E5 runs")
    bad = sub[sub["judge_model"] != expected_judge]
    if len(bad):
        raise ValueError(f"{len(bad)} April rows judged by {sorted(bad['judge_model'].unique())}, not {expected_judge}")
    out = sub[["run_id", "brief_id", "model_id", "condition", "judge_model", *[f"j_{d}" for d in DIMS]]]
    out = out.rename(columns={f"j_{d}": d for d in DIMS}).reset_index(drop=True)
    for d in DIMS:
        out[d] = pd.to_numeric(out[d])
    return out


def find_transcript(data_root: Path, run_id: str) -> Path:
    """``transcripts/<run_id>_*.jsonl``; FileNotFoundError when absent or ambiguous."""
    hits = sorted((data_root / "transcripts").glob(f"{run_id}_*.jsonl"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one transcript for {run_id}, found {len(hits)}")
    return hits[0]


def load_briefs(briefs_dir: Path) -> dict[str, dict]:
    import yaml

    briefs: dict[str, dict] = {}
    for path in sorted(briefs_dir.glob("*.yaml")):
        with open(path, encoding="utf-8") as fh:
            brief = yaml.safe_load(fh)
        briefs[brief["id"]] = brief
    return briefs


def probe_accuracy_per_turn(
    data_root: Path,
    run_ids: list[str],
    briefs_dir: Path,
    turns: list[int] = PROBE_TURNS,
) -> pd.DataFrame:
    """Keyword probe accuracy of the probe after each turn, per run.

    Columns: ``run_id, turn, overall_accuracy, constraint_recall,
    banned_recall, repetition``. The probe after assistant turn t has
    ``turn_index == t`` in ``probes.extract_probes``. ValueError when a run
    lacks a probe for any turn in ``turns`` or its brief is unknown.
    """
    briefs = load_briefs(briefs_dir)
    rows = []
    for run_id in run_ids:
        transcript = read_jsonl_first(find_transcript(data_root, run_id))
        md = transcript.get("metadata", {})
        brief = briefs.get(md.get("brief_id"))
        if brief is None:
            raise ValueError(f"{run_id}: brief {md.get('brief_id')!r} not found in {briefs_dir}")
        recs = analyze_probes_for_run(transcript, brief)
        by_turn = {r["turn_index"]: r for r in recs}
        for t in turns:
            if t not in by_turn:
                raise ValueError(f"{run_id}: no probe after turn {t} (found turns {sorted(by_turn)})")
            r = by_turn[t]
            rows.append(
                {
                    "run_id": run_id,
                    "turn": t,
                    "overall_accuracy": float(r["overall_accuracy"]),
                    "constraint_recall": float(r["constraint_recall"]),
                    "banned_recall": float(r["banned_recall"]),
                    "repetition": md.get("repetition"),
                }
            )
    return pd.DataFrame(rows, columns=["run_id", "turn", "overall_accuracy", "constraint_recall", "banned_recall", "repetition"])


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def assemble_long(per_turn: pd.DataFrame, april: pd.DataFrame, probes: pd.DataFrame) -> pd.DataFrame:
    """One row per (run, turn 1..6) with scores, source, format and probe recall.

    Turns 1..5 come from ``per_turn``; turn 6 from ``april``. The probe of
    turn t is the probe after turn t; turn 6 takes probe 5.
    """
    pt = per_turn[["run_id", "brief_id", "model_id", "condition", "turn", *DIMS]].copy()
    pt["source"] = SOURCE_PER_TURN
    pt["format"] = FORMAT_REFINEMENT
    ap = april[["run_id", "brief_id", "model_id", "condition", *DIMS]].copy()
    ap["turn"] = FINAL_TURN
    ap["source"] = SOURCE_T6_APRIL
    ap["format"] = FORMAT_FINAL
    # The April row carries the authoritative brief/model/condition for turn 6;
    # the per-turn files must agree on every run.
    meta_pt = pt[["run_id", "brief_id", "model_id", "condition"]].drop_duplicates()
    meta_ap = ap[["run_id", "brief_id", "model_id", "condition"]]
    merged = meta_pt.merge(meta_ap, on="run_id", suffixes=("", "_april"))
    for c in ("brief_id", "model_id", "condition"):
        bad = merged[merged[c] != merged[f"{c}_april"]]
        if len(bad):
            raise ValueError(f"{len(bad)} runs disagree on {c} between the E5 files and the parquet, e.g. {bad['run_id'].head(3).tolist()}")
    long = pd.concat([pt, ap], ignore_index=True)
    long["probe_turn"] = long["turn"].clip(upper=max(PROBE_TURNS))
    pr = probes[["run_id", "turn", "overall_accuracy"]].rename(columns={"turn": "probe_turn", "overall_accuracy": "probe_accuracy"})
    long = long.merge(pr, on=["run_id", "probe_turn"], how="left")
    long["violates"] = long["constraint_adherence"] < ADHERENCE_THRESHOLD
    long["knows"] = (long["probe_accuracy"] >= PROBE_ACCURACY_THRESHOLD).where(long["probe_accuracy"].notna())
    long["kbv"] = (long["violates"] & (long["probe_accuracy"] >= PROBE_ACCURACY_THRESHOLD)).where(long["probe_accuracy"].notna())
    long["model_id"] = pd.Categorical(long["model_id"], categories=_ordered(long["model_id"], MODEL_ORDER))
    long["condition"] = pd.Categorical(long["condition"], categories=_ordered(long["condition"], CONDITION_ORDER))
    return long.sort_values(["model_id", "condition", "turn", "run_id"]).reset_index(drop=True)


def _ordered(values: pd.Series, preferred: list[str]) -> list[str]:
    present = list(pd.unique(values.astype(str)))
    return [v for v in preferred if v in present] + sorted(v for v in present if v not in preferred)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def t_interval(values: pd.Series | np.ndarray, level: float = 0.95) -> tuple[float, float, float, float]:
    """(mean, se, ci_lo, ci_hi) with a t interval, df = n - 1. NaN when n < 2."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n == 0:
        return (math.nan, math.nan, math.nan, math.nan)
    mean = float(x.mean())
    if n < 2:
        return (mean, math.nan, math.nan, math.nan)
    se = float(x.std(ddof=1) / math.sqrt(n))
    q = float(stats.t.ppf(0.5 + level / 2, n - 1))
    return (mean, se, mean - q * se, mean + q * se)


def paired_delta(later: pd.Series | np.ndarray, earlier: pd.Series | np.ndarray) -> dict:
    """Paired statistics of ``later - earlier`` (aligned arrays)."""
    a = np.asarray(later, dtype=float)
    b = np.asarray(earlier, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired_delta needs aligned arrays")
    d = a - b
    n = len(d)
    mean, se, lo, hi = t_interval(d)
    if n == 0:
        p_t, p_w = math.nan, math.nan
    elif np.all(d == 0):
        p_t, p_w = 1.0, 1.0
    else:
        with warnings.catch_warnings():
            # scipy warns about precision loss when most deltas are equal; the
            # integer deltas here are exact, so the warning carries no information.
            warnings.simplefilter("ignore", RuntimeWarning)
            p_t = float(stats.ttest_1samp(d, 0.0).pvalue) if n >= 2 else math.nan
            p_w = float(stats.wilcoxon(d, zero_method="wilcox", method="approx").pvalue)
    return {
        "n": int(n),
        "mean_delta": mean,
        "se": se,
        "ci_lo": lo,
        "ci_hi": hi,
        "p_paired_t": p_t,
        "p_wilcoxon": p_w,
        "n_down": int((d < 0).sum()),
        "n_same": int((d == 0).sum()),
        "n_up": int((d > 0).sum()),
    }


def step_share(numerator_mean: float, denominator_mean: float) -> float:
    """``numerator / denominator``; NaN when the denominator is 0 or NaN."""
    if denominator_mean is None or math.isnan(denominator_mean) or denominator_mean == 0:
        return math.nan
    return float(numerator_mean / denominator_mean)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def cells_table(long: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, condition, turn)."""
    rows = []
    for (model_id, condition, turn), g in long.groupby(["model_id", "condition", "turn"], sort=True, observed=True):
        row = {
            "model_id": str(model_id),
            "condition": str(condition),
            "turn": int(turn),
            "source": g["source"].iloc[0],
            "format": g["format"].iloc[0],
            "n": int(len(g)),
        }
        for d in DIMS:
            mean, se, lo, hi = t_interval(g[d])
            row[f"mean_{d}"] = mean
            row[f"se_{d}"] = se
            row[f"ci_lo_{d}"] = lo
            row[f"ci_hi_{d}"] = hi
        k = int(g["violates"].sum())
        lo, hi = wilson_ci(k, len(g))
        row["share_ca_lt4"] = k / len(g)
        row["share_ca_lt4_ci_lo"] = lo
        row["share_ca_lt4_ci_hi"] = hi
        with_probe = g[g["probe_accuracy"].notna()]
        kbv_n = int(len(with_probe))
        kbv_k = int(with_probe["kbv"].astype(bool).sum()) if kbv_n else 0
        lo, hi = wilson_ci(kbv_k, kbv_n)
        row["kbv_rate"] = kbv_k / kbv_n if kbv_n else math.nan
        row["kbv_n"] = kbv_n
        row["kbv_ci_lo"] = lo
        row["kbv_ci_hi"] = hi
        row["probe_recall_mean"] = float(with_probe["probe_accuracy"].mean()) if kbv_n else math.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=list(COLUMNS["e5_per_turn_cells"]))


def wide_scores(long: pd.DataFrame, t6_check: pd.DataFrame) -> pd.DataFrame:
    """One row per (run, dimension) with columns t1..t5, t6april, t6check.

    ``t6check`` is NaN for runs outside the check set.
    """
    recs = []
    chk = t6_check.set_index("run_id") if len(t6_check) else None
    meta = long[["run_id", "brief_id", "model_id", "condition"]].drop_duplicates("run_id")
    piv = {d: long.pivot(index="run_id", columns="turn", values=d) for d in DIMS}
    for _, m in meta.iterrows():
        rid = m["run_id"]
        for d in DIMS:
            row = {"run_id": rid, "brief_id": m["brief_id"], "model_id": str(m["model_id"]), "condition": str(m["condition"]), "dimension": d}
            for t in PER_TURN_TURNS:
                row[f"t{t}"] = float(piv[d].loc[rid, t])
            row["t6april"] = float(piv[d].loc[rid, FINAL_TURN])
            row["t6check"] = float(chk.loc[rid, d]) if chk is not None and rid in chk.index else math.nan
            recs.append(row)
    return pd.DataFrame(recs)


def deltas_table(wide: pd.DataFrame) -> pd.DataFrame:
    """Paired contrasts per (model, condition, dimension), plus pooled rows.

    Pooled rows: ``model_id = pooled`` per condition (both models), and
    ``model_id = pooled, condition = pooled`` (every run). Each contrast
    uses the runs that have both sides; ``t6check-t5`` therefore uses the
    check ids only.
    """
    groups: list[tuple[str, str, pd.DataFrame]] = []
    for (model_id, condition), g in wide.groupby(["model_id", "condition"], sort=False):
        groups.append((str(model_id), str(condition), g))
    for condition, g in wide.groupby("condition", sort=False):
        groups.append((POOLED, str(condition), g))
    groups.append((POOLED, POOLED, wide))

    rows = []
    for model_id, condition, g in groups:
        for d in DIMS:
            gd = g[g["dimension"] == d]
            stats_by_contrast: dict[str, dict] = {}
            for contrast, (later, earlier, same_date, fmt) in CONTRASTS.items():
                pair = gd[[later, earlier]].dropna()
                st = paired_delta(pair[later], pair[earlier])
                stats_by_contrast[contrast] = st
                rows.append(
                    {
                        "model_id": model_id,
                        "condition": condition,
                        "dimension": d,
                        "contrast": contrast,
                        **st,
                        "step_share": math.nan,
                        "step_share_denominator_ci_excludes_zero": pd.NA,
                        "same_judge_date": same_date,
                        "format_matched": fmt,
                    }
                )
            # step share lives on the t2-t1 row of this (model, condition, dim)
            denom = stats_by_contrast[STEP_SHARE_DENOMINATOR]
            for row in rows[-len(CONTRASTS):]:
                if row["contrast"] == STEP_SHARE_CONTRAST:
                    row["step_share"] = step_share(stats_by_contrast[STEP_SHARE_CONTRAST]["mean_delta"], denom["mean_delta"])
                    row["step_share_denominator_ci_excludes_zero"] = ci_excludes_zero(denom["ci_lo"], denom["ci_hi"])
    out = pd.DataFrame(rows, columns=list(COLUMNS["e5_per_turn_deltas"]))
    out["step_share_denominator_ci_excludes_zero"] = out["step_share_denominator_ci_excludes_zero"].astype("boolean")
    return out


def ci_excludes_zero(lo: float, hi: float) -> bool:
    """True when the interval [lo, hi] does not contain 0. False when a bound is NaN."""
    if lo is None or hi is None or math.isnan(lo) or math.isnan(hi):
        return False
    return bool(lo > 0 or hi < 0)


def april_judge_dates(data_root: Path, run_ids: list[str]) -> dict:
    """UTC dates of the April ``scores/judge_<run_id>.jsonl`` timestamps of ``run_ids``.

    Returns ``{"dates": sorted ISO dates, "n_files": int, "n_missing_files": int,
    "n_without_timestamp": int}``. Nothing is hard-coded; missing files are
    counted, not fabricated.
    """
    dates: set[str] = set()
    n_files = n_missing = n_no_ts = 0
    for run_id in run_ids:
        path = data_root / "scores" / f"judge_{run_id}.jsonl"
        if not path.exists():
            n_missing += 1
            continue
        n_files += 1
        ts = read_jsonl_first(path).get("timestamp")
        try:
            dates.add(dt.datetime.fromtimestamp(float(ts), dt.timezone.utc).date().isoformat())
        except (TypeError, ValueError):
            n_no_ts += 1
    return {"dates": sorted(dates), "n_files": n_files, "n_missing_files": n_missing, "n_without_timestamp": n_no_ts}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _utc(ts: float | None) -> str | None:
    if ts is None or (isinstance(ts, float) and math.isnan(ts)):
        return None
    return dt.datetime.fromtimestamp(float(ts), dt.timezone.utc).isoformat()


def run(
    data_root: Path = DEFAULT_DATA_ROOT,
    out_dir: Path = DEFAULT_OUT_DIR,
    briefs_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    briefs_dir = Path(briefs_dir) if briefs_dir is not None else data_root.parent / "briefs"
    e5 = data_root / E5_DIR

    run_ids = read_id_file(e5 / RUN_IDS_FILE)
    check_ids = read_id_file(e5 / T6_CHECK_IDS_FILE)
    extra_check = sorted(set(check_ids) - set(run_ids))
    if extra_check:
        raise ValueError(f"turn-6 check ids outside run_ids.txt: {extra_check}")

    per_turn = load_turn_dir(e5 / PER_TURN_SUBDIR, FILE_PREFIX, OPUS)
    outside = sorted(set(per_turn["run_id"]) - set(run_ids))
    per_turn = per_turn[per_turn["run_id"].isin(run_ids) & per_turn["turn"].isin(PER_TURN_TURNS)].reset_index(drop=True)
    check_complete(per_turn, run_ids, PER_TURN_TURNS, "per-turn files")

    t6_check = load_turn_dir(e5 / T6_CHECK_SUBDIR, FILE_PREFIX, OPUS)
    outside_check = sorted(set(t6_check["run_id"]) - set(check_ids))
    t6_check = t6_check[t6_check["run_id"].isin(check_ids) & (t6_check["turn"] == FINAL_TURN)].reset_index(drop=True)
    check_complete(t6_check, check_ids, [FINAL_TURN], "turn-6 check files")

    april = load_april(data_root, run_ids, OPUS)
    probes = probe_accuracy_per_turn(data_root, run_ids, briefs_dir)

    long = assemble_long(per_turn, april, probes)
    rows_before_stray = int(len(long))
    stray = stray_run_ids(data_root)
    long, stray_info = drop_stray(long, data_root, stray=stray)
    t6_check = t6_check[~t6_check["run_id"].isin(stray)].reset_index(drop=True)
    kept_ids = sorted(set(long["run_id"]))
    kept_check_ids = sorted(set(t6_check["run_id"]))
    # The categorical order must survive the filter.
    long["model_id"] = pd.Categorical(long["model_id"].astype(str), categories=_ordered(long["model_id"].astype(str), MODEL_ORDER))
    long["condition"] = pd.Categorical(long["condition"].astype(str), categories=_ordered(long["condition"].astype(str), CONDITION_ORDER))

    cells = cells_table(long)
    wide = wide_scores(long, t6_check)
    deltas = deltas_table(wide)

    rep = probes.drop_duplicates("run_id")["repetition"]
    ts = pd.to_numeric(per_turn["timestamp"], errors="coerce")
    ts6 = pd.to_numeric(t6_check["timestamp"], errors="coerce")
    april_dates = april_judge_dates(data_root, kept_ids)
    # Facts computed from the tables, for the provenance notes below.
    recall_min = float(cells["probe_recall_mean"].min())
    recall_max = float(cells["probe_recall_mean"].max())
    kbv_equals_share = bool(np.allclose(cells["kbv_rate"], cells["share_ca_lt4"]))
    check_cell_ns = sorted(deltas.loc[(deltas["contrast"] == "t6check-t5") & (deltas["model_id"] != POOLED), "n"].unique().tolist())
    inputs = [
        f"{E5_DIR}/{PER_TURN_SUBDIR}/{FILE_PREFIX}_t<K>_<run_id>.jsonl, K in 1..5",
        f"{E5_DIR}/{T6_CHECK_SUBDIR}/{FILE_PREFIX}_t6_<run_id>.jsonl",
        f"{E5_DIR}/{RUN_IDS_FILE}",
        f"{E5_DIR}/{T6_CHECK_IDS_FILE}",
        "aggregated/all_scores.parquet (j_* of the same runs = turn 6, April, Opus)",
        "transcripts/<run_id>_*.jsonl (restatement probes after turns 1..5)",
        f"{briefs_dir.name}/*.yaml (hard_constraints and banned_moves for the keyword probe scorer)",
        "scores/judge_*.jsonl and openweight_subjects/scores/judge_*.jsonl (stray rule timestamps)",
    ]
    filters = {
        "sample": stray_info,
        "stray_rule": STRAY_RULE,
        "repetition_rule": REPETITION_RULE,
        "repetitions_observed": {str(k): int(v) for k, v in rep.value_counts(dropna=False).sort_index().items()},
        "run_ids": {"file": f"{E5_DIR}/{RUN_IDS_FILE}", "n": len(run_ids), "n_after_stray": len(kept_ids)},
        "t6_check_ids": {"file": f"{E5_DIR}/{T6_CHECK_IDS_FILE}", "n": len(check_ids), "n_after_stray": len(kept_check_ids), "ids": kept_check_ids},
        "per_turn_files_outside_run_ids_ignored": outside,
        "t6_check_files_outside_check_ids_ignored": outside_check,
        "guards": [
            f"judge_model must be {OPUS} on every E5 file and every April row",
            "cost_estimated must be false on every E5 file",
            "turn_index must equal the K in the file name",
            "every (run, turn) pair in run_ids x 1..5 and check_ids x 6 must exist exactly once",
            "every run must have an April parquet row and a probe after each of turns 1..5",
        ],
        "kbv": f"constraint_adherence_t < {ADHERENCE_THRESHOLD} and probe_t overall_accuracy >= {PROBE_ACCURACY_THRESHOLD}; probe_t is the restatement probe after assistant turn t (drift_bench.analysis.probes keyword scorer); turn 6 uses probe 5",
        "kbv_note": (
            f"per-turn kbv_rate {'equals' if kbv_equals_share else 'does NOT equal'} share_ca_lt4 in every cell: "
            f"probe_recall_mean ranges {recall_min:.3f} to {recall_max:.3f} over the {len(cells)} cells under the keyword scorer, "
            f"so the recall filter (>= {PROBE_ACCURACY_THRESHOLD}) removes "
            f"{'no run' if kbv_equals_share else 'some runs'}"
        ),
        "t6check_note": (
            f"per-cell t6check-t5 rows have n = {check_cell_ns} and are not for quoting; "
            f"quote the model_id = {POOLED}, condition = {POOLED} row (n = {len(kept_check_ids)})"
        ),
    }
    judges = {
        "per_turn": {"judge_model": OPUS, "judge_response_model": sorted(per_turn["judge_response_model"].dropna().astype(str).unique().tolist()), "temperature": 0},
        "t6_check": {"judge_model": OPUS, "judge_response_model": sorted(t6_check["judge_response_model"].dropna().astype(str).unique().tolist()), "temperature": 0},
        "t6_april": {"judge_model": OPUS, "judge_response_model": "not recorded in April (field predates it); same alias"},
    }
    dates = {
        "per_turn_judge_utc": {"first": _utc(ts.min()), "last": _utc(ts.max())},
        "t6_check_judge_utc": {"first": _utc(ts6.min()), "last": _utc(ts6.max())},
        "t6_april": {
            "judge_dates_utc": april_dates["dates"],
            "n_judge_files": april_dates["n_files"],
            "n_missing_judge_files": april_dates["n_missing_files"],
            "n_without_timestamp": april_dates["n_without_timestamp"],
            "source": "UTC date of the timestamp in scores/judge_<run_id>.jsonl for each of the runs (the parquet carries no timestamp)",
        },
        "note": "record timestamps are call completions",
    }
    base = {"inputs": inputs, "filters": filters, "judges": judges, "dates": dates}

    write_csv_with_provenance(
        cells,
        out_dir / "e5_per_turn_cells.csv",
        {
            **base,
            "description": "C8: per-turn means, shares and KBV per (model, condition, turn 1..6); turns 1..5 are the E5 Opus re-scores, turn 6 the published April Opus score (different format, C14)",
            "ci_methods": {"mean": CI_T, "share_ca_lt4": CI_WILSON, "kbv_rate": CI_WILSON},
            "row_counts": {
                "per_turn_files": int(len(per_turn)),
                "t6_check_files": int(len(t6_check)),
                "april_rows": int(len(april)),
                "probe_rows": int(len(probes)),
                "long_rows_before_stray": rows_before_stray,
                "rows_in": int(len(long)),
                "cells_out": int(len(cells)),
            },
            "columns": COLUMNS["e5_per_turn_cells"],
        },
    )
    write_csv_with_provenance(
        deltas,
        out_dir / "e5_per_turn_deltas.csv",
        {
            **base,
            "description": "C8/C14: paired within-run contrasts per (model, condition, dimension); pooled rows per condition and over everything; t6check-t5 on the 20 check ids",
            "ci_methods": {"mean_delta": CI_PAIRED, "p_paired_t": PAIRED_T_METHOD, "p_wilcoxon": WILCOXON_METHOD},
            "contrasts": {k: {"later": v[0], "earlier": v[1], "same_judge_date": v[2], "format_matched": v[3]} for k, v in CONTRASTS.items()},
            "step_share": f"mean({STEP_SHARE_CONTRAST}) / mean({STEP_SHARE_DENOMINATOR}) on {STEP_SHARE_CONTRAST} rows; NaN when the denominator is 0",
            "pooled": f"model_id = {POOLED} pools both models within a condition; condition = {POOLED} too pools every run",
            "row_counts": {
                "wide_rows_in": int(len(wide)),
                "runs_in": len(kept_ids),
                "check_runs_in": len(kept_check_ids),
                "deltas_out": int(len(deltas)),
            },
            "columns": COLUMNS["e5_per_turn_deltas"],
        },
    )
    return {"e5_per_turn_cells": cells, "e5_per_turn_deltas": deltas}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--briefs-dir", type=Path, default=None)
    args = ap.parse_args()
    tables = run(args.data_root, args.out_dir, args.briefs_dir)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 300):
        c = tables["e5_per_turn_cells"]
        print(c[["model_id", "condition", "turn", "n", "mean_constraint_adherence", "share_ca_lt4", "mean_complexity_inflation", "mean_alternative_coverage", "kbv_rate", "probe_recall_mean"]])
        d = tables["e5_per_turn_deltas"]
        print(d[d["dimension"] == "constraint_adherence"][["model_id", "condition", "contrast", "n", "mean_delta", "ci_lo", "ci_hi", "p_wilcoxon", "step_share"]])


if __name__ == "__main__":
    main()
