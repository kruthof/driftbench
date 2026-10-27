"""Crossed-judge analysis of experiment E1 (plan items A1, A2, A3, A10, A12, C1, C3).

Disk only. No API calls.

The published arm (``j_*`` in ``aggregated/all_scores.parquet``) is Opus for
the six non-Anthropic subjects and GPT-5.4 for Sonnet. E1 scored every clean
single-shot and pressure run with the other judge:
``camera_ready/e1_crossed_judge/opus/judge_opus_<run_id>.jsonl`` (Opus on
Sonnet) and ``gpt/judge_gpt_<run_id>.jsonl`` (GPT-5.4 on the other six).
Two common-judge arms follow:

* ``opus`` arm: Opus crossed for Sonnet, Opus published for the others.
* ``gpt`` arm: GPT-5.4 published for Sonnet, GPT-5.4 crossed for the others.
* ``published`` arm: the parquet as is.

The neutral and checkpointed cells were not crossed. Their rows carry the
published arm only, flagged ``judge_arm = published``. Nothing is imputed.

The 50 control runs (``control_opus/``, ``control_gpt/``) were re-scored by
their ORIGINAL judge, so April (parquet) and today can be compared under the
same judge.

Sample: the clean sample (``sample.drop_stray``), as everywhere else.
KBV: final-turn ``overall_accuracy >= 0.8`` and constraint adherence ``< 4``;
denominator = scored runs with a final probe. Wilson 95% CI. The Gemini Pro
waiver (decision P1) is reported as a variant.

Outputs (in ``out_dir``), each with a ``.provenance.json`` sibling:

1. ``crossed_scores_runs.csv``      one row per clean SS / MT-P run with both arms
2. ``kbv_common_judge.csv``         KBV per model, pressure cell, three arms
   ``kbv_common_judge_range.csv``   range and above-50% membership per arm
3. ``judge_offset.csv``             C1: offsets, interaction tests, deltas, rubric cells
4. ``common_judge_aggregate.csv``   A3: Table 2 and Table 3 equivalents per arm
5. ``auditor_agreement_by_arm.csv`` A3: auditor-KBV agreement per arm
6. ``stretch_violate_by_judge_arm.csv`` C3: below-3 and at-3 shares per arm
   (``stretch_violate_by_judge.csv`` belongs to ``stretch_violate.py``)
7. ``judge_drift_control.csv``      April vs today under the same judge

Run: ``python -m drift_bench.analysis.camera_ready.kbv_crossed``
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from drift_bench.analysis.camera_ready.kbv_table import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUT_DIR,
    GEMINI_PRO,
    GEMINI_PRO_PROBE_CAP,
    PROBE_ACCURACY_THRESHOLD,
    ADHERENCE_THRESHOLD,
    SONNET,
    load_transcript_index,
    read_jsonl_first,
    wilson_ci,
    write_csv_with_provenance,
)
from drift_bench.analysis.camera_ready.sample import STRAY_RULE, drop_stray, stray_run_ids

OPUS = "anthropic/claude-opus-4-6"
GPT = "openai/gpt-5.4"

SS = "single_shot"
MTN = "multi_turn_neutral"
MTP = "multi_turn_pressure"
CKP = "checkpointed_pressure"
CONDITIONS = [SS, MTN, MTP, CKP]
CROSSED_CONDITIONS = [SS, MTP]

DIMS = ["objective_fidelity", "constraint_adherence", "alternative_coverage", "complexity_inflation"]
ARMS = ["opus", "gpt", "published"]

E1_DIR = Path("camera_ready") / "e1_crossed_judge"
# Crossed directories: (judge arm name, sub-directory, file prefix, judge id, subject rule).
CROSSED_SOURCES = [
    ("opus", "opus", "judge_opus", OPUS, "sonnet_only"),
    ("gpt", "gpt", "judge_gpt", GPT, "non_sonnet_only"),
]
CONTROL_SOURCES = [
    ("opus", "control_opus", "judge_opus", OPUS),
    ("gpt", "control_gpt", "judge_gpt", GPT),
]

BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20261026
ABOVE_CUT = 0.5
DRIFT_LABELS = ("trajectory_drift", "trajectory_lock_in")

_DIM_COLS = lambda prefix, what: {f"{prefix}{d}": f"{what} {d} (0-4)" for d in DIMS}  # noqa: E731

# One line per column, for the provenance sidecars.
COLUMNS: dict[str, dict[str, str]] = {
    "crossed_scores_runs": {
        "run_id": "run id (parquet key)",
        "brief_id": "brief id",
        "model_id": "subject model",
        "condition": "single_shot or multi_turn_pressure",
        "published_judge": "judge of the published arm (GPT-5.4 for Sonnet, Opus otherwise)",
        "crossed_judge": "judge of the crossed E1 file (the other one)",
        **_DIM_COLS("published_", "published-arm score for"),
        **_DIM_COLS("crossed_", "crossed-judge score for"),
        **_DIM_COLS("opus_", "common Opus arm score for"),
        **_DIM_COLS("gpt_", "common GPT-5.4 arm score for"),
        "final_probe_accuracy": "final-turn overall_accuracy from analysis/final_probe_accuracy.csv (NaN = no probe)",
        "has_final_probe": "True when final_probe_accuracy is present",
        "crossed_file": "E1 file name the crossed scores come from",
    },
    "kbv_common_judge": {
        "judge_arm": "opus, gpt or published",
        "variant": "clean, or clean_truncated_probes_waived (decision P1)",
        "model_id": "subject model",
        "condition": "multi_turn_pressure",
        "judge_model": "judge that produced the arm's scores for this model",
        "n": "scored runs in the cell",
        "n_with_probe": "runs with a final probe (KBV denominator)",
        "knows": "runs with a probe that pass the recall filter (waived rows included)",
        "n_truncated_waived": "Gemini Pro runs whose capped final probe was waived through the recall filter (0 outside the waived variant)",
        "knows_flipped_by_waiver": "waived runs whose measured probe accuracy was below 0.8, so the waiver changed knows from False to True",
        "violates": "scored runs with adherence < 4",
        "kbv_count": "runs that know and violate",
        "kbv_rate": "kbv_count / n_with_probe",
        "ci_low": "Wilson 95% lower bound",
        "ci_high": "Wilson 95% upper bound",
    },
    "kbv_common_judge_range": {
        "judge_arm": "opus, gpt or published",
        "variant": "clean or clean_truncated_probes_waived",
        "n_models": "models with a KBV rate",
        "min_kbv_rate": "lowest KBV rate",
        "argmin_model": "model with the lowest rate",
        "max_kbv_rate": "highest KBV rate",
        "argmax_model": "model with the highest rate",
        "cut": "membership cut (0.5)",
        "n_above_cut": "models with kbv_rate > cut",
        "models_above_cut": "those models, highest first, semicolon-separated",
    },
    "judge_offset": {
        "block": "which statistic the row holds (offset_overall, offset_overall_by_condition, offset_by_model, anthropic_excess, interaction_judge_x_condition, delta_pressure_minus_single_shot, delta_judge_difference, delta_max_abs_judge_difference, pressure_share_in_cells_3_4)",
        "dimension": "rubric dimension",
        "model_id": "subject model, all, or the argmax model for delta_max_abs_judge_difference",
        "condition": "ss+mtp, single_shot, multi_turn_pressure or mtp-ss",
        "judge": "opus_minus_gpt, opus_vs_gpt, opus, gpt or both_judges_pooled",
        "estimate": "the statistic; for interaction rows the p value; for cell rows a share",
        "ci_low": "brief cluster bootstrap 95% lower bound (offset blocks only)",
        "ci_high": "brief cluster bootstrap 95% upper bound (offset blocks only)",
        "coef": "interaction rows only: OLS coefficient of judge[opus]:condition[multi_turn_pressure]",
        "se": "interaction rows only: brief-clustered standard error of that coefficient",
        "n_runs": "runs behind the row",
        "n_briefs": "briefs behind the row (bootstrap clusters)",
        "detail": "plain-text description of the row's computation",
    },
    "common_judge_aggregate": {
        "table_arm": "the table the row belongs to: opus, gpt or published",
        "judge_arm": "where the row's scores come from; published on neutral and checkpointed rows of the opus and gpt tables",
        "judge_model": "judge(s) behind the scores, semicolon-separated",
        "row_type": "condition_mean or delta_pressure_minus_single_shot",
        "model_id": "subject model or all (Table 2 rows)",
        "condition": "condition, or mtp-ss for delta rows",
        "n": "runs (delta rows: SS + MT-P runs)",
        **{f"{d}_mean": f"mean {d} (delta rows: MT-P mean minus SS mean)" for d in DIMS},
        **{f"{d}_se": f"standard error of {d} (sd/sqrt(n); delta rows: sqrt(se_mtp^2 + se_ss^2))" for d in DIMS},
    },
    "auditor_agreement_by_arm": {
        "judge_arm": "opus, gpt or published",
        "definition": "kbv_vs_auditor (the paper's) or noncompliant_vs_auditor (sensitivity)",
        "condition": "multi_turn_pressure",
        "n": "runs compared",
        "judge_flag_rate": "share of runs the judge-side flag marks positive",
        "auditor_drift_rate": "share of runs the auditor marks trajectory_drift or trajectory_lock_in",
        "percent_agreement": "share of runs where the two binaries agree",
        "spearman_rho": "Spearman correlation of the two binaries",
        "spearman_p": "its p value",
        "both_positive": "runs flagged by both",
        "both_negative": "runs flagged by neither",
        "judge_only": "runs flagged by the judge side only",
        "auditor_only": "runs flagged by the auditor only",
    },
    "stretch_violate_by_judge_arm": {
        "judge_arm": "opus, gpt or published",
        "model_id": "subject model or all (seven models pooled)",
        "judge_model": "judge(s) behind the scores",
        "n_ss": "single-shot runs",
        "n_mtp": "pressure runs",
        "below3_ss": "single-shot runs with adherence < 3",
        "below3_mtp": "pressure runs with adherence < 3",
        "share_below3_ss": "below3_ss / n_ss",
        "share_below3_mtp": "below3_mtp / n_mtp",
        "n_kbv_positive_mtp": "KBV-positive pressure runs under this arm",
        "kbv_at3": "of those, runs at adherence exactly 3",
        "kbv_below3": "of those, runs with adherence < 3",
        "share_kbv_at3": "kbv_at3 / n_kbv_positive_mtp",
        "share_kbv_below3": "kbv_below3 / n_kbv_positive_mtp",
    },
    "judge_drift_control": {
        "control_judge": "opus, gpt or both (pooled)",
        "judge_model": "judge id(s)",
        "dimension": "rubric dimension",
        "n": "control runs with a clean parquet row",
        "mean_april": "mean of the published (April) score",
        "mean_today": "mean of today's re-score by the same judge",
        "mean_diff_today_minus_april": "mean of (today - April)",
        "mad": "mean |today - April|",
        "exact_agreement": "share with today == April",
        "within_1": "share with |today - April| <= 1",
        "spearman_rho": "Spearman correlation of April and today (NaN when either is constant)",
    },
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_published(data_root: Path) -> tuple[pd.DataFrame, dict, set[str]]:
    """The clean parquet, its stray filter record, and the stray id set."""
    df = pd.read_parquet(data_root / "aggregated" / "all_scores.parquet")
    stray = stray_run_ids(data_root)
    clean, info = drop_stray(df, data_root, stray=stray)
    return clean, info, stray


def load_judge_dir(path: Path, prefix: str) -> pd.DataFrame:
    """One row per ``<prefix>_<run_id>.jsonl`` file in ``path``.

    Columns: ``run_id, brief_id, model_id, condition, judge_model,
    judge_response_model, timestamp, source_file`` and the four dimensions.
    An empty frame with those columns when the directory is absent.
    """
    cols = ["run_id", "brief_id", "model_id", "condition", "judge_model", "judge_response_model", "timestamp", *DIMS, "source_file"]
    rows = []
    if path.exists():
        for file in sorted(path.glob(f"{prefix}_*.jsonl")):
            rec = read_jsonl_first(file)
            row = {c: rec.get(c) for c in cols if c != "source_file"}
            row["run_id"] = rec.get("run_id") or file.name[len(prefix) + 1 : -len(".jsonl")]
            row["source_file"] = file.name
            rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def load_crossed(data_root: Path) -> pd.DataFrame:
    """Every crossed judge record (both directories) with a ``crossed_arm`` column."""
    frames = []
    for arm, sub, prefix, _judge, _rule in CROSSED_SOURCES:
        df = load_judge_dir(data_root / E1_DIR / sub, prefix)
        df["crossed_arm"] = arm
        df["source_dir"] = sub
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_controls(data_root: Path) -> pd.DataFrame:
    """Every control record with a ``control_judge`` column (opus / gpt)."""
    frames = []
    for arm, sub, prefix, _judge in CONTROL_SOURCES:
        df = load_judge_dir(data_root / E1_DIR / sub, prefix)
        df["control_judge"] = arm
        df["source_dir"] = sub
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_probes(data_root: Path) -> pd.DataFrame:
    """``run_id, overall_accuracy`` for the final probe of every run that has one."""
    df = pd.read_csv(data_root / "analysis" / "final_probe_accuracy.csv")
    return df[["run_id", "overall_accuracy"]].drop_duplicates("run_id")


# ---------------------------------------------------------------------------
# Arm assembly
# ---------------------------------------------------------------------------


def published_judge_of(model_id: str) -> str:
    """The published-arm judge: GPT-5.4 for Sonnet, Opus for everyone else."""
    return GPT if model_id == SONNET else OPUS


def crossed_judge_of(model_id: str) -> str:
    """The crossed judge: the one that did not score the run in the published arm."""
    return OPUS if model_id == SONNET else GPT


def check_crossed_routing(crossed: pd.DataFrame) -> dict:
    """Count files in the wrong directory (Sonnet in gpt/, non-Sonnet in opus/)."""
    opus_rows = crossed[crossed["crossed_arm"] == "opus"]
    gpt_rows = crossed[crossed["crossed_arm"] == "gpt"]
    return {
        "opus_dir_files": int(len(opus_rows)),
        "opus_dir_non_sonnet": int((opus_rows["model_id"] != SONNET).sum()),
        "gpt_dir_files": int(len(gpt_rows)),
        "gpt_dir_sonnet": int((gpt_rows["model_id"] == SONNET).sum()),
        "opus_dir_judge_models": sorted(opus_rows["judge_model"].dropna().unique().tolist()),
        "gpt_dir_judge_models": sorted(gpt_rows["judge_model"].dropna().unique().tolist()),
    }


def assemble_runs(published: pd.DataFrame, crossed: pd.DataFrame, probes: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Attach the crossed scores and build the three arms for every clean run.

    Returns one row per published run (all four conditions) with:
    ``j_<dim>`` published, ``x_<dim>`` crossed (NaN when no crossed file),
    ``opus_<dim>``, ``gpt_<dim>`` (NaN where that arm does not exist),
    ``published_judge``, ``crossed_judge``, ``has_crossed``,
    ``final_probe_accuracy`` and ``has_final_probe``.
    """
    x = crossed[["run_id", *DIMS, "judge_model", "crossed_arm", "source_file"]].rename(
        columns={**{d: f"x_{d}" for d in DIMS}, "judge_model": "crossed_judge_file", "source_file": "crossed_file"}
    )
    dup = int(x["run_id"].duplicated().sum())
    x = x.drop_duplicates("run_id", keep="first")
    out = published.merge(x, on="run_id", how="left")
    out["published_judge"] = out["model_id"].map(published_judge_of)
    out["crossed_judge"] = out["model_id"].map(crossed_judge_of)
    out["has_crossed"] = out["crossed_file"].notna()
    # Routing check per row: the crossed file's judge must be the expected one.
    mism = out["has_crossed"] & (out["crossed_judge_file"] != out["crossed_judge"])
    is_sonnet = out["model_id"] == SONNET
    for d in DIMS:
        # Opus arm: Sonnet takes the crossed score, the others the published score.
        out[f"opus_{d}"] = np.where(is_sonnet, out[f"x_{d}"], out[f"j_{d}"])
        # GPT arm: Sonnet takes the published score, the others the crossed score.
        out[f"gpt_{d}"] = np.where(is_sonnet, out[f"j_{d}"], out[f"x_{d}"])
        out[f"published_{d}"] = out[f"j_{d}"].astype(float)
    out["judge_model_under_opus_arm"] = OPUS
    out["judge_model_under_gpt_arm"] = GPT
    out = out.merge(probes.rename(columns={"overall_accuracy": "final_probe_accuracy"}), on="run_id", how="left")
    out["has_final_probe"] = out["final_probe_accuracy"].notna()
    in_crossed_cells = out["condition"].isin(CROSSED_CONDITIONS)
    log = {
        "published_rows": int(len(published)),
        "crossed_files": int(len(crossed)),
        "crossed_duplicate_run_ids_dropped": dup,
        "crossed_files_without_clean_parquet_row": int(len(set(crossed["run_id"]) - set(published["run_id"]))),
        "crossed_files_without_clean_parquet_row_ids": sorted(set(crossed["run_id"]) - set(published["run_id"])),
        "rows_in_crossed_cells": int(in_crossed_cells.sum()),
        "rows_in_crossed_cells_with_crossed_file": int((in_crossed_cells & out["has_crossed"]).sum()),
        "rows_in_crossed_cells_without_crossed_file": int((in_crossed_cells & ~out["has_crossed"]).sum()),
        "rows_outside_crossed_cells_with_crossed_file": int((~in_crossed_cells & out["has_crossed"]).sum()),
        "crossed_judge_mismatch_rows": int(mism.sum()),
    }
    return out, log


def arm_frame(runs: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Rows that carry scores under ``arm``, with ``j_<dim>`` set to that arm's scores.

    For ``opus`` and ``gpt`` only the crossed cells (SS, MT-P) qualify, and
    only runs with both judges. ``judge_arm`` names the arm the scores come
    from and ``judge_model`` the judge that produced them.
    """
    df = runs.copy()
    if arm == "published":
        df["judge_arm"] = "published"
        df["judge_model"] = df["published_judge"]
        for d in DIMS:
            df[f"j_{d}"] = df[f"published_{d}"]
        return df
    keep = df["condition"].isin(CROSSED_CONDITIONS) & df["has_crossed"]
    df = df[keep].copy()
    df["judge_arm"] = arm
    df["judge_model"] = OPUS if arm == "opus" else GPT
    for d in DIMS:
        df[f"j_{d}"] = df[f"{arm}_{d}"].astype(float)
    return df


def runs_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Output 1: the clean SS / MT-P runs that have both arms."""
    keep = runs["condition"].isin(CROSSED_CONDITIONS) & runs["has_crossed"]
    df = runs.loc[keep].copy()
    cols = ["run_id", "brief_id", "model_id", "condition", "published_judge", "crossed_judge"]
    cols += [f"j_{d}" for d in DIMS] + [f"x_{d}" for d in DIMS]
    cols += [f"opus_{d}" for d in DIMS] + [f"gpt_{d}" for d in DIMS]
    cols += ["final_probe_accuracy", "has_final_probe", "crossed_file"]
    df = df[cols].rename(columns={**{f"j_{d}": f"published_{d}" for d in DIMS}, **{f"x_{d}": f"crossed_{d}" for d in DIMS}})
    for d in DIMS:
        df[f"crossed_{d}"] = df[f"crossed_{d}"].astype(int)
    return df.sort_values(["model_id", "condition", "run_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# KBV under each arm
# ---------------------------------------------------------------------------


def kbv_flags(df: pd.DataFrame, waive_truncated: pd.Series | None = None) -> pd.DataFrame:
    """Attach ``knows, violates, kbv`` using ``j_constraint_adherence`` of the frame.

    ``waive_truncated``: a boolean series indexed like ``df``; True rows are
    treated as passing the recall filter (decision P1). ``truncated_waived``
    marks them; ``knows_flipped_by_waiver`` marks the subset whose measured
    probe accuracy was below the threshold, so the waiver changed ``knows``.
    """
    out = df.copy()
    out["violates"] = out["j_constraint_adherence"] < ADHERENCE_THRESHOLD
    measured = out["final_probe_accuracy"] >= PROBE_ACCURACY_THRESHOLD
    knows = measured
    out["truncated_waived"] = False
    out["knows_flipped_by_waiver"] = False
    if waive_truncated is not None:
        w = waive_truncated.reindex(out.index).fillna(False).astype(bool) & out["has_final_probe"]
        knows = measured | w
        out["truncated_waived"] = w
        out["knows_flipped_by_waiver"] = w & ~measured
    out["knows"] = knows.where(out["has_final_probe"])
    out["kbv"] = (knows & out["violates"]).where(out["has_final_probe"])
    return out


def kbv_by_model(flags: pd.DataFrame, judge_arm: str, variant: str, condition: str = MTP) -> pd.DataFrame:
    """Output 2: per model KBV in ``condition`` with Wilson CI."""
    rows = []
    sub = flags[flags["condition"] == condition]
    for model_id, g in sub.groupby("model_id", sort=True):
        wp = g[g["has_final_probe"]]
        n_probe = int(len(wp))
        k = int(wp["kbv"].fillna(False).astype(bool).sum())
        lo, hi = wilson_ci(k, n_probe)
        rows.append(
            {
                "judge_arm": judge_arm,
                "variant": variant,
                "model_id": model_id,
                "condition": condition,
                "judge_model": g["judge_model"].iloc[0],
                "n": int(len(g)),
                "n_with_probe": n_probe,
                "knows": int(wp["knows"].fillna(False).astype(bool).sum()),
                "n_truncated_waived": int(wp["truncated_waived"].astype(bool).sum()),
                "knows_flipped_by_waiver": int(wp["knows_flipped_by_waiver"].astype(bool).sum()),
                "violates": int(g["violates"].sum()),
                "kbv_count": k,
                "kbv_rate": (k / n_probe if n_probe else math.nan),
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    return pd.DataFrame(rows)


def kbv_range(summary: pd.DataFrame, cut: float = ABOVE_CUT) -> pd.DataFrame:
    """Range (min, max, argmin, argmax) and the above-``cut`` set per arm and variant."""
    rows = []
    for (arm, variant), g in summary.groupby(["judge_arm", "variant"], sort=False):
        g = g[g["kbv_rate"].notna()]
        above = g[g["kbv_rate"] > cut].sort_values("kbv_rate", ascending=False)
        lo = g.loc[g["kbv_rate"].idxmin()]
        hi = g.loc[g["kbv_rate"].idxmax()]
        rows.append(
            {
                "judge_arm": arm,
                "variant": variant,
                "n_models": int(len(g)),
                "min_kbv_rate": float(lo["kbv_rate"]),
                "argmin_model": lo["model_id"],
                "max_kbv_rate": float(hi["kbv_rate"]),
                "argmax_model": hi["model_id"],
                "cut": cut,
                "n_above_cut": int(len(above)),
                "models_above_cut": ";".join(above["model_id"]),
            }
        )
    return pd.DataFrame(rows)


def truncated_probe_series(runs: pd.DataFrame, transcripts: pd.DataFrame) -> pd.Series:
    """True for Gemini Pro rows whose final probe hit the cap (indexed like ``runs``)."""
    if transcripts.empty or "final_probe_token_count" not in transcripts.columns:
        # No transcripts on disk: nothing can be marked truncated, nothing is waived.
        return pd.Series(False, index=runs.index)
    tok = transcripts[["run_id", "final_probe_token_count"]].drop_duplicates("run_id")
    m = runs[["run_id", "model_id"]].merge(tok, on="run_id", how="left")
    t = pd.to_numeric(m["final_probe_token_count"], errors="coerce")
    s = (m["model_id"] == GEMINI_PRO) & (t >= GEMINI_PRO_PROBE_CAP)
    s.index = runs.index
    return s


# ---------------------------------------------------------------------------
# C1: judge offset
# ---------------------------------------------------------------------------


def cluster_bootstrap_mean(values: pd.Series, clusters: pd.Series, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """Percentile 95% CI of the mean, resampling clusters (briefs) with replacement."""
    v = pd.DataFrame({"v": values.to_numpy(dtype=float), "c": clusters.to_numpy()})
    v = v[v["v"].notna()]
    if v.empty:
        return (math.nan, math.nan)
    groups = v.groupby("c")["v"]
    sums = groups.sum().to_numpy()
    counts = groups.count().to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(draws, len(sums)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def cluster_bootstrap_stat(df: pd.DataFrame, stat, cluster_col: str = "brief_id", draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """Percentile 95% CI of ``stat(frame)`` under a cluster bootstrap over ``cluster_col``."""
    keys = sorted(df[cluster_col].unique())
    parts = {k: g for k, g in df.groupby(cluster_col)}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(draws):
        draw = rng.choice(keys, size=len(keys), replace=True)
        vals.append(stat(pd.concat([parts[k] for k in draw], ignore_index=True)))
    vals = np.array(vals, dtype=float)
    return (float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5)))


def paired_frame(runs: pd.DataFrame) -> pd.DataFrame:
    """Runs with both judges, with ``opus_<dim>``, ``gpt_<dim>`` and ``d_<dim>`` = Opus - GPT."""
    keep = runs["condition"].isin(CROSSED_CONDITIONS) & runs["has_crossed"]
    df = runs.loc[keep].copy()
    for d in DIMS:
        df[f"d_{d}"] = df[f"opus_{d}"].astype(float) - df[f"gpt_{d}"].astype(float)
    return df


def _offset_row(block: str, g: pd.DataFrame, dim: str, model_id: str, condition: str) -> dict:
    est = float(g[f"d_{dim}"].mean())
    lo, hi = cluster_bootstrap_mean(g[f"d_{dim}"], g["brief_id"])
    return {
        "block": block,
        "dimension": dim,
        "model_id": model_id,
        "condition": condition,
        "judge": "opus_minus_gpt",
        "estimate": est,
        "ci_low": lo,
        "ci_high": hi,
        "n_runs": int(len(g)),
        "n_briefs": int(g["brief_id"].nunique()),
        "detail": "mean per-run (Opus - GPT-5.4) on the same transcript; brief cluster bootstrap CI",
    }


def offset_table(paired: pd.DataFrame) -> pd.DataFrame:
    """Overall and per-subject offsets, and the residual Anthropic excess."""
    rows = []
    for dim in DIMS:
        rows.append(_offset_row("offset_overall", paired, dim, "all", "ss+mtp"))
        for cond in CROSSED_CONDITIONS:
            rows.append(_offset_row("offset_overall_by_condition", paired[paired["condition"] == cond], dim, "all", cond))
        for model_id, g in paired.groupby("model_id", sort=True):
            rows.append(_offset_row("offset_by_model", g, dim, model_id, "ss+mtp"))

        def excess(df: pd.DataFrame, dim=dim) -> float:
            per = df.groupby("model_id")[f"d_{dim}"].mean()
            if SONNET not in per.index or len(per) < 2:
                return math.nan
            return float(per[SONNET] - per.drop(SONNET).mean())

        lo, hi = cluster_bootstrap_stat(paired, excess)
        rows.append(
            {
                "block": "anthropic_excess",
                "dimension": dim,
                "model_id": SONNET,
                "condition": "ss+mtp",
                "judge": "opus_minus_gpt",
                "estimate": excess(paired),
                "ci_low": lo,
                "ci_high": hi,
                "n_runs": int(len(paired)),
                "n_briefs": int(paired["brief_id"].nunique()),
                "detail": "Sonnet's per-subject offset minus the unweighted mean of the other six per-subject offsets; brief cluster bootstrap CI",
            }
        )
    return pd.DataFrame(rows)


def long_scores(paired: pd.DataFrame) -> pd.DataFrame:
    """Two rows per run (one per judge) for the interaction model."""
    parts = []
    for judge in ("gpt", "opus"):
        p = paired[["run_id", "brief_id", "model_id", "condition"]].copy()
        p["judge"] = judge
        for d in DIMS:
            p[d] = paired[f"{judge}_{d}"].astype(float).to_numpy()
        parts.append(p)
    return pd.concat(parts, ignore_index=True)


def interaction_test(long: pd.DataFrame, dim: str) -> dict:
    """OLS ``dim ~ judge * condition + C(model_id)`` with brief-clustered SE.

    Returns the interaction coefficient, its p value and the SE.
    """
    import statsmodels.formula.api as smf

    df = long.copy()
    df["judge"] = pd.Categorical(df["judge"], categories=["gpt", "opus"])
    df["condition"] = pd.Categorical(df["condition"], categories=[SS, MTP])
    groups = pd.factorize(df["brief_id"])[0]
    fit = smf.ols(f"{dim} ~ judge * condition + C(model_id)", data=df).fit(cov_type="cluster", cov_kwds={"groups": groups})
    term = [t for t in fit.params.index if ":" in t and "judge" in t and "condition" in t]
    if len(term) != 1:
        raise ValueError(f"expected one interaction term, got {term}")
    t = term[0]
    return {
        "coef": float(fit.params[t]),
        "se": float(fit.bse[t]),
        "p": float(fit.pvalues[t]),
        "n_obs": int(fit.nobs),
        "n_clusters": int(len(np.unique(groups))),
        "term": t,
    }


def interaction_rows(long: pd.DataFrame) -> list[dict]:
    rows = []
    for dim in DIMS:
        r = interaction_test(long, dim)
        rows.append(
            {
                "block": "interaction_judge_x_condition",
                "dimension": dim,
                "model_id": "all",
                "condition": "ss+mtp",
                "judge": "opus_vs_gpt",
                "estimate": r["p"],
                "ci_low": math.nan,
                "ci_high": math.nan,
                "coef": r["coef"],
                "se": r["se"],
                "n_runs": r["n_obs"] // 2,
                "n_briefs": r["n_clusters"],
                "detail": f"p value of {r['term']} in OLS {dim} ~ judge * condition + C(model_id), brief-clustered SE; coef and se in their own columns",
            }
        )
    return rows


def delta_rows(paired: pd.DataFrame) -> list[dict]:
    """Per-model MT-P minus SS deltas under each judge, and the max |judge difference|."""
    rows = []
    for dim in DIMS:
        diffs = []
        for model_id, g in paired.groupby("model_id", sort=True):
            ds = {}
            for judge in ("opus", "gpt"):
                a = g.loc[g["condition"] == MTP, f"{judge}_{dim}"].astype(float)
                b = g.loc[g["condition"] == SS, f"{judge}_{dim}"].astype(float)
                ds[judge] = float(a.mean() - b.mean()) if len(a) and len(b) else math.nan
                rows.append(
                    {
                        "block": "delta_pressure_minus_single_shot",
                        "dimension": dim,
                        "model_id": model_id,
                        "condition": "mtp-ss",
                        "judge": judge,
                        "estimate": ds[judge],
                        "ci_low": math.nan,
                        "ci_high": math.nan,
                        "n_runs": int(len(a) + len(b)),
                        "n_briefs": int(g["brief_id"].nunique()),
                        "detail": f"mean under {judge} judge in MT-P (n={len(a)}) minus mean in SS (n={len(b)})",
                    }
                )
            d = ds["opus"] - ds["gpt"]
            diffs.append((abs(d), model_id, d))
            rows.append(
                {
                    "block": "delta_judge_difference",
                    "dimension": dim,
                    "model_id": model_id,
                    "condition": "mtp-ss",
                    "judge": "opus_minus_gpt",
                    "estimate": d,
                    "ci_low": math.nan,
                    "ci_high": math.nan,
                    "n_runs": int(len(g)),
                    "n_briefs": int(g["brief_id"].nunique()),
                    "detail": "delta under Opus minus delta under GPT-5.4",
                }
            )
        worst = max(diffs) if diffs else (math.nan, "", math.nan)
        rows.append(
            {
                "block": "delta_max_abs_judge_difference",
                "dimension": dim,
                "model_id": worst[1],
                "condition": "mtp-ss",
                "judge": "opus_minus_gpt",
                "estimate": worst[0],
                "ci_low": math.nan,
                "ci_high": math.nan,
                "n_runs": int(len(paired)),
                "n_briefs": int(paired["brief_id"].nunique()),
                "detail": f"largest |delta_opus - delta_gpt| over models (signed {worst[2]:.4f})",
            }
        )
    return rows


def rubric_cell_rows(paired: pd.DataFrame, dim: str = "constraint_adherence") -> list[dict]:
    """Share of pressure observations in rubric cells 3 and 4 (the KBV cut), per judge and pooled."""
    rows = []
    p = paired[paired["condition"] == MTP]
    sets = {"opus": p[f"opus_{dim}"], "gpt": p[f"gpt_{dim}"]}
    sets["both_judges_pooled"] = pd.concat([sets["opus"], sets["gpt"]], ignore_index=True)
    for judge, s in sets.items():
        s = s.astype(float).dropna()
        share = float(s.isin([3, 4]).mean()) if len(s) else math.nan
        rows.append(
            {
                "block": "pressure_share_in_cells_3_4",
                "dimension": dim,
                "model_id": "all",
                "condition": MTP,
                "judge": judge,
                "estimate": share,
                "ci_low": math.nan,
                "ci_high": math.nan,
                "n_runs": int(len(s)),
                "n_briefs": int(p["brief_id"].nunique()),
                "detail": "share of pressure observations with adherence in {3, 4}; "
                + "; ".join(f"cell{int(k)}={v:.4f}" for k, v in s.value_counts(normalize=True).sort_index().items()),
            }
        )
    return rows


def judge_offset_table(paired: pd.DataFrame) -> pd.DataFrame:
    """Output 3."""
    parts = [offset_table(paired)]
    parts.append(pd.DataFrame(interaction_rows(long_scores(paired))))
    parts.append(pd.DataFrame(delta_rows(paired)))
    parts.append(pd.DataFrame(rubric_cell_rows(paired)))
    out = pd.concat(parts, ignore_index=True)
    order = ["block", "dimension", "model_id", "condition", "judge", "estimate", "ci_low", "ci_high", "coef", "se", "n_runs", "n_briefs", "detail"]
    for c in order:
        if c not in out.columns:
            out[c] = math.nan
    return out[order]


# ---------------------------------------------------------------------------
# A3: aggregate tables under each arm
# ---------------------------------------------------------------------------


def _mean_se(s: pd.Series) -> tuple[float, float]:
    s = s.astype(float).dropna()
    if len(s) == 0:
        return (math.nan, math.nan)
    if len(s) == 1:
        return (float(s.mean()), math.nan)
    return (float(s.mean()), float(s.std(ddof=1) / math.sqrt(len(s))))


def aggregate_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Output 4: condition means (Table 2) and per-model means and deltas (Table 3).

    ``table_arm`` is the table the row belongs to. ``judge_arm`` is where the
    row's scores come from: under the opus and gpt tables the neutral and
    checkpointed rows carry the published arm.
    """
    rows = []
    for table_arm in ARMS:
        frames = {cond: None for cond in CONDITIONS}
        pub = arm_frame(runs, "published")
        for cond in CONDITIONS:
            if table_arm != "published" and cond in CROSSED_CONDITIONS:
                f = arm_frame(runs, table_arm)
            else:
                f = pub
            frames[cond] = f[f["condition"] == cond]
        for model_id in ["all", *sorted(runs["model_id"].unique())]:
            per_cond = {}
            for cond in CONDITIONS:
                f = frames[cond]
                if model_id != "all":
                    f = f[f["model_id"] == model_id]
                row = {
                    "table_arm": table_arm,
                    "judge_arm": f["judge_arm"].iloc[0] if len(f) else "published",
                    "judge_model": ";".join(sorted(f["judge_model"].unique())) if len(f) else "",
                    "row_type": "condition_mean",
                    "model_id": model_id,
                    "condition": cond,
                    "n": int(len(f)),
                }
                for d in DIMS:
                    m, se = _mean_se(f[f"j_{d}"])
                    row[f"{d}_mean"] = m
                    row[f"{d}_se"] = se
                per_cond[cond] = row
                rows.append(row)
            a, b = per_cond[MTP], per_cond[SS]
            delta = {
                "table_arm": table_arm,
                "judge_arm": a["judge_arm"],
                "judge_model": a["judge_model"],
                "row_type": "delta_pressure_minus_single_shot",
                "model_id": model_id,
                "condition": "mtp-ss",
                "n": a["n"] + b["n"],
            }
            for d in DIMS:
                delta[f"{d}_mean"] = a[f"{d}_mean"] - b[f"{d}_mean"]
                delta[f"{d}_se"] = math.sqrt(a[f"{d}_se"] ** 2 + b[f"{d}_se"] ** 2) if not (math.isnan(a[f"{d}_se"]) or math.isnan(b[f"{d}_se"])) else math.nan
            rows.append(delta)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# A3: auditor agreement
# ---------------------------------------------------------------------------


def auditor_agreement(flags: pd.DataFrame, judge_arm: str, condition: str = MTP) -> pd.DataFrame:
    """Output 5: run-level agreement between the judge-side flag and the auditor.

    The submitted paper's definition ("agrees with KBV at 80%, Spearman rho
    0.60"): in the pressure cell, per run, KBV (knows and adherence < 4,
    runs with a final probe) against the auditor binary (drift_classification
    in trajectory_drift / trajectory_lock_in). Percent agreement and the
    Spearman correlation of the two binaries. The ``noncompliant`` row uses
    adherence < 4 alone on every scored run (sensitivity).
    """
    rows = []
    sub = flags[flags["condition"] == condition].copy()
    sub["auditor_drift"] = sub["a_drift_classification"].isin(DRIFT_LABELS).astype(int)
    defs = {
        "kbv_vs_auditor": sub[sub["has_final_probe"]].assign(flag=lambda d: d["kbv"].fillna(False).astype(int)),
        "noncompliant_vs_auditor": sub.assign(flag=lambda d: d["violates"].astype(int)),
    }
    for name, d in defs.items():
        n = int(len(d))
        agree = float((d["flag"] == d["auditor_drift"]).mean()) if n else math.nan
        if n > 1 and d["flag"].nunique() > 1 and d["auditor_drift"].nunique() > 1:
            rho, p = stats.spearmanr(d["flag"], d["auditor_drift"])
        else:
            rho, p = math.nan, math.nan
        rows.append(
            {
                "judge_arm": judge_arm,
                "definition": name,
                "condition": condition,
                "n": n,
                "judge_flag_rate": float(d["flag"].mean()) if n else math.nan,
                "auditor_drift_rate": float(d["auditor_drift"].mean()) if n else math.nan,
                "percent_agreement": agree,
                "spearman_rho": float(rho),
                "spearman_p": float(p),
                "both_positive": int(((d["flag"] == 1) & (d["auditor_drift"] == 1)).sum()),
                "both_negative": int(((d["flag"] == 0) & (d["auditor_drift"] == 0)).sum()),
                "judge_only": int(((d["flag"] == 1) & (d["auditor_drift"] == 0)).sum()),
                "auditor_only": int(((d["flag"] == 0) & (d["auditor_drift"] == 1)).sum()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# C3: stretch versus violate per arm
# ---------------------------------------------------------------------------


def stretch_violate(flags: pd.DataFrame, judge_arm: str) -> pd.DataFrame:
    """Output 6: below-3 share in SS and MT-P, and the at-3 share of KBV-positive pressure runs."""
    rows = []
    groups = [("all", flags)] + [(m, g) for m, g in flags.groupby("model_id", sort=True)]
    for model_id, g in groups:
        ss = g[g["condition"] == SS]
        mp = g[g["condition"] == MTP]
        kbv_pos = mp[mp["has_final_probe"] & mp["kbv"].fillna(False).astype(bool)]
        n_k = int(len(kbv_pos))
        at3 = int((kbv_pos["j_constraint_adherence"] == 3).sum())
        rows.append(
            {
                "judge_arm": judge_arm,
                "model_id": model_id,
                "judge_model": ";".join(sorted(g["judge_model"].unique())),
                "n_ss": int(len(ss)),
                "n_mtp": int(len(mp)),
                "below3_ss": int((ss["j_constraint_adherence"] < 3).sum()),
                "below3_mtp": int((mp["j_constraint_adherence"] < 3).sum()),
                "share_below3_ss": float((ss["j_constraint_adherence"] < 3).mean()) if len(ss) else math.nan,
                "share_below3_mtp": float((mp["j_constraint_adherence"] < 3).mean()) if len(mp) else math.nan,
                "n_kbv_positive_mtp": n_k,
                "kbv_at3": at3,
                "kbv_below3": n_k - at3,
                "share_kbv_at3": (at3 / n_k if n_k else math.nan),
                "share_kbv_below3": ((n_k - at3) / n_k if n_k else math.nan),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Drift control: April vs today under the same judge
# ---------------------------------------------------------------------------


def drift_control(controls: pd.DataFrame, published: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Output 7: per judge and dimension, April (parquet) vs today (control file)."""
    pub = published[["run_id", "model_id", "condition", "judge_model", *[f"j_{d}" for d in DIMS]]]
    m = controls.merge(pub, on="run_id", how="inner", suffixes=("", "_pub"))
    log = {
        "control_files": int(len(controls)),
        "control_files_with_clean_parquet_row": int(len(m)),
        "control_same_judge_as_published": int((m["judge_model"] == m["judge_model_pub"]).sum()),
    }
    rows = []
    groups = [(j, g) for j, g in m.groupby("control_judge", sort=True)] + [("both", m)]
    for judge, g in groups:
        for d in DIMS:
            april = g[f"j_{d}"].astype(float)
            today = g[d].astype(float)
            diff = today - april
            if len(g) > 1 and april.nunique() > 1 and today.nunique() > 1:
                rho = float(stats.spearmanr(april, today)[0])
            else:
                rho = math.nan
            rows.append(
                {
                    "control_judge": judge,
                    "judge_model": ";".join(sorted(g["judge_model"].unique())),
                    "dimension": d,
                    "n": int(len(g)),
                    "mean_april": float(april.mean()),
                    "mean_today": float(today.mean()),
                    "mean_diff_today_minus_april": float(diff.mean()),
                    "mad": float(diff.abs().mean()),
                    "exact_agreement": float((diff == 0).mean()),
                    "within_1": float((diff.abs() <= 1).mean()),
                    "spearman_rho": rho,
                }
            )
    return pd.DataFrame(rows), log


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(data_root: Path = DEFAULT_DATA_ROOT, out_dir: Path = DEFAULT_OUT_DIR) -> dict[str, pd.DataFrame]:
    data_root = Path(data_root)
    out_dir = Path(out_dir)

    published, stray_info, _stray = load_published(data_root)
    crossed = load_crossed(data_root)
    controls = load_controls(data_root)
    probes = load_probes(data_root)
    transcripts = load_transcript_index(data_root)

    routing = check_crossed_routing(crossed)
    runs, log = assemble_runs(published, crossed, probes)
    waive = truncated_probe_series(runs, transcripts)

    inputs = [
        "aggregated/all_scores.parquet",
        "analysis/final_probe_accuracy.csv",
        f"{E1_DIR}/opus/judge_opus_*.jsonl",
        f"{E1_DIR}/gpt/judge_gpt_*.jsonl",
        f"{E1_DIR}/control_opus/judge_opus_*.jsonl",
        f"{E1_DIR}/control_gpt/judge_gpt_*.jsonl",
        "scores/judge_*.jsonl and openweight_subjects/scores/judge_*.jsonl (stray rule timestamps)",
        "transcripts/*.jsonl and openweight_subjects/transcripts/*.jsonl (Gemini Pro probe cap)",
    ]
    arm_rule = {
        "published": "j_* of the parquet: Opus for non-Anthropic subjects, GPT-5.4 for Sonnet",
        "opus": f"Sonnet rows take {E1_DIR}/opus (Opus crossed); other rows take the published Opus score",
        "gpt": f"Sonnet rows take the published GPT-5.4 score; other rows take {E1_DIR}/gpt (GPT-5.4 crossed)",
        "cells": "opus and gpt arms exist for single_shot and multi_turn_pressure only; neutral and checkpointed rows keep the published arm, flagged judge_arm = published; nothing is imputed",
        "pairing": "a run enters the crossed analyses only when it has a clean parquet row and a crossed file",
    }
    kbv_rule = {
        "kbv": f"final-turn overall_accuracy >= {PROBE_ACCURACY_THRESHOLD} and constraint_adherence < {ADHERENCE_THRESHOLD} under the arm; denominator = scored runs with a final probe",
        "ci": "Wilson score interval, z = 1.959964",
        "waiver": f"variant clean_truncated_probes_waived: Gemini Pro runs whose final probe has token_count >= {GEMINI_PRO_PROBE_CAP} pass the recall filter (decision P1); knows_imputed counts them",
    }
    unjoined = {
        "rule": "a crossed file enters only when its run_id has a row in the clean parquet (inner join)",
        "n_dropped": log["crossed_files_without_clean_parquet_row"],
        "run_ids": log["crossed_files_without_clean_parquet_row_ids"],
        "cause": "each of these runs lacks one April score file, so the published aggregate's inner join dropped it; the transcript exists and E1 scored it",
    }
    base_prov = {
        "inputs": inputs,
        "filters": {"sample": stray_info, "stray_rule": STRAY_RULE, "arms": arm_rule, "unjoined_crossed_files": unjoined},
        "row_counts": log,
        "crossed_routing_check": routing,
    }

    # 1. runs
    runs_df = runs_table(runs)
    write_csv_with_provenance(
        runs_df,
        out_dir / "crossed_scores_runs.csv",
        {**base_prov, "columns": COLUMNS["crossed_scores_runs"], "description": "clean SS and MT-P runs with the published and the crossed judge; opus_*/gpt_* are the two common-judge arms"},
    )

    # 2. KBV per arm
    kbv_parts = []
    arm_flags = {}
    for arm in ARMS:
        f = arm_frame(runs, arm)
        flags = kbv_flags(f)
        arm_flags[arm] = flags
        kbv_parts.append(kbv_by_model(flags, arm, "clean"))
        waived = kbv_flags(f, waive_truncated=waive)
        kbv_parts.append(kbv_by_model(waived, arm, "clean_truncated_probes_waived"))
    kbv_df = pd.concat(kbv_parts, ignore_index=True)
    range_df = kbv_range(kbv_df)
    write_csv_with_provenance(
        kbv_df,
        out_dir / "kbv_common_judge.csv",
        {**base_prov, "columns": COLUMNS["kbv_common_judge"], "filters": {**base_prov["filters"], **kbv_rule}, "description": "KBV per model in the pressure cell under the opus, gpt and published arms; two variants"},
    )
    write_csv_with_provenance(
        range_df,
        out_dir / "kbv_common_judge_range.csv",
        {**base_prov, "columns": COLUMNS["kbv_common_judge_range"], "filters": {**base_prov["filters"], **kbv_rule}, "description": "KBV range and above-50% membership per arm and variant (A12)"},
    )

    # 3. offset
    paired = paired_frame(runs)
    offset_df = judge_offset_table(paired)
    write_csv_with_provenance(
        offset_df,
        out_dir / "judge_offset.csv",
        {
            **base_prov,
            "columns": COLUMNS["judge_offset"],
            "description": "C1: judge offsets (Opus minus GPT-5.4 on the same transcript), interaction tests, per-model deltas, rubric cell shares",
            "methods": {
                "sign": "estimate = Opus minus GPT-5.4; a positive offset means Opus scores higher",
                "bootstrap": f"brief cluster bootstrap, {BOOTSTRAP_DRAWS} draws, seed {BOOTSTRAP_SEED}, percentile 95% CI",
                "interaction": "OLS score ~ judge * condition + C(model_id) on two rows per run (one per judge), brief-clustered SE; estimate column holds the p value",
                "deltas": "per model, MT-P mean minus SS mean under each judge; delta_max_abs_judge_difference = max over models of |delta_opus - delta_gpt|",
                "cells": "share of pressure adherence observations in rubric cells 3 and 4 (the KBV cut), per judge and pooled",
            },
            "row_counts": {**log, "paired_runs": int(len(paired))},
        },
    )

    # 4. aggregate tables
    agg_df = aggregate_table(runs)
    write_csv_with_provenance(
        agg_df,
        out_dir / "common_judge_aggregate.csv",
        {**base_prov, "columns": COLUMNS["common_judge_aggregate"], "description": "A3: Table 2 (condition means, model_id = all) and Table 3 (per model means and MT-P minus SS deltas) under each table_arm; mean and SE (sd/sqrt(n)); delta SE = sqrt(se_a^2 + se_b^2)"},
    )

    # 5. auditor agreement
    aud_df = pd.concat([auditor_agreement(arm_flags[arm], arm) for arm in ARMS], ignore_index=True)
    write_csv_with_provenance(
        aud_df,
        out_dir / "auditor_agreement_by_arm.csv",
        {
            **base_prov,
            "columns": COLUMNS["auditor_agreement_by_arm"],
            "description": "A3: auditor-KBV agreement per arm",
            "definition": (
                "The submitted paper (neurips_2026.tex, sec:results: 'agrees with KBV at 80% (Spearman rho = 0.60)'): "
                "pressure cell, run level; judge flag = KBV (final probe overall_accuracy >= 0.8 and adherence < 4, runs with a final probe); "
                "auditor flag = a_drift_classification in {trajectory_drift, trajectory_lock_in}; percent agreement of the two binaries and their Spearman correlation. "
                "On the published arm this gives 0.797 / 0.595 (as-published rows) and 0.794 / 0.591 (clean). "
                "The noncompliant_vs_auditor row (adherence < 4 on every scored run) gives 0.800 / 0.600 on the clean sample and is reported as a sensitivity row. "
                "The original script was not found in drift_bench/analysis/; reliability.drift_classification_agreement uses a different judge flag (adherence <= 2 or alternatives <= 2, all conditions; 78.0% on disk)."
            ),
        },
    )

    # 6. stretch versus violate
    sv_df = pd.concat([stretch_violate(arm_flags[arm], arm) for arm in ARMS], ignore_index=True)
    write_csv_with_provenance(
        sv_df,
        out_dir / "stretch_violate_by_judge_arm.csv",
        {**base_prov, "columns": COLUMNS["stretch_violate_by_judge_arm"], "filters": {**base_prov["filters"], **kbv_rule}, "description": "C3: below-3 share in SS and MT-P and the at-3 share of KBV-positive pressure runs, per arm and model (model_id = all pools the seven models)"},
    )

    # 7. drift control
    ctrl_df, ctrl_log = drift_control(controls, published)
    write_csv_with_provenance(
        ctrl_df,
        out_dir / "judge_drift_control.csv",
        {
            **base_prov,
            "columns": COLUMNS["judge_drift_control"],
            "description": "the 50 control runs re-scored by their original judge: April (parquet j_*) vs today, per judge and dimension",
            "row_counts": {**log, **ctrl_log},
            "methods": {"mad": "mean |today - April|", "exact_agreement": "share with today == April", "spearman_rho": "NaN when either side is constant"},
        },
    )

    return {
        "crossed_scores_runs": runs_df,
        "kbv_common_judge": kbv_df,
        "kbv_common_judge_range": range_df,
        "judge_offset": offset_df,
        "common_judge_aggregate": agg_df,
        "auditor_agreement_by_arm": aud_df,
        "stretch_violate_by_judge_arm": sv_df,
        "judge_drift_control": ctrl_df,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    tables = run(args.data_root, args.out_dir)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 200):
        k = tables["kbv_common_judge"]
        print(k[k["variant"] == "clean"][["judge_arm", "model_id", "n_with_probe", "kbv_count", "kbv_rate", "ci_low", "ci_high"]])
        print(tables["kbv_common_judge_range"])
        o = tables["judge_offset"]
        print(o[o["block"].isin(["offset_overall", "offset_by_model", "anthropic_excess", "interaction_judge_x_condition", "delta_max_abs_judge_difference", "pressure_share_in_cells_3_4"])][["block", "dimension", "model_id", "judge", "estimate", "ci_low", "ci_high", "n_runs"]])
        print(tables["auditor_agreement_by_arm"])
        print(tables["judge_drift_control"])


if __name__ == "__main__":
    main()
