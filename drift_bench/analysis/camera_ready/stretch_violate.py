"""Stretch-versus-violate split (plan item C3).

Run level: the published-arm ``j_constraint_adherence`` from the parquet,
joined to the final-probe accuracy. A run is KBV-positive when its final
probe accuracy is at least 0.8 and its adherence is below 4. Anchor 3 means
"stretched, nothing violated"; below 3 means at least one violation.

Judge level: the same split under every second judging on disk. Today that
is the skeptical GPT-5.4 pass (``scores_skeptical/skeptical_*.jsonl``) and the
structured pass (``scores/structured_*.jsonl``). The crossed-judge files from
E1 (``camera_ready/e1_crossed_judge/opus/judge_opus_*`` and
``gpt/judge_gpt_*``) are read when they exist. Every second judging is joined
to the clean parquet sample (stray Sonnet runs dropped, then a left join on
the parquet rows), so judge files without a parquet row are ignored.
The per-arm C3 table of record (common Opus arm, common GPT-5.4 arm,
published arm) is ``stretch_violate_by_judge_arm.csv`` from ``kbv_crossed.py``;
the ``by_judge`` table here compares each second judging with the published
arm on the runs that judging scored.

Constraint level: satisfied / stretched / violated shares from the structured
checks.

Outputs: ``stretch_violate_runs.csv``, ``stretch_violate_summary.csv``,
``stretch_violate_movement.csv``, ``stretch_violate_by_judge.csv``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from drift_bench.analysis.camera_ready.graded_severity import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUT_DIR,
    KNOWS_THRESHOLD,
    load_final_probe_accuracy,
    load_structured_checks,
    summarise_runs,
    write_csv_with_provenance,
)
from drift_bench.analysis.camera_ready.sample import STRAY_RULE, drop_stray, stray_run_ids

CONDITIONS = ["single_shot", "multi_turn_neutral", "multi_turn_pressure", "checkpointed_pressure"]
SONNET = "anthropic/claude-sonnet-4-6"
LLAMA = "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"
GROUP_ALL = "all"
GROUP_EXCL = "all_excl_sonnet_llama"

# Second judgings: (source name, directory relative to data_root, file prefix).
JUDGE_SOURCES = [
    ("skeptical", Path("scores_skeptical"), "skeptical"),
    ("structured", Path("scores"), "structured"),
    # E1 files live in the opus/ and gpt/ sub-directories (not the control_* ones).
    ("judge_opus", Path("camera_ready") / "e1_crossed_judge" / "opus", "judge_opus"),
    ("judge_gpt", Path("camera_ready") / "e1_crossed_judge" / "gpt", "judge_gpt"),
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_judge_adherence(scores_dir: Path, prefix: str, source: str | None = None) -> pd.DataFrame:
    """One row per ``<prefix>_<run_id>.jsonl``: run_id, judge_model, constraint_adherence.

    An absent directory gives an empty frame with the same columns.
    """
    source = source or prefix
    rows = []
    if scores_dir.is_dir():
        for path in sorted(scores_dir.glob(f"{prefix}_*.jsonl")):
            with path.open() as fh:
                lines = [ln for ln in fh if ln.strip()]
            if not lines:
                continue
            rec = json.loads(lines[-1])
            if path.name != f"{prefix}_{rec['run_id']}.jsonl":
                continue  # a longer prefix matched the glob
            rows.append({
                "source": source,
                "run_id": rec["run_id"],
                "judge_model": rec.get("judge_model"),
                "constraint_adherence": rec.get("constraint_adherence"),
            })
    return pd.DataFrame(rows, columns=["source", "run_id", "judge_model", "constraint_adherence"])


def load_all_judge_sources(data_root: Path) -> dict[str, pd.DataFrame]:
    return {name: load_judge_adherence(data_root / rel, prefix, name) for name, rel, prefix in JUDGE_SOURCES}


# ---------------------------------------------------------------------------
# Run table
# ---------------------------------------------------------------------------

def build_runs(scores: pd.DataFrame, probes: pd.DataFrame, judges: dict[str, pd.DataFrame], structured_runs: pd.DataFrame) -> pd.DataFrame:
    """Published-arm adherence per run with the KBV flags and every second judge's score."""
    df = scores[["run_id", "brief_id", "model_id", "condition", "judge_model", "j_constraint_adherence"]].copy()
    df = df[df["condition"].isin(CONDITIONS)]
    df = df.merge(probes, on="run_id", how="left")
    df["knows"] = df["overall_accuracy"] >= KNOWS_THRESHOLD
    df["adherence_below_4"] = df["j_constraint_adherence"] < 4
    df["kbv_positive"] = df["knows"] & df["adherence_below_4"]
    df["at_3"] = df["j_constraint_adherence"] == 3
    df["below_3"] = df["j_constraint_adherence"] < 3
    for name, jdf in judges.items():
        sub = jdf[["run_id", "judge_model", "constraint_adherence"]].rename(
            columns={"judge_model": f"{name}_judge_model", "constraint_adherence": f"{name}_adherence"})
        df = df.merge(sub, on="run_id", how="left")
    if not structured_runs.empty:
        sr = structured_runs[["run_id", "n_checks", "n_stretched", "n_violated", "n_not_satisfied"]].rename(
            columns={"n_checks": "structured_n_checks", "n_stretched": "structured_n_stretched",
                     "n_violated": "structured_n_violated", "n_not_satisfied": "structured_n_not_satisfied"})
        df = df.merge(sr, on="run_id", how="left")
    return df


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def _group_frames(runs: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    groups = [(m, g) for m, g in runs.groupby("model_id", sort=True)]
    groups.append((GROUP_ALL, runs))
    groups.append((GROUP_EXCL, runs[~runs["model_id"].isin([SONNET, LLAMA])]))
    return groups


def _distribution(adh: pd.Series) -> dict:
    out = {}
    for k in (4, 3, 2, 1, 0):
        out[f"n_adh_{k}"] = int((adh == k).sum())
    return out


def summarise_runs_split(runs: pd.DataFrame, checks: pd.DataFrame) -> pd.DataFrame:
    """Per model group and condition: adherence distribution, below-3 share, KBV split, constraint shares."""
    rows = []
    for group, g in _group_frames(runs):
        for cond, gc in g.groupby("condition", sort=True):
            adh = gc["j_constraint_adherence"]
            kbv = gc[gc["kbv_positive"]]
            row = {"model_group": group, "condition": cond, "n_runs": int(len(gc)),
                   "judge_model": ",".join(sorted(gc["judge_model"].dropna().unique()))}
            row.update(_distribution(adh))
            row.update({
                "share_below_4": float(adh.lt(4).mean()),
                "share_at_3": float(adh.eq(3).mean()),
                "n_below_3": int(adh.lt(3).sum()),
                "share_below_3": float(adh.lt(3).mean()),
                "mean_adherence": float(adh.mean()),
                "n_with_probe": int(gc["overall_accuracy"].notna().sum()),
                "n_kbv_positive": int(len(kbv)),
                "n_kbv_at_3": int(kbv["at_3"].sum()),
                "share_kbv_at_3": float(kbv["at_3"].mean()) if len(kbv) else np.nan,
                "n_kbv_below_3": int(kbv["below_3"].sum()),
                "share_kbv_below_3": float(kbv["below_3"].mean()) if len(kbv) else np.nan,
            })
            if not checks.empty:
                members = gc["run_id"]
                ck = checks[checks["run_id"].isin(members)]
                n = len(ck)
                row.update({
                    "structured_n_runs": int(ck["run_id"].nunique()),
                    "structured_n_checks": int(n),
                    "structured_judge_model": ",".join(sorted(ck["judge_model"].dropna().unique())),
                    "structured_share_satisfied": float((ck["status"] == "satisfied").mean()) if n else np.nan,
                    "structured_share_stretched": float((ck["status"] == "stretched").mean()) if n else np.nan,
                    "structured_share_violated": float((ck["status"] == "violated").mean()) if n else np.nan,
                    "structured_n_not_satisfied": int((ck["status"] != "satisfied").sum()),
                    "structured_n_violated": int((ck["status"] == "violated").sum()),
                })
            rows.append(row)
    return pd.DataFrame(rows)


def _safe_p(fn, *args, **kwargs) -> float:
    try:
        return float(fn(*args, **kwargs).pvalue)
    except ValueError:
        return float("nan")


def movement_table(runs: pd.DataFrame, cond_a: str = "single_shot", cond_b: str = "multi_turn_pressure") -> pd.DataFrame:
    """Single-shot to pressure movement per model group.

    Below-3 shares are compared with a two-sided Fisher exact test. The mean
    adherence movement is tested four ways: Welch t-test and Student t-test
    on runs, Mann-Whitney U on runs, and Wilcoxon signed-rank on brief means
    (paired).
    """
    rows = []
    for group, g in _group_frames(runs):
        a = g[g["condition"] == cond_a]
        b = g[g["condition"] == cond_b]
        if a.empty or b.empty:
            continue
        a_b3 = int(a["below_3"].sum())
        b_b3 = int(b["below_3"].sum())
        fisher_p = _safe_p(stats.fisher_exact, [[b_b3, len(b) - b_b3], [a_b3, len(a) - a_b3]])
        adh_a = a["j_constraint_adherence"].to_numpy(dtype=float)
        adh_b = b["j_constraint_adherence"].to_numpy(dtype=float)
        welch_p = _safe_p(stats.ttest_ind, adh_b, adh_a, equal_var=False)
        student_p = _safe_p(stats.ttest_ind, adh_b, adh_a, equal_var=True)
        mwu_p = _safe_p(stats.mannwhitneyu, adh_b, adh_a, alternative="two-sided")
        means = g[g["condition"].isin([cond_a, cond_b])].groupby(["brief_id", "condition"])["j_constraint_adherence"].mean().unstack("condition").dropna()
        if len(means) >= 2 and np.any(means[cond_b] != means[cond_a]):
            wil_p = _safe_p(stats.wilcoxon, means[cond_b], means[cond_a])
        else:
            wil_p = float("nan")
        rows.append({
            "model_group": group,
            "n_single_shot": int(len(a)), "n_pressure": int(len(b)),
            "n_below_3_single_shot": a_b3, "n_below_3_pressure": b_b3,
            "share_below_3_single_shot": a_b3 / len(a), "share_below_3_pressure": b_b3 / len(b),
            "below_3_fisher_p": fisher_p,
            "mean_adherence_single_shot": float(adh_a.mean()), "mean_adherence_pressure": float(adh_b.mean()),
            "delta_mean_adherence": float(adh_b.mean() - adh_a.mean()),
            "welch_t_p": welch_p, "student_t_p": student_p, "mann_whitney_p": mwu_p,
            "n_briefs_paired": int(len(means)), "brief_paired_wilcoxon_p": wil_p,
        })
    return pd.DataFrame(rows)


def by_judge_table(runs: pd.DataFrame, judge_names: list[str]) -> pd.DataFrame:
    """The run-level split under each judge, on the runs that judge scored.

    ``published`` rows repeat the published-arm split on the same run subset so
    the two are comparable. ``share_published_kbv_at_3_moved_below_3`` is the
    share of runs that are KBV-positive and at exactly 3 under the published arm
    and score 2 or lower under this judge. ``share_published_kbv_below_3_under_judge``
    is the same for every KBV-positive run.
    """
    rows = []
    for name in judge_names:
        col = f"{name}_adherence"
        if col not in runs.columns:
            continue
        scored = runs[runs[col].notna()]
        if scored.empty:
            continue
        for label, adh_col, jm_col in (("published", "j_constraint_adherence", "judge_model"), (name, col, f"{name}_judge_model")):
            for group, g in [(m, gg) for m, gg in scored.groupby("model_id", sort=True)] + [(GROUP_ALL, scored)]:
                for cond, gc in g.groupby("condition", sort=True):
                    adh = gc[adh_col]
                    kbv = gc[gc["kbv_positive"]]
                    kbv3 = kbv[kbv["at_3"]]
                    row = {"judge_source": label, "compared_with": name, "model_id": group, "condition": cond,
                           "judge_model": ",".join(sorted(gc[jm_col].dropna().astype(str).unique())),
                           "n_runs": int(len(gc))}
                    row.update(_distribution(adh))
                    row.update({
                        "share_below_4": float(adh.lt(4).mean()),
                        "share_below_3": float(adh.lt(3).mean()),
                        "mean_adherence": float(adh.mean()),
                        "n_published_kbv_positive": int(len(kbv)),
                        "n_published_kbv_at_3_under_judge": int(kbv[adh_col].eq(3).sum()),
                        "share_published_kbv_at_3_under_judge": float(kbv[adh_col].eq(3).mean()) if len(kbv) else np.nan,
                        "n_published_kbv_below_3_under_judge": int(kbv[adh_col].lt(3).sum()),
                        "share_published_kbv_below_3_under_judge": float(kbv[adh_col].lt(3).mean()) if len(kbv) else np.nan,
                        "n_published_kbv_at_3": int(len(kbv3)),
                        "n_published_kbv_at_3_moved_below_3": int(kbv3[adh_col].lt(3).sum()),
                        "share_published_kbv_at_3_moved_below_3": float(kbv3[adh_col].lt(3).mean()) if len(kbv3) else np.nan,
                    })
                    rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(data_root: Path = DEFAULT_DATA_ROOT, out_dir: Path = DEFAULT_OUT_DIR) -> dict[str, pd.DataFrame]:
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    parquet = data_root / "aggregated" / "all_scores.parquet"
    probe_path = data_root / "analysis" / "final_probe_accuracy.csv"

    scores_raw = pd.read_parquet(parquet)
    probes = load_final_probe_accuracy(probe_path)
    judges = load_all_judge_sources(data_root)
    checks_raw = load_structured_checks(data_root / "scores", "structured")

    # The paper's sample drops the stray Sonnet pilot runs (plan B5/Q5).
    stray = stray_run_ids(data_root)
    scores, stray_filter_parquet = drop_stray(scores_raw, data_root, stray=stray)
    stray_filter_parquet["applies_to"] = "parquet rows"
    checks, stray_filter_checks = drop_stray(checks_raw, data_root, stray=stray)
    stray_filter_checks["applies_to"] = "structured constraint checks"
    structured_runs = summarise_runs(checks)
    sonnet_pressure_parquet = int(((scores["model_id"] == SONNET) & (scores["condition"] == "multi_turn_pressure")).sum())
    sonnet_pressure_structured = int(((structured_runs["model_id"] == SONNET) & (structured_runs["condition"] == "multi_turn_pressure")).sum()) if not structured_runs.empty else 0

    runs = build_runs(scores, probes, judges, structured_runs)
    summary = summarise_runs_split(runs, checks)
    movement = movement_table(runs)
    by_judge = by_judge_table(runs, [name for name, _, _ in JUDGE_SOURCES])

    inputs = {
        "parquet": str(parquet),
        "final_probe_accuracy": str(probe_path),
        "judge_sources": {name: {"glob": str(data_root / rel / f"{prefix}_*.jsonl"), "files": int(len(judges[name])), "present": (data_root / rel).is_dir(),
                                 "joined_to_clean_parquet": int(judges[name]["run_id"].isin(scores["run_id"]).sum()),
                                 "join": "drop_stray on the parquet, then left join of the parquet rows on run_id; judge files without a parquet row are ignored"}
                          for name, rel, prefix in JUDGE_SOURCES},
    }
    rules = {
        "conditions": f"parquet rows with condition in {CONDITIONS}",
        "sample": STRAY_RULE,
        "denominators": (f"clean sample (stray Sonnet runs dropped): run-level rows count parquet rows (Sonnet pressure n = {sonnet_pressure_parquet}); "
                         f"graded_severity.py and the constraint-level columns here count runs with a structured file (Sonnet pressure n = {sonnet_pressure_structured})"),
        "knows": f"final-turn overall_accuracy >= {KNOWS_THRESHOLD}; runs without a probe row count as not knowing",
        "kbv_positive": "knows and j_constraint_adherence < 4",
        "at_3": "j_constraint_adherence == 3", "below_3": "j_constraint_adherence < 3",
        GROUP_EXCL: f"all models except {SONNET} and {LLAMA}",
        "constraint_level": "structured checks pooled over the runs in the group; status lower-cased",
        "movement_tests": "Fisher exact (two-sided) on below-3 counts; Welch t, Student t, Mann-Whitney U on run adherence; Wilcoxon signed-rank on brief means",
        "by_judge": "rows restricted to runs the second judge scored; published rows repeat the published arm on that subset",
    }
    filters = [stray_filter_parquet, stray_filter_checks]
    rows_in = {"parquet_loaded": int(len(scores_raw)), "parquet_after_stray_filter": int(len(scores)),
               "structured_checks_loaded": int(len(checks_raw)), "structured_checks_after_stray_filter": int(len(checks)),
               "probe_rows": int(len(probes))}
    write_csv_with_provenance(runs, out_dir / "stretch_violate_runs.csv",
                              {"inputs": inputs, "rules": rules, "filters": filters, "rows_in": rows_in})
    write_csv_with_provenance(summary, out_dir / "stretch_violate_summary.csv",
                              {"inputs": inputs, "rules": rules, "filters": filters, "rows_in": rows_in})
    write_csv_with_provenance(movement, out_dir / "stretch_violate_movement.csv",
                              {"inputs": inputs, "rules": rules, "filters": filters, "rows_in": rows_in})
    write_csv_with_provenance(by_judge, out_dir / "stretch_violate_by_judge.csv",
                              {"inputs": inputs, "rules": rules, "filters": filters, "rows_in": rows_in})
    return {"stretch_violate_runs": runs, "stretch_violate_summary": summary,
            "stretch_violate_movement": movement, "stretch_violate_by_judge": by_judge}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    frames = run(args.data_root, args.out_dir)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    s = frames["stretch_violate_summary"]
    print(s[["model_group", "condition", "n_runs", "n_adh_4", "n_adh_3", "n_adh_2", "n_adh_1", "n_adh_0",
             "share_below_3", "n_kbv_positive", "share_kbv_at_3", "n_kbv_below_3"]].to_string(index=False))
    print(frames["stretch_violate_movement"].to_string(index=False))
    bj = frames["stretch_violate_by_judge"]
    print(bj[bj["condition"] == "multi_turn_pressure"][["judge_source", "compared_with", "model_id", "n_runs", "share_below_3",
          "n_published_kbv_positive", "share_published_kbv_below_3_under_judge", "n_published_kbv_at_3",
          "share_published_kbv_at_3_moved_below_3"]].to_string(index=False))


if __name__ == "__main__":
    main()
