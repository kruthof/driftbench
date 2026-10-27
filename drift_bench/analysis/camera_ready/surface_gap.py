"""Surface fidelity gap (SFG) per model, condition and judge arm (writing gap WA-2).

SFG = objective fidelity minus the mean of constraint adherence and
alternative coverage. A large positive gap means the proposal still sounds
on-task while it fails on constraints and alternatives. The submitted paper
printed +0.80 for Llama and +0.18 for Sonnet (pressure, published arm, as
published; ``analysis/surface_gap_summary.csv``).

Input: the crossed-judge run table that ``kbv_crossed.py`` writes as
``crossed_scores_runs.csv``. This module rebuilds it from the raw files with
``kbv_crossed``'s own loaders, so it does not depend on module run order and
never reads another module's output. One row per clean single-shot or
pressure run that has both judge arms, with ``opus_*``, ``gpt_*`` and
``published_*`` scores. Disk only, no API calls.

Output: ``surface_gap.csv``, one row per judge arm x condition x model (and
a pooled ``all`` row): the three dimension means, the SFG, ``n`` and a
brief-clustered bootstrap 95% CI of the SFG (2,000 draws, fixed seed).

Run: ``python -m drift_bench.analysis.camera_ready.surface_gap``
"""

from __future__ import annotations

import argparse

import datetime as dt
import json
import math
from pathlib import Path

import pandas as pd

from drift_bench.analysis.camera_ready import kbv_crossed as kc

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
DEFAULT_OUT_DIR = DEFAULT_DATA_ROOT / "camera_ready" / "analysis"
CSV_NAME = "surface_gap.csv"

ARMS = ["opus", "gpt", "published"]
ARM_JUDGES = {"opus": kc.OPUS, "gpt": kc.GPT}
CONDITIONS = list(kc.CROSSED_CONDITIONS)
GAP_DIMS = ("objective_fidelity", "constraint_adherence", "alternative_coverage")
BOOTSTRAP_DRAWS = kc.BOOTSTRAP_DRAWS
BOOTSTRAP_SEED = kc.BOOTSTRAP_SEED


def run_sfg(df: pd.DataFrame, arm: str) -> pd.Series:
    """Run-level SFG under ``arm``: fidelity - (adherence + coverage) / 2."""
    fid = pd.to_numeric(df[f"{arm}_objective_fidelity"], errors="coerce")
    adh = pd.to_numeric(df[f"{arm}_constraint_adherence"], errors="coerce")
    cov = pd.to_numeric(df[f"{arm}_alternative_coverage"], errors="coerce")
    return fid - (adh + cov) / 2.0


def load_crossed_runs(data_root: Path) -> tuple[pd.DataFrame, dict]:
    """The ``crossed_scores_runs`` table, rebuilt from the raw files.

    Same loaders and same assembly as ``kbv_crossed.run``. Returns the table
    and a log with the sample filter and the join counts.
    """
    published, stray_info, _ = kc.load_published(data_root)
    crossed = kc.load_crossed(data_root)
    probes = kc.load_probes(data_root)
    runs, log = kc.assemble_runs(published, crossed, probes)
    return kc.runs_table(runs), {"sample": stray_info, "join": log}


def surface_gap_table(runs: pd.DataFrame, arms: list[str] = ARMS, conditions: list[str] = CONDITIONS,
                      draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    """One row per judge arm x condition x model (plus ``all``).

    A run enters an arm only when all three of the arm's scores are present.
    """
    rows = []
    for arm in arms:
        needed = [f"{arm}_{d}" for d in GAP_DIMS]
        sub = runs.dropna(subset=needed)
        sub = sub[sub["condition"].isin(conditions)].copy()
        sub["sfg"] = run_sfg(sub, arm)
        for cond in conditions:
            c = sub[sub["condition"] == cond]
            for model_id in ["all", *sorted(c["model_id"].unique())]:
                g = c if model_id == "all" else c[c["model_id"] == model_id]
                if g.empty:
                    continue
                means = {d: float(pd.to_numeric(g[f"{arm}_{d}"]).mean()) for d in GAP_DIMS}
                lo, hi = kc.cluster_bootstrap_mean(g["sfg"], g["brief_id"], draws, seed)
                if arm == "published" and "published_judge" in g.columns:
                    judge_model = ";".join(sorted(g["published_judge"].astype(str).unique()))
                else:
                    judge_model = ARM_JUDGES.get(arm, "")
                rows.append({
                    "judge_arm": arm,
                    "judge_model": judge_model,
                    "condition": cond,
                    "model_id": model_id,
                    "n": int(len(g)),
                    "n_briefs": int(g["brief_id"].nunique()),
                    "fidelity_mean": means["objective_fidelity"],
                    "adherence_mean": means["constraint_adherence"],
                    "coverage_mean": means["alternative_coverage"],
                    "sfg": means["objective_fidelity"] - (means["constraint_adherence"] + means["alternative_coverage"]) / 2.0,
                    "sfg_se": float(g["sfg"].std(ddof=1) / math.sqrt(len(g))) if len(g) > 1 else math.nan,
                    "sfg_ci_low": lo,
                    "sfg_ci_high": hi,
                })
    return pd.DataFrame(rows)


def run(data_root: Path = DEFAULT_DATA_ROOT, out_dir: Path = DEFAULT_OUT_DIR) -> dict[str, pd.DataFrame]:
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    runs, log = load_crossed_runs(data_root)
    table = surface_gap_table(runs)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / CSV_NAME
    table.to_csv(path, index=False)
    in_cells = runs["condition"].isin(CONDITIONS)
    prov = {
        "csv": path.name,
        "date": dt.date.today().isoformat(),
        "inputs": [
            "aggregated/all_scores.parquet",
            "analysis/final_probe_accuracy.csv",
            f"{kc.E1_DIR}/opus/judge_opus_*.jsonl",
            f"{kc.E1_DIR}/gpt/judge_gpt_*.jsonl",
            "scores/judge_*.jsonl and openweight_subjects/scores/judge_*.jsonl (stray rule timestamps)",
        ],
        "upstream": "the run table is kbv_crossed.runs_table(kbv_crossed.assemble_runs(...)), the same rows kbv_crossed.py writes "
                    "to crossed_scores_runs.csv; rebuilt here from the raw files",
        "filters": [
            {"rule": "clean sample: the 30 stray Sonnet runs dropped (sample.py)", "detail": log["sample"]},
            {"rule": "runs in single_shot or multi_turn_pressure with a crossed judge file (kbv_crossed.runs_table)",
             "rows_out": int(len(runs)), "join_counts": log["join"]},
            {"rule": f"condition in {CONDITIONS}", "rows_in": int(len(runs)), "rows_out": int(in_cells.sum())},
            {"rule": "per arm: rows with non-missing <arm>_objective_fidelity, <arm>_constraint_adherence, <arm>_alternative_coverage",
             "rows_by_arm": {arm: int(runs.dropna(subset=[f"{arm}_{d}" for d in GAP_DIMS]).shape[0]) for arm in ARMS}},
        ],
        "definitions": {
            "sfg": "mean objective_fidelity - (mean constraint_adherence + mean alternative_coverage) / 2 over the cell's runs; "
                   "equal to the mean of the run-level gap",
            "sfg_ci": f"percentile 95% CI of the mean run-level gap under a brief-clustered bootstrap, {BOOTSTRAP_DRAWS} draws, seed {BOOTSTRAP_SEED}",
            "sfg_se": "standard error of the run-level gap (no clustering), for reference",
            "arms": "opus: Opus 4.6 on every subject (crossed for Sonnet); gpt: GPT-5.4 on every subject (crossed for the six others); "
                    "published: the submission's cross-family arm, on the same paired runs",
            "submitted_values": "pressure, published arm, as published (analysis/surface_gap_summary.csv): Llama +0.80, Sonnet +0.18",
        },
        "rows_out": int(len(table)),
    }
    with open(out_dir / f"{CSV_NAME}.provenance.json", "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=2, default=str)
    return {"surface_gap": table}


def main() -> None:
    parser = argparse.ArgumentParser(description="surface_gap (camera-ready). No API calls.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    t = run(args.data_root, args.out_dir)["surface_gap"]
    cols = ["judge_arm", "model_id", "n", "fidelity_mean", "adherence_mean", "coverage_mean", "sfg", "sfg_ci_low", "sfg_ci_high"]
    with pd.option_context("display.width", 220, "display.max_rows", 100):
        print(t[t["condition"] == "multi_turn_pressure"][cols].to_string(index=False))


if __name__ == "__main__":
    main()
