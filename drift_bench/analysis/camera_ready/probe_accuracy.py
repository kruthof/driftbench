"""Pooled final-probe accuracy (plan item B4; writing gap WA-3).

The submitted paper said: "six of seven models achieving 100% recall and
Gemini Pro at 81% (97.3% across all models)". The value is the mean
final-turn ``overall_accuracy`` of ``analysis/final_probe_accuracy.csv``.
This module writes it on four samples:

* ``probe_file``: every row of ``final_probe_accuracy.csv``, labelled with
  the file's own model and condition. The file holds 87 runs of the
  rigor-pressure follow-up (``multi_turn_pressure_rigor``) and runs without
  a parquet row, so its all-rows mean is not the paper's sample.
* ``as_published``: the probe rows of the runs in the parquet (2,146 rows;
  the submission's sample).
* ``clean``: the 30 stray Sonnet runs dropped (``kbv_table.build_samples``).
* ``clean_untruncated_probes``: ``clean`` minus the Gemini Pro runs whose
  final probe stopped at the 1,020-token cap.

Rows: sample x scope x model x aggregation. Scopes: ``all_rows`` (no
condition filter), ``multi_turn`` (neutral, pressure, checkpointed) and
each of the three multi-turn conditions. Model ``all`` with aggregation
``run_mean`` pools the runs; aggregation ``model_mean`` is the unweighted
mean of the per-model values. Each row has the mean accuracy, ``n``, a
brief-clustered bootstrap 95% CI (run_mean only), and the share of runs at
or above the 0.8 recall cut.

Disk only, no API calls.
Run: ``python -m drift_bench.analysis.camera_ready.probe_accuracy``
"""

from __future__ import annotations

import argparse

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from drift_bench.analysis.camera_ready.kbv_table import (
    GEMINI_PRO,
    GEMINI_PRO_PROBE_CAP,
    PROBE_ACCURACY_THRESHOLD,
    build_samples,
    filter_rules,
    load_final_probes,
    load_judge_timestamps,
    load_scores,
    load_transcript_index,
    truncated_probe_mask,
)

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
DEFAULT_OUT_DIR = DEFAULT_DATA_ROOT / "camera_ready" / "analysis"
CSV_NAME = "probe_accuracy_summary.csv"

MULTI_TURN = ["multi_turn_neutral", "multi_turn_pressure", "checkpointed_pressure"]
# None means no condition filter.
SCOPES: dict[str, list[str] | None] = {
    "all_rows": None,
    "multi_turn": MULTI_TURN,
    **{c: [c] for c in MULTI_TURN},
}
SAMPLES = ["probe_file", "as_published", "clean", "clean_untruncated_probes"]
PROBE_COLUMNS = ["run_id", "turn_index", "overall_accuracy", "constraint_recall", "banned_recall"]
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20261026


def cluster_bootstrap_mean(values: pd.Series, clusters: pd.Series, draws: int = BOOTSTRAP_DRAWS,
                           seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
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


def probe_rows(sample: pd.DataFrame, probes: pd.DataFrame) -> pd.DataFrame:
    """The probe rows whose run is in ``sample`` (inner join on run_id).

    ``sample`` supplies ``brief_id``, ``model_id``, ``condition`` and, when
    present, ``final_probe_token_count``; the probe file's own labels are
    dropped so the sample's labels win.
    """
    keep = [c for c in PROBE_COLUMNS if c in probes.columns]
    p = probes[keep].drop_duplicates("run_id")
    left = [c for c in ("run_id", "brief_id", "model_id", "condition", "final_probe_token_count") if c in sample.columns]
    return sample[left].merge(p, on="run_id", how="inner")


def file_rows(probes: pd.DataFrame, transcripts: pd.DataFrame) -> pd.DataFrame:
    """Every row of the probe file, with the transcript's final probe token count."""
    out = probes.drop_duplicates("run_id").copy()
    if "final_probe_token_count" not in out.columns and "final_probe_token_count" in transcripts.columns:
        out = out.merge(transcripts[["run_id", "final_probe_token_count"]].drop_duplicates("run_id"), on="run_id", how="left")
    return out


def _cell(g: pd.DataFrame, threshold: float, draws: int, seed: int) -> dict:
    acc = pd.to_numeric(g["overall_accuracy"], errors="coerce")
    valid = acc.notna()
    lo, hi = cluster_bootstrap_mean(acc, g["brief_id"], draws, seed)
    trunc = int(truncated_probe_mask(g).sum()) if "final_probe_token_count" in g.columns else 0
    return {
        "n": int(valid.sum()),
        "n_models": int(g["model_id"].nunique()),
        "n_briefs": int(g["brief_id"].nunique()),
        "mean_accuracy": float(acc.mean()) if valid.any() else math.nan,
        "ci_low": lo,
        "ci_high": hi,
        "n_at_or_above_cut": int((acc[valid] >= threshold).sum()),
        "share_at_or_above_cut": float((acc[valid] >= threshold).mean()) if valid.any() else math.nan,
        "mean_constraint_recall": float(pd.to_numeric(g["constraint_recall"], errors="coerce").mean()) if "constraint_recall" in g.columns else math.nan,
        "mean_banned_recall": float(pd.to_numeric(g["banned_recall"], errors="coerce").mean()) if "banned_recall" in g.columns else math.nan,
        "n_final_probe_truncated": trunc,
    }


def summarize(rows: pd.DataFrame, sample: str, threshold: float = PROBE_ACCURACY_THRESHOLD,
              draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    """One row per scope x model (run_mean), plus one model_mean row per scope.

    ``rows`` needs ``brief_id, model_id, condition, overall_accuracy``.
    A scope with no rows is skipped.
    """
    out = []
    for scope, conds in SCOPES.items():
        s = rows if conds is None else rows[rows["condition"].isin(conds)]
        if s.empty:
            continue
        models = sorted(s["model_id"].unique())
        per_model = []
        for model_id in ["all", *models]:
            g = s if model_id == "all" else s[s["model_id"] == model_id]
            row = {"sample": sample, "scope": scope, "model_id": model_id, "aggregation": "run_mean",
                   **_cell(g, threshold, draws, seed)}
            out.append(row)
            if model_id != "all":
                per_model.append(row)
        # Unweighted mean over models. No CI; the count column is left blank
        # because a share averaged over models has no single numerator.
        out.append({
            "sample": sample, "scope": scope, "model_id": "all", "aggregation": "model_mean",
            "n": int(sum(r["n"] for r in per_model)),
            "n_models": len(per_model),
            "n_briefs": int(s["brief_id"].nunique()),
            "mean_accuracy": float(np.mean([r["mean_accuracy"] for r in per_model])),
            "ci_low": math.nan,
            "ci_high": math.nan,
            "n_at_or_above_cut": math.nan,
            "share_at_or_above_cut": float(np.mean([r["share_at_or_above_cut"] for r in per_model])),
            "mean_constraint_recall": float(np.mean([r["mean_constraint_recall"] for r in per_model])),
            "mean_banned_recall": float(np.mean([r["mean_banned_recall"] for r in per_model])),
            "n_final_probe_truncated": int(sum(r["n_final_probe_truncated"] for r in per_model)),
        })
    return pd.DataFrame(out)


def run(data_root: Path = DEFAULT_DATA_ROOT, out_dir: Path = DEFAULT_OUT_DIR) -> dict[str, pd.DataFrame]:
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    scores = load_scores(data_root)
    judge_ts = load_judge_timestamps(data_root)
    transcripts = load_transcript_index(data_root)
    probes = load_final_probes(data_root)
    samples, log = build_samples(scores, judge_ts, transcripts)

    joined: dict[str, int] = {}
    tables = []
    for name in SAMPLES:
        rows = file_rows(probes, transcripts) if name == "probe_file" else probe_rows(samples[name], probes)
        joined[name] = int(len(rows))
        tables.append(summarize(rows, name))
    table = pd.concat(tables, ignore_index=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / CSV_NAME
    table.to_csv(path, index=False)
    rules = filter_rules()
    in_parquet = probes["run_id"].isin(scores["run_id"])
    prov = {
        "csv": path.name,
        "date": dt.date.today().isoformat(),
        "inputs": [
            "aggregated/all_scores.parquet",
            "analysis/final_probe_accuracy.csv",
            "scores/judge_*.jsonl",
            "openweight_subjects/scores/judge_*.jsonl",
            "transcripts/*.jsonl",
            "openweight_subjects/transcripts/*.jsonl",
        ],
        "filters": {
            "samples": {
                "probe_file": "every row of analysis/final_probe_accuracy.csv (one row per run_id), with the file's model_id, condition and brief_id",
                **{k: rules[k] for k in SAMPLES if k != "probe_file"},
            },
            "join": "for the parquet samples: final_probe_accuracy.csv rows whose run_id is in the sample (inner join); the sample's model_id, condition and brief_id are used",
            "scopes": {k: ("no condition filter" if v is None else v) for k, v in SCOPES.items()},
            "truncated_probe": f"model_id == {GEMINI_PRO} and the final probe assistant message has token_count >= {GEMINI_PRO_PROBE_CAP}",
        },
        "row_counts": {
            **log,
            "probe_rows_in_file": int(len(probes)),
            "probe_rows_in_file_with_parquet_row": int(in_parquet.sum()),
            "probe_rows_in_file_without_parquet_row_by_condition": probes.loc[~in_parquet, "condition"].value_counts().to_dict(),
            "probe_rows_joined_by_sample": joined,
        },
        "definitions": {
            "mean_accuracy": "mean of final-turn overall_accuracy over the runs in the cell",
            "run_mean": "runs pooled; the CI is the percentile 95% CI of the run mean under a brief-clustered bootstrap, "
                        f"{BOOTSTRAP_DRAWS} draws, seed {BOOTSTRAP_SEED}",
            "model_mean": "unweighted mean of the per-model run_mean rows of the same sample and scope; no CI; n_at_or_above_cut left blank",
            "share_at_or_above_cut": f"share of runs with overall_accuracy >= {PROBE_ACCURACY_THRESHOLD} (the KBV recall cut)",
            "n_final_probe_truncated": "runs in the cell whose final probe hit the Gemini Pro cap (see filters.truncated_probe)",
            "submitted_value": "97.3% (submitted text: 'six of seven models achieving 100% recall and Gemini Pro at 81% "
                               "(97.3% across all models)') = sample as_published, scope multi_turn, model all, run_mean "
                               "(the model_mean row gives the same value to one decimal)",
        },
        "rows_out": int(len(table)),
    }
    with open(out_dir / f"{CSV_NAME}.provenance.json", "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=2, default=str)
    return {"probe_accuracy_summary": table}


def main() -> None:
    parser = argparse.ArgumentParser(description="probe_accuracy (camera-ready). No API calls.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    t = run(args.data_root, args.out_dir)["probe_accuracy_summary"]
    cols = ["sample", "scope", "aggregation", "n", "mean_accuracy", "ci_low", "ci_high", "share_at_or_above_cut", "n_final_probe_truncated"]
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(t[t["model_id"] == "all"][cols].to_string(index=False))


if __name__ == "__main__":
    main()
