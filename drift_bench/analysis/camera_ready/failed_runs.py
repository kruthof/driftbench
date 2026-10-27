"""Characterise the failed and anomalous runs of the main benchmark (plan item E9).

``metadata.error`` is null in every main transcript and the main
``run_summary.json`` was overwritten by a later rigor-pressure pass, so the
"12 failed runs" cannot be read off a single artifact. This script derives
them from the design instead: 38 briefs x 4 conditions x 7 models x 2
repetitions = 2,128 intended rows. Every intended row that the published
parquet lacks is a failed run, either at the subject stage (no transcript)
or at the scoring stage (transcript but no judge or auditor record).

Two caveats, also recorded in the provenance files:

* The subject failures are inferred from the design, not read from a log.
  Only the open-weight ``run_summary.json`` survives; it confirms the 5
  Qwen failures. The 3 commercial subject failures rest on the design
  count alone.
* Sonnet's 19 scored runs without a transcript never show up as a design
  shortfall: Sonnet is over-sampled (334 parquet rows against 304 intended),
  so every Sonnet design cell still holds at least two transcripts.

The 87 ``multi_turn_pressure_rigor`` transcripts that sit in the main
transcript directory belong to follow-up B. They are not anomalies and are
not listed in ``failed_runs.csv``; the summary reports their count.

Outputs (in ``out_dir``):

* ``failed_runs.csv``: one row per anomaly, with
  ``run_id, brief_id, model_id, condition, anomaly, detail``.
* ``failed_runs_summary.csv``: counts per anomaly plus the design shortfall,
  the input row counts and the two caveats above.
* ``run_summaries.csv``: the contents of every ``run_summary.json`` on disk.

Run: ``python -m drift_bench.analysis.camera_ready.failed_runs``
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from drift_bench.analysis.camera_ready.kbv_table import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUT_DIR,
    MAIN_CONDITIONS,
    SONNET,
    STRAY_JUDGE_DATE,
    load_score_records,
    load_scores,
    load_transcript_index,
    write_csv_with_provenance,
)

# Non-probe assistant messages a complete transcript must hold. Checkpointed
# runs carry two extra CHECKPOINT answers, so they hold 8, not 6.
REQUIRED_ASSISTANT_TURNS = {
    "single_shot": 1,
    "multi_turn_neutral": 6,
    "multi_turn_pressure": 6,
    "checkpointed_pressure": 8,
}
INTENDED_REPS = 2
N_BRIEFS = 38

RUN_SUMMARY_DIRS = ("", "openweight_subjects", "monitored", "followup_a", "followup_b")
SCORE_COLUMNS = [
    "j_objective_fidelity",
    "j_constraint_adherence",
    "j_alternative_coverage",
    "j_complexity_inflation",
    "a_objective_fidelity",
    "a_constraint_adherence",
    "a_alternative_coverage",
    "a_complexity_inflation",
    "a_recoverability",
    "a_drift_classification",
]


def _row(run_id, brief_id, model_id, condition, anomaly, detail) -> dict:
    return {
        "run_id": run_id or "",
        "brief_id": brief_id,
        "model_id": model_id,
        "condition": condition,
        "anomaly": anomaly,
        "detail": detail,
    }


def design_shortfall(
    transcripts: pd.DataFrame,
    models: list[str],
    briefs: list[str],
    conditions: list[str] = MAIN_CONDITIONS,
    reps: int = INTENDED_REPS,
) -> list[dict]:
    """One anomaly per intended (brief, condition, model, rep) with no transcript."""
    counts = transcripts.groupby(["brief_id", "condition", "model_id"]).size()
    rows = []
    for brief in briefs:
        for condition in conditions:
            for model in models:
                have = int(counts.get((brief, condition, model), 0))
                for _ in range(max(0, reps - have)):
                    rows.append(
                        _row(
                            None,
                            brief,
                            model,
                            condition,
                            "missing_transcript_in_design_cell",
                            f"cell has {have} transcripts, design intends {reps}",
                        )
                    )
    return rows


def short_transcripts(transcripts: pd.DataFrame, required: dict = REQUIRED_ASSISTANT_TURNS) -> list[dict]:
    """Transcripts with fewer non-probe assistant turns than their condition needs."""
    rows = []
    for t in transcripts.itertuples(index=False):
        need = required.get(t.condition)
        if need is not None and t.n_assistant < need:
            rows.append(
                _row(
                    t.run_id,
                    t.brief_id,
                    t.model_id,
                    t.condition,
                    "transcript_short",
                    f"{t.n_assistant} non-probe assistant turns, condition needs {need}",
                )
            )
    return rows


def metadata_errors(transcripts: pd.DataFrame) -> list[dict]:
    rows = []
    for t in transcripts.itertuples(index=False):
        if t.error not in (None, "") and not (isinstance(t.error, float) and pd.isna(t.error)):
            rows.append(_row(t.run_id, t.brief_id, t.model_id, t.condition, "metadata_error", str(t.error)))
    return rows


def _cross_check(
    left: pd.DataFrame, right_ids: set, anomaly: str, detail: str
) -> list[dict]:
    rows = []
    for t in left[~left["run_id"].isin(right_ids)].itertuples(index=False):
        rows.append(_row(t.run_id, t.brief_id, t.model_id, t.condition, anomaly, detail))
    return rows


def null_scores(scores: pd.DataFrame, columns: list[str] = SCORE_COLUMNS) -> list[dict]:
    rows = []
    present = [c for c in columns if c in scores.columns]
    bad = scores[scores[present].isna().any(axis=1)]
    for t in bad.itertuples(index=False):
        nulls = [c for c in present if pd.isna(getattr(t, c))]
        rows.append(_row(t.run_id, t.brief_id, t.model_id, t.condition, "null_score_in_parquet", ",".join(nulls)))
    return rows


def stray_rows(judge: pd.DataFrame) -> list[dict]:
    """Sonnet judge records dated 2026-04-16 UTC (the aborted pilot pass)."""
    rows = []
    sel = judge[(judge["model_id"] == SONNET) & (judge["judge_date"] == STRAY_JUDGE_DATE)]
    for t in sel.itertuples(index=False):
        rows.append(
            _row(t.run_id, t.brief_id, t.model_id, t.condition, "stray_pilot_row", f"judge timestamp date {t.judge_date} UTC")
        )
    return rows


def split_main_conditions(transcripts: pd.DataFrame, conditions: list[str] = MAIN_CONDITIONS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a transcript index into main-condition rows and the rest.

    The rest is the follow-up B rigor-pressure pass that shares the main
    transcript directory. Those rows are reported by count only.
    """
    is_main = transcripts["condition"].isin(conditions)
    return transcripts[is_main], transcripts[~is_main]


def load_run_summaries(data_root: Path) -> pd.DataFrame:
    rows = []
    for rel in RUN_SUMMARY_DIRS:
        path = data_root / rel / "run_summary.json" if rel else data_root / "run_summary.json"
        if not path.exists():
            continue
        with open(path) as fh:
            s = json.load(fh)
        rows.append(
            {
                "file": str(path.relative_to(data_root)),
                "total_runs": s.get("total_runs"),
                "failed_runs": s.get("failed_runs"),
                "judge_scores": s.get("judge_scores"),
                "auditor_scores": s.get("auditor_scores"),
                "conditions": ";".join(s.get("conditions", []) or []),
                "models": ";".join(s.get("models", []) or []),
                "repetitions": s.get("repetitions"),
            }
        )
    return pd.DataFrame(rows)


def compute(data_root: Path = DEFAULT_DATA_ROOT) -> dict[str, pd.DataFrame]:
    """Build the anomaly table without writing anything."""
    data_root = Path(data_root)
    scores = load_scores(data_root)
    transcripts = load_transcript_index(data_root)
    judge = load_score_records(data_root, "judge")
    auditor = load_score_records(data_root, "auditor")

    models = sorted(scores["model_id"].unique())
    briefs = sorted(scores["brief_id"].unique())
    main_t, foreign = split_main_conditions(transcripts)
    main_j = judge[judge["condition"].isin(MAIN_CONDITIONS)]
    main_a = auditor[auditor["condition"].isin(MAIN_CONDITIONS)]

    rows: list[dict] = []
    rows += design_shortfall(main_t, models, briefs)
    rows += short_transcripts(main_t)
    rows += metadata_errors(main_t)
    rows += _cross_check(main_t, set(main_j["run_id"]), "transcript_without_judge", "no judge_*.jsonl for this transcript")
    rows += _cross_check(main_j, set(main_t["run_id"]), "judge_without_transcript", "judge_*.jsonl exists, no transcript on disk")
    rows += _cross_check(main_j, set(main_a["run_id"]), "judge_without_auditor", "no auditor_*.jsonl; the inner merge drops this run from the parquet")
    rows += _cross_check(main_a, set(main_j["run_id"]), "auditor_without_judge", "auditor_*.jsonl exists, no judge_*.jsonl")
    rows += _cross_check(main_j, set(scores["run_id"]), "judge_not_in_parquet", "judge record missing from all_scores.parquet")
    rows += null_scores(scores)
    rows += stray_rows(judge)

    anomalies = pd.DataFrame(rows, columns=["run_id", "brief_id", "model_id", "condition", "anomaly", "detail"])

    # Failed runs = intended rows absent from the parquet. Subject failures
    # have no transcript; scoring failures have a transcript but no parquet row.
    n_design = len(briefs) * len(MAIN_CONDITIONS) * len(models) * INTENDED_REPS
    missing = anomalies[anomalies["anomaly"] == "missing_transcript_in_design_cell"]
    subject_failures = int(len(missing))
    missing_by_model = ", ".join(f"{m.split('/')[-1]} {n}" for m, n in missing["model_id"].value_counts().sort_index().items())
    scoring_failures = int(
        anomalies["anomaly"].isin(["transcript_without_judge", "judge_without_auditor"]).sum()
    )
    non_sonnet_parquet = int((scores["model_id"] != SONNET).sum())
    non_sonnet_design = len(briefs) * len(MAIN_CONDITIONS) * (len(models) - 1) * INTENDED_REPS
    sonnet_rows = scores[scores["model_id"] == SONNET]
    sonnet_design = len(briefs) * len(MAIN_CONDITIONS) * INTENDED_REPS
    sonnet_no_transcript = int((~sonnet_rows["run_id"].isin(transcripts["run_id"])).sum())
    summaries = load_run_summaries(data_root)
    ow = summaries[summaries["file"] == "openweight_subjects/run_summary.json"]
    ow_failed = int(ow["failed_runs"].iloc[0]) if len(ow) and pd.notna(ow["failed_runs"].iloc[0]) else None
    foreign_conditions = ", ".join(f"{c} {n}" for c, n in foreign["condition"].value_counts().sort_index().items())
    summary_rows = [
        {"quantity": "rows_in:transcripts", "value": int(len(transcripts)), "detail": "transcripts/*.jsonl + openweight_subjects/transcripts/*.jsonl"},
        {"quantity": "rows_in:judge_files", "value": int(len(judge)), "detail": "judge_*.jsonl in scores/ and openweight_subjects/scores/"},
        {"quantity": "rows_in:auditor_files", "value": int(len(auditor)), "detail": "auditor_*.jsonl in scores/ and openweight_subjects/scores/"},
        {"quantity": "rows_in:parquet_rows", "value": int(len(scores)), "detail": "aggregated/all_scores.parquet"},
        {"quantity": "design_rows", "value": n_design, "detail": f"{len(briefs)} briefs x {len(MAIN_CONDITIONS)} conditions x {len(models)} models x {INTENDED_REPS} reps"},
        {"quantity": "parquet_rows", "value": int(len(scores)), "detail": "aggregated/all_scores.parquet"},
        {
            "quantity": "subject_failures",
            "value": subject_failures,
            "detail": (
                "intended runs with no transcript (missing_transcript_in_design_cell); inferred from the design, "
                "not logged: metadata.error is null everywhere and the main run_summary.json was overwritten by the "
                f"rigor-pressure pass; by model: {missing_by_model}; openweight_subjects/run_summary.json reports "
                f"failed_runs = {ow_failed}, which confirms the Qwen count"
            ),
        },
        {"quantity": "scoring_failures", "value": scoring_failures, "detail": "transcripts dropped before the parquet (transcript_without_judge + judge_without_auditor)"},
        {"quantity": "failed_runs", "value": subject_failures + scoring_failures, "detail": "subject_failures + scoring_failures"},
        {"quantity": "non_sonnet_design_shortfall", "value": non_sonnet_design - non_sonnet_parquet, "detail": f"{non_sonnet_design} intended non-Sonnet rows minus {non_sonnet_parquet} in the parquet"},
        {
            "quantity": "sonnet_scored_runs_without_transcript",
            "value": sonnet_no_transcript,
            "detail": (
                f"Sonnet parquet rows with no transcript on disk; never a design shortfall because over-sampling masks "
                f"them ({len(sonnet_rows)} Sonnet parquet rows against {sonnet_design} intended; every Sonnet design "
                "cell still holds at least two transcripts); listed as judge_without_transcript"
            ),
        },
        {
            "quantity": "transcripts_outside_main_conditions",
            "value": int(len(foreign)),
            "detail": (
                f"transcripts in the main transcript dir on other conditions ({foreign_conditions}); the follow-up B "
                "rigor-pressure pass, a known separate experiment; not listed in failed_runs.csv"
            ),
        },
    ]
    for anomaly, n in anomalies["anomaly"].value_counts().sort_index().items():
        summary_rows.append({"quantity": f"anomaly:{anomaly}", "value": int(n), "detail": "rows in failed_runs.csv"})
    summary = pd.DataFrame(summary_rows)

    return {
        "failed_runs": anomalies,
        "failed_runs_summary": summary,
        "run_summaries": summaries,
    }


def rows_in_from_summary(summary: pd.DataFrame) -> dict[str, int]:
    """The ``rows_in:*`` counts of a ``failed_runs_summary`` table as a dict."""
    sel = summary[summary["quantity"].str.startswith("rows_in:")]
    return {q.split(":", 1)[1]: int(v) for q, v in zip(sel["quantity"], sel["value"])}


def failed_run_count(data_root: Path = DEFAULT_DATA_ROOT) -> int:
    """The number of failed runs (subject plus scoring failures)."""
    summary = compute(data_root)["failed_runs_summary"]
    return int(summary.loc[summary["quantity"] == "failed_runs", "value"].iloc[0])


def run(data_root: Path = DEFAULT_DATA_ROOT, out_dir: Path = DEFAULT_OUT_DIR) -> dict[str, pd.DataFrame]:
    data_root, out_dir = Path(data_root), Path(out_dir)
    tables = compute(data_root)
    summary = tables["failed_runs_summary"].set_index("quantity")
    prov = {
        "rows_in": rows_in_from_summary(tables["failed_runs_summary"]),
        "inputs": [
            "aggregated/all_scores.parquet",
            "transcripts/*.jsonl",
            "openweight_subjects/transcripts/*.jsonl",
            "scores/judge_*.jsonl",
            "scores/auditor_*.jsonl",
            "openweight_subjects/scores/judge_*.jsonl",
            "openweight_subjects/scores/auditor_*.jsonl",
            "*/run_summary.json",
        ],
        "filters": {
            "design": f"{N_BRIEFS} briefs x {MAIN_CONDITIONS} x 7 models x {INTENDED_REPS} reps",
            "required_assistant_turns": REQUIRED_ASSISTANT_TURNS,
            "stray_pilot_row": f"model_id == {SONNET} and judge timestamp date == {STRAY_JUDGE_DATE} UTC",
            "main_conditions_only": "cross-checks use transcripts and score files on the four main conditions",
            "failed_runs": "missing_transcript_in_design_cell + transcript_without_judge + judge_without_auditor",
            "subject_failures_inferred": summary.loc["subject_failures", "detail"],
            "sonnet_missing_transcripts_masked": summary.loc["sonnet_scored_runs_without_transcript", "detail"],
            "transcripts_outside_main_conditions": (
                f"{int(summary.loc['transcripts_outside_main_conditions', 'value'])} transcripts on other conditions "
                "(follow-up B rigor pressure) are excluded from failed_runs.csv and counted in failed_runs_summary.csv only"
            ),
        },
    }
    write_csv_with_provenance(tables["failed_runs"], out_dir / "failed_runs.csv", {**prov, "description": "one row per anomaly"})
    write_csv_with_provenance(tables["failed_runs_summary"], out_dir / "failed_runs_summary.csv", {**prov, "description": "counts per anomaly and the design shortfall"})
    write_csv_with_provenance(tables["run_summaries"], out_dir / "run_summaries.csv", {**prov, "description": "every run_summary.json on disk; the top-level one was overwritten by a rigor-pressure pass"})
    return tables


def main() -> None:
    tables = run()
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 80):
        print(tables["failed_runs_summary"])
        print(tables["run_summaries"])
        f = tables["failed_runs"]
        print(f[f["anomaly"] != "stray_pilot_row"])


if __name__ == "__main__":
    main()
