"""Aggregate per-run JSONL score files into a unified Parquet table."""

from pathlib import Path

import jsonlines
import pandas as pd


def _load_score_files(scores_dir: Path, prefix: str) -> list[dict]:
    """Load all JSONL score files matching a prefix (e.g., 'judge_', 'auditor_')."""
    records = []
    for path in sorted(scores_dir.glob(f"{prefix}*.jsonl")):
        with jsonlines.open(path) as reader:
            for record in reader:
                records.append(record)
    return records


def aggregate_scores(
    scores_dir: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Merge judge and auditor scores into a single DataFrame and save as Parquet."""
    # Load judge scores from per-run files
    judge_records = []
    for record in _load_score_files(scores_dir, "judge_"):
        judge_records.append({
            "run_id": record["run_id"],
            "brief_id": record["brief_id"],
            "model_id": record["model_id"],
            "condition": record["condition"],
            "judge_model": record["judge_model"],
            "j_objective_fidelity": record["objective_fidelity"],
            "j_constraint_adherence": record["constraint_adherence"],
            "j_alternative_coverage": record["alternative_coverage"],
            "j_complexity_inflation": record["complexity_inflation"],
            "j_summary": record["summary"],
        })
    judge_df = pd.DataFrame(judge_records)

    # Load auditor scores from per-run files
    auditor_records = []
    for record in _load_score_files(scores_dir, "auditor_"):
        auditor_records.append({
            "run_id": record["run_id"],
            "a_objective_fidelity": record["objective_fidelity"],
            "a_constraint_adherence": record["constraint_adherence"],
            "a_alternative_coverage": record["alternative_coverage"],
            "a_complexity_inflation": record["complexity_inflation"],
            "a_recoverability": record["recoverability"],
            "a_drift_classification": record["drift_classification"],
        })
    auditor_df = pd.DataFrame(auditor_records)

    # Merge on run_id
    if not judge_df.empty and not auditor_df.empty:
        merged = judge_df.merge(auditor_df, on="run_id", how="inner")
    elif not judge_df.empty:
        merged = judge_df
    elif not auditor_df.empty:
        merged = auditor_df
    else:
        merged = pd.DataFrame()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not merged.empty:
        merged.to_parquet(output_path, index=False)

    return merged
