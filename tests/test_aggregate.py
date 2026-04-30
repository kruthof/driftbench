"""Aggregate per-run JSONL score files into a unified DataFrame."""

from pathlib import Path

import jsonlines
import pytest

from drift_bench.analysis.aggregate import aggregate_scores
from drift_bench.judges.auditor import AuditorScore
from drift_bench.judges.judge import JudgeScore


def _write_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, mode="w") as writer:
        writer.write(payload)


def _judge_record(run_id: str, **overrides) -> dict:
    base = JudgeScore(
        run_id=run_id,
        brief_id="test_01",
        model_id="openai/gpt-5.4",
        condition="multi_turn_pressure",
        judge_model="anthropic/claude-opus-4-6",
        objective_fidelity=4,
        constraint_adherence=3,
        alternative_coverage=2,
        complexity_inflation=1,
        summary="s",
        violations=[],
        optional_extras_flagged=[],
    ).to_dict()
    base.update(overrides)
    return base


def _auditor_record(run_id: str, **overrides) -> dict:
    base = AuditorScore(
        run_id=run_id,
        brief_id="test_01",
        model_id="openai/gpt-5.4",
        condition="multi_turn_pressure",
        auditor_model="anthropic/claude-opus-4-6",
        objective_fidelity=3,
        constraint_adherence=3,
        alternative_coverage=2,
        complexity_inflation=2,
        recoverability=3,
        drift_classification="mild_drift",
        drift_events=[],
        corrected_proposal="",
    ).to_dict()
    base.update(overrides)
    return base


def test_aggregate_inner_joins_judge_and_auditor(tmp_path: Path) -> None:
    scores_dir = tmp_path / "scores"
    out = tmp_path / "agg.parquet"
    _write_jsonl(scores_dir / "judge_run1.jsonl", _judge_record("run1"))
    _write_jsonl(scores_dir / "auditor_run1.jsonl", _auditor_record("run1"))

    df = aggregate_scores(scores_dir, out)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["run_id"] == "run1"
    assert row["j_constraint_adherence"] == 3
    assert row["a_complexity_inflation"] == 2
    assert row["a_drift_classification"] == "mild_drift"
    assert out.exists()


def test_aggregate_drops_unmatched_runs_under_inner_join(tmp_path: Path) -> None:
    """When both sides have data, aggregation is an inner join on run_id."""
    scores_dir = tmp_path / "scores"
    _write_jsonl(scores_dir / "judge_run1.jsonl", _judge_record("run1"))
    _write_jsonl(scores_dir / "judge_run2.jsonl", _judge_record("run2"))
    _write_jsonl(scores_dir / "auditor_run1.jsonl", _auditor_record("run1"))

    df = aggregate_scores(scores_dir, tmp_path / "agg.parquet")
    assert set(df["run_id"]) == {"run1"}


def test_aggregate_judge_only_when_no_auditor(tmp_path: Path) -> None:
    scores_dir = tmp_path / "scores"
    _write_jsonl(scores_dir / "judge_run1.jsonl", _judge_record("run1"))

    df = aggregate_scores(scores_dir, tmp_path / "agg.parquet")
    assert len(df) == 1
    assert "j_constraint_adherence" in df.columns
    assert "a_constraint_adherence" not in df.columns


def test_aggregate_empty_dir_returns_empty_dataframe(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    df = aggregate_scores(empty, tmp_path / "agg.parquet")
    assert df.empty
    # No file written when nothing to aggregate.
    assert not (tmp_path / "agg.parquet").exists()
