"""Score dataclasses round-trip through dict (used by per-run jsonl files)."""

from dataclasses import asdict

import pytest

from drift_bench.judges.auditor import AuditorScore
from drift_bench.judges.judge import JudgeScore
from drift_bench.judges.judge_blind import BlindJudgeScore
from drift_bench.judges.judge_structured import StructuredJudgeScore
from drift_bench.judges.structure_extractor import StructuralCounts


def _judge_score() -> JudgeScore:
    return JudgeScore(
        run_id="run1",
        brief_id="test_01",
        model_id="openai/gpt-5.4",
        condition="multi_turn_pressure",
        judge_model="anthropic/claude-opus-4-6",
        objective_fidelity=4,
        constraint_adherence=3,
        alternative_coverage=2,
        complexity_inflation=1,
        summary="ok",
        violations=["v1"],
        optional_extras_flagged=[],
    )


def _auditor_score() -> AuditorScore:
    return AuditorScore(
        run_id="run1",
        brief_id="test_01",
        model_id="openai/gpt-5.4",
        condition="multi_turn_pressure",
        auditor_model="anthropic/claude-opus-4-6",
        objective_fidelity=4,
        constraint_adherence=3,
        alternative_coverage=2,
        complexity_inflation=1,
        recoverability=3,
        drift_classification="mild_drift",
        drift_events=[{"turn": 2, "description": "added a confound"}],
        corrected_proposal="rewrite",
    )


@pytest.mark.parametrize(
    "score, cls",
    [
        (_judge_score(), JudgeScore),
        (_auditor_score(), AuditorScore),
    ],
)
def test_score_round_trip(score, cls) -> None:
    record = score.to_dict()
    recovered = cls(**{k: record[k] for k in cls.__dataclass_fields__})
    assert recovered == score


def test_blind_judge_score_round_trip() -> None:
    s = BlindJudgeScore(
        run_id="run1",
        brief_id="test_01",
        model_id="openai/gpt-5.4",
        condition="single_shot",
        judge_model="anthropic/claude-opus-4-6",
        objective_fidelity=4,
        constraint_adherence=4,
        alternative_coverage=3,
        complexity_inflation=0,
        summary="aligned",
        violations=[],
        optional_extras_flagged=[],
    )
    record = s.to_dict()
    rec = BlindJudgeScore(**{k: record[k] for k in BlindJudgeScore.__dataclass_fields__})
    assert rec == s


def test_structured_judge_score_round_trip() -> None:
    s = StructuredJudgeScore(
        run_id="run1",
        brief_id="test_01",
        model_id="openai/gpt-5.4",
        condition="multi_turn_pressure",
        judge_model="anthropic/claude-opus-4-6",
        objective_fidelity=3,
        constraint_adherence=2,
        alternative_coverage=1,
        complexity_inflation=2,
        constraint_checks=[{"constraint": "c1", "status": "satisfied"}],
        direction_checks=[{"direction": "d1", "status": "evaluated"}],
        summary="partial",
        violations=["v"],
        optional_extras_flagged=["e"],
    )
    record = s.to_dict()
    rec = StructuredJudgeScore(**{k: record[k] for k in StructuredJudgeScore.__dataclass_fields__})
    assert rec == s


def test_structural_counts_total_is_derived() -> None:
    """total_structural is computed in __post_init__; round-trip must agree."""
    s = StructuralCounts(
        run_id="r",
        brief_id="b",
        model_id="m",
        condition="c",
        judge_model="j",
        stages=2,
        components=3,
        datasets_resources=1,
        sub_experiments=4,
        dependencies=5,
    )
    assert s.total_structural == 15
    rec = StructuralCounts(**{
        k: v for k, v in asdict(s).items() if k in StructuralCounts.__dataclass_fields__
    })
    assert rec.total_structural == 15


def test_judge_score_extra_field_in_record_is_ignored() -> None:
    """Forward-compat: a record with an extra field must still load via the
    {k: record[k] for k in fields} pattern used by score_all()."""
    s = _judge_score()
    record = s.to_dict()
    record["future_field"] = "value-that-old-code-doesnt-know-about"
    rec = JudgeScore(**{k: record[k] for k in JudgeScore.__dataclass_fields__})
    assert rec == s
