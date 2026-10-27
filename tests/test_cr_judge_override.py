"""X1: judge_override wins over config routing; default routing unchanged.

Pure-function tests. The one scoring path exercised patches ``_call_llm``
so no network call happens.
"""

import asyncio
import json
from pathlib import Path

import jsonlines
import pytest
import yaml

from drift_bench.judges.auditor import AuditorPipeline
from drift_bench.judges.judge import JudgePipeline, JudgeScore, score_from_record
from drift_bench.judges.judge_blind import BlindJudgePipeline
from drift_bench.judges.judge_skeptical import DEFAULT_SKEPTICAL_JUDGE, SkepticalJudgePipeline
from drift_bench.judges.judge_structured import StructuredJudgePipeline
import litellm

from drift_bench.rescore import (
    ZERO_COST_LIMIT,
    is_published_shape,
    RescoreRefused,
    check_output_dir,
    is_published_scores_dir,
    load_transcript_dicts,
    per_turn_collision,
    prefix_collides_with_published,
    rescore,
    summary_path,
)
from drift_bench.runners.base import Transcript

OVERRIDE = "anthropic/claude-opus-4-6"
SONNET = "anthropic/claude-sonnet-4-6"
GPT = "openai/gpt-5.4"


@pytest.fixture(scope="module")
def judges_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)["models"]["judges"]


@pytest.mark.parametrize("cls", [JudgePipeline, StructuredJudgePipeline, BlindJudgePipeline])
def test_default_routing_is_unchanged(cls, config_path: Path, judges_config: dict) -> None:
    pipeline = cls(config_path)
    assert pipeline.judge_override is None
    assert pipeline._select_judge_model(SONNET) == judges_config["for_anthropic_runs"]
    assert pipeline._select_judge_model(GPT) == judges_config["default"]


@pytest.mark.parametrize("cls", [JudgePipeline, StructuredJudgePipeline, BlindJudgePipeline])
def test_override_wins_for_every_subject_family(cls, config_path: Path) -> None:
    pipeline = cls(config_path, judge_override=OVERRIDE)
    for subject in (SONNET, GPT, "gemini/gemini-3.1-pro-preview", "together_ai/x/y"):
        assert pipeline._select_judge_model(subject) == OVERRIDE


def test_skeptical_default_and_override(config_path: Path) -> None:
    default = SkepticalJudgePipeline(config_path)
    assert default._select_judge_model(SONNET) == DEFAULT_SKEPTICAL_JUDGE
    assert default.judge_model == DEFAULT_SKEPTICAL_JUDGE
    overridden = SkepticalJudgePipeline(config_path, judge_override=OVERRIDE)
    assert overridden._select_judge_model(GPT) == OVERRIDE
    assert overridden.judge_model == OVERRIDE


def test_auditor_override_wins(config_path: Path) -> None:
    default = AuditorPipeline(config_path)
    assert default._select_auditor_model(SONNET) == default.default_auditor_model
    overridden = AuditorPipeline(config_path, auditor_override=OVERRIDE)
    for subject in (SONNET, GPT, "gemini/gemini-3.1-pro-preview"):
        assert overridden._select_auditor_model(subject) == OVERRIDE


@pytest.mark.parametrize(
    "cls, default_prefix",
    [
        (JudgePipeline, "judge"),
        (StructuredJudgePipeline, "structured"),
        (BlindJudgePipeline, "blind"),
        (SkepticalJudgePipeline, "skeptical"),
        (AuditorPipeline, "auditor"),
    ],
)
def test_output_prefix_default_and_custom(cls, default_prefix: str, config_path: Path, tmp_path: Path) -> None:
    default = cls(config_path)
    assert default._output_path(tmp_path, "r1") == tmp_path / f"{default_prefix}_r1.jsonl"
    custom = cls(config_path, output_prefix=f"{default_prefix}_opus")
    assert custom._output_path(tmp_path, "r1") == tmp_path / f"{default_prefix}_opus_r1.jsonl"
    # The crossed file never collides with the published one.
    assert custom._output_path(tmp_path, "r1") != default._output_path(tmp_path, "r1")


def test_score_from_record_accepts_old_records_without_new_keys() -> None:
    """Published judge files predate turn_index / judge_response_model / cost_usd."""
    old_record = {
        "run_id": "r1", "brief_id": "b", "model_id": GPT, "condition": "single_shot",
        "judge_model": OVERRIDE, "objective_fidelity": 4, "constraint_adherence": 3,
        "alternative_coverage": 2, "complexity_inflation": 1, "summary": "s",
        "violations": [], "optional_extras_flagged": [], "timestamp": 1.0,
        "judge_latency_ms": 2.0, "raw_response": "{}", "unknown_future_key": "x",
    }
    score = score_from_record(JudgeScore, old_record)
    assert score.turn_index is None
    assert score.judge_response_model is None
    assert score.cost_usd == 0.0


# ---------------------------------------------------------------------------
# rescore.py: resumability, filters and the recorded judge id (no network)
# ---------------------------------------------------------------------------

def _write_transcript(path_dir: Path, sample_transcript: Transcript, run_id: str, model_id: str, condition: str) -> None:
    t = Transcript.from_dict(sample_transcript.to_dict())
    t.metadata.run_id = run_id
    t.metadata.model_id = model_id
    t.metadata.condition = condition
    t.save(path_dir)


def _fake_llm_factory(seen: list[dict]):
    async def fake_call_llm(self, model_id, messages, temperature=0.0, max_tokens=0, response_format=None):
        seen.append({"model_id": model_id, "messages": messages})
        content = json.dumps({
            "objective_fidelity": 4, "constraint_adherence": 4,
            "alternative_coverage": 3, "complexity_inflation": 0,
            "summary": "fake", "violations": [], "optional_extras_flagged": [],
        })
        return {
            "content": content, "input_tokens": 10, "output_tokens": 5,
            "cost": 0.01, "latency_ms": 1.0,
            "finish_reason": "stop", "response_model": "claude-opus-4-6-20260101",
        }
    return fake_call_llm


def test_load_transcript_dicts_filters(tmp_path: Path, sample_transcript: Transcript) -> None:
    _write_transcript(tmp_path, sample_transcript, "a1", SONNET, "single_shot")
    _write_transcript(tmp_path, sample_transcript, "a2", GPT, "single_shot")
    _write_transcript(tmp_path, sample_transcript, "a3", SONNET, "multi_turn_neutral")
    assert {t["metadata"]["run_id"] for t in load_transcript_dicts(tmp_path)} == {"a1", "a2", "a3"}
    got = load_transcript_dicts(tmp_path, conditions=["single_shot"], models=[SONNET])
    assert [t["metadata"]["run_id"] for t in got] == ["a1"]


def test_prefix_collision_rule(tmp_path: Path) -> None:
    assert prefix_collides_with_published("judge_opus", tmp_path) is None  # empty dir
    (tmp_path / "judge_0123456789ab.jsonl").write_text("{}\n")  # published shape
    assert prefix_collides_with_published("judge_opus", tmp_path) == "judge"
    assert prefix_collides_with_published("crossed", tmp_path) is None
    assert prefix_collides_with_published("judge", tmp_path) is None  # same prefix is not a collision
    # A directory holding only the prefix's own files is not a collision (resume case).
    own = tmp_path / "own"
    own.mkdir()
    (own / "judge_opus_r1.jsonl").write_text("{}\n")
    (own / "judge_opus_t2_r1.jsonl").write_text("{}\n")
    assert prefix_collides_with_published("judge_opus", own) is None
    with pytest.raises(RescoreRefused):
        check_output_dir(tmp_path, "judge_opus", per_turn=False)
    check_output_dir(tmp_path, "judge_opus", per_turn=False, allow_published_dir=True)  # override


def test_per_turn_and_final_files_never_share_a_directory(tmp_path: Path) -> None:
    assert per_turn_collision("judge_opus", tmp_path, per_turn=True) is None
    (tmp_path / "judge_opus_t1_r1.jsonl").write_text("{}\n")
    (tmp_path / "judge_opus_t2_r1.jsonl").write_text("{}\n")
    # Resuming a per-turn run next to its own per-turn files is fine.
    assert per_turn_collision("judge_opus", tmp_path, per_turn=True) is None
    # A final run next to per-turn files is refused.
    assert per_turn_collision("judge_opus", tmp_path, per_turn=False)
    with pytest.raises(RescoreRefused):
        check_output_dir(tmp_path, "judge_opus", per_turn=False)
    # And a per-turn run next to final files is refused.
    other = tmp_path / "final"
    other.mkdir()
    (other / "judge_opus_r1.jsonl").write_text("{}\n")
    assert per_turn_collision("judge_opus", other, per_turn=True)
    with pytest.raises(RescoreRefused):
        check_output_dir(other, "judge_opus", per_turn=True)
    assert per_turn_collision("judge_opus", other, per_turn=False) is None


def test_sibling_crossed_prefix_is_not_a_collision(tmp_path: Path) -> None:
    """E1 layout: judge_opus_* and judge_gpt_* may coexist; judge_<id> is the published shape."""
    assert is_published_shape("judge_0123456789ab.jsonl", "judge")
    assert not is_published_shape("judge_opus_0123456789ab.jsonl", "judge")
    assert not is_published_shape("judge_opus_t2_0123456789ab.jsonl", "judge")
    assert not is_published_shape("judge_gpt_0123456789ab.jsonl", "judge")

    opus_dir = tmp_path / "opus"
    opus_dir.mkdir()
    (opus_dir / "judge_opus_0123456789ab.jsonl").write_text("{}\n")
    assert prefix_collides_with_published("judge_gpt", opus_dir) is None
    check_output_dir(opus_dir, "judge_gpt", per_turn=False)  # accepted

    published = tmp_path / "published"
    published.mkdir()
    (published / "judge_0123456789ab.jsonl").write_text("{}\n")
    assert prefix_collides_with_published("judge_gpt", published) == "judge"
    with pytest.raises(RescoreRefused):
        check_output_dir(published, "judge_gpt", per_turn=False)


def test_published_scores_dirs_are_refused(project_root: Path, tmp_path: Path) -> None:
    published = project_root / "drift_bench" / "data" / "scores"
    assert is_published_scores_dir(published)
    assert is_published_scores_dir(project_root / "drift_bench" / "data" / "openweight_subjects" / "scores")
    assert not is_published_scores_dir(tmp_path)
    with pytest.raises(RescoreRefused):
        check_output_dir(published, "judge_opus", per_turn=False)


def test_summary_path_names() -> None:
    d = Path("/x")
    assert summary_path(d, "judge_opus", None) == d / "rescore_summary_judge_opus.json"
    assert summary_path(d, "judge_opus", [1, 2, 3]) == d / "rescore_summary_judge_opus_t1-2-3.json"


def _complete_record(run_id: str, judge_model: str) -> dict:
    return {
        "run_id": run_id, "brief_id": "test_01", "model_id": GPT, "condition": "multi_turn_pressure",
        "judge_model": judge_model, "objective_fidelity": 4, "constraint_adherence": 4,
        "alternative_coverage": 4, "complexity_inflation": 0, "summary": "done",
        "violations": [], "optional_extras_flagged": [],
    }


def test_rescore_uses_override_and_skips_existing(
    tmp_path: Path, sample_transcript: Transcript, sample_brief: dict, config_path: Path, monkeypatch,
) -> None:
    dir_a = tmp_path / "transcripts"
    dir_b = tmp_path / "openweight_transcripts"
    scores_dir = tmp_path / "e1_crossed_judge"
    briefs_dir = tmp_path / "briefs"
    briefs_dir.mkdir()
    with open(briefs_dir / "test_01.yaml", "w") as f:
        yaml.safe_dump(sample_brief, f)
    _write_transcript(dir_a, sample_transcript, "r_sonnet", SONNET, "multi_turn_pressure")
    _write_transcript(dir_b, sample_transcript, "r_gpt", GPT, "multi_turn_pressure")
    _write_transcript(dir_a, sample_transcript, "r_done", GPT, "multi_turn_pressure")
    _write_transcript(dir_a, sample_transcript, "r_corrupt", GPT, "multi_turn_pressure")

    scores_dir.mkdir()
    # Loadable existing output: skipped.
    with jsonlines.open(scores_dir / "judge_opus_r_done.jsonl", mode="w") as w:
        w.write(_complete_record("r_done", "stale"))
    # Corrupt existing output (missing required keys): re-scored.
    with jsonlines.open(scores_dir / "judge_opus_r_corrupt.jsonl", mode="w") as w:
        w.write({"run_id": "r_corrupt"})

    seen: list[dict] = []
    monkeypatch.setattr(JudgePipeline, "_call_llm", _fake_llm_factory(seen))

    summary = asyncio.run(rescore(
        transcripts_dirs=[dir_a, dir_b],
        scores_dir=scores_dir,
        judge_override=OVERRIDE,
        output_prefix="judge_opus",
        mode="judge",
        conditions=["multi_turn_pressure"],
        config_path=config_path,
        briefs_dir=briefs_dir,
        max_budget=10.0,
        argv=["rescore", "--test"],
    ))

    assert summary["scored"] == 3
    assert summary["skipped_existing"] == 1
    assert summary["failed"] == 0
    assert summary["zero_cost"] == 0
    assert summary["n_cost_estimated"] == 0
    assert summary["judge_models_used"] == [OVERRIDE]
    assert summary["judge_response_models"] == ["claude-opus-4-6-20260101"]
    # Both transcript directories were read; every call went to the override,
    # including the Sonnet run that config routing would send to GPT-5.4.
    assert {c["model_id"] for c in seen} == {OVERRIDE}
    assert len(seen) == 3

    for run_id in ("r_sonnet", "r_gpt", "r_corrupt"):
        with jsonlines.open(scores_dir / f"judge_opus_{run_id}.jsonl") as r:
            record = next(iter(r))
        assert record["judge_model"] == OVERRIDE
        assert record["judge_response_model"] == "claude-opus-4-6-20260101"
        assert record["turn_index"] is None
        assert record["cost_usd"] == pytest.approx(0.01)
        assert record["cost_estimated"] is False

    # The loadable file was left alone.
    with jsonlines.open(scores_dir / "judge_opus_r_done.jsonl") as r:
        assert next(iter(r))["judge_model"] == "stale"
    # No published-prefix file was written.
    assert not list(scores_dir.glob("judge_r_*.jsonl"))

    # Summary record appended, one per invocation.
    spath = scores_dir / "rescore_summary_judge_opus.json"
    assert spath.exists()
    with open(spath) as f:
        records = json.load(f)
    assert len(records) == 1
    assert records[0]["argv"] == ["rescore", "--test"]
    assert records[0]["litellm_version"]
    assert records[0]["utc_date"]
    assert records[0]["budget_spent_usd"] == pytest.approx(0.03)

    # Second invocation: everything skipped, second summary record appended.
    summary2 = asyncio.run(rescore(
        transcripts_dirs=[dir_a, dir_b], scores_dir=scores_dir, judge_override=OVERRIDE,
        output_prefix="judge_opus", conditions=["multi_turn_pressure"],
        config_path=config_path, briefs_dir=briefs_dir, argv=["rescore"],
    ))
    assert summary2["skipped_existing"] == 4 and summary2["scored"] == 0
    with open(spath) as f:
        assert len(json.load(f)) == 2


def test_dry_run_makes_no_calls_and_writes_nothing(
    tmp_path: Path, sample_transcript: Transcript, sample_brief: dict, config_path: Path, monkeypatch,
) -> None:
    async def explode(**kwargs):
        raise AssertionError("network call during --dry-run")

    monkeypatch.setattr(litellm, "acompletion", explode)
    transcripts_dir = tmp_path / "transcripts"
    briefs_dir = tmp_path / "briefs"
    briefs_dir.mkdir()
    with open(briefs_dir / "test_01.yaml", "w") as f:
        yaml.safe_dump(sample_brief, f)
    _write_transcript(transcripts_dir, sample_transcript, "d1", SONNET, "multi_turn_pressure")
    _write_transcript(transcripts_dir, sample_transcript, "d2", GPT, "multi_turn_pressure")
    _write_transcript(transcripts_dir, sample_transcript, "d3", GPT, "single_shot")
    scores_dir = tmp_path / "out"
    scores_dir.mkdir()
    with jsonlines.open(scores_dir / "judge_opus_d2.jsonl", mode="w") as w:
        w.write(_complete_record("d2", OVERRIDE))

    summary = asyncio.run(rescore(
        transcripts_dirs=[transcripts_dir], scores_dir=scores_dir, judge_override=OVERRIDE,
        output_prefix="judge_opus", conditions=["multi_turn_pressure"],
        config_path=config_path, briefs_dir=briefs_dir, dry_run=True,
    ))
    assert summary["dry_run"] is True
    assert summary["transcripts"] == 2 and summary["tasks"] == 1 and summary["skipped_existing"] == 1
    assert summary["judge_models"] == [OVERRIDE]
    assert summary["unpriced_judge_models"] == []
    assert summary["estimated_cost_usd"] > 0
    assert summary["estimated_input_tokens"] > summary["assumed_prompt_overhead_tokens"]
    # Nothing written: no new score file, no summary record.
    assert sorted(p.name for p in scores_dir.iterdir()) == ["judge_opus_d2.jsonl"]

    # Per-turn dry run counts one task per (turn, run); an unpriced judge is reported.
    summary_t = asyncio.run(rescore(
        transcripts_dirs=[transcripts_dir], scores_dir=tmp_path / "turns", judge_override="together_ai/x/y",
        output_prefix="judge_x", conditions=["multi_turn_pressure"], turn_indices=[1, 2],
        config_path=config_path, briefs_dir=briefs_dir, dry_run=True,
    ))
    assert summary_t["tasks"] == 4
    assert summary_t["unpriced_judge_models"] == ["together_ai/x/y"]
    assert summary_t["estimated_cost_usd"] == 0.0
    assert not (tmp_path / "turns").exists()


def test_rescore_refuses_prefix_collision(
    tmp_path: Path, sample_transcript: Transcript, sample_brief: dict, config_path: Path,
) -> None:
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    (scores_dir / "judge_0123456789ab.jsonl").write_text("{}\n")  # published shape
    with pytest.raises(RescoreRefused):
        asyncio.run(rescore(
            transcripts_dirs=[tmp_path], scores_dir=scores_dir, judge_override=OVERRIDE,
            output_prefix="judge_opus", config_path=config_path, briefs_dir=tmp_path,
        ))


def test_rescore_aborts_after_too_many_zero_cost_calls(
    tmp_path: Path, sample_transcript: Transcript, sample_brief: dict, config_path: Path, monkeypatch,
) -> None:
    transcripts_dir = tmp_path / "transcripts"
    briefs_dir = tmp_path / "briefs"
    briefs_dir.mkdir()
    with open(briefs_dir / "test_01.yaml", "w") as f:
        yaml.safe_dump(sample_brief, f)
    n_runs = ZERO_COST_LIMIT + 5
    for i in range(n_runs):
        _write_transcript(transcripts_dir, sample_transcript, f"z{i:02d}", GPT, "single_shot")

    seen: list[dict] = []
    fake = _fake_llm_factory(seen)

    async def zero_cost(self, *args, **kwargs):
        result = await fake(self, *args, **kwargs)
        result["cost"] = 0.0
        return result

    # An estimated cost is carried into the record and counted in the summary.
    async def estimated(self, *args, **kwargs):
        result = await fake(self, *args, **kwargs)
        result["cost_estimated"] = True
        return result

    monkeypatch.setattr(JudgePipeline, "_call_llm", estimated)
    est_summary = asyncio.run(rescore(
        transcripts_dirs=[transcripts_dir], scores_dir=tmp_path / "est", judge_override=OVERRIDE,
        output_prefix="judge_opus", config_path=config_path, briefs_dir=briefs_dir, limit=2,
    ))
    assert est_summary["n_cost_estimated"] == 2 and est_summary["scored"] == 2
    with jsonlines.open(tmp_path / "est" / "judge_opus_z00.jsonl") as r:
        assert next(iter(r))["cost_estimated"] is True

    monkeypatch.setattr(JudgePipeline, "_call_llm", zero_cost)
    summary = asyncio.run(rescore(
        transcripts_dirs=[transcripts_dir], scores_dir=tmp_path / "out", judge_override=OVERRIDE,
        output_prefix="judge_opus", config_path=config_path, briefs_dir=briefs_dir, concurrency=1,
    ))
    assert summary["aborted_zero_cost"] is True
    assert summary["zero_cost"] == ZERO_COST_LIMIT + 1
    assert summary["scored"] == ZERO_COST_LIMIT + 1
    assert summary["aborted"] == n_runs - (ZERO_COST_LIMIT + 1)


def test_run_id_include_and_exclude_filters(tmp_path: Path, sample_transcript: Transcript) -> None:
    """E1: --exclude-run-ids-file drops the stray runs; --run-ids-file picks the control subset."""
    from drift_bench.rescore import load_run_ids_file

    _write_transcript(tmp_path, sample_transcript, "a1", SONNET, "single_shot")
    _write_transcript(tmp_path, sample_transcript, "a2", SONNET, "single_shot")
    _write_transcript(tmp_path, sample_transcript, "a3", SONNET, "multi_turn_pressure")
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("# stray runs\na2\n\n a3  # trailing comment\nnot_present\n")
    ids = load_run_ids_file(ids_file)
    assert ids == {"a2", "a3", "not_present"}

    kept = load_transcript_dicts(tmp_path, exclude_run_ids=ids)
    assert [t["metadata"]["run_id"] for t in kept] == ["a1"]
    only = load_transcript_dicts(tmp_path, run_ids=ids)
    assert [t["metadata"]["run_id"] for t in only] == ["a2", "a3"]
    # Empty sets mean no filter; condition filter still applies on top.
    assert len(load_transcript_dicts(tmp_path, run_ids=set(), exclude_run_ids=set())) == 3
    both = load_transcript_dicts(tmp_path, conditions=["single_shot"], run_ids=ids, exclude_run_ids={"a3"})
    assert [t["metadata"]["run_id"] for t in both] == ["a2"]
