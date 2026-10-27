"""X5: custom prompts_dir, rubrics_path and calibration_path are honoured.

A custom directory holding one template overrides that template and
falls back to drift_bench/prompts for the rest.
"""

from pathlib import Path

import pytest
import yaml

from drift_bench.judges.judge import DEFAULT_RUBRICS_PATH, JudgePipeline
from drift_bench.judges.auditor import AuditorPipeline
from drift_bench.prompts.loader import (
    PROMPTS_DIR,
    get_default_prompts_dir,
    get_env,
    get_restatement_probe,
    get_turn_prompts,
    render_system_prompt,
    render_template,
    set_default_prompts_dir,
)
from drift_bench.runners.base import BaseRunner, Message, RunMetadata, Transcript

MARKER = "CUSTOM-SYSTEM-PROMPT-MARKER"


@pytest.fixture
def custom_prompts_dir(tmp_path: Path) -> Path:
    d = tmp_path / "prompts_custom"
    d.mkdir()
    (d / "system.j2").write_text(f"{MARKER} fidelity\n")
    return d


def test_custom_dir_overrides_one_template_and_falls_back_for_others(custom_prompts_dir: Path) -> None:
    assert MARKER in render_system_prompt(prompts_dir=custom_prompts_dir)
    assert MARKER not in render_system_prompt()
    # Not in the custom dir: comes from the built-in directory.
    assert get_restatement_probe(prompts_dir=custom_prompts_dir) == get_restatement_probe()
    assert render_template("checkpoint.j2", prompts_dir=custom_prompts_dir) == render_template("checkpoint.j2")


def test_default_env_is_unchanged_by_custom_dir(custom_prompts_dir: Path) -> None:
    get_env(custom_prompts_dir)
    assert get_env() is get_env(PROMPTS_DIR)
    assert get_env(custom_prompts_dir) is not get_env()


def test_missing_prompts_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        get_env(tmp_path / "does_not_exist")


def test_base_runner_uses_custom_system_prompt(config_path: Path, custom_prompts_dir: Path) -> None:
    runner = BaseRunner(config_path, prompts_dir=custom_prompts_dir)
    assert MARKER in runner.system_prompt
    assert runner.prompts_dir == custom_prompts_dir
    default = BaseRunner(config_path)
    assert MARKER not in default.system_prompt
    assert default.prompts_dir is None


def test_set_default_prompts_dir_reaches_turn_prompts(sample_brief: dict, tmp_path: Path) -> None:
    """The condition runners read turn prompts via the loader default (pipeline --prompts-dir)."""
    d = tmp_path / "p"
    d.mkdir()
    (d / "single_shot.j2").write_text("SWE-SINGLE-SHOT {{ brief }}\n")
    try:
        set_default_prompts_dir(d)
        assert get_default_prompts_dir() == d
        prompts = get_turn_prompts("single_shot", sample_brief)
        assert len(prompts) == 1 and prompts[0].startswith("SWE-SINGLE-SHOT")
        # Fallback still works for a template the custom dir lacks.
        assert len(get_turn_prompts("multi_turn_pressure", sample_brief)) == 6
    finally:
        set_default_prompts_dir(None)
    assert get_default_prompts_dir() == PROMPTS_DIR
    assert not get_turn_prompts("single_shot", sample_brief)[0].startswith("SWE-SINGLE-SHOT")


def test_judge_uses_custom_template_rubrics_and_calibration(
    config_path: Path, sample_brief: dict, tmp_path: Path, sample_transcript: Transcript,
) -> None:
    d = tmp_path / "prompts_swe"
    d.mkdir()
    (d / "judge.j2").write_text(
        "SWE-JUDGE\n{{ brief }}\n{{ transcript }}\n{{ final_proposal }}\n{{ rubrics }}\n{{ calibration_examples }}\n"
    )
    rubrics = tmp_path / "rubrics_swe.yaml"
    rubrics.write_text(yaml.safe_dump({"swe_rubric_key": "code must compile"}))
    calibration = tmp_path / "calibration_swe.yaml"
    calibration.write_text(yaml.safe_dump({"examples": [{
        "label": "swe-gold", "brief_summary": "b", "proposal_summary": "p",
        "scores": {"objective_fidelity": 4, "constraint_adherence": 4,
                   "alternative_coverage": 4, "complexity_inflation": 0},
        "rationale": "r",
    }]}))

    judge = JudgePipeline(config_path, prompts_dir=d, rubrics_path=rubrics, calibration_path=calibration)
    assert judge.rubrics == {"swe_rubric_key": "code must compile"}
    prompt = judge.build_prompt(sample_transcript, sample_brief)
    assert prompt.startswith("SWE-JUDGE")
    assert "swe_rubric_key" in prompt
    assert "Example (swe-gold)" in prompt
    assert "turn 2 final assistant" in prompt

    default = JudgePipeline(config_path)
    assert default.rubrics_path == DEFAULT_RUBRICS_PATH
    assert not default.build_prompt(sample_transcript, sample_brief).startswith("SWE-JUDGE")


def test_auditor_accepts_custom_paths(config_path: Path, tmp_path: Path) -> None:
    rubrics = tmp_path / "r.yaml"
    rubrics.write_text(yaml.safe_dump({"k": 1}))
    auditor = AuditorPipeline(config_path, rubrics_path=rubrics)
    assert auditor.rubrics == {"k": 1}
    assert AuditorPipeline(config_path).rubrics_path == DEFAULT_RUBRICS_PATH


def test_run_experiment_resets_default_prompts_dir(tmp_path: Path, config_path: Path, monkeypatch) -> None:
    """pipeline.run_experiment sets the loader default for its run only (try/finally)."""
    import asyncio
    from drift_bench import pipeline

    d = tmp_path / "p"
    d.mkdir()
    seen: dict = {}

    async def fake_body(**kwargs):
        seen["during"] = get_default_prompts_dir()
        raise RuntimeError("stop here")

    monkeypatch.setattr(pipeline, "_run_experiment_body", fake_body)
    with pytest.raises(RuntimeError):
        asyncio.run(pipeline.run_experiment(config_path, tmp_path, tmp_path, prompts_dir=d))
    assert seen["during"] == d
    assert get_default_prompts_dir() == PROMPTS_DIR
