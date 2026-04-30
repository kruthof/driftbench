"""Cross-family judge/auditor routing.

These tests pin the methodology: for every subject in every shipped config,
both judge and auditor must come from a different model family than the
subject (cross-family rule), AND from a different family from each other
(so judge↔auditor agreement is not a single model agreeing with itself).

The third invariant is currently violated for OpenAI and Anthropic subjects
— both routes resolve to the same model family. Those parameter cases are
marked ``xfail(strict=True)`` so the test goes XPASS once the routing is
fixed and prompts the marker to be removed.
"""

from pathlib import Path
from typing import Iterator

import pytest
import yaml

from drift_bench.judges.auditor import AuditorPipeline
from drift_bench.judges.judge import JudgePipeline


def _family(model_id: str) -> str:
    return model_id.split("/", 1)[0]


def _config_subjects(path: Path) -> list[str]:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    subjects = cfg.get("models", {}).get("subjects") or []
    return [s["id"] for s in subjects]


def _independence_xfail(subject_id: str) -> bool:
    """Subjects where the current routing returns the same family for judge and auditor."""
    return ("openai" in subject_id) or ("anthropic" in subject_id)


def _all_subject_cases(all_config_paths: list[Path]) -> Iterator[tuple[Path, str]]:
    seen: set[tuple[str, str]] = set()
    for cfg in all_config_paths:
        for subject in _config_subjects(cfg):
            key = (cfg.name, subject)
            if key in seen:
                continue
            seen.add(key)
            yield cfg, subject


def test_each_config_has_subjects(all_config_paths: list[Path]) -> None:
    assert all_config_paths, "no config files found"
    for cfg in all_config_paths:
        assert _config_subjects(cfg), f"{cfg.name} declares no subjects"


@pytest.fixture(scope="module")
def judge(config_path: Path) -> JudgePipeline:
    return JudgePipeline(config_path)


@pytest.fixture(scope="module")
def auditor(config_path: Path) -> AuditorPipeline:
    return AuditorPipeline(config_path)


def _subject_params(all_config_paths: list[Path]) -> list[pytest.param]:
    params = []
    for cfg, subject in _all_subject_cases(all_config_paths):
        params.append(pytest.param(subject, id=f"{cfg.stem}::{subject}"))
    return params


def pytest_generate_tests(metafunc):
    """Parametrise routing tests over every (config, subject) pair on disk."""
    if "subject_id" in metafunc.fixturenames:
        cfg_paths = sorted((Path(__file__).resolve().parents[1] / "drift_bench").glob("config*.yaml"))
        metafunc.parametrize(
            "subject_id",
            [pytest.param(s, id=s) for s in {sid for _, sid in _all_subject_cases(cfg_paths)}],
        )


def test_judge_is_cross_family(judge: JudgePipeline, subject_id: str) -> None:
    judge_model = judge._select_judge_model(subject_id)
    assert _family(judge_model) != _family(subject_id), (
        f"judge {judge_model} shares family with subject {subject_id}"
    )


def test_auditor_is_cross_family(auditor: AuditorPipeline, subject_id: str) -> None:
    auditor_model = auditor._select_auditor_model(subject_id)
    assert _family(auditor_model) != _family(subject_id), (
        f"auditor {auditor_model} shares family with subject {subject_id}"
    )


def test_judge_and_auditor_are_independent(
    judge: JudgePipeline,
    auditor: AuditorPipeline,
    subject_id: str,
    request: pytest.FixtureRequest,
) -> None:
    """Judge and auditor must come from different families.

    Otherwise the inter-rater reliability story collapses: kappa would
    measure the same model agreeing with itself across two prompt variants.

    Currently fails for OpenAI and Anthropic subjects — see code review.
    """
    if _independence_xfail(subject_id):
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason=(
                    "judge and auditor resolve to the same family for "
                    f"{subject_id} — routing rules in JudgePipeline and "
                    "AuditorPipeline need a single shared cross-family helper. "
                    "Remove this xfail once fixed."
                ),
            )
        )
    judge_model = judge._select_judge_model(subject_id)
    auditor_model = auditor._select_auditor_model(subject_id)
    assert _family(judge_model) != _family(auditor_model), (
        f"judge ({judge_model}) and auditor ({auditor_model}) share a family "
        f"for subject {subject_id}"
    )


def test_routing_is_deterministic(judge: JudgePipeline, subject_id: str) -> None:
    """Calling the router twice with the same subject yields the same model."""
    a = judge._select_judge_model(subject_id)
    b = judge._select_judge_model(subject_id)
    assert a == b
