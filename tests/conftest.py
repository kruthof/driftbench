"""Shared fixtures for drift_bench tests.

Tests are pure-function: no LLM calls, no network, no API keys required.
Fixtures point at real config / brief / prompt files so the tests catch
drift between code and data.
"""

from pathlib import Path

import pytest

from drift_bench.runners.base import Message, RunMetadata, Transcript

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRIFT_BENCH_DIR = PROJECT_ROOT / "drift_bench"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def briefs_dir() -> Path:
    return DRIFT_BENCH_DIR / "briefs"


@pytest.fixture(scope="session")
def config_path() -> Path:
    return DRIFT_BENCH_DIR / "config.yaml"


@pytest.fixture(scope="session")
def all_config_paths() -> list[Path]:
    """All config files that ship subject lists."""
    return sorted(DRIFT_BENCH_DIR.glob("config*.yaml"))


@pytest.fixture
def sample_brief() -> dict:
    """Minimal brief that satisfies the schema."""
    return {
        "id": "test_01",
        "domain": "test domain",
        "objective": "Test that the schema validator accepts a minimal brief with text long enough.",
        "hard_constraints": [
            "constraint one",
            "constraint two",
            "constraint three",
        ],
        "success_criteria": [
            "criterion one",
            "criterion two",
        ],
        "plausible_directions": [
            "direction one",
            "direction two",
        ],
        "banned_moves": [
            "banned one",
            "banned two",
        ],
    }


@pytest.fixture
def sample_transcript() -> Transcript:
    """Multi-turn transcript with a probe pair embedded after turn 1."""
    metadata = RunMetadata(
        run_id="abc123",
        brief_id="test_01",
        model_id="openai/gpt-5.4",
        condition="multi_turn_pressure",
        repetition=0,
        start_time=1000.0,
        end_time=1100.0,
        total_input_tokens=500,
        total_output_tokens=300,
        total_cost_usd=0.05,
    )
    messages = [
        Message(role="system", content="system prompt", timestamp=1000.0),
        Message(role="user", content="turn 1 user", timestamp=1001.0),
        Message(role="assistant", content="turn 1 assistant", timestamp=1002.0),
        Message(role="user", content="restate constraints", timestamp=1003.0, is_probe=True),
        Message(role="assistant", content="probe answer", timestamp=1004.0, is_probe=True),
        Message(role="user", content="turn 2 user", timestamp=1005.0),
        Message(role="assistant", content="turn 2 final assistant", timestamp=1006.0),
    ]
    return Transcript(metadata=metadata, messages=messages)
