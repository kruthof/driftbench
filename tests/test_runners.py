"""Runner internals: probe filtering, transcript serialisation, registry."""

from pathlib import Path

import jsonlines

from drift_bench.pipeline import CONDITION_RUNNERS
from drift_bench.runners.base import BaseRunner, Message, RunMetadata, Transcript


def test_build_messages_list_filters_probes(config_path: Path, sample_transcript: Transcript) -> None:
    """Probe messages must never enter the conversation context the model sees."""
    runner = BaseRunner(config_path)
    msgs = runner._build_messages_list(sample_transcript.messages)

    # No probe messages survive the filter.
    assert all("probe" not in m["content"].lower() for m in msgs)
    # Original ordering is preserved for non-probe messages.
    expected_contents = [
        m.content for m in sample_transcript.messages if not m.is_probe
    ]
    assert [m["content"] for m in msgs] == expected_contents
    # Output shape is litellm-compatible.
    assert all(set(m.keys()) == {"role", "content"} for m in msgs)


def test_build_messages_list_preserves_order(config_path: Path) -> None:
    """Filter preserves relative order even when probes are interleaved."""
    runner = BaseRunner(config_path)
    msgs = [
        Message(role="user", content="a", is_probe=False),
        Message(role="user", content="probe1", is_probe=True),
        Message(role="assistant", content="b", is_probe=False),
        Message(role="user", content="probe2", is_probe=True),
        Message(role="assistant", content="c", is_probe=False),
    ]
    out = runner._build_messages_list(msgs)
    assert [m["content"] for m in out] == ["a", "b", "c"]


def test_transcript_round_trip(tmp_path: Path, sample_transcript: Transcript) -> None:
    """save() then read back yields the same fields."""
    path = sample_transcript.save(tmp_path)
    assert path.exists()
    with jsonlines.open(path) as reader:
        record = next(iter(reader))

    assert record["metadata"]["run_id"] == sample_transcript.metadata.run_id
    assert record["metadata"]["brief_id"] == sample_transcript.metadata.brief_id
    assert record["metadata"]["condition"] == sample_transcript.metadata.condition
    assert len(record["messages"]) == len(sample_transcript.messages)
    # Probe flags survive the round-trip.
    probe_count = sum(1 for m in record["messages"] if m["is_probe"])
    assert probe_count == 2


def test_run_metadata_default_run_id_is_unique() -> None:
    """Default factory must yield distinct ids per instance (used as filename key)."""
    ids = {RunMetadata().run_id for _ in range(64)}
    assert len(ids) == 64


def test_condition_runner_registry_is_well_formed() -> None:
    """Pipeline registry must point at importable runner classes for every documented condition."""
    expected = {
        "single_shot",
        "multi_turn_neutral",
        "multi_turn_pressure",
        "multi_turn_pressure_rigor",
        "multi_turn_pressure_monitored",
        "checkpointed_pressure",
    }
    assert set(CONDITION_RUNNERS) == expected
    for cls in CONDITION_RUNNERS.values():
        assert hasattr(cls, "run"), f"{cls!r} has no run() method"


def test_provider_lookup_falls_back_to_prefix(config_path: Path) -> None:
    """Unknown model_id falls back to the slash-prefix split."""
    runner = BaseRunner(config_path)
    assert runner._get_provider("openai/gpt-5.4") == "openai"
    # An ID not in config: fall back to prefix.
    assert runner._get_provider("totally_made_up/model") == "totally_made_up"
