"""X2: turn-indexed scoring truncates the transcript after assistant turn k.

Synthetic transcript with probe pairs after turns 1 and 2. No network:
the one scoring call patches ``_call_llm``.
"""

import asyncio
import json
from pathlib import Path

import jsonlines
import pytest

from drift_bench.judges.judge import (
    JudgePipeline,
    score_output_path,
    truncate_messages_at_turn,
    truncate_transcript_at_turn,
    truncate_transcript_dict_at_turn,
)
from drift_bench.judges.judge_structured import StructuredJudgePipeline
from drift_bench.runners.base import Message, RunMetadata, Transcript


def _three_turn_transcript() -> Transcript:
    msgs = [
        Message(role="system", content="sys"),
        Message(role="user", content="U1"),
        Message(role="assistant", content="A1"),
        Message(role="user", content="probe-q-1", is_probe=True),
        Message(role="assistant", content="probe-a-1", is_probe=True),
        Message(role="user", content="U2"),
        Message(role="assistant", content="A2"),
        Message(role="user", content="probe-q-2", is_probe=True),
        Message(role="assistant", content="probe-a-2", is_probe=True),
        Message(role="user", content="U3"),
        Message(role="assistant", content="A3"),
    ]
    meta = RunMetadata(run_id="t3", brief_id="test_01", model_id="openai/gpt-5.4",
                       condition="multi_turn_pressure")
    return Transcript(metadata=meta, messages=msgs)


@pytest.mark.parametrize("k, last_content, n_msgs", [(1, "A1", 3), (2, "A2", 7), (3, "A3", 11)])
def test_truncate_keeps_k_assistant_turns(k: int, last_content: str, n_msgs: int) -> None:
    t = _three_turn_transcript()
    kept = truncate_messages_at_turn(t.messages, k)
    assert len(kept) == n_msgs
    assert kept[-1].content == last_content
    assert not kept[-1].is_probe
    assistant_turns = [m for m in kept if m.role == "assistant" and not m.is_probe]
    assert len(assistant_turns) == k
    # Nothing after the k-th assistant message survives, probes included.
    assert all(m.content != f"probe-q-{k}" for m in kept)


def test_truncate_out_of_range_raises() -> None:
    t = _three_turn_transcript()
    with pytest.raises(ValueError):
        truncate_messages_at_turn(t.messages, 4)
    with pytest.raises(ValueError):
        truncate_messages_at_turn(t.messages, 0)


def test_truncate_transcript_does_not_mutate_original() -> None:
    t = _three_turn_transcript()
    n = len(t.messages)
    cut = truncate_transcript_at_turn(t, 1)
    assert len(t.messages) == n
    assert cut.metadata is t.metadata
    assert [m.content for m in cut.messages] == ["sys", "U1", "A1"]


def test_truncate_dict_variant_matches_object_variant() -> None:
    t = _three_turn_transcript()
    d = t.to_dict()
    cut = truncate_transcript_dict_at_turn(d, 2)
    assert [m["content"] for m in cut["messages"]] == [m.content for m in truncate_transcript_at_turn(t, 2).messages]
    assert len(d["messages"]) == 11  # original untouched
    assert cut["metadata"] == d["metadata"]


def test_judge_formatting_after_truncation(config_path: Path) -> None:
    judge = JudgePipeline(config_path)
    cut = truncate_transcript_at_turn(_three_turn_transcript(), 2)
    text = judge._format_transcript_for_judge(cut)
    assert "Turn 2 (Assistant)" in text and "A2" in text
    assert "Turn 3" not in text and "A3" not in text
    assert "probe" not in text
    assert judge._extract_final_proposal(cut) == "A2"


def test_score_output_path_infix() -> None:
    d = Path("/x")
    assert score_output_path(d, "judge", "r1", None) == d / "judge_r1.jsonl"
    assert score_output_path(d, "judge_opus", "r1", 3) == d / "judge_opus_t3_r1.jsonl"


def _fake_llm(seen: list[dict]):
    async def fake(self, model_id, messages, temperature=0.0, max_tokens=0, response_format=None):
        seen.append({"model_id": model_id, "prompt": messages[-1]["content"]})
        return {
            "content": json.dumps({"objective_fidelity": 3, "constraint_adherence": 3,
                                   "alternative_coverage": 3, "complexity_inflation": 1,
                                   "summary": "s", "violations": [], "optional_extras_flagged": []}),
            "input_tokens": 1, "output_tokens": 1, "cost": 0.0, "latency_ms": 0.5,
            "finish_reason": "stop", "response_model": "m",
        }
    return fake


def test_score_transcript_with_turn_index_writes_t_file(
    config_path: Path, sample_brief: dict, tmp_path: Path, monkeypatch,
) -> None:
    seen: list[dict] = []
    monkeypatch.setattr(JudgePipeline, "_call_llm", _fake_llm(seen))
    judge = JudgePipeline(config_path, output_prefix="judge_opus")
    t = _three_turn_transcript()

    score = asyncio.run(judge.score_transcript(t, sample_brief, tmp_path, turn_index=2))

    assert score.turn_index == 2
    prompt = seen[0]["prompt"]
    assert "A2" in prompt and "A3" not in prompt and "probe-a" not in prompt
    path = tmp_path / "judge_opus_t2_t3.jsonl"
    assert path.exists()
    with jsonlines.open(path) as r:
        record = next(iter(r))
    assert record["turn_index"] == 2
    assert record["run_id"] == "t3"
    # Untruncated scoring keeps today's file name and no turn_index.
    score_full = asyncio.run(judge.score_transcript(t, sample_brief, tmp_path))
    assert score_full.turn_index is None
    assert (tmp_path / "judge_opus_t3.jsonl").exists()
    assert "A3" in seen[1]["prompt"]


def test_structured_judge_turn_index_uses_kth_proposal(
    config_path: Path, sample_brief: dict, tmp_path: Path, monkeypatch,
) -> None:
    seen: list[dict] = []
    monkeypatch.setattr(StructuredJudgePipeline, "_call_llm", _fake_llm(seen))
    judge = StructuredJudgePipeline(config_path)
    d = _three_turn_transcript().to_dict()

    score = asyncio.run(judge.score_transcript(d, sample_brief, tmp_path, turn_index=1))

    assert score.turn_index == 1
    assert "A1" in seen[0]["prompt"] and "A2" not in seen[0]["prompt"]
    assert (tmp_path / "structured_t1_t3.jsonl").exists()
